"""Metrics on target graphs, communication and counterfactual seed pairs."""

from collections import defaultdict
from itertools import combinations

import torch

from scripts.audit_dataset import measurements, scores
from sdr_mamba.data_v2 import PIXEL_BINS, WINDOW_BINS
from sdr_mamba.model_v2 import raw_readout_gradient, transported_sources, zero_neighbor
from sdr_mamba.scan import scan
from sdr_mamba.state_transport import initial_states


def rms(tensor):
    return None if tensor is None else tensor.detach().square().mean().sqrt().item()


def mask_bins(distance, bins, outside=False):
    for lo, hi in bins:
        yield (f"{lo}-{hi}" if hi is not None else f"{lo}+",
               (distance >= lo) if hi is None else (distance >= lo) & (distance <= hi))
    if outside:
        yield "outside", distance < 0


def gradients(model, trace, loss_by_frame, final_loss, metadata):
    rows = []
    for index, (frame, readout_loss) in enumerate(zip(trace, loss_by_frame)):
        current = raw_readout_gradient(readout_loss, frame).reshape_as(frame.state)
        views = {"next": frame.state, "initial": frame.initial, "tokens": frame.tokens,
                 "received_features": frame.received_features}
        keys = [key for key, value in views.items() if value is not None]
        back = dict(zip(keys, torch.autograd.grad(final_loss, [views[key] for key in keys],
                                                  retain_graph=True, allow_unused=True)))
        b, hg, wg, *_ = frame.state.shape
        token_grad = back["tokens"].detach().unflatten(0, (b, hg, wg, 4))
        scanned_norm = scan(frame.normalized.detach())
        # Each slice is the exact directional VJP into the shared projection.
        weights = torch.einsum("bhwdlo,bhwdli->doi", token_grad, scanned_norm)
        biases = token_grad.sum((0, 1, 2, 4))
        for direction in range(4):
            received = frame.initial if frame.initial is not None else frame.received_features
            sent = frame.state if frame.sent_features is None else frame.sent_features
            incoming = None if received is None else rms(received[:, :, :, direction])
            outgoing = rms(sent[:, :, :, direction])

            def direction_grad(name):
                gradient = back.get(name)
                return rms(None if gradient is None else gradient[:, :, :, direction])

            earlier = None if index == 0 else trace[index - 1]
            source = None if earlier is None else (
                earlier.state if earlier.sent_features is None else earlier.sent_features
            )
            earlier_rms = None if source is None else rms(source[:, :, :, direction])
            rows.append({
                **metadata, "round": index + 1, "direction": direction,
                "current_raw_readout_grad_rms": rms(current[:, :, :, direction]),
                "future_state_transport_grad_rms": direction_grad("next"),
                "initial_state_grad_rms": direction_grad("initial"),
                "feature_message_grad_rms": direction_grad("received_features"),
                "final_state_rms": rms(frame.state[:, :, :, direction]),
                "received_rms": incoming, "sent_rms": outgoing,
                "received_to_previous_sent_rms": incoming / earlier_rms if earlier_rms and incoming is not None else None,
                "directional_shared_projection_weight_grad_l2": weights[direction].norm().item(),
                "directional_shared_projection_bias_grad_l2": biases[direction].norm().item(),
            })
    return rows


class DiagnosticTotals:
    def __init__(self):
        self._interventions = defaultdict(lambda: [0.0] * 5)
        self._norms = defaultdict(lambda: [0.0] * 5)

    @torch.no_grad()
    def add_interventions(self, model, trace, graph_distance):
        for turn in range(2, min(len(trace), 4) + 1):
            frame, previous = trace[turn - 1], trace[turn - 2].state
            true_probability = frame.logits.sigmoid()
            true_prediction = frame.logits >= 0
            sources = transported_sources(previous)
            for variant, initial, other in (
                ("zero_neighbor", zero_neighbor(previous, model.gate, model.mixing), torch.zeros_like(sources)),
                ("shuffled_neighbor", initial_states(previous, "shuffled", model.gate, model.mixing),
                 transported_sources(previous, shuffled=True)),
            ):
                alternative = model.step(frame.features, initial)
                delta = (true_probability - alternative.logits.sigmoid()).abs()
                flips = true_prediction != (alternative.logits >= 0)
                for label, mask in mask_bins(graph_distance, WINDOW_BINS, outside=True):
                    windows = mask[:, ::model.k, ::model.k]
                    key = (turn, variant, label)
                    size = int(windows.sum()) * sources.shape[3] * sources.shape[4] * sources.shape[5] * sources.shape[6]
                    self.add(self._interventions[key], (
                        delta[mask].sum().item(), int(flips[mask].sum()), int(mask.sum()),
                        (sources[windows] - other[windows]).square().sum().item(), size,
                    ))

    @torch.no_grad()
    def add_norms(self, trace, graph_distance, k):
        graph = graph_distance[:, ::k, ::k]
        for turn, frame in enumerate(trace, 1):
            received = frame.initial if frame.initial is not None else frame.received_features
            previous = trace[turn - 2] if turn > 1 else None
            prior = None if previous is None else (
                previous.state if previous.sent_features is None else previous.sent_features
            )
            sent = frame.state if frame.sent_features is None else frame.sent_features
            final = frame.state
            for label, mask in mask_bins(graph, WINDOW_BINS, outside=True):
                for direction in range(4):
                    state = final[..., direction, :, :, :][mask]
                    self.add(self._norms[turn, label, direction], (
                        state.square().sum().item(),
                        sent[..., direction, :, :, :][mask].square().sum().item(),
                        0 if received is None else received[..., direction, :, :, :][mask].square().sum().item(),
                        0 if prior is None else prior[..., direction, :, :, :][mask].square().sum().item(),
                        state.numel(),
                    ))

    @staticmethod
    def add(total, values):
        for i, value in enumerate(values):
            total[i] += value

    def interventions(self, info):
        return [
            {**info, "round": turn, "replacement": variant, "graph_bucket": label,
             "mean_abs_probability_delta": delta / count if count else None,
             "flip_rate": flips / count if count else None,
             "neighbor_state_delta_rms": (norm / n_state) ** .5 if n_state else None,
             "pixels": int(count)}
            for (turn, variant, label), (delta, flips, count, norm, n_state) in self._interventions.items()
        ]

    def state_norms(self, info):
        return [
            {**info, "round": turn, "graph_bucket": label, "direction": direction,
             "final_state_rms": (state / count) ** .5 if count else None,
             "sent_message_rms": (sent / count) ** .5 if count else None,
             "received_message_rms": (received / count) ** .5 if count else None,
             "received_to_previous_local_sent_rms": (received / prior) ** .5 if prior else None,
             "elements": count}
            for (turn, label, direction), (state, sent, received, prior, count) in self._norms.items()
        ]


def paired_rows(predictions, targets, metadata):
    result = []
    for map_id, (pred, label) in enumerate(zip(predictions.unflatten(0, (-1, 3)),
                                               targets.unflatten(0, (-1, 3)))):
        for left, right in combinations(range(3), 2):
            change = pred[left] ^ pred[right]
            gt_change = label[left] ^ label[right]
            result.append({
                **metadata, "map_id": map_id, "seed_slot_a": left, "seed_slot_b": right,
                "iou_a": scores(pred[left], label[left])["iou"],
                "iou_b": scores(pred[right], label[right])["iou"],
                "swapped_iou_a": scores(pred[left], label[right])["iou"],
                "swapped_iou_b": scores(pred[right], label[left])["iou"],
                "gt_change_rate": gt_change.float().mean().item(),
                "prediction_change_rate": change.float().mean().item(),
                "change_mask_iou": scores(change, gt_change)["iou"],
            })
    return result


def distance_rows(pred, target, data, metadata):
    rows = []
    for axis, bins in (("window", WINDOW_BINS), ("pixel", PIXEL_BINS)):
        for label, mask in mask_bins(data[axis], bins, outside=True):
            counts = scores(pred[mask], target[mask])
            if axis == "pixel" and label != "outside":
                # Pixel BFS distances are defined only for target pixels.
                for key in ("iou", "dice", "precision"):
                    counts[key] = None
            rows.append({**metadata, "distance_axis": axis, "bucket": label,
                         "pixels": int(mask.sum()), **counts})
    return rows


def component_rows(predictions, targets, metadata):
    target = targets.unflatten(0, (-1, 3))
    pred = predictions.unflatten(0, (-1, 3))
    rows = []
    for index, (p, groups) in enumerate(zip(pred, target)):
        for seed_slot, component_slot in ((i, j) for i in range(3) for j in range(3)):
            label = groups[component_slot]
            counts = scores(p[seed_slot], label)
            rows.append({
                **metadata, "map_id": index, "seed_slot": seed_slot, "component_slot": component_slot,
                "role": "target" if seed_slot == component_slot else "distractor", **counts,
            })
    return rows
