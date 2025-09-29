#!/usr/bin/env python3
"""Convert QLinearConv weights in an ONNX model to CMSIS-NN friendly layout.

This script rewrites every QLinearConv weight tensor so that:

* Regular/grouped convolutions store weights in OHWI order
  (out, height, width, in-per-group) as expected by CMSIS-NN s8 kernels.
* Depthwise convolutions are reshaped to OHWI with n=1: [1, kH, kW, M],
  where M is the number of output channels.
* Weight zero-points are folded into the tensor data and reset to zero, matching
  the runtime assumption that CMSIS receives symmetric int8 weights.

In addition to weight repacking, the converter performs graph-level fusions to
remove redundant DequantizeLinear/activation/QuantizeLinear chains following a
QLinearConv when their effect can be captured by the quantized convolution
itself.  This reduces the number of runtime buffers and improves inference
latency for CMSIS-NN backends.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


class Layout(str):
    STANDARD = "standard"
    OHWI = "ohwi"
    DEPTHWISE_KHWC = "depthwise_khwc"


logger = logging.getLogger(__name__)


def _get_attribute_i(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


def _collect_tensor_shapes(model: onnx.ModelProto) -> Dict[str, List[Optional[int]]]:
    """Collect static tensor shapes from the graph if available."""

    def _extract_shapes(graph: onnx.GraphProto, mapping: Dict[str, List[Optional[int]]]) -> None:
        for value_info in list(graph.value_info) + list(graph.input) + list(graph.output):
            if not value_info.type.HasField("tensor_type"):
                continue
            tensor_type = value_info.type.tensor_type
            dims: List[Optional[int]] = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    dims.append(int(dim.dim_value))
                else:
                    dims.append(None)
            mapping[value_info.name] = dims

    shapes: Dict[str, List[Optional[int]]] = {}
    _extract_shapes(model.graph, shapes)

    for subgraph in model.graph.node:
        for attr in subgraph.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _extract_shapes(attr.g, shapes)
    return shapes


def _detect_layout(
    shape: Tuple[int, int, int, int],
    group: int,
    in_channels: Optional[int],
) -> Layout:
    if in_channels is not None and group > 0:
        # Standard/grouped (OIHW or OHWI)
        if shape[1] * group == in_channels:
            return Layout.STANDARD
        # Depthwise cases (accept both pre-OHWI KHWC and OHWI-with-C=1)
        if group == in_channels:
            # KHWC depthwise (pre-normalized)
            if shape[3] == in_channels:
                logger.info("Detected depthwise KHWC layout for shape %s", shape)
                return Layout.DEPTHWISE_KHWC
            # OHWI depthwise variant: [M, kH, kW, 1]
            if shape[0] == in_channels and shape[3] == 1:
                logger.info("Detected depthwise OHWI layout for shape %s", shape)
                return Layout.DEPTHWISE_KHWC
        if shape[3] * group == in_channels:
            return Layout.OHWI

    if group > 1 and shape[0] <= group and shape[3] == group:
        logger.info(
            "Detected depthwise KHWC layout via fallback for shape %s group %d", shape, group
        )
        return Layout.DEPTHWISE_KHWC

    if group > 1 and shape[0] == group and shape[3] == 1:
        logger.info(
            "Detected depthwise OHWI layout via fallback for shape %s group %d", shape, group
        )
        return Layout.DEPTHWISE_KHWC

    if group > 1 and shape[1] == 1:
        return Layout.STANDARD

    if group > 0 and shape[3] * group == shape[0] * shape[1]:
        return Layout.OHWI

    return Layout.STANDARD


def _convert_weight(
    weight: onnx.TensorProto,
    weight_zp: onnx.TensorProto | None,
    group: int,
    in_channels: Optional[int],
) -> Tuple[onnx.TensorProto, onnx.TensorProto | None]:
    array = numpy_helper.to_array(weight)
    if array.ndim != 4:
        raise ValueError(f"unsupported weight rank {array.ndim} for tensor {weight.name}")

    if array.dtype not in (np.int8, np.uint8):
        raise ValueError(
            f"weights for {weight.name} must be int8/uint8, got {array.dtype}"
        )

    layout = _detect_layout(tuple(int(dim) for dim in array.shape), group, in_channels)

    adjusted = array.astype(np.int32)

    # Normalize to OHWI
    if layout == Layout.STANDARD:
        # OIHW -> OHWI
        out_channels, in_channels_per_group, kernel_h, kernel_w = adjusted.shape
        reordered = np.transpose(adjusted, (0, 2, 3, 1))
    elif layout == Layout.OHWI:
        out_channels, kernel_h, kernel_w, in_channels_per_group = adjusted.shape
        reordered = adjusted
    elif layout == Layout.DEPTHWISE_KHWC:
        # Handle depthwise variants in robust order:
        d0, kernel_h, kernel_w, d3 = adjusted.shape
        # Case A: square OHWI-like [M,kH,kW,M] with M == in_channels/group -> collapse to diagonal (1,kH,kW,M)
        if ((in_channels is not None and d0 == d3 == in_channels) or (in_channels is None and d0 == d3)):
            out_channels = d0
            in_channels_per_group = out_channels
            diag = np.zeros((1, kernel_h, kernel_w, out_channels), dtype=np.int32)
            for m in range(out_channels):
                diag[0, :, :, m] = adjusted[m, :, :, m]
            reordered = diag
        # Case B: KHWC canonical (mult, kH, kW, C_in)
        elif d3 > 1:
            channel_multiplier, _, _, input_channels = adjusted.shape
            out_channels = channel_multiplier * input_channels
            swapped = np.transpose(adjusted, (1, 2, 3, 0))  # (kH, kW, C_in, mult)
            reordered = swapped.reshape(kernel_h, kernel_w, out_channels)
            reordered = reordered.reshape(1, kernel_h, kernel_w, out_channels)
            in_channels_per_group = out_channels
        # Case C: OHWI variant [M,kH,kW,1] -> (1,kH,kW,M)
        elif d3 == 1:
            M = d0
            reordered = np.transpose(adjusted, (3, 1, 2, 0))  # (1, kH, kW, M)
            out_channels = M
            in_channels_per_group = out_channels
        else:
            # Fallback: keep OHWI
            out_channels, kernel_h, kernel_w, in_channels_per_group = adjusted.shape
            reordered = adjusted
    else:
        raise ValueError(f"unsupported weight layout for tensor {weight.name}")

    zp_original: np.ndarray | None = None
    zp_array = None
    if weight_zp is not None:
        zp_original = numpy_helper.to_array(weight_zp)
        zp_array = zp_original.astype(np.int32)

    if zp_array is not None:
        if zp_array.size == 0:
            zp_array = np.array(0, dtype=np.int32)
        elif zp_array.size == 1:
            zp_array = np.array(int(zp_array.reshape(())), dtype=np.int32)
        else:
            # Determine per-output-channel count from the reordered tensor shape
            # Standard OHWI: per-channel along axis 0; Depthwise OHWI: per-channel along axis 3
            oc = reordered.shape[0] if reordered.shape[0] != 1 else reordered.shape[3]
            if zp_array.size != oc:
                raise ValueError(
                    f"weight zero-point size {zp_array.size} does not match output channels {oc}"
                )
            if reordered.shape[0] != 1:
                zp_array = zp_array.reshape((oc, 1, 1, 1))
            else:
                zp_array = zp_array.reshape((1, 1, 1, oc))

    # Use the actual reordered shape to set dims
    new_dims = list(reordered.shape)

    if zp_array is not None:
        reordered = reordered - zp_array

    np.clip(reordered, -128, 127, out=reordered)
    adjusted_int8 = reordered.astype(np.int8)

    new_weight = numpy_helper.from_array(adjusted_int8, name=weight.name)
    new_weight.dims[:] = new_dims

    new_zp = None
    if weight_zp is not None:
        assert zp_original is not None
        zeros = np.zeros_like(zp_original)
        new_zp = numpy_helper.from_array(zeros, name=weight_zp.name)

    return new_weight, new_zp


def convert_model(model: onnx.ModelProto) -> None:
    try:
        inferred = shape_inference.infer_shapes(model)
    except Exception:
        inferred = None

    tensor_shapes: Dict[str, List[Optional[int]]] = {}
    if inferred is not None:
        tensor_shapes.update(_collect_tensor_shapes(inferred))
        original_shapes = _collect_tensor_shapes(model)
        for name, dims in original_shapes.items():
            tensor_shapes.setdefault(name, dims)
    else:
        tensor_shapes.update(_collect_tensor_shapes(model))

    initializer_map: Dict[str, onnx.TensorProto] = {init.name: init for init in model.graph.initializer}

    for node in model.graph.node:
        if node.op_type != "QLinearConv" or len(node.input) < 6:
            continue

        weight_name = node.input[3]
        zp_name = node.input[5]

        if weight_name not in initializer_map:
            raise KeyError(f"initializer {weight_name} not found for node {node.name or weight_name}")

        weight_tensor = initializer_map[weight_name]
        zp_tensor = initializer_map.get(zp_name)

        group = _get_attribute_i(node, "group", 1)
        input_shape = tensor_shapes.get(node.input[0])
        in_channels = None
        if input_shape and len(input_shape) >= 2 and input_shape[1] is not None:
            in_channels = int(input_shape[1])

        new_weight, new_zp = _convert_weight(weight_tensor, zp_tensor, group, in_channels)

        weight_tensor.CopyFrom(new_weight)
        if new_zp is not None and zp_tensor is not None:
            zp_tensor.CopyFrom(new_zp)

    # After weight normalization, apply high-level fusions to reduce runtime buffers
    _fuse_activation_chain_into_qlinearconv(model)
    _normalize_first_qlinearconv_padding(model)


def _get_initializer_map(model: onnx.ModelProto) -> Dict[str, onnx.TensorProto]:
    return {init.name: init for init in model.graph.initializer}


def _to_numpy(init: onnx.TensorProto) -> np.ndarray:
    return numpy_helper.to_array(init)


def _const_scalar_from_input(initializers: Dict[str, onnx.TensorProto], name: str) -> Optional[float]:
    if name in initializers:
        arr = _to_numpy(initializers[name])
        return float(arr.reshape(()))
    return None


def _build_consumers(model: onnx.ModelProto) -> Dict[str, List[onnx.NodeProto]]:
    cons: Dict[str, List[onnx.NodeProto]] = {}
    for n in model.graph.node:
        for inp in n.input:
            cons.setdefault(inp, []).append(n)
    return cons


def _remove_nodes(model: onnx.ModelProto, nodes: List[onnx.NodeProto]) -> None:
    keep = [n for n in model.graph.node if n not in nodes]
    del model.graph.node[:]
    model.graph.node.extend(keep)


def _prune_unused_initializers(model: onnx.ModelProto) -> None:
    used: set[str] = set()
    for n in model.graph.node:
        for name in n.input:
            used.add(name)
    keep = [init for init in model.graph.initializer if init.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)


def _single_consumer(
    consumers: Dict[str, List[onnx.NodeProto]], name: str
) -> Optional[onnx.NodeProto]:
    cs = consumers.get(name, [])
    if len(cs) == 1:
        return cs[0]
    return None


def _read_clip_bounds(
    node: onnx.NodeProto, inits: Dict[str, onnx.TensorProto]
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None
    for a in node.attribute:
        if a.name == "min":
            clip_min = float(a.f)
        elif a.name == "max":
            clip_max = float(a.f)
    if clip_min is None and len(node.input) >= 2:
        clip_min = _const_scalar_from_input(inits, node.input[1])
    if clip_max is None and len(node.input) >= 3:
        clip_max = _const_scalar_from_input(inits, node.input[2])
    return clip_min, clip_max


def _fuse_activation_chain_into_qlinearconv(model: onnx.ModelProto) -> None:
    """Fuse QLinearConv -> DequantizeLinear -> (Clip|Relu) -> QuantizeLinear chains.

    The fused QLinearConv adopts the QuantizeLinear scale/zero-point so that the
    quantized accumulator saturates to the same range that the activation
    enforced in floating point.  This mirrors the behaviour of ReLU and
    ReLU6-like activations for unsigned outputs while eliminating the need for
    additional buffers.
    """
    inits = _get_initializer_map(model)
    consumers = _build_consumers(model)

    to_remove: List[onnx.NodeProto] = []
    for conv in list(model.graph.node):
        if conv.op_type != "QLinearConv" or len(conv.output) == 0:
            continue
        conv_out = conv.output[0]
        dlin = _single_consumer(consumers, conv_out)
        if dlin is None or dlin.op_type != "DequantizeLinear" or len(dlin.output) == 0:
            continue

        act = _single_consumer(consumers, dlin.output[0])
        if act is None or len(act.output) == 0:
            continue

        activation_kind: Optional[str] = None
        clip_bounds: Tuple[Optional[float], Optional[float]] = (None, None)

        if act.op_type == "Clip":
            clip_bounds = _read_clip_bounds(act, inits)
            if clip_bounds is None:
                continue
            clip_min, clip_max = clip_bounds
            if clip_min is None or clip_max is None:
                continue
            # Only accept non-negative clamps; the upper bound is optional but
            # must be finite.  This covers ReLU (max=None) and ReLU6-style
            # activations.
            if clip_min < -1e-5:
                continue
            if clip_max <= 0.0:
                continue
            activation_kind = "clip"
        elif act.op_type == "Relu":
            activation_kind = "relu"
            clip_bounds = (0.0, None)
        else:
            continue

        qlin = _single_consumer(consumers, act.output[0])
        if qlin is None or qlin.op_type != "QuantizeLinear" or len(qlin.input) < 3:
            continue

        # Ensure the convolution provides output scale/zero-point tensors.
        if len(conv.input) < 8:
            continue

        old_y_scale_name = conv.input[6]
        new_y_scale_name = qlin.input[1]

        if old_y_scale_name not in inits or new_y_scale_name not in inits:
            continue

        old_scale_arr = numpy_helper.to_array(inits[old_y_scale_name]).astype(np.float32)
        new_scale_arr = numpy_helper.to_array(inits[new_y_scale_name]).astype(np.float32)

        if old_scale_arr.size != 1 or new_scale_arr.size != 1:
            continue

        old_scale = float(old_scale_arr.reshape(()))
        new_scale = float(new_scale_arr.reshape(()))

        if old_scale <= 0.0 or new_scale <= 0.0:
            continue

        # The QuantizeLinear output zero-point must correspond to the
        # activation's lower bound.  For ReLU and Clip(0,x) this is typically 0
        # for uint8 tensors.
        new_y_zp_name = qlin.input[2]
        if new_y_zp_name not in inits:
            continue
        new_y_zp_arr = numpy_helper.to_array(inits[new_y_zp_name])
        if new_y_zp_arr.size != 1:
            continue
        if activation_kind in {"relu", "clip"}:
            zp_val = float(new_y_zp_arr.reshape(()))
            # Enforce non-negative lower bounds; otherwise the activation cannot
            # be represented by quantized saturation.
            if zp_val < -1e-5:
                continue

        scale_ratio = new_scale / old_scale

        w_scale_name = conv.input[4]
        if w_scale_name not in inits:
            continue
        w_scale_arr = numpy_helper.to_array(inits[w_scale_name]).astype(np.float32)
        w_scale_arr *= scale_ratio

        if len(conv.input) >= 9:
            bias_name = conv.input[8]
            if bias_name in inits:
                bias_arr = numpy_helper.to_array(inits[bias_name]).astype(np.float64)
                bias_arr *= (old_scale / new_scale)
                bias_arr = np.round(bias_arr)
                bias_arr = np.clip(
                    bias_arr,
                    np.iinfo(np.int32).min,
                    np.iinfo(np.int32).max,
                ).astype(np.int32)
                inits[bias_name] = numpy_helper.from_array(bias_arr, bias_name)
                for idx, init in enumerate(model.graph.initializer):
                    if init.name == bias_name:
                        del model.graph.initializer[idx]
                        model.graph.initializer.insert(idx, inits[bias_name])
                        break

        inits[w_scale_name] = numpy_helper.from_array(w_scale_arr, w_scale_name)
        for idx, init in enumerate(model.graph.initializer):
            if init.name == w_scale_name:
                del model.graph.initializer[idx]
                model.graph.initializer.insert(idx, inits[w_scale_name])
                break

        conv.input[6] = new_y_scale_name
        conv.input[7] = new_y_zp_name
        conv.output[0] = qlin.output[0]

        to_remove.extend([dlin, act, qlin])

    if to_remove:
        _remove_nodes(model, to_remove)
        _prune_unused_initializers(model)


def _normalize_first_qlinearconv_padding(model: onnx.ModelProto) -> None:
    """Make first QLinearConv padding symmetric to avoid odd effective input sizes.

    If pads attribute is [t,l,b,r] and asymmetric, set it to [max(t,b), max(l,r), max(t,b), max(l,r)].
    """
    for node in model.graph.node:
        if node.op_type != "QLinearConv":
            continue
        for a in node.attribute:
            if a.name == "pads" and len(a.ints) >= 4:
                t, l, b, r = list(a.ints[:4])
                st = max(t, b)
                sl = max(l, r)
                if (t, l, b, r) != (st, sl, st, sl):
                    a.ints[:] = [st, sl, st, sl]
                return


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite QLinearConv weights for CMSIS-NN")
    parser.add_argument("--input", required=True, help="Path to the source ONNX model")
    parser.add_argument("--output", required=True, help="Destination path for the converted model")
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    model = onnx.load(args.input)
    convert_model(model)
    onnx.save(model, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
