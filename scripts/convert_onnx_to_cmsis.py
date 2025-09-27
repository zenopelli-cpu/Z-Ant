#!/usr/bin/env python3
"""Convert QLinearConv weights in an ONNX model to CMSIS-NN friendly layout.

This script rewrites every QLinearConv weight tensor so that:

* Regular/grouped convolutions store weights in OHWI order (out, height, width, in).
* Depthwise convolutions follow the same OHWI layout so that CMSIS detects them as
  already-normalized tensors instead of performing the KHWC fallback conversion.
* Weight zero-points are folded into the tensor data and reset to zero, matching the
  runtime assumption that CMSIS receives symmetric int8 weights.

The resulting model can be consumed directly by the CMSIS execution path without
additional repacking at prediction time.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


class Layout(str):
    STANDARD = "standard"
    OHWI = "ohwi"
    DEPTHWISE_KHWC = "depthwise_khwc"


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
        if shape[1] * group == in_channels:
            return Layout.STANDARD
        if shape[3] * group == in_channels:
            return Layout.OHWI
        if group == in_channels and shape[3] == in_channels:
            return Layout.DEPTHWISE_KHWC

    if group > 1 and shape[0] <= group and shape[3] == group:
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

    if layout == Layout.STANDARD:
        out_channels, in_channels_per_group, kernel_h, kernel_w = adjusted.shape
        reordered = np.transpose(adjusted, (0, 2, 3, 1))
    elif layout == Layout.OHWI:
        out_channels, kernel_h, kernel_w, in_channels_per_group = adjusted.shape
        reordered = adjusted
    elif layout == Layout.DEPTHWISE_KHWC:
        channel_multiplier, kernel_h, kernel_w, input_channels = adjusted.shape
        out_channels = channel_multiplier * input_channels
        in_channels_per_group = 1
        swapped = np.transpose(adjusted, (3, 1, 2, 0))
        reordered = swapped.reshape(out_channels, kernel_h, kernel_w, 1)
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
        elif zp_array.size == out_channels:
            zp_array = zp_array.reshape((out_channels, 1, 1, 1))
        else:
            raise ValueError(
                f"weight zero-point size {zp_array.size} does not match output channels {out_channels}"
            )

    new_dims = [out_channels, kernel_h, kernel_w, in_channels_per_group]

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


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite QLinearConv weights for CMSIS-NN")
    parser.add_argument("--input", required=True, help="Path to the source ONNX model")
    parser.add_argument("--output", required=True, help="Destination path for the converted model")
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    model = onnx.load(args.input)
    convert_model(model)
    onnx.save(model, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
