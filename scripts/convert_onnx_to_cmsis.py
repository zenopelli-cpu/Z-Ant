#!/usr/bin/env python3
"""Convert QLinearConv weights in an ONNX model to CMSIS-NN friendly layout.

This script rewrites every QLinearConv weight tensor so that:

* Regular/grouped convolutions store weights in OHWI order (out, height, width, in).
* Depthwise convolutions store weights in [1, height, width, out] order as expected by
  the CMSIS-NN depthwise wrapper.
* Weight zero-points are folded into the tensor data and reset to zero, matching the
  runtime assumption that CMSIS receives symmetric int8 weights.

The resulting model can be consumed directly by the CMSIS execution path without
additional repacking at prediction time.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, Tuple

import numpy as np
import onnx
from onnx import numpy_helper


def _get_attribute_i(node: onnx.NodeProto, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


def _broadcast_zero_point(
    zp: np.ndarray, out_channels: int, *, depthwise: bool
) -> np.ndarray:
    if zp.size == 0:
        return np.array(0, dtype=np.int32)
    if zp.size == 1:
        return np.array(int(zp.reshape(())), dtype=np.int32)
    if zp.size != out_channels:
        raise ValueError(
            f"weight zero-point size {zp.size} does not match output channels {out_channels}"
        )

    if depthwise:
        return zp.reshape((1, 1, 1, out_channels)).astype(np.int32)
    return zp.reshape((out_channels, 1, 1, 1)).astype(np.int32)


def _convert_weight(
    weight: onnx.TensorProto,
    weight_zp: onnx.TensorProto | None,
    group: int,
) -> Tuple[onnx.TensorProto, onnx.TensorProto | None]:
    array = numpy_helper.to_array(weight)
    if array.ndim != 4:
        raise ValueError(f"unsupported weight rank {array.ndim} for tensor {weight.name}")

    if array.dtype not in (np.int8, np.uint8):
        raise ValueError(
            f"weights for {weight.name} must be int8/uint8, got {array.dtype}"
        )

    out_channels, in_channels, kernel_h, kernel_w = array.shape
    depthwise = group == out_channels and in_channels == 1

    if depthwise:
        reordered = array.transpose(1, 2, 3, 0)  # [1, kh, kw, Cout]
        new_dims = [1, kernel_h, kernel_w, out_channels]
    else:
        reordered = array.transpose(0, 2, 3, 1)  # [O, kh, kw, I]
        new_dims = [out_channels, kernel_h, kernel_w, in_channels]

    zp_array = None
    if weight_zp is not None:
        zp_array = numpy_helper.to_array(weight_zp).astype(np.int32)

    broadcast = (
        _broadcast_zero_point(zp_array, out_channels, depthwise=depthwise)
        if zp_array is not None
        else np.array(0, dtype=np.int32)
    )

    adjusted = reordered.astype(np.int32) - broadcast
    np.clip(adjusted, -128, 127, out=adjusted)
    adjusted_int8 = adjusted.astype(np.int8)

    new_weight = numpy_helper.from_array(adjusted_int8, name=weight.name)
    new_weight.dims[:] = new_dims

    new_zp = None
    if weight_zp is not None:
        zeros = np.zeros_like(zp_array, dtype=np.int8)
        new_zp = numpy_helper.from_array(zeros, name=weight_zp.name)

    return new_weight, new_zp


def convert_model(model: onnx.ModelProto) -> None:
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

        new_weight, new_zp = _convert_weight(weight_tensor, zp_tensor, group)

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
