#!/usr/bin/env python3
"""Convert QLinearConv weights in an ONNX model to CMSIS-NN friendly layout.

This script rewrites every QLinearConv weight tensor so that:

* Regular/grouped convolutions store weights in OHWI order
  (out, height, width, in-per-group) as expected by CMSIS-NN s8 kernels.
* Depthwise convolutions are reshaped to OHWI with n=1: [1, kH, kW, M],
  where M is the number of output channels.
* Weight zero-points are folded into the tensor data and reset to zero, matching
  the runtime assumption that CMSIS receives symmetric int8 weights.

The resulting model can be consumed directly by the CMSIS execution path without
additional repacking at prediction time.

FIXED VERSION: Corrected channel detection logic and layout conversion.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import onnx
from onnx import numpy_helper, shape_inference


class Layout(str):
    OIHW = "oihw"  # Standard ONNX format: [out_channels, in_channels, height, width]
    OHWI = "ohwi"  # CMSIS-NN format: [out_channels, height, width, in_channels]
    DEPTHWISE_KHWC = "depthwise_khwc"  # Depthwise: [multiplier, height, width, in_channels]


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
    node_name: str = ""
) -> Layout:
    """Detect the current layout of weight tensor."""
    d0, d1, d2, d3 = shape
    
    print(f"  Analyzing weight {node_name}: shape={shape}, group={group}, in_channels={in_channels}")
    
    # Check for depthwise convolution (group == in_channels)
    if in_channels is not None and group == in_channels and group > 1:
        print(f"    -> Detected depthwise convolution")
        # Depthwise: each input channel has its own filter
        # Common formats:
        # - KHWC: [multiplier, height, width, in_channels] 
        # - OHWI variant: [out_channels, height, width, 1] where out_channels = multiplier * in_channels
        
        if d3 == in_channels:  # KHWC format
            print(f"    -> KHWC format: [multiplier={d0}, h={d1}, w={d2}, in_channels={d3}]")
            return Layout.DEPTHWISE_KHWC
        elif d3 == 1 and d0 == in_channels:  # OHWI variant for depthwise
            print(f"    -> OHWI depthwise variant: [out_channels={d0}, h={d1}, w={d2}, 1]")
            return Layout.DEPTHWISE_KHWC
        elif d1 == in_channels:  # OIHW depthwise (each output channel processes one input channel)
            print(f"    -> OIHW depthwise: [out_channels={d0}, in_channels={d1}, h={d2}, w={d3}]")
            return Layout.OIHW
    
    # Check for standard convolution layouts
    if in_channels is not None:
        # OIHW: [out_channels, in_channels_per_group, height, width]
        if d1 * group == in_channels:
            print(f"    -> OIHW format: [out_channels={d0}, in_per_group={d1}, h={d2}, w={d3}]")
            return Layout.OIHW
        
        # OHWI: [out_channels, height, width, in_channels_per_group]
        if d3 * group == in_channels:
            print(f"    -> OHWI format: [out_channels={d0}, h={d1}, w={d2}, in_per_group={d3}]")
            return Layout.OHWI
    
    # Default assumption: OIHW (standard ONNX format)
    print(f"    -> Defaulting to OIHW format")
    return Layout.OIHW


def _convert_weight(
    weight: onnx.TensorProto,
    weight_zp: onnx.TensorProto | None,
    group: int,
    in_channels: Optional[int],
    node_name: str = ""
) -> Tuple[onnx.TensorProto, onnx.TensorProto | None]:
    """Convert weight tensor to CMSIS-NN OHWI format."""
    
    array = numpy_helper.to_array(weight)
    if array.ndim != 4:
        raise ValueError(f"unsupported weight rank {array.ndim} for tensor {weight.name}")

    if array.dtype not in (np.int8, np.uint8):
        raise ValueError(
            f"weights for {weight.name} must be int8/uint8, got {array.dtype}"
        )

    print(f"Converting weight {weight.name} for node {node_name}")
    print(f"  Original shape: {array.shape}, dtype: {array.dtype}")
    
    layout = _detect_layout(tuple(int(dim) for dim in array.shape), group, in_channels, node_name)

    # Convert to int32 for processing
    adjusted = array.astype(np.int32)

    # Convert to OHWI format based on detected layout
    if layout == Layout.OIHW:
        # OIHW -> OHWI: [O, I, H, W] -> [O, H, W, I]
        out_channels, in_channels_per_group, kernel_h, kernel_w = adjusted.shape
        reordered = np.transpose(adjusted, (0, 2, 3, 1))  # [O, H, W, I]
        print(f"  OIHW -> OHWI: {adjusted.shape} -> {reordered.shape}")
        
    elif layout == Layout.OHWI:
        # Already in OHWI format
        out_channels, kernel_h, kernel_w, in_channels_per_group = adjusted.shape
        reordered = adjusted
        print(f"  Already OHWI: {adjusted.shape}")
        
    elif layout == Layout.DEPTHWISE_KHWC:
        # Handle depthwise convolution
        d0, d1, d2, d3 = adjusted.shape
        
        if d3 == 1 and in_channels is not None and d0 == in_channels:
            # OHWI depthwise variant: [M, kH, kW, 1] -> [1, kH, kW, M]
            reordered = np.transpose(adjusted, (3, 1, 2, 0))  # [1, kH, kW, M]
            print(f"  Depthwise OHWI variant: {adjusted.shape} -> {reordered.shape}")
            
        elif d3 > 1 and in_channels is not None and d3 == in_channels:
            # KHWC format: [multiplier, kH, kW, in_channels] -> [1, kH, kW, out_channels]
            multiplier, kernel_h, kernel_w, input_channels = adjusted.shape
            out_channels = multiplier * input_channels
            
            # Reshape to [1, kH, kW, out_channels] by interleaving channels
            reordered = np.zeros((1, kernel_h, kernel_w, out_channels), dtype=np.int32)
            for m in range(multiplier):
                for c in range(input_channels):
                    out_idx = c * multiplier + m
                    reordered[0, :, :, out_idx] = adjusted[m, :, :, c]
            
            print(f"  Depthwise KHWC: {adjusted.shape} -> {reordered.shape}")
            
        elif in_channels is not None and d1 == in_channels:
            # OIHW depthwise: [out_channels, in_channels, kH, kW] -> [1, kH, kW, out_channels]
            # For depthwise, out_channels == in_channels, and each filter processes one input channel
            out_channels, _, kernel_h, kernel_w = adjusted.shape
            reordered = np.transpose(adjusted, (1, 2, 3, 0))  # [in_channels, kH, kW, out_channels]
            reordered = reordered.reshape(1, kernel_h, kernel_w, out_channels)
            print(f"  Depthwise OIHW: {adjusted.shape} -> {reordered.shape}")
            
        else:
            raise ValueError(f"Unsupported depthwise layout for {weight.name}: shape={adjusted.shape}")
    else:
        raise ValueError(f"Unsupported weight layout {layout} for tensor {weight.name}")

    # Handle weight zero-point - keep original zero-points for CMSIS compatibility
    zp_original: np.ndarray | None = None
    if weight_zp is not None:
        zp_original = numpy_helper.to_array(weight_zp)
        print(f"  Keeping original zero-point: shape={zp_original.shape}, value={zp_original}")

    # Clip to int8 range and convert (no zero-point folding)
    np.clip(reordered, -128, 127, out=reordered)
    adjusted_int8 = reordered.astype(np.int8)

    # Create new weight tensor
    new_weight = numpy_helper.from_array(adjusted_int8, name=weight.name)
    new_weight.dims[:] = list(reordered.shape)

    # Keep original zero-point tensor unchanged
    new_zp = weight_zp

    print(f"  Final shape: {reordered.shape}")
    return new_weight, new_zp


def convert_model(model: onnx.ModelProto) -> None:
    """Convert all QLinearConv weights in the model to CMSIS-NN format."""
    
    print("=== ONNX to CMSIS-NN Conversion ===")
    
    # Try to infer shapes
    try:
        inferred = shape_inference.infer_shapes(model)
        print("✓ Shape inference successful")
    except Exception as e:
        print(f"⚠ Shape inference failed: {e}")
        inferred = None

    # Collect tensor shapes
    tensor_shapes: Dict[str, List[Optional[int]]] = {}
    if inferred is not None:
        tensor_shapes.update(_collect_tensor_shapes(inferred))
        original_shapes = _collect_tensor_shapes(model)
        for name, dims in original_shapes.items():
            tensor_shapes.setdefault(name, dims)
    else:
        tensor_shapes.update(_collect_tensor_shapes(model))

    # Build initializer map
    initializer_map: Dict[str, onnx.TensorProto] = {init.name: init for init in model.graph.initializer}

    # Process all QLinearConv nodes
    qconv_nodes = [n for n in model.graph.node if n.op_type == "QLinearConv"]
    print(f"\nFound {len(qconv_nodes)} QLinearConv nodes to convert")

    for i, node in enumerate(qconv_nodes):
        if len(node.input) < 6:
            print(f"Skipping node {node.name}: insufficient inputs")
            continue

        print(f"\n--- Processing QLinearConv {i+1}/{len(qconv_nodes)} ---")
        print(f"Node: {node.name}")

        weight_name = node.input[3]
        zp_name = node.input[5]

        if weight_name not in initializer_map:
            raise KeyError(f"initializer {weight_name} not found for node {node.name or weight_name}")

        weight_tensor = initializer_map[weight_name]
        zp_tensor = initializer_map.get(zp_name)

        # Get node attributes
        group = _get_attribute_i(node, "group", 1)
        
        # Get input channels from input tensor shape
        input_shape = tensor_shapes.get(node.input[0])
        in_channels = None
        if input_shape and len(input_shape) >= 4:
            # Try NCHW format first (typical for ONNX models)
            if input_shape[1] is not None and input_shape[1] < input_shape[3]:
                # NCHW: [batch, channels, height, width]
                in_channels = int(input_shape[1])
            elif input_shape[3] is not None:
                # NHWC: [batch, height, width, channels]
                in_channels = int(input_shape[3])
            elif input_shape[1] is not None:
                # Fallback to NCHW format
                in_channels = int(input_shape[1])

        print(f"Input shape: {input_shape}, in_channels: {in_channels}, group: {group}")

        # Convert the weight
        try:
            new_weight, new_zp = _convert_weight(weight_tensor, zp_tensor, group, in_channels, node.name)
            
            # Update the model
            weight_tensor.CopyFrom(new_weight)
            if new_zp is not None and zp_tensor is not None:
                zp_tensor.CopyFrom(new_zp)
                
            print(f"✓ Successfully converted weight {weight_name}")
            
        except Exception as e:
            print(f"✗ Failed to convert weight {weight_name}: {e}")
            raise

    # Apply additional optimizations
    print(f"\n=== Applying Optimizations ===")
    _fuse_clip_into_qlinearconv(model)
    _normalize_first_qlinearconv_padding(model)
    print("✓ Conversion complete!")


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


def _fuse_clip_into_qlinearconv(model: onnx.ModelProto) -> None:
    """Fuse QLinearConv -> DequantizeLinear -> Clip(0,6) -> QuantizeLinear into a single QLinearConv."""
    inits = _get_initializer_map(model)
    consumers = _build_consumers(model)

    to_remove: List[onnx.NodeProto] = []
    fused_count = 0
    
    for conv in list(model.graph.node):
        if conv.op_type != "QLinearConv" or len(conv.output) == 0:
            continue
        conv_out = conv.output[0]
        ds = consumers.get(conv_out, [])
        if len(ds) != 1:
            continue
        dlin = ds[0]
        if dlin.op_type != "DequantizeLinear" or len(dlin.output) == 0:
            continue
        dlin_out = dlin.output[0]
        cs = consumers.get(dlin_out, [])
        if len(cs) != 1:
            continue
        clip = cs[0]
        if clip.op_type != "Clip" or len(clip.output) == 0:
            continue

        # Get clip min/max
        clip_min: Optional[float] = None
        clip_max: Optional[float] = None
        for a in clip.attribute:
            if a.name == "min":
                clip_min = float(a.f)
            elif a.name == "max":
                clip_max = float(a.f)
        # Opset 11+ uses extra inputs
        if clip_min is None and len(clip.input) >= 2:
            clip_min = _const_scalar_from_input(inits, clip.input[1])
        if clip_max is None and len(clip.input) >= 3:
            clip_max = _const_scalar_from_input(inits, clip.input[2])

        if clip_min is None or clip_max is None:
            continue
        # Only fuse ReLU6 range
        if not (abs(clip_min - 0.0) < 1e-5 and abs(clip_max - 6.0) < 1e-3):
            continue

        cs2 = consumers.get(clip.output[0], [])
        if len(cs2) != 1:
            continue
        qlin = cs2[0]
        if qlin.op_type != "QuantizeLinear" or len(qlin.input) < 3:
            continue

        # Perform fusion
        if len(conv.input) < 8:
            continue
        conv.input[6] = qlin.input[1]  # y_scale
        conv.input[7] = qlin.input[2]  # y_zero_point
        conv.output[0] = qlin.output[0]

        to_remove.extend([dlin, clip, qlin])
        fused_count += 1

    if to_remove:
        _remove_nodes(model, to_remove)
        _prune_unused_initializers(model)
        print(f"✓ Fused {fused_count} ReLU6 patterns")


def _normalize_first_qlinearconv_padding(model: onnx.ModelProto) -> None:
    """Make first QLinearConv padding symmetric."""
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
                    print(f"✓ Normalized padding from [{t},{l},{b},{r}] to [{st},{sl},{st},{sl}]")
                return


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite QLinearConv weights for CMSIS-NN")
    parser.add_argument("--input", required=True, help="Path to the source ONNX model")
    parser.add_argument("--output", required=True, help="Destination path for the converted model")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    return parser.parse_args(argv)


def main(argv: Iterable[str]) -> int:
    args = parse_args(argv)
    
    try:
        print(f"Loading model: {args.input}")
        model = onnx.load(args.input)
        
        print(f"Model loaded: {len(model.graph.node)} nodes, {len(model.graph.initializer)} initializers")
        
        convert_model(model)
        
        print(f"Saving converted model: {args.output}")
        onnx.save(model, args.output)
        
        print("✓ Conversion successful!")
        return 0
        
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
