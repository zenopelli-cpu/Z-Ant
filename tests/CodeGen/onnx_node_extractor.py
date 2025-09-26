#!/usr/bin/env python3
"""
ONNX Node Extractor

This script takes an ONNX neural network and:
1. Extracts each node as a separate ONNX model
2. For a given input, computes and saves the input/output values for each node
3. Saves everything in organized folders with JSON metadata
4. Maintains proper types, especially for quantized operations (QLinearOps output uint8)
"""

import onnx
import onnxruntime as ort
import numpy as np
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
import logging
import re
from onnx import numpy_helper

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ONNXNodeExtractor:
    def __init__(self, model_path: str, output_dir: str = None):
        self.model_path = Path(model_path)
        # If no output directory specified, use the same folder as the model
        if output_dir is None:
            self.output_dir = self.model_path.parent / "extracted_nodes"
        else:
            self.output_dir = Path(output_dir)
        self.model = None
        self.session = None
        self.intermediate_values = {}
        # Cache for tensor type information
        self.tensor_type_cache = {}
        
    def sanitize_filename(self, name: str) -> str:
        """Sanitize a string to be safe for use as a filename"""
        if not name:
            return "unnamed"
        
        # Replace problematic characters with underscores _
        # This includes: / \ : * ? " < > | . -
        sanitized = re.sub(r'[/\\:*?"<>|.\-]', '_', name)
        
        # Replace multiple consecutive underscores with single underscore
        sanitized = re.sub(r'_+', '_', sanitized)
        
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        
        # Ensure it's not empty after sanitization
        if not sanitized:
            return "unnamed"
            
        # Limit length to avoid filesystem issues
        return sanitized[:100]
    
    def load_model(self):
        """Load the ONNX model and build tensor type cache"""
        logger.info(f"Loading ONNX model from {self.model_path}")
        self.model = onnx.load(str(self.model_path))
        onnx.checker.check_model(self.model)
        
        # Create inference session
        self.session = ort.InferenceSession(str(self.model_path))
        logger.info(f"Model loaded successfully. Found {len(self.model.graph.node)} nodes")
        
        # Build tensor type cache from the original model
        self._build_tensor_type_cache()
        
    def _build_tensor_type_cache(self):
        """Build a cache of tensor names to their types from the original model"""
        logger.info("Building tensor type cache...")
        
        # Add input types
        for input_info in self.model.graph.input:
            self.tensor_type_cache[input_info.name] = input_info.type.tensor_type.elem_type
            
        # Add output types  
        for output_info in self.model.graph.output:
            self.tensor_type_cache[output_info.name] = output_info.type.tensor_type.elem_type
            
        # Add value_info types
        for value_info in self.model.graph.value_info:
            self.tensor_type_cache[value_info.name] = value_info.type.tensor_type.elem_type
            
        # Add initializer types
        for initializer in self.model.graph.initializer:
            self.tensor_type_cache[initializer.name] = initializer.data_type
            
        # Infer types for node outputs based on operation types
        for node in self.model.graph.node:
            for output_name in node.output:
                if output_name and output_name not in self.tensor_type_cache:
                    inferred_type = self._infer_tensor_type_from_op(node, output_name)
                    self.tensor_type_cache[output_name] = inferred_type
                    logger.debug(f"Inferred type {self._type_to_string(inferred_type)} for tensor '{output_name}' from {node.op_type}")
        
        logger.info(f"Built tensor type cache with {len(self.tensor_type_cache)} entries")
    
    def _type_to_string(self, tensor_type: int) -> str:
        """Convert ONNX tensor type to readable string"""
        type_map = {
            onnx.TensorProto.FLOAT: "float32",
            onnx.TensorProto.UINT8: "uint8", 
            onnx.TensorProto.INT8: "int8",
            onnx.TensorProto.UINT16: "uint16",
            onnx.TensorProto.INT16: "int16",
            onnx.TensorProto.INT32: "int32",
            onnx.TensorProto.INT64: "int64",
            onnx.TensorProto.STRING: "string",
            onnx.TensorProto.BOOL: "bool",
            onnx.TensorProto.FLOAT16: "float16",
            onnx.TensorProto.DOUBLE: "float64",
            onnx.TensorProto.UINT32: "uint32",
            onnx.TensorProto.UINT64: "uint64"
        }
        return type_map.get(tensor_type, f"unknown({tensor_type})")
    
    def create_output_directories(self):
        """Create organized output directory structure"""
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "individual_nodes").mkdir(exist_ok=True)
        (self.output_dir / "node_data").mkdir(exist_ok=True)
        
    def get_intermediate_outputs(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get intermediate outputs for all nodes in the network"""
        logger.info("Computing intermediate outputs for all nodes...")
        
        try:
            # For quantized models, we need to run inference node by node
            # instead of trying to create a single model with all outputs
            intermediate_outputs = {}
            
            # First, get the final outputs normally
            outputs = self.session.run(None, input_data)
            output_names = [output.name for output in self.session.get_outputs()]
            intermediate_outputs.update(dict(zip(output_names, outputs)))
            
            # Try the original approach first for non-quantized models
            model_with_outputs = onnx.ModelProto()
            model_with_outputs.CopyFrom(self.model)
            
            # Add all intermediate tensors as outputs with correct types
            existing_outputs = {output.name for output in model_with_outputs.graph.output}
            intermediate_tensors_added = []
            
            for node in model_with_outputs.graph.node:
                for output_name in node.output:
                    if output_name and output_name not in existing_outputs:
                        # Use cached type information to create proper value_info
                        tensor_type = self.tensor_type_cache.get(output_name, onnx.TensorProto.FLOAT)
                        
                        # Create value_info with correct type
                        value_info = onnx.helper.make_tensor_value_info(
                            output_name, tensor_type, None
                        )
                        
                        model_with_outputs.graph.output.append(value_info)
                        intermediate_tensors_added.append(output_name)
                        logger.debug(f"Added intermediate output '{output_name}' with type {self._type_to_string(tensor_type)}")
            
            # Save temporary model and run inference
            temp_model_path = self.output_dir / "temp_model.onnx"
            onnx.save(model_with_outputs, temp_model_path)
            
            temp_session = ort.InferenceSession(str(temp_model_path))
            temp_outputs = temp_session.run(None, input_data)
            temp_output_names = [output.name for output in temp_session.get_outputs()]
            
            intermediate_outputs.update(dict(zip(temp_output_names, temp_outputs)))
            logger.info(f"Successfully computed {len(intermediate_outputs)} intermediate outputs")
            
        except Exception as e:
            logger.warning(f"Failed to get all intermediate outputs using unified approach: {e}")
            logger.info("Falling back to cumulative execution for quantized model...")
            
            # Fallback: run cumulative models (slower but works for quantized models)
            try:
                intermediate_outputs = self._get_intermediate_outputs_cumulative(input_data)
            except Exception as e2:
                logger.warning(f"Cumulative approach also failed: {e2}")
                # Final fallback: just get final outputs
                outputs = self.session.run(None, input_data)
                output_names = [output.name for output in self.session.get_outputs()]
                intermediate_outputs = dict(zip(output_names, outputs))
        
        finally:
            # Clean up temporary file
            temp_model_path = self.output_dir / "temp_model.onnx"
            if temp_model_path.exists():
                temp_model_path.unlink()
        
        return intermediate_outputs

    def _infer_tensor_type_from_op(self, node: onnx.NodeProto, output_name: str) -> int:
        """Infer the tensor type for operations, especially quantized ones"""
        # QLinear operations that output uint8
        qlinear_ops_uint8 = {
            'QuantizeLinear', 'QLinearConv', 'QLinearMatMul', 'QLinearAdd',
            'QLinearMul', 'QLinearAveragePool', 'QLinearGlobalAveragePool',
            'QLinearConcat', 'QLinearLeakyRelu', 'QLinearSigmoid'
        }
        
        # Operations that typically output int8 (signed quantized)
        qlinear_ops_int8 = {
            # Add any specific ops that output int8 if needed
        }
        
        # Operations that output float32
        float_ops = {
            'DequantizeLinear', 'Conv', 'MatMul', 'Add', 'Mul', 'Relu', 'Sigmoid',
            'AveragePool', 'GlobalAveragePool', 'MaxPool', 'BatchNormalization',
            'Softmax', 'Reshape', 'Transpose', 'Concat', 'Split', 'Squeeze', 'Unsqueeze'
        }
        
        if node.op_type in qlinear_ops_uint8:
            return onnx.TensorProto.UINT8
        elif node.op_type in qlinear_ops_int8:
            return onnx.TensorProto.INT8
        elif node.op_type in float_ops:
            return onnx.TensorProto.FLOAT
        elif node.op_type == 'Cast':
            # For Cast operations, check the 'to' attribute
            for attr in node.attribute:
                if attr.name == 'to':
                    return attr.i
            return onnx.TensorProto.FLOAT  # Default if no 'to' attribute
        else:
            # For unknown operations, try to infer from input types
            # If all inputs are quantized, output might be quantized too
            input_types = []
            for input_name in node.input:
                if input_name in self.tensor_type_cache:
                    input_types.append(self.tensor_type_cache[input_name])
            
            if all(t in [onnx.TensorProto.UINT8, onnx.TensorProto.INT8] for t in input_types):
                return onnx.TensorProto.UINT8  # Default quantized output
            else:
                return onnx.TensorProto.FLOAT  # Default to float

    def _get_intermediate_outputs_cumulative(self, input_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Get intermediate outputs by creating cumulative models for quantized networks"""
        intermediate_outputs = {}
        
        logger.info("Running cumulative execution for quantized model...")
        
        # For quantized models, we'll create cumulative models that include nodes up to each point
        for target_node_idx in range(len(self.model.graph.node)):
            try:
                # Create a model that includes all nodes up to target_node_idx
                cumulative_model = self._create_cumulative_model(target_node_idx)
                
                if cumulative_model is None:
                    logger.warning(f"Skipping cumulative model up to node {target_node_idx}")
                    continue
                
                # Save and run the cumulative model
                temp_model_path = self.output_dir / f"temp_cumulative_{target_node_idx}.onnx"
                onnx.save(cumulative_model, temp_model_path)
                
                try:
                    cumulative_session = ort.InferenceSession(str(temp_model_path))
                    
                    # Run with original inputs
                    outputs = cumulative_session.run(None, input_data)
                    output_names = [output.name for output in cumulative_session.get_outputs()]
                    
                    # Store the final output of this cumulative model
                    target_node = self.model.graph.node[target_node_idx]
                    for output_name in target_node.output:
                        if output_name in output_names:
                            output_idx = output_names.index(output_name)
                            actual_output = outputs[output_idx]
                            intermediate_outputs[output_name] = actual_output
                            
                            # Verify the output type matches expected
                            expected_type = self.tensor_type_cache.get(output_name, onnx.TensorProto.FLOAT)
                            expected_type_name = self._type_to_string(expected_type)
                            
                            logger.debug(f"Node {target_node_idx} ({target_node.op_type}): "
                                       f"output '{output_name}' - expected: {expected_type_name}, "
                                       f"actual: {actual_output.dtype}")
                    
                    logger.debug(f"Successfully executed cumulative model up to node {target_node_idx} ({target_node.op_type})")
                    
                except Exception as e:
                    logger.warning(f"Failed to execute cumulative model up to node {target_node_idx}: {e}")
                
                finally:
                    # Clean up temp file
                    if temp_model_path.exists():
                        temp_model_path.unlink()
                        
            except Exception as e:
                logger.warning(f"Failed to create cumulative model up to node {target_node_idx}: {e}")
                continue
        
        return intermediate_outputs

    def _create_cumulative_model(self, target_node_idx: int) -> onnx.ModelProto:
        """Create a model that includes all nodes up to and including target_node_idx"""
        try:
            # Create new graph with nodes up to target_node_idx
            included_nodes = self.model.graph.node[:target_node_idx + 1]
            target_node = self.model.graph.node[target_node_idx]
            
            # Create new graph
            new_graph = onnx.helper.make_graph(
                nodes=included_nodes,
                name=f"cumulative_up_to_{target_node_idx}",
                inputs=self.model.graph.input,  # Keep original inputs
                outputs=[]
            )
            
            # Add all original initializers
            new_graph.initializer.extend(self.model.graph.initializer)
            
            # Add all original value_info
            new_graph.value_info.extend(self.model.graph.value_info)
            
            # Set the target node's outputs as the graph outputs with correct types
            for output_name in target_node.output:
                if output_name:
                    # Use cached type information
                    tensor_type = self.tensor_type_cache.get(output_name, onnx.TensorProto.FLOAT)
                    
                    # Try to find existing value info first
                    value_info = None
                    for vi in self.model.graph.value_info:
                        if vi.name == output_name:
                            value_info = vi
                            break
                    
                    # If not found in value_info, check the original graph outputs
                    if value_info is None:
                        for vi in self.model.graph.output:
                            if vi.name == output_name:
                                value_info = vi
                                break
                    
                    if value_info is None:
                        # Create value info with correct cached type
                        value_info = onnx.helper.make_tensor_value_info(
                            output_name, tensor_type, None
                        )
                        logger.debug(f"Created value_info for {output_name} with cached type {self._type_to_string(tensor_type)} for {target_node.op_type}")
                    else:
                        # Verify the type matches our cache
                        existing_type = value_info.type.tensor_type.elem_type
                        if existing_type != tensor_type:
                            logger.debug(f"Type mismatch for {output_name}: existing={self._type_to_string(existing_type)}, cached={self._type_to_string(tensor_type)}")
                    
                    new_graph.output.append(value_info)
            
            # Create model
            new_model = onnx.helper.make_model(new_graph)
            new_model.opset_import.extend(self.model.opset_import)
            
            return new_model
            
        except Exception as e:
            logger.error(f"Failed to create cumulative model up to node {target_node_idx}: {e}")
            return None
    
    def extract_single_node(self, node_idx: int, node: onnx.NodeProto, input_data: Dict[str, np.ndarray], intermediate_outputs: Dict[str, np.ndarray]) -> Tuple[str, Dict[str, Any]]:
        """Extract a single node as an individual ONNX model with correct types"""
        logger.info(f"Extracting node {node_idx}: {node.op_type}")
        
        # Create new graph with just this node
        new_graph = onnx.helper.make_graph(
            nodes=[node],
            name=f"single_node_{node_idx}_{node.op_type}",
            inputs=[],  # Will be filled below
            outputs=[]  # Will be filled below
        )
        
        # Find input and output value infos with correct types
        input_value_infos = []
        output_value_infos = []
        
        # Get input value infos with proper types
        for input_name in node.input:
            if input_name:  # Skip empty strings
                value_info = self._find_value_info_with_type(input_name)
                if value_info:
                    input_value_infos.append(value_info)
        
        # Get output value infos with proper types
        for output_name in node.output:
            if output_name:  # Skip empty strings
                value_info = self._find_value_info_with_type(output_name)
                if value_info:
                    output_value_infos.append(value_info)
        
        # Determine which input stays as runtime input: first non-initializer
        initializer_names = {init.name for init in self.model.graph.initializer}
        kept_input_name = None
        for input_name in node.input:
            if input_name and input_name not in initializer_names:
                kept_input_name = input_name
                break

        # Update graph outputs with correct types
        new_graph.output.extend(output_value_infos)

        # Copy relevant initializers that are directly referenced
        for initializer in self.model.graph.initializer:
            if initializer.name in node.input:
                new_graph.initializer.append(initializer)

        # For each input: keep one as real input, convert others to constants
        for vi in input_value_infos:
            inp_name = vi.name
            if kept_input_name is not None and inp_name == kept_input_name:
                new_graph.input.append(vi)
                continue

            # If already provided as initializer above, skip
            if any(init.name == inp_name for init in new_graph.initializer):
                continue

            # Try to materialize as constant from intermediate outputs or provided input data
            arr = None
            if inp_name in intermediate_outputs:
                arr = intermediate_outputs[inp_name]
            elif inp_name in input_data:
                arr = input_data[inp_name]
            else:
                # Fallback: create zeros using value_info shape and type if available
                shape = []
                tt = vi.type.tensor_type
                if tt.shape.dim:
                    for dim in tt.shape.dim:
                        if dim.dim_value:
                            shape.append(dim.dim_value)
                        else:
                            shape.append(1)
                else:
                    shape = [1]
                
                # Use the correct dtype based on tensor type
                tensor_type = tt.elem_type
                if tensor_type == onnx.TensorProto.UINT8:
                    arr = np.zeros(shape, dtype=np.uint8)
                elif tensor_type == onnx.TensorProto.INT8:
                    arr = np.zeros(shape, dtype=np.int8)
                elif tensor_type == onnx.TensorProto.INT32:
                    arr = np.zeros(shape, dtype=np.int32)
                elif tensor_type == onnx.TensorProto.INT64:
                    arr = np.zeros(shape, dtype=np.int64)
                else:
                    arr = np.zeros(shape, dtype=np.float32)

            try:
                # Create tensor with correct type preservation
                if arr.dtype == np.uint8:
                    tensor_proto = numpy_helper.from_array(arr, name=inp_name)
                elif arr.dtype == np.int8:
                    tensor_proto = numpy_helper.from_array(arr, name=inp_name)
                elif arr.dtype in [np.int32, np.int64]:
                    tensor_proto = numpy_helper.from_array(arr, name=inp_name)
                else:
                    # For float types, ensure float32
                    tensor_proto = numpy_helper.from_array(arr.astype(np.float32, copy=False), name=inp_name)
                
                new_graph.initializer.append(tensor_proto)
                logger.debug(f"Created constant for '{inp_name}' with dtype {arr.dtype}")
                
            except Exception as e:
                logger.warning(f"Failed to bake constant for input '{inp_name}' of node {node_idx}: {e}. Leaving as graph input.")
                new_graph.input.append(vi)
         
        # Create new model
        new_model = onnx.helper.make_model(new_graph)
        new_model.opset_import.extend(self.model.opset_import)
        
        # Save the individual node model
        sanitized_node_name = self.sanitize_filename(node.name if node.name else "unnamed")
        node_filename = f"node_{node_idx:03d}_{node.op_type}_{sanitized_node_name}.onnx"
        node_path = self.output_dir / "individual_nodes" / node_filename
        
        try:
            onnx.save(new_model, node_path)
            
            # Create metadata with type information
            metadata = {
                "node_index": node_idx,
                "op_type": node.op_type,
                "node_name": node.name if node.name else "unnamed",
                "inputs": [{"name": inp, "type": self._type_to_string(self.tensor_type_cache.get(inp, onnx.TensorProto.FLOAT))} for inp in node.input if inp],
                "outputs": [{"name": out, "type": self._type_to_string(self.tensor_type_cache.get(out, onnx.TensorProto.FLOAT))} for out in node.output if out],
                "attributes": self._extract_attributes(node),
                "model_path": str(node_path.relative_to(self.output_dir))
            }
            
            logger.info(f"Successfully extracted node {node_idx} with proper types")
            return node_filename, metadata
            
        except Exception as e:
            logger.error(f"Failed to extract node {node_idx}: {e}")
            return None, None
    
    def _find_value_info_with_type(self, tensor_name: str) -> onnx.ValueInfoProto:
        """Find value info for a tensor by name, using cached type information"""
        # Check in graph inputs
        for value_info in self.model.graph.input:
            if value_info.name == tensor_name:
                return value_info
        
        # Check in graph outputs
        for value_info in self.model.graph.output:
            if value_info.name == tensor_name:
                return value_info
        
        # Check in value_info
        for value_info in self.model.graph.value_info:
            if value_info.name == tensor_name:
                return value_info
        
        # Check in initializers and create value info with correct type
        for initializer in self.model.graph.initializer:
            if initializer.name == tensor_name:
                return onnx.helper.make_tensor_value_info(
                    tensor_name, initializer.data_type, initializer.dims
                )
        
        # Create value info with cached type or default to float
        cached_type = self.tensor_type_cache.get(tensor_name, onnx.TensorProto.FLOAT)
        return onnx.helper.make_tensor_value_info(
            tensor_name, cached_type, None
        )
    
    def _extract_attributes(self, node: onnx.NodeProto) -> Dict[str, Any]:
        """Extract node attributes to a serializable format"""
        attributes = {}
        for attr in node.attribute:
            attr_value = onnx.helper.get_attribute_value(attr)
            # Convert non-serializable types to JSON-serializable formats
            if isinstance(attr_value, np.ndarray):
                attr_value = attr_value.tolist()
            elif isinstance(attr_value, bytes):
                attr_value = attr_value.decode('utf-8', errors='ignore')
            elif hasattr(attr_value, '__iter__') and not isinstance(attr_value, (str, bytes)):
                # Handle lists/tuples that might contain numpy arrays or bytes
                try:
                    attr_value = [
                        item.tolist() if isinstance(item, np.ndarray) 
                        else item.decode('utf-8', errors='ignore') if isinstance(item, bytes)
                        else item
                        for item in attr_value
                    ]
                except (TypeError, AttributeError):
                    # If conversion fails, convert to string
                    attr_value = str(attr_value)
            attributes[attr.name] = attr_value
        return attributes
    
    def save_node_data(self, input_data: Dict[str, np.ndarray], 
                      intermediate_outputs: Dict[str, np.ndarray]):
        """Save input/output data for each node with proper type information"""
        logger.info("Saving node input/output data...")
        
        for node_idx, node in enumerate(self.model.graph.node):
            node_data = {
                "name": node.name if node.name else "unnamed",
                "type": "exact",
                "input": [],
                "output": [],
                "expected_class": 0,
            }
            
            # Determine the kept input (first non-initializer), and save its data
            kept_input_name = None
            initializer_names = {init.name for init in self.model.graph.initializer}
            for input_name in node.input:
                if input_name and input_name not in initializer_names:
                    kept_input_name = input_name
                    break

            if kept_input_name:
                if kept_input_name in input_data:
                    node_data["input"] = input_data[kept_input_name].flatten().tolist()
                elif kept_input_name in intermediate_outputs:
                    node_data["input"] = intermediate_outputs[kept_input_name].flatten().tolist()
                else:
                    for initializer in self.model.graph.initializer:
                        if initializer.name == kept_input_name:
                            tensor_data = onnx.numpy_helper.to_array(initializer)
                            node_data["input"] = tensor_data.flatten().tolist()
                            # node_data["input_types"] = [str(tensor_data.dtype)]
                            break
             
            # Collect output data with type information
            for output_name in node.output:
                if output_name and output_name in intermediate_outputs:
                    output_data = intermediate_outputs[output_name]
                    node_data["output"] = output_data.flatten().tolist()
                    # node_data["output_types"] = [str(output_data.dtype)]
                    
                    # Verify QLinear operations output uint8
                    expected_type = self.tensor_type_cache.get(output_name, onnx.TensorProto.FLOAT)
                    expected_dtype = self._onnx_type_to_numpy_dtype(expected_type)
                    if expected_dtype != output_data.dtype:
                        logger.warning(f"Type mismatch for {node.op_type} output '{output_name}': "
                                     f"expected {expected_dtype}, got {output_data.dtype}")

            # Save node data
            sanitized_node_name = self.sanitize_filename(node.name if node.name else "unnamed")
            data_filename = f"node_{node_idx:03d}_{node.op_type}_{sanitized_node_name}_data.json"
            data_path = self.output_dir / "node_data" / data_filename
            
            with open(data_path, 'w') as f:
                json.dump([node_data], f, indent=2)
        
        logger.info(f"Saved data for {len(self.model.graph.node)} nodes")
    
    def _onnx_type_to_numpy_dtype(self, onnx_type: int) -> np.dtype:
        """Convert ONNX tensor type to numpy dtype"""
        type_map = {
            onnx.TensorProto.FLOAT: np.float32,
            onnx.TensorProto.UINT8: np.uint8,
            onnx.TensorProto.INT8: np.int8,
            onnx.TensorProto.UINT16: np.uint16,
            onnx.TensorProto.INT16: np.int16,
            onnx.TensorProto.INT32: np.int32,
            onnx.TensorProto.INT64: np.int64,
            onnx.TensorProto.BOOL: np.bool_,
            onnx.TensorProto.FLOAT16: np.float16,
            onnx.TensorProto.DOUBLE: np.float64,
            onnx.TensorProto.UINT32: np.uint32,
            onnx.TensorProto.UINT64: np.uint64
        }
        return type_map.get(onnx_type, np.float32)
    
    def generate_random_input(self) -> Dict[str, np.ndarray]:
        """Generate random input data based on model input specifications with correct data types"""
        input_data = {}
        
        for input_info in self.model.graph.input:
            # Skip inputs that are initializers (weights/biases)
            if any(init.name == input_info.name for init in self.model.graph.initializer):
                continue
            
            # Get shape from type info
            shape = []
            if input_info.type.tensor_type.shape.dim:
                for dim in input_info.type.tensor_type.shape.dim:
                    if dim.dim_value:
                        shape.append(dim.dim_value)
                    else:
                        # Use a default size for dynamic dimensions
                        shape.append(1)
            else:
                # Default shape if not specified
                shape = [1, 3, 224, 224]  # Common for image models
            
            # Get data type and generate appropriate data
            elem_type = input_info.type.tensor_type.elem_type
            
            if elem_type == onnx.TensorProto.FLOAT:
                data = np.random.randn(*shape).astype(np.float32)
                logger.info(f"Generated float32 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.DOUBLE:
                data = np.random.randn(*shape).astype(np.float64)
                logger.info(f"Generated float64 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.INT8:
                data = np.random.randint(-128, 128, shape).astype(np.int8)
                logger.info(f"Generated int8 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.UINT8:
                data = np.random.randint(0, 256, shape).astype(np.uint8)
                logger.info(f"Generated uint8 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.INT16:
                data = np.random.randint(-32768, 32768, shape).astype(np.int16)
                logger.info(f"Generated int16 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.UINT16:
                data = np.random.randint(0, 65536, shape).astype(np.uint16)
                logger.info(f"Generated uint16 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.INT32:
                data = np.random.randint(-1000, 1000, shape).astype(np.int32)
                logger.info(f"Generated int32 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.UINT32:
                data = np.random.randint(0, 1000, shape).astype(np.uint32)
                logger.info(f"Generated uint32 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.INT64:
                data = np.random.randint(0, 100, shape).astype(np.int64)
                logger.info(f"Generated int64 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.UINT64:
                data = np.random.randint(0, 100, shape).astype(np.uint64)
                logger.info(f"Generated uint64 input '{input_info.name}' with shape {shape}")
            elif elem_type == onnx.TensorProto.FLOAT16:
                data = np.random.randn(*shape).astype(np.float16)
                logger.info(f"Generated float16 input '{input_info.name}' with shape {shape}")
            else:
                # Default to float32 for unknown types
                data = np.random.randn(*shape).astype(np.float32)
                logger.warning(f"Unknown data type {elem_type}, defaulting to float32 for input '{input_info.name}' with shape {shape}")
            
            input_data[input_info.name] = data
        
        return input_data
    
    def run_extraction(self, input_data: Dict[str, np.ndarray] = None):
        """Main method to run the complete extraction process"""
        self.load_model()
        self.create_output_directories()
        
        # Use provided input data or generate random data
        if input_data is None:
            input_data = self.generate_random_input()
        
        # Get intermediate outputs
        intermediate_outputs = self.get_intermediate_outputs(input_data)
        
        # Verify quantized operation outputs
        self._verify_quantized_outputs(intermediate_outputs)
        
        # Extract individual nodes
        extracted_nodes = []
        for node_idx, node in enumerate(self.model.graph.node):
            node.name = node.name if node.name else "unnamed"
            filename, metadata = self.extract_single_node(node_idx, node, input_data, intermediate_outputs)
            if metadata:
                extracted_nodes.append(metadata)
        
        # Save node input/output data
        self.save_node_data(input_data, intermediate_outputs)
        
        # Convert extracted_nodes to JSON-serializable format
        json_serializable_nodes = []
        for node_info in extracted_nodes:
            serializable_node = {}
            for key, value in node_info.items():
                if isinstance(value, bytes):
                    # Convert bytes to string or skip
                    serializable_node[key] = value.decode('utf-8', errors='ignore')
                elif isinstance(value, np.ndarray):
                    # Convert numpy arrays to lists
                    serializable_node[key] = value.tolist()
                elif hasattr(value, '__dict__'):
                    # For complex objects, convert to string representation
                    serializable_node[key] = str(value)
                else:
                    serializable_node[key] = value
            json_serializable_nodes.append(serializable_node)

        # Save summary with type information
        summary = {
            "original_model": str(self.model_path),
            "total_nodes": len(self.model.graph.node),
            "extracted_nodes": len(extracted_nodes),
            "input_shape": {name: list(data.shape) for name, data in input_data.items()},
            "input_types": {name: str(data.dtype) for name, data in input_data.items()},
            "tensor_type_cache_size": len(self.tensor_type_cache),
            "quantized_operations": self._count_quantized_operations(),
            "nodes": json_serializable_nodes  # Use the serializable version
        }

        summary_path = self.output_dir / "extraction_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Extraction complete! Results saved to {self.output_dir}")
        logger.info(f"- Individual node models: {len(extracted_nodes)}")
        logger.info(f"- Node data files: {len(self.model.graph.node)}")
        logger.info(f"- Summary: {summary_path}")
        logger.info(f"- Quantized operations found: {summary['quantized_operations']}")
    
    def _verify_quantized_outputs(self, intermediate_outputs: Dict[str, np.ndarray]):
        """Verify that QLinear operations output uint8 as expected"""
        logger.info("Verifying quantized operation outputs...")
        
        qlinear_ops = {
            'QuantizeLinear', 'QLinearConv', 'QLinearMatMul', 'QLinearAdd',
            'QLinearMul', 'QLinearAveragePool', 'QLinearGlobalAveragePool',
            'QLinearConcat', 'QLinearLeakyRelu', 'QLinearSigmoid'
        }
        
        verification_results = []
        
        for node in self.model.graph.node:
            if node.op_type in qlinear_ops:
                for output_name in node.output:
                    if output_name and output_name in intermediate_outputs:
                        output_data = intermediate_outputs[output_name]
                        expected_type = np.uint8
                        actual_type = output_data.dtype
                        
                        result = {
                            "node_type": node.op_type,
                            "output_name": output_name,
                            "expected_dtype": str(expected_type),
                            "actual_dtype": str(actual_type),
                            "matches": actual_type == expected_type
                        }
                        
                        verification_results.append(result)
                        
                        if not result["matches"]:
                            logger.warning(f"QLinear operation {node.op_type} output '{output_name}' "
                                         f"has dtype {actual_type}, expected {expected_type}")
                        else:
                            logger.debug(f"✓ QLinear operation {node.op_type} output '{output_name}' "
                                       f"correctly has dtype {actual_type}")
        
        # Save verification results
        if verification_results:
            verification_path = self.output_dir / "quantized_verification.json"
            with open(verification_path, 'w') as f:
                json.dump(verification_results, f, indent=2)
            
            passed = sum(1 for r in verification_results if r["matches"])
            total = len(verification_results)
            logger.info(f"Quantized operation verification: {passed}/{total} passed")
    
    def _count_quantized_operations(self) -> Dict[str, int]:
        """Count quantized operations in the model"""
        qlinear_ops = {
            'QuantizeLinear', 'QLinearConv', 'QLinearMatMul', 'QLinearAdd',
            'QLinearMul', 'QLinearAveragePool', 'QLinearGlobalAveragePool',
            'QLinearConcat', 'QLinearLeakyRelu', 'QLinearSigmoid', 'DequantizeLinear'
        }
        
        op_counts = {}
        for node in self.model.graph.node:
            if node.op_type in qlinear_ops:
                op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        
        return op_counts


def main():
    parser = argparse.ArgumentParser(description="Extract individual nodes from ONNX neural network with proper type handling")
    parser.add_argument("--model", help=" Name of your model. Automatic path is datasets/models/my_model/my_model.onnx")
    parser.add_argument("-o", "--output", help="Output directory (default: same folder as model)")
    parser.add_argument("--input-data", help="Path to numpy file with input data (optional)")
    parser.add_argument("--verify-types", action="store_true", help="Enable extra type verification logging")

    args = parser.parse_args()

    model_name = args.model
    model_path = f"datasets/models/{model_name}/{model_name}.onnx"
    
    # Set debug logging if verification requested
    if args.verify_types:
        logging.getLogger(__name__).setLevel(logging.DEBUG)
    
    # Load custom input data if provided
    input_data = None
    if args.input_data:
        input_data = np.load(args.input_data, allow_pickle=True).item()
        logger.info(f"Loaded custom input data from {args.input_data}")
    
    # Run extraction
    extractor = ONNXNodeExtractor(model_path, args.output)
    extractor.run_extraction(input_data)


if __name__ == "__main__":
    main()