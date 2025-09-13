//! The aim of this is to detect well-defined sequences of nodes and sobsitute them with the fused version of those.
//! For Example:
//! Imagine in your graph you have the following sequence
//!     ...
//!     DeQuantLinear
//!     Pad
//!     Conv
//!     QuantLinear
//!     ...
//! it can be sobsitute with:
//!     ...
//!     qlinarConv
//!     ...
//! qlinarConv is a known operation the ONNX format, but for example if I want I can fuse the following:
//!     ...
//!     Conv
//!     Relu
//!     ...
//! into a custom operation called fused_Conv_Relu.
//!
//! RULES:
//!     - The name of the operations must be a concatenation of the string "fused" with names of the operations in the same order of the fusion, so
//!     the sequence: opA -> opB -> opC will be fused, if possible, into "fused_opA_opB_opC"
//!     - The pattern matcher will only fuse a math operation into an already existing math kernel
//!     - The names of the input and output tensors will be the same of the input and output tensors of the fused sequence
//!

