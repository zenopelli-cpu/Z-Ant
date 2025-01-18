//! These operations reduce a tensor to a smaller shape by aggregating values. Like:
//!    Sum: Compute the sum of elements along specific dimensions.
//!   Mean: Compute the average.
//!    Min/Max: Find the minimum or maximum value.
//!    Prod: Compute the product of elements.
//!    Standard Deviation and Variance: Statistical operations.
//!
const std = @import("std");
const Tensor = @import("tensor").Tensor; // Import Tensor type
const pkg_allocator = @import("pkgAllocator").allocator;
const TensorMathError = @import("errorHandler").TensorMathError;
const Converter = @import("typeC");

/// Performs the mean of a given tensor. It is a reduction operation, collapsing the whole tenosr into a single value.
pub fn mean(comptime T: anytype, tensor: *Tensor(T)) f32 {
    var res: f32 = 0;

    for (tensor.data) |*d| {
        res += Converter.convert(T, f32, d.*);
    }
    res = res / Converter.convert(usize, f32, tensor.size);
    return res;
}
