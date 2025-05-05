const std = @import("std");
const zant = @import("../../../../zant.zig");
const Uops = zant.uops;
const UOpBuilder = Uops.UOpBuilder;
const DType = Uops.DType;
const Any = Uops.Any;

const Tensor = zant.core.tensor.Tensor; // Import Tensor type
const pkg_allocator = zant.utils.allocator.allocator;
const error_handler = zant.utils.error_handler;
const TensorMathError = error_handler.TensorMathError;
const TensorError = error_handler.TensorError;

//--------------------------------------------------------------------------
// lowerCast  ─  emit UOps for an ONNX Cast (op-set 23)
// • No shape change  →  one flat loop, same index for X and Y
// • Per-element conversion handled by a single CAST micro-op
// • ‘saturate’ flag is only relevant for float-8 targets
//--------------------------------------------------------------------------

pub fn lowerCast(
    b: *UOpBuilder,
    X_id: usize, // SSA id of input tensor X
    shape: []const usize, // logical shape  (rank-long)
    stride: []const isize, // X’s element-stride vector
    to_dtype: DType, // target scalar type (attr "to")
    saturate: bool, // attr "saturate"  (true by default)
) usize {

    // ── 1. Debug / view -------------------------------------------------
    _ = b.push(.SHAPE, .i32, &.{X_id}, null); // optional

    const id_viewX = b.push(.VIEW, to_dtype, &.{X_id}, // dtype here is
        Any{
            .view_meta = .{
                .shape = shape, // irrelevant; view
                .strides = stride,
            },
        }); // just forwards X

    // ── 2. Output buffer ----------------------------------------------
    const id_Y = b.push(.DEFINE_GLOBAL, to_dtype, &.{}, Any{ .shape = shape });

    // ---- element count ------------------------------------------------
    var nelem: usize = 1;
    for (shape) |d| nelem *= d;

    // ── 3. Flat loop over all elements --------------------------------
    const id_rng = b.push(.RANGE, .i32, &.{}, Any{ .loop_bounds = .{ .start = 0, .end = nelem } });

    const id_pX = b.push(.GEP, to_dtype, &.{ id_viewX, id_rng }, Any{ .mem_info = .{ .base = id_viewX, .offset = 0, .stride = 1 } });

    const id_vX = b.push(.LOAD, to_dtype, &.{id_pX}, null);

    const id_vY = b.push(.CAST, to_dtype, &.{id_vX}, Any{ .cast_meta = .{ .to = to_dtype, .saturate = saturate } });

    const id_pY = b.push(.GEP, to_dtype, &.{ id_Y, id_rng }, Any{ .mem_info = .{ .base = id_Y, .offset = 0, .stride = 1 } });

    _ = b.push(.STORE, to_dtype, &.{ id_pY, id_vY }, null);

    _ = b.push(.ENDRANGE, .bool, &.{id_rng}, null);

    return id_Y; // SSA id of the casted output tensor
}
