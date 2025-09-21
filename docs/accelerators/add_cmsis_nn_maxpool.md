### Add a new CMSIS-NN op (example: MaxPool) and integrate it

This repo already wires CMSIS-NN Helium for quantized convolution (`arm_convolve_s8`) via a thin Zig shim and build glue. Here’s how to add another CMSIS-NN op end-to-end using MaxPool as the concrete example.

What you’ll touch:
- `src/Core/Tensor/Accelerators/stm32n6/cmsis_nn.zig` (externs + feature probes)
- `build.zig` (add CMSIS-NN Pooling sources)
- `src/Core/Tensor/TensorMath/op_maxPool.zig` (dispatch to accelerated path)

Prereqs:
- Build with STM32N6 accelerator and CMSIS-NN enabled, e.g.:
  - `-Dstm32n6_accel=true -Dstm32n6_use_cmsis=true`
  - CMSIS include paths are auto-detected from `third_party/` or can be overridden with `-Dstm32n6_cmsis_path=/path/to/CMSIS`.

---

## 1) Add CMSIS-NN MaxPool externs

Extend the STM32N6 CMSIS-NN shim to expose pooling. Mirror the `conv` pattern to keep linkage and stubs centralized.

Edit `src/Core/Tensor/Accelerators/stm32n6/cmsis_nn.zig` and add a PoolParams struct plus a new `pool` namespace with `arm_max_pool_s8`.

Insert near the existing ConvParams/conv block:

```zig
pub const PoolParams = extern struct {
    stride: extern struct { h: i32, w: i32 },
    padding: extern struct { h: i32, w: i32 },
    activation: extern struct { min: i32, max: i32 },
};

pub const pool = if (is_enabled) struct {
    extern fn arm_max_pool_s8(
        ctx: *const Context,
        pool_params: *const PoolParams,
        input_dims: *const Dims,
        input_data: [*]const i8,
        filter_dims: *const Dims,
        output_dims: *const Dims,
        output_data: [*]i8,
    ) callconv(.C) i32;
} else struct {
    pub fn arm_max_pool_s8(
        ctx: *const Context,
        pool_params: *const PoolParams,
        input_dims: *const Dims,
        input_data: [*]const i8,
        filter_dims: *const Dims,
        output_dims: *const Dims,
        output_data: [*]i8,
    ) callconv(.C) i32 {
        _ = ctx;
        _ = pool_params;
        _ = input_dims;
        _ = input_data;
        _ = filter_dims;
        _ = output_dims;
        _ = output_data;
        return 0;
    }
};

pub inline fn supportsMaxPoolS8() bool {
    return is_enabled;
}
```

Notes:
- CMSIS-NN MaxPool does not require a workspace; pass a `Context` with `.buf = null` and `.size = 0`.
- CMSIS-NN expects NHWC. If your tensors are NCHW, you may need a temporary reorder (see below). Start with the simple case and fall back when unsupported.

---

## 2) Link the CMSIS-NN Pooling kernel in the build

Add the MaxPool source to the embedded C bridge so the symbol is available when `-Dstm32n6_use_cmsis=true`.

Edit `build.zig`, in `configureStm32n6Support(...)` right after the Convolution sources are added, append:

```zig
// CMSIS-NN Pooling (MaxPool s8)
if (std.fs.cwd().access("third_party/CMSIS-NN/Source/PoolingFunctions/arm_max_pool_s8.c", .{})) |_| {
    step.addCSourceFile(.{
        .file = b.path("third_party/CMSIS-NN/Source/PoolingFunctions/arm_max_pool_s8.c"),
        .flags = c_flags,
    });
} else |err| {
    if (err != error.FileNotFound) @panic("unexpected error probing arm_max_pool_s8.c");
}
```

If you vendor CMSIS somewhere else, either:
- set `-Dstm32n6_cmsis_path=/path/to/CMSIS`, or
- copy the needed files under `third_party/CMSIS-NN` with the same subdir layout.

---

## 3) Dispatch to CMSIS-NN from the MaxPool operator

Wire an accelerated path in `src/Core/Tensor/TensorMath/op_maxPool.zig` following the QLinearConv pattern.

At the top, import the accelerator modules:

```zig
const accelerators = zant.core.tensor_math.Accelerators; // or @import("../Accelerators/mod.zig") in this file style
const cmsis_nn = @import("../Accelerators/stm32n6/cmsis_nn.zig");
```

Add a small helper before `lean_onnx_maxpool`:

```zig
fn tryAcceleratedMaxPoolS8(
    input: *Tensor(i8),
    output: *Tensor(i8),
    kernel_h: usize, kernel_w: usize,
    stride_h: usize, stride_w: usize,
    pad_top: usize, pad_left: usize,
) bool {
    if (!accelerators.canUseCmsisHelium()) return false;
    if (!cmsis_nn.supportsMaxPoolS8()) return false;
    if (input.shape.len != 4 or output.shape.len != 4) return false; // NCHW only for now

    const dims = cmsis_nn.Dims;
    const ctx = cmsis_nn.Context{ .buf = null, .size = 0 };

    // NCHW logical dims; CMSIS expects NHWC. If layouts mismatch at runtime,
    // add a temporary reorder here and revert after. Start strict and bail out.
    const n = @as(i32, @intCast(input.shape[0]));
    const c = @as(i32, @intCast(input.shape[1]));
    const h = @as(i32, @intCast(input.shape[2]));
    const w = @as(i32, @intCast(input.shape[3]));

    const input_dims = dims{ .n = n, .h = h, .w = w, .c = c };
    const filter_dims = dims{ .n = 1, .h = @as(i32, @intCast(kernel_h)), .w = @as(i32, @intCast(kernel_w)), .c = 1 };
    const output_dims = dims{
        .n = n,
        .h = @as(i32, @intCast(output.shape[2])),
        .w = @as(i32, @intCast(output.shape[3])),
        .c = c,
    };

    const PoolParams = cmsis_nn.PoolParams;
    const pool_params = PoolParams{
        .stride = .{ .h = @as(i32, @intCast(stride_h)), .w = @as(i32, @intCast(stride_w)) },
        .padding = .{ .h = @as(i32, @intCast(pad_top)), .w = @as(i32, @intCast(pad_left)) },
        .activation = .{ .min = -128, .max = 127 },
    };

    const status = cmsis_nn.pool.arm_max_pool_s8(
        &ctx,
        &pool_params,
        &input_dims,
        @as([*]const i8, @ptrCast(input.data.ptr)),
        &filter_dims,
        &output_dims,
        @as([*]i8, @ptrCast(output.data.ptr)),
    );
    return status == cmsis_nn.ARM_CMSIS_NN_SUCCESS;
}
```

Call it at the start of `lean_onnx_maxpool` when types/params match:

```zig
// inside lean_onnx_maxpool, after output allocation and before the Zig loop
if (T == i8 and kernel_shape.len == 2 and strides.len == 2 and dilations.len == 2 and dilations[0] == 1 and dilations[1] == 1) {
    const pad_top: usize = if (pads.len > 0) pads[0] else 0;
    const pad_left: usize = if (pads.len > 1) pads[1] else pad_top;
    if (tryAcceleratedMaxPoolS8(input, output, kernel_shape[0], kernel_shape[1], strides[0], strides[1], pad_top, pad_left)) return;
}
```

Fallbacks:
- If acceleration isn’t available or shapes/dtypes don’t match, the original Zig reference loop runs unchanged.

---

## 4) Build & run

Example static lib build (same flags used by conv acceleration):

```bash
zig build lib -Dmodel=beer -Dstm32n6_accel=true -Dstm32n6_use_cmsis=true -Doptimize=ReleaseFast
```

You can also reuse the QEMU-based flow under `scripts/test_beer_comparison.sh` as a template: it already wires CMSIS include/source discovery and links your generated `libzant.a` together with CMSIS-NN.

---

## Notes, limits, and future tweaks

- Layout: CMSIS-NN pooling APIs are NHWC-centric. If your runtime tensors are NCHW, add a compact reorder in the accelerated path until a permanent NHWC path exists. Keep it gated behind the same `accelerators.canUseCmsisHelium()`/`supportsMaxPoolS8()` checks.
- Dilation/ceil_mode: start with `dilations = 1` and let the operator’s existing output-shape logic handle `ceil_mode`. Expand support as needed.
- Types: `arm_max_pool_s8` covers int8. For uint8 or f32 inputs, keep using the Zig path or add additional externs if present upstream.

---

References:
- [CMSIS-NN Pooling API (official docs)](https://arm-software.github.io/CMSIS-NN/latest/group__Pooling.html)
- CMSIS-NN sources (look under `Source/PoolingFunctions/arm_max_pool_s8.c` in your CMSIS-NN checkout)


