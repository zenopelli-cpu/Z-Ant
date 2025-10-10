# Nicla-SmartTrashCollector (Example)

> Minimal 5-class waste sorting on **Nicla Vision** using **Z-Ant** runtime.
> Pipeline: GC2145 camera (160×120 RGB565) → resize to 96×96 → NCHW float → `predict()` → top-1 → RGB LEDs and serial log.

## Requirements
- Arduino core: `arduino:mbed_nicla` (tested 4.4.1)
- `arduino-cli`, `dfu-util` (only if using XIP flashing script)
- Zig 0.14.x (only to build `libzant.a` via Z-Ant)
- Z-Ant checked out (this repo)

## Build the Z-Ant static library
> We do **not** commit `libzant.a`. Build it and copy it under this example’s local library folder.

```bash
# From repo root
zig build lib-gen  -Dmodel="YOUR_MODEL" -Dxip=true -Ddynamic -Ddo_export
zig build lib-test -Dmodel="YOUR_MODEL" -Dxip=true -Ddynamic -Ddo_export
zig build lib      -Dmodel="YOUR_MODEL" -Dtarget=thumb-freestanding -Dcpu=cortex_m7 -Dxip=true -Doptimize=ReleaseSmall

# Copy the generated library:
cp zig-out/YOUR_MODEL/libzant.a examples/Nicla-SmartTrashCollector/ZantLib/src/cortex-m7/

