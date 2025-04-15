const std = @import("std");
const zant = @import("../../../../zant.zig");

const Tensor = zant.core.tensor.Tensor;
const TensorError = zant.utils.error_handler.TensorError;
const TensorMathError = zant.utils.error_handler.TensorMathError;

const pkg_allocator = zant.utils.allocator.allocator;

pub const PadMode = enum {
    constant,
    reflect,
    edge,
    wrap,
};

/// Calculates the shape of the output tensor after applying padding.
/// Takes the input tensor shape, the pads array (1D, [start_pad_axis_0, start_pad_axis_1, ..., end_pad_axis_0, end_pad_axis_1, ...]),
/// and optional axes to apply padding to.
/// Returns a new slice allocated using pkg_allocator containing the output shape.
pub fn get_pads_output_shape(
    allocator: std.mem.Allocator,
    input_shape: []const usize,
    pad_values: []const i64,
    axes: ?[]const isize,
) ![]usize {
    const rank = input_shape.len;
    var output_rank = rank; // Rank of the output shape
    var effective_rank = rank; // Rank used for padding calculation
    var effective_input_shape = input_shape; // Shape used for calculation
    var temp_effective_input_shape: []usize = undefined; // Buffer if we prepend shape
    var temp_shape_allocated = false;

    // Check pad length IF axes is null
    if (axes == null) {
        if (pad_values.len == rank * 2) {
            // Standard case: pads match input rank
            effective_rank = rank;
            output_rank = rank;
        } else if (pad_values.len == (rank + 1) * 2) {
            // Special case: pads imply rank + 1, assume leading batch dim
            effective_rank = rank + 1;
            output_rank = rank + 1;
            // Create effective shape [1] ++ input_shape
            temp_effective_input_shape = try pkg_allocator.alloc(usize, effective_rank);
            temp_shape_allocated = true;
            temp_effective_input_shape[0] = 1;
            @memcpy(temp_effective_input_shape[1..], input_shape);
            effective_input_shape = temp_effective_input_shape;
        } else {
            // Mismatched pads length
            return TensorMathError.InvalidPaddingShape;
        }
    }

    var output_shape = try allocator.alloc(usize, output_rank);
    errdefer allocator.free(output_shape);
    // Initialize output shape based on effective input shape
    if (effective_rank == output_rank) {
        @memcpy(output_shape, effective_input_shape);
    } else { // effective_rank = rank + 1, output_rank = rank + 1
        @memcpy(output_shape, effective_input_shape); // Copy the [1, ...] shape
    }
    // Free temp buffer if it was used
    if (temp_shape_allocated) {
        pkg_allocator.free(temp_effective_input_shape);
    }

    if (axes) |ax| {
        if (ax.len > effective_rank) {
            // Should have been caught earlier, but defensive check
            return TensorMathError.InvalidInput;
        }
        // Check pad_values length against axes length
        if (pad_values.len != ax.len * 2) { // Check specifically for axes case
            return TensorMathError.InvalidPaddingShape;
        }
        var axis_seen = try std.DynamicBitSet.initEmpty(allocator, effective_rank);
        defer axis_seen.deinit();

        for (ax, 0..) |axis_raw, i| {
            const axis: usize = if (axis_raw >= 0) @intCast(axis_raw) else @intCast(@as(isize, @intCast(rank)) + axis_raw);
            if (axis >= rank) {
                return TensorMathError.AxisOutOfRange;
            }
            if (axis_seen.isSet(axis)) {
                // Axis repeated, undefined behavior according to spec, return error
                return TensorMathError.InvalidInput;
            }
            axis_seen.set(axis);

            const pad_start = pad_values[i];
            const pad_end = pad_values[ax.len + i]; // ONNX spec: [x1_begin, x2_begin, ..., x1_end, x2_end, ...]

            if (@as(isize, @intCast(output_shape[axis])) + pad_start + pad_end <= 0) {
                // Resulting dimension size is non-positive
                return TensorMathError.InvalidPaddingSize;
            }
            output_shape[axis] = @intCast(@as(isize, @intCast(output_shape[axis])) + pad_start + pad_end);
        }
    } else {
        // No axes provided, apply padding to all axes
        // This condition is already checked at the top if axes is null
        // if (pad_values.len != rank * 2) { // Redundant if top check exists
        //      return TensorMathError.InvalidPaddingShape;
        // }
        for (0..effective_rank) |axis| {
            const pad_start = pad_values[axis];
            const pad_end = pad_values[effective_rank + axis];

            if (@as(isize, @intCast(output_shape[axis])) + pad_start + pad_end <= 0) {
                // Resulting dimension size is non-positive
                return TensorMathError.InvalidPaddingSize;
            }
            output_shape[axis] = @intCast(@as(isize, @intCast(output_shape[axis])) + pad_start + pad_end);
        }
    }

    return output_shape;
}

// Helper function to get the effective input coordinate for a given output coordinate and axis,
// considering the padding mode and pad amounts.
fn get_input_coord(
    out_coord: usize,
    axis_len_in: usize,
    pad_start: i64,
    pad_end: i64,
    mode: PadMode,
) ?isize {
    const axis_len_out = @as(isize, @intCast(axis_len_in)) + pad_start + pad_end;
    const coord: isize = @intCast(out_coord);
    const len_in: isize = @intCast(axis_len_in);

    // Check if the coordinate is within the original data region
    if (coord >= pad_start and coord < axis_len_out - pad_end) {
        return @as(isize, @intCast(coord - pad_start));
    }

    // If not in original data region, calculate based on mode
    switch (mode) {
        .constant => return null, // Signal to use constant_value
        .edge => {
            if (coord < pad_start) {
                return 0; // Clamp to start edge
            } else { // coord >= axis_len_out - pad_end
                return len_in - 1; // Clamp to end edge
            }
        },
        .reflect => {
            // Standard reflect mode (mirrors without repeating boundary)
            if (len_in == 1) return 0; // Reflecting a single element is just the element itself

            const rel_coord = coord - pad_start;
            if (rel_coord < 0) {
                // Reflect before start: e.g., rel_coord -1 -> 0, -2 -> 1
                return @as(isize, @intCast(-rel_coord));
            } else if (rel_coord >= len_in) {
                // Reflect after end: e.g., rel_coord len_in -> len_in - 2, len_in + 1 -> len_in - 3
                return @as(isize, @intCast((2 * len_in) - rel_coord - 2));
            } else {
                // Should be unreachable due to the initial check, but for completeness
                return @as(isize, @intCast(rel_coord));
            }
        },
        .wrap => {
            if (coord < pad_start) {
                // Wrap before the start
                return @as(isize, @intCast(@mod(coord - pad_start, len_in)));
            } else { // coord >= axis_len_out - pad_end
                // Wrap after the end
                return @as(isize, @intCast(@mod(coord - pad_start, len_in)));
            }
        },
    }
}

// Helper struct to manage multi-dimensional iteration
const IndexIterator = struct {
    shape: []const usize,
    strides: []usize,
    current_indices: []usize,
    current_flat_index: usize,
    size: usize,
    rank: usize,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, shape_slice: []const usize) !IndexIterator {
        const rank = shape_slice.len;
        var size: usize = 1;
        if (rank == 0) {
            size = 0; // Handle scalar tensor case if needed, though padding usually implies rank > 0
        } else {
            for (shape_slice) |dim| size *= dim;
        }

        var strides = try allocator.alloc(usize, rank);
        errdefer allocator.free(strides);
        var current_stride: usize = 1;
        var i = rank;
        while (i > 0) {
            i -= 1;
            strides[i] = current_stride;
            current_stride *= shape_slice[i];
        }

        var current_indices = try allocator.alloc(usize, rank);
        // errdefer handled below
        _ = &current_indices; // Mark as used to silence linter, contents are mutated later

        return IndexIterator{
            .shape = shape_slice,
            .strides = strides,
            .current_indices = current_indices,
            .current_flat_index = 0,
            .size = size,
            .rank = rank,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *IndexIterator) void {
        self.allocator.free(self.current_indices);
        self.allocator.free(self.strides);
    }

    // Advances the iterator to the next set of indices and returns true, or returns false if iteration is complete.
    pub fn next(self: *IndexIterator) bool {
        if (self.current_flat_index >= self.size) {
            return false;
        }
        if (self.current_flat_index == 0) { // First call
            @memset(self.current_indices, 0);
            self.current_flat_index += 1;
            return true;
        }

        // Increment indices like an odometer
        var i = self.rank;
        while (i > 0) {
            i -= 1;
            self.current_indices[i] += 1;
            if (self.current_indices[i] < self.shape[i]) {
                self.current_flat_index += 1;
                return true; // Successfully incremented this dimension
            }
            // Current dimension wrapped around, reset to 0 and carry over to the next dimension
            self.current_indices[i] = 0;
        }

        // If we reach here, all dimensions have wrapped around, iteration is complete
        // Set flat index beyond size to ensure future next() calls return false immediately
        self.current_flat_index = self.size + 1;
        return false;
    }

    // Calculates the flat index for the current indices.
    pub fn getFlatIndex(self: *const IndexIterator) usize {
        if (self.rank == 0) return 0; // Handle scalar case

        var flat_index: usize = 0;
        for (0..self.rank) |i| {
            flat_index += self.current_indices[i] * self.strides[i];
        }
        return flat_index;
    }
};

/// Performs padding operation without allocating the output tensor.
/// Output tensor must be pre-allocated with the correct shape and zero-initialized if necessary.
pub fn pads_lean(
    comptime T: anytype,
    data: *const Tensor(T),
    pad_values: []const i64,
    mode: PadMode,
    constant_value: ?T, // Only used for constant mode
    axes: ?[]const isize, // Optional axes to apply padding
    output: *Tensor(T),
) !void {
    const input_rank = data.shape.len;
    const output_rank = output.shape.len;

    // Allow output rank to be input_rank + 1 (prepended dim case)
    if (output_rank != input_rank and output_rank != input_rank + 1) {
        // Only allow output rank == input rank or output rank == input rank + 1
        return TensorMathError.OutputTensorWrongShape;
    }
    // Validate pad_values length against the effective rank (output rank) if axes is null
    if (axes == null and pad_values.len != output_rank * 2) {
        // This check might seem redundant given get_pads_output_shape, but good defense
        return TensorMathError.InvalidPaddingShape;
    }
    // Validate pad_values length against axes length if axes is provided
    if (axes != null and pad_values.len != axes.?.len * 2) {
        return TensorMathError.InvalidPaddingShape;
    }

    const const_val = if (mode == .constant) constant_value orelse @as(T, 0) else @as(T, undefined);

    // Store pad_start and pad_end for each *output* axis
    var pads_per_axis = try pkg_allocator.alloc([2]i64, output_rank);
    defer pkg_allocator.free(pads_per_axis);
    // Initialize with zero padding
    for (0..output_rank) |i| {
        pads_per_axis[i] = .{ @as(i64, 0), @as(i64, 0) };
    }

    const prepended_dim = (output_rank == input_rank + 1);

    if (axes) |ax| {
        // axes are relative to input dimensions
        if (ax.len > input_rank) return TensorMathError.InvalidInput; // Should be guaranteed by caller
        var axis_map = std.AutoHashMap(usize, void).init(pkg_allocator); // Track seen output axes
        defer axis_map.deinit();

        for (ax, 0..) |axis_raw, i| {
            const resolved_input_axis: usize = if (axis_raw >= 0) @intCast(axis_raw) else @intCast(@as(isize, @intCast(input_rank)) + axis_raw);
            if (resolved_input_axis >= input_rank) return TensorMathError.AxisOutOfRange;

            // Map input axis to output axis
            const output_axis = if (prepended_dim) resolved_input_axis + 1 else resolved_input_axis;

            if (axis_map.contains(output_axis)) return TensorMathError.InvalidInput; // Repeated axis applied to same output dim

            const p_start = pad_values[i];
            const p_end = pad_values[ax.len + i];
            try axis_map.put(output_axis, {});
            pads_per_axis[output_axis] = .{ p_start, p_end };
        }
    } else {
        // No axes specified, apply to all output dimensions
        // pad_values length must match output_rank * 2 (checked earlier)
        for (0..output_rank) |axis| {
            pads_per_axis[axis] = .{ pad_values[axis], pad_values[output_rank + axis] };
        }
    }

    // Iterate through each element of the output tensor
    var out_iter = try IndexIterator.init(pkg_allocator, output.shape);
    defer out_iter.deinit();
    // Input indices buffer has size input_rank
    var in_indices = try pkg_allocator.alloc(usize, input_rank);
    defer pkg_allocator.free(in_indices);

    while (out_iter.next()) {
        const out_flat_index = out_iter.getFlatIndex();
        var use_constant = false;
        var calculated_in_indices = true; // Track if we successfully got all needed input coords

        // Calculate potential input indices based on output indices and padding rules.
        for (0..output_rank) |output_axis| {
            const out_coord = out_iter.current_indices[output_axis];
            const pad_start = pads_per_axis[output_axis][0];
            const pad_end = pads_per_axis[output_axis][1];

            // Determine corresponding input axis, if any
            const maybe_input_axis: ?usize = if (!prepended_dim) output_axis else if (output_axis == 0) null // Prepended dimension has no direct input counterpart
                else output_axis - 1; // Shift subsequent axes

            // Determine the effective input size for this dimension for get_input_coord
            const axis_len_in: usize = if (maybe_input_axis) |ia| data.shape[ia] else 1; // Treat prepended dim as having size 1 input

            // Calculate the source coordinate using the effective input length
            const maybe_in_coord = get_input_coord(out_coord, axis_len_in, pad_start, pad_end, mode);

            if (maybe_in_coord) |in_coord| {
                // Check bounds relative to effective input length
                if (in_coord < 0 or @as(usize, @intCast(in_coord)) >= axis_len_in) {
                    return TensorMathError.UnexpectedError;
                }
                // Store input index if this output axis corresponds to an actual input axis
                if (maybe_input_axis) |ia| {
                    in_indices[ia] = @intCast(in_coord);
                } else {
                    // This coordinate is in the prepended dimension. get_input_coord should return 0
                    // for edge/reflect/wrap based on axis_len_in=1. If it's not 0, something is wrong.
                    if (in_coord != 0) {
                        std.debug.print("Error: get_input_coord returned non-zero index {} for prepended dimension (axis_len_in=1)", .{in_coord});
                        return TensorMathError.UnexpectedError;
                    }
                    // No need to store index as it doesn't map to an input dimension.
                }
            } else {
                // get_input_coord returned null, only valid for constant mode in padding area
                if (mode == .constant) {
                    use_constant = true;
                    calculated_in_indices = false; // Input indices are now irrelevant
                    break; // Stop calculating indices for this output element
                } else {
                    // Should not happen for non-constant modes
                    std.debug.print("Error: get_input_coord returned null for non-constant mode {}", .{mode});
                    return TensorMathError.UnexpectedError;
                }
            }
        }

        // Assign value to output tensor element
        if (use_constant) {
            output.data[out_flat_index] = const_val;
        } else if (calculated_in_indices) {
            // Calculate flat index using input_rank, in_indices, and data.shape/strides
            var in_flat_index: usize = 0;
            var current_stride: usize = 1;
            var i = input_rank; // Iterate based on input rank
            while (i > 0) {
                i -= 1;
                in_flat_index += in_indices[i] * current_stride;
                // Stride calculation MUST use input shape
                current_stride *= data.shape[i];
            }

            // Safeguard: Check if calculated index is within bounds of input data
            if (in_flat_index >= data.data.len) {
                std.debug.print("Error: Calculated in_flat_index {} out of bounds (size {}). In Indices: {any}, Input Shape: {any}", .{ in_flat_index, data.data.len, in_indices, data.shape });
                return TensorMathError.UnexpectedError;
            }

            output.data[out_flat_index] = data.data[in_flat_index];
        } else {
            // This case implies (!use_constant and !calculated_in_indices), which should be unreachable
            // if the logic in the loop is correct.
            std.debug.print("Error: Reached unreachable state in pads_lean (use_constant=false, calculated_in_indices=false)", .{});
            return TensorMathError.UnexpectedError;
        }
    }
}

/// Pads a tensor according to the specified mode and padding values.
/// Allocates and returns a new tensor with the padded data.
pub fn pads(
    comptime T: anytype,
    data: *const Tensor(T),
    pads_tensor: *const Tensor(i64),
    mode_str: ?[]const u8,
    constant_value: ?T,
    axes_tensor: ?*const Tensor(isize),
) !Tensor(T) {
    // --- Input Validation ---
    const rank = data.shape.len;

    // Validate pads tensor
    if (pads_tensor.shape.len != 1) {
        return TensorMathError.InvalidPaddingShape;
    }
    const num_pad_values = pads_tensor.shape[0];

    // Validate axes tensor if provided
    var axes_data: ?[]const isize = null;
    var num_axes: usize = rank; // Default to all axes
    if (axes_tensor) |ax_tensor| {
        if (ax_tensor.shape.len != 1) {
            return TensorMathError.InvalidInput;
        }
        num_axes = ax_tensor.shape[0];
        if (num_axes > rank) {
            return TensorMathError.InvalidInput;
        }
        if (num_pad_values != num_axes * 2) {
            return TensorMathError.InvalidPaddingShape;
        }
        // Check for valid axis values and duplicates
        var axis_seen = try std.DynamicBitSet.initEmpty(pkg_allocator, rank);
        defer axis_seen.deinit();
        for (ax_tensor.data) |axis_raw| {
            const axis: usize = if (axis_raw >= 0) @intCast(axis_raw) else @intCast(@as(isize, @intCast(rank)) + axis_raw);
            if (axis >= rank) {
                return TensorMathError.AxisOutOfRange;
            }
            if (axis_seen.isSet(axis)) {
                return TensorMathError.InvalidInput; // Axis repeated
            }
            axis_seen.set(axis);
        }
        axes_data = ax_tensor.data; // Use the data directly
    } else {
        // No axes tensor provided, check if pads length matches rank * 2
        if (num_pad_values != rank * 2) {
            return TensorMathError.InvalidPaddingShape;
        }
        axes_data = null; // Explicitly null
    }

    // Parse mode
    var mode: PadMode = .constant; // Default
    if (mode_str) |m_str| {
        if (std.ascii.eqlIgnoreCase(m_str, "constant")) {
            mode = .constant;
        } else if (std.ascii.eqlIgnoreCase(m_str, "reflect")) {
            mode = .reflect;
        } else if (std.ascii.eqlIgnoreCase(m_str, "edge")) {
            mode = .edge;
        } else if (std.ascii.eqlIgnoreCase(m_str, "wrap")) {
            mode = .wrap;
        } else {
            return TensorMathError.UnsupportedMode;
        }
    }

    // --- Calculate Output Shape & Allocate ---
    const output_shape = try get_pads_output_shape(pkg_allocator, data.shape, pads_tensor.data, axes_data);
    defer pkg_allocator.free(output_shape); // get_pads_output_shape allocates this

    // Check for zero-sized dimensions which might indicate invalid padding amounts
    for (output_shape) |dim_size| {
        if (dim_size == 0) {

            // This case might be valid if the input also had zero dimensions,
            // but often indicates negative padding reducing dimension to zero or less.
            // get_pads_output_shape should have caught negative results, so this is likely zero.
            // Depending on desired behavior for zero-sized dims, might return error or empty tensor.
            // For now, let's allow zero-sized dims if allocation succeeds. -Marco
            break;
        }
    }

    var output = try Tensor(T).fromShape(&pkg_allocator, output_shape);
    errdefer output.deinit(); // Ensure deallocation if pads_lean fails

    // --- Perform Padding ---
    try pads_lean(T, data, pads_tensor.data, mode, constant_value, axes_data, &output);

    return output;
}
