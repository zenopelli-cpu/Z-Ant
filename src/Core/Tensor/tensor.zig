//! Tensor has a crucial role in all the project. Is the foundamental class around witch everything
//! is constructed. A tensor is a multi-dimensional array or a mathematical object that generalizes
//! the concept of scalars, vectors, and matrices to higher dimensions. A scalar is a 0-dimensional
//! tensor, a vector is a 1-dimensional tensor, and a matrix is a 2-dimensional tensor. Tensors can extend
//! to even higher dimensions (3D, 4D, etc.).

const std = @import("std");
const tMath = @import("tensor_m");
const Architectures = @import("architectures").Architectures;
const TensorError = @import("errorHandler").TensorError;
const ArgumentError = @import("errorHandler").ArgumentError;

///Class Tensor.
///Return a generic type structure
pub fn Tensor(comptime T: type) type {
    return struct {
        data: []T, //contains all the data of the tensor in a monodimensional array
        size: usize, //dimension of the tensor, equal to data.len
        shape: []usize, //defines the multidimensional structure of the tensor
        allocator: *const std.mem.Allocator, //allocator used in the memory initialization of the tensor

        ///Method used to initialize an undefined Tensor. It just set the allocator.
        /// More usefull methods are:
        ///  - fromArray()
        ///  - copy()
        ///  - fromShape()
        pub fn init(allocator: *const std.mem.Allocator) !@This() {
            return @This(){
                .data = &[_]T{},
                .size = 0,
                .shape = &[_]usize{},
                .allocator = allocator,
            };
        }

        ///Free all the possible allocation, use it every time you create a new Tensor ( defer yourTensor.deinit() )
        pub fn deinit(self: *@This()) void {
            if (self.size > 0) {
                if (self.data.len > 0) {
                    self.allocator.free(self.data);
                    self.data = &[_]T{};
                }
                if (self.shape.len > 0) {
                    self.allocator.free(self.shape);
                    self.shape = &[_]usize{};
                }
            }
        }

        ///Given a multidimensional array with its shape, returns the equivalent Tensor.
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn fromArray(allocator: *const std.mem.Allocator, inputArray: anytype, shape: []usize) !@This() {

            // Calculate total size based on shape
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }

            // Allocate memory for tensor shape
            const tensorShape = try allocator.alloc(usize, shape.len);
            @memcpy(tensorShape, shape);

            // Allocate memory for tensor data
            const tensorData = try allocator.alloc(T, total_size);

            // Flatten the input array into tensor data
            _ = flattenArray(T, inputArray, tensorData, 0);

            // Return the new tensor
            return @This(){
                .data = tensorData,
                .size = total_size,
                .shape = tensorShape,
                .allocator = allocator,
            };
        }

        /// Given the Tensor (self) returns the equivalent multidimensional array.
        /// See constructMultidimensionalArray() in this file.
        /// IMPORTANT: Remember to cal yourAllocator.free(yourMultidimArray) otherwise it generates a memory leak!
        pub fn toArray(self: @This(), comptime dimension: usize) !MagicalReturnType(T, dimension) {
            if (dimension == 1) {
                return self.data;
            }
            return constructMultidimensionalArray(self.allocator, T, self.data, self.shape, 0, dimension);
        }

        /// Returns a Tensor witch is the copy of this Tensor (self).
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn copy(self: *@This()) !Tensor(T) {
            return try Tensor(T).fromArray(self.allocator, self.data, self.shape);
        }

        /// Return a all-zero tensor starting from the given shape
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn fromShape(allocator: *const std.mem.Allocator, shape: []usize) !@This() {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }

            const tensorShape = try allocator.alloc(usize, shape.len);
            @memcpy(tensorShape, shape);

            const tensorData = try allocator.alloc(T, total_size);
            @memset(tensorData, 0);

            return @This(){
                .data = tensorData,
                .size = total_size,
                .shape = tensorShape,
                .allocator = allocator,
            };
        }

        /// Given any array and its shape it reshape the tensor and update .data
        pub fn fill(self: *@This(), inputArray: anytype, shape: []usize) !void {

            //deinitialize data e shape
            self.deinit(); //if the Tensor has been just init() this function does nothing

            //than, filling with the new values
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }
            const tensorShape = try self.allocator.alloc(usize, shape.len);
            @memcpy(tensorShape, shape);

            const tensorData = try self.allocator.alloc(T, total_size);
            _ = flattenArray(T, inputArray, tensorData, 0);

            self.data = tensorData;
            self.size = total_size;
            self.shape = tensorShape;
        }

        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///--------------------------------------------------------------------------getters and setters---------------------------------------------------------------
        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///Set the shape of a Tensor.
        pub fn setShape(self: *@This(), shape: []usize) !void {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }
            self.shape = shape;
            self.size = total_size;
        }

        ///Returns the size of the Tensor.
        pub fn getSize(self: *@This()) usize {
            return self.size;
        }

        ///Given an index, return the value at self.data[index].
        /// Errors:
        ///     - error.IndexOutOfBounds;
        pub fn get(self: *const @This(), idx: usize) !T {
            if (idx >= self.data.len) {
                return error.IndexOutOfBounds;
            }
            return self.data[idx];
        }

        ///Set to value the data at self.data[idx].
        /// Errors:
        ///     - error.IndexOutOfBounds;
        pub fn set(self: *@This(), idx: usize, value: T) !void {
            if (idx >= self.data.len) {
                return error.IndexOutOfBounds;
            }
            self.data[idx] = value;
        }

        /// Given the coordinates (indices) it returns the correspondant value in the
        /// multidimensional array.
        /// See flatten_index().
        pub fn get_at(self: *const @This(), indices: []const usize) !T {
            const idx = try self.flatten_index(indices);
            return self.get(idx);
        }

        /// Given the the value and the coordinates (indices), it sets the value in
        /// the multidimensional array at the specified coordinates.
        /// See flatten_index().
        pub fn set_at(self: *@This(), indices: []const usize, value: T) !void {
            const idx = try self.flatten_index(indices);
            return self.set(idx, value);
        }

        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///-------------------------------------------------------------------------------------utils------------------------------------------------------------------
        ///------------------------------------------------------------------------------------------------------------------------------------------------------------
        ///Starting from the monodimensional array self.data and the shape self.shape, it returns the equivalent multidimensional array
        fn constructMultidimensionalArray(
            allocator: *const std.mem.Allocator,
            comptime ElementType: type,
            data: []ElementType,
            shape: []usize,
            comptime depth: usize,
            comptime dimension: usize,
        ) !MagicalReturnType(ElementType, dimension - depth) {
            if (depth == dimension - 1) {
                return data;
            }

            const current_dim = shape[depth];
            var result = try allocator.alloc(
                MagicalReturnType(ElementType, dimension - depth - 1),
                current_dim,
            );

            // defer allocator.free(result); ??????????? MARCO : era già commentata, ci va o meno la .free()? non credo vada liberato perchè è lui stesso l'array multidim.
            // non andrebbe però creato un metodo freeMultidimensionalArray() che fa la stessa cosa ma librando spazio?
            // AGGIORANEMENTO: nei tests_tensor mi è bastato fare: line 197 -> defer allocator.free(array_from_tensor);

            var offset: usize = 0;
            const sub_array_size = calculateProduct(shape[(depth + 1)..]);

            for (0..current_dim) |i| {
                result[i] = try constructMultidimensionalArray(
                    allocator,
                    ElementType,
                    data[offset .. offset + sub_array_size],
                    shape,
                    depth + 1,
                    dimension,
                );
                offset += sub_array_size;
            }

            return result;
        }

        fn MagicalReturnType(comptime DataType: type, comptime dim_count: usize) type {
            return if (dim_count == 1) []DataType else []MagicalReturnType(DataType, dim_count - 1);
        }

        fn calculateProduct(slices: []usize) usize {
            var product: usize = 1;
            for (slices) |elem| {
                product *= elem;
            }
            return product;
        }

        /// Modify, if possible, the shape of a tensor, use it wisely.
        /// Errors:
        ///     - TensorError.InputArrayWrongSize
        pub fn reshape(self: *@This(), shape: []usize) !void {
            var total_size: usize = 1;
            for (shape) |dim| {
                total_size *= dim;
            }
            if (total_size != self.size) {
                return TensorError.InputArrayWrongSize;
            }

            self.allocator.free(self.shape);
            const tensorShape = try self.allocator.alloc(usize, shape.len);
            // copy elements of shape
            @memcpy(tensorShape, shape);

            self.shape = tensorShape;
        }

        /// Given the coordinates (indices) of a multidimensional Tensor returns the correspondant potition in the monodimensional space of self.data
        pub fn flatten_index(self: *const @This(), indices: []const usize) !usize {
            var idx: usize = 0;
            var stride: usize = 1;

            if (indices.len != self.shape.len) {
                return error.InvalidIndexLength;
            }

            for (0..self.shape.len) |i| {
                const rev_idx = self.shape.len - 1 - i;
                const index = indices[rev_idx];

                // Controllo per indice fuori dai limiti
                if (index >= self.shape[rev_idx]) {
                    return error.IndexOutOfBounds;
                }

                idx += index * stride;
                stride *= self.shape[rev_idx];
            }

            return idx;
        }

        pub fn slice(self: *Tensor(T), start_indices: []usize, slice_shape: []usize) !Tensor(T) {
            // Validate input
            if (start_indices.len != self.shape.len) return TensorError.InvalidSliceIndices;
            if (slice_shape.len != self.shape.len) return TensorError.InvalidSliceShape;

            // Verify that the slice is within bounds
            for (0..self.shape.len) |i| {
                if (start_indices[i] + slice_shape[i] > self.shape[i]) return TensorError.SliceOutOfBounds;
            }

            // Calculate the total size of the new tensor
            var new_size: usize = 1;
            for (slice_shape) |dim| {
                new_size *= dim;
            }

            // Allocate data for the new tensor
            const new_data = try self.allocator.alloc(T, new_size);

            // Prepare for copying data
            const num_dims = self.shape.len;

            // Strides for the original tensor
            const strides = try self.getStrides();
            defer self.allocator.free(strides);

            // Recursive function to copy data
            const indices = try self.allocator.alloc(usize, num_dims);
            defer self.allocator.free(indices);

            for (indices) |*idx| idx.* = 0;

            var new_data_index: usize = 0;

            try copy_data_recursive(
                self,
                new_data,
                &new_data_index,
                start_indices,
                slice_shape,
                indices,
                0,
            );

            // Create the new tensor
            var new_tensor = Tensor(T){
                .data = new_data,
                .shape = try self.allocator.dupe(usize, slice_shape),
                .size = new_size,
                .allocator = self.allocator,
            };

            _ = &new_tensor;

            return new_tensor;
        }

        // Recursive function to copy data
        fn copy_data_recursive(
            self: *Tensor(T),
            new_data: []T,
            new_data_index: *usize,
            start_indices: []usize,
            slice_shape: []usize,
            indices: []usize,
            dim: usize,
        ) !void {
            if (dim == self.shape.len) {
                // Calculate the index in the original tensor
                var self_indices = try self.allocator.alloc(usize, self.shape.len);
                defer self.allocator.free(self_indices);

                for (0..self.shape.len) |i| {
                    self_indices[i] = start_indices[i] + indices[i];
                }

                const flat_index = try self.get_flat_index(self_indices);
                new_data[new_data_index.*] = self.data[flat_index];
                new_data_index.* += 1;
            } else {
                for (0..slice_shape[dim]) |i| {
                    indices[dim] = i;
                    try copy_data_recursive(
                        self,
                        new_data,
                        new_data_index,
                        start_indices,
                        slice_shape,
                        indices,
                        dim + 1,
                    );
                }
            }
        }

        // Helper function to calculate the flat index from multi-dimensional indices
        fn get_flat_index(self: *Tensor(T), indices: []usize) !usize {
            if (indices.len != self.shape.len) return TensorError.InvalidIndices;

            var flat_index: usize = 0;
            var stride: usize = 1;

            var i: usize = self.shape.len - 1;
            while (true) {
                flat_index += indices[i] * stride;
                stride *= self.shape[i];
                if (i == 0) break;
                i -= 1;
            }

            return flat_index;
        }

        // Function to calculate strides for the tensor
        pub fn getStrides(self: *Tensor(T)) ![]usize {
            const num_dims = self.shape.len;
            var strides = try self.allocator.alloc(usize, num_dims);
            strides[num_dims - 1] = 1;
            var i: usize = num_dims - 1;
            while (i > 0) {
                strides[i - 1] = strides[i] * self.shape[i];
                i -= 1;
            }
            return strides;
        }

        /// Prints all the possible details of a tensor.
        /// Very usefull in debugging.
        pub fn info(self: *@This()) void {
            std.debug.print("\ntensor infos: ", .{});
            std.debug.print("\n  data type:{}", .{@TypeOf(self.data[0])});
            std.debug.print("\n  size:{}", .{self.size});
            std.debug.print("\n shape.len:{} shape: [ ", .{self.shape.len});
            for (0..self.shape.len) |i| {
                std.debug.print("{} ", .{self.shape[i]});
            }
            std.debug.print("] ", .{});
            //self.print();
        }

        /// Prints all the array self.data in an array.
        pub fn print(self: *@This()) void {
            std.debug.print("\n  tensor data: ", .{});
            for (0..self.size) |i| {
                std.debug.print("{} ", .{self.data[i]});
            }
            std.debug.print("\n", .{});
        }

        /// Print the Tensor() to console in a more readable way.
        pub fn printMultidim(self: *@This()) void {
            // Allocate array to store the indices
            self._printMultidimHelper(0, 0);
        }

        fn _printMultidimHelper(self: *@This(), offset: usize, idx: usize) void {
            // Print opening bracket with a number of tab that is equals to idx
            for (0..idx) |_| {
                std.debug.print("    ", .{});
            }
            std.debug.print("[", .{});

            if (idx == self.shape.len - 1) {
                for (0..self.shape[self.shape.len - 1]) |i| {
                    const local_idx = offset + i;
                    std.debug.print("{}, ", .{self.data[local_idx]});
                }
                std.debug.print("],\n", .{});
            } else {
                std.debug.print("\n", .{});
                for (0..self.shape[idx]) |i| {
                    self._printMultidimHelper(offset + self.shape[idx + 1] * i, idx + 1);
                }
                std.debug.print("\n", .{});

                for (0..idx) |_| {
                    std.debug.print("    ", .{});
                }
                std.debug.print("]", .{});
                if (idx != 0) {
                    std.debug.print(",\n", .{});
                }
            }
        }

        /// Performs Element-wise binary division of two tensors.
        pub fn div(lhs: *@This(), rhs: *@This()) !Tensor(T) {
            if (lhs.size != rhs.size) {
                return TensorError.MismatchedShape;
            }

            const allocator = lhs.allocator;
            const result = try Tensor(T).fromShape(allocator, lhs.shape);

            for (0..lhs.size) |i| {
                result.data[i] = lhs.data[i] / rhs.data[i];
            }

            return result;
        }
        /// Set all tensor values to zero.
        pub fn setToZero(self: *@This()) !void {
            if (self.size == 0) {
                return TensorError.TensorNotInitialized;
            }
            @memset(self.data, 0);
        }

        /// Gather elements from the tensor along an axis using the provided indices.
        /// The axis parameter specifies the axis along which the elements will be gathered.
        /// The indices tensor must have the same number of dimensions as the input tensor, except for the axis dimension.
        /// The shape of the output tensor is the same as the shape of the indices tensor, with the axis dimension removed.
        /// The output tensor is created by copying elements from the input tensor using the indices tensor.
        pub fn gather(self: *@This(), indices: Tensor(usize), selected_axis: isize) !@This() {
            // Validate that the axis is within the tensor's dimensions
            const number_dimensions: isize = @intCast(self.shape.len);
            if (selected_axis >= number_dimensions or selected_axis <= -1 * number_dimensions) {
                return TensorError.InvalidAxis;
            }

            // If axis is negative, convert it to a positive index
            const axis: usize = @intCast(if (selected_axis < 0) number_dimensions + selected_axis else selected_axis);

            // Calculate the shape of the output tensor:
            // [data.shape[0..axis], indices.shape..., data.shape[axis+1..]]
            const output_shape_len = self.shape.len - 1 + indices.shape.len;
            const output_shape = try self.allocator.alloc(usize, output_shape_len);

            // Copy the dimensions before the axis
            for (0..axis) |i| {
                output_shape[i] = self.shape[i];
            }

            // Copy the indices tensor's shape
            for (0..indices.shape.len) |i| {
                output_shape[axis + i] = indices.shape[i];
            }

            // Copy the dimensions after the axis
            for (0..(self.shape.len - axis - 1)) |i| {
                output_shape[axis + indices.shape.len + i] = self.shape[axis + 1 + i];
            }

            // Compute the total number of elements in each segment
            var outer_size: usize = 1;
            for (0..axis) |i| outer_size *= self.shape[i];

            var indices_size: usize = 1;
            for (0..indices.shape.len) |i| indices_size *= indices.shape[i];

            var inner_size: usize = 1;
            for (axis + 1..self.shape.len) |i| inner_size *= self.shape[i];

            // Compute the total size of the output tensor
            const output_total_size = outer_size * indices_size * inner_size;

            // Allocate memory for the output tensor's data
            const output_data = try self.allocator.alloc(T, output_total_size);

            // Get strides for the input tensor
            const data_strides = try self.getStrides();
            defer self.allocator.free(data_strides);

            // Iterate over each "outer" segment
            for (0..outer_size) |outer_idx| {
                // Iterate over each index in the indices tensor
                for (0..indices_size) |idx| {
                    // Retrieve the gather index from the indices tensor
                    const gather_idx = try indices.get(idx);

                    // Validate the gather index
                    if (gather_idx >= self.shape[axis]) {
                        return TensorError.IndexOutOfBounds;
                    }

                    // Calculate the correct data_offset
                    const data_offset = (outer_idx * self.shape[axis] + gather_idx) * inner_size;

                    // Calculate the starting offset in the output tensor
                    const output_offset = (outer_idx * indices_size + idx) * inner_size;

                    // Debug Prints (optional, can be commented out after debugging)
                    std.debug.print("Outer Index: {}, Gather Index: {}, Data Offset: {}, Output Offset: {}\n", .{ outer_idx, gather_idx, data_offset, output_offset });
                    std.debug.print("Copying from input data[{}] = {}\n", .{ data_offset, self.data[data_offset] });

                    // Perform the data copy using std.mem.copy
                    @memcpy(output_data[output_offset .. output_offset + inner_size], self.data[data_offset .. data_offset + inner_size]);

                    std.debug.print("Copied to output data[{}] = {}\n", .{ output_offset, output_data[output_offset] });
                }
            }

            // Create and return the new tensor with the gathered data and calculated shape
            return @This(){
                .data = output_data,
                .size = output_total_size,
                .shape = output_shape,
                .allocator = self.allocator,
            };
        }

        fn calculateStrides(shape: []const usize) ![]usize {
            const allocator = std.heap.page_allocator;
            const strides = try allocator.alloc(usize, shape.len);
            var stride: usize = 1;
            var i: usize = shape.len;
            while (i > 0) {
                i -= 1;
                strides[i] = stride;
                stride *= shape[i];
            }
            return strides;
        }

        /// Implements the ONNX slice operator (https://onnx.ai/onnx/operators/onnx__Slice.html)
        /// Takes a tensor and extracts a slice along multiple axes.
        /// starts: Starting indices for each axis
        /// ends: Ending indices for each axis (exclusive)
        /// axes: Which axes to slice (if null, assumes [0,1,2,...])
        /// steps: Step sizes for each axis (if null, assumes all 1s)
        pub fn slice_onnx(self: *Tensor(T), starts: []const i64, ends: []const i64, axes: ?[]const i64, steps: ?[]const i64) !Tensor(T) {
            // Validate input lengths
            if (starts.len != ends.len) return TensorError.InvalidSliceIndices;
            if (axes) |a| {
                if (a.len != starts.len) return TensorError.InvalidSliceIndices;
            }
            if (steps) |s| {
                if (s.len != starts.len) return TensorError.InvalidSliceIndices;
            }

            // Create arrays to store the actual indices and steps for each dimension
            var actual_starts = try self.allocator.alloc(i64, self.shape.len);
            defer self.allocator.free(actual_starts);
            var actual_ends = try self.allocator.alloc(i64, self.shape.len);
            defer self.allocator.free(actual_ends);
            var actual_steps = try self.allocator.alloc(i64, self.shape.len);
            defer self.allocator.free(actual_steps);

            // Initialize with defaults (full range, step 1)
            for (0..self.shape.len) |i| {
                actual_starts[i] = 0;
                actual_ends[i] = @intCast(self.shape[i]);
                actual_steps[i] = 1;
            }

            // Update with provided values
            for (starts, 0..) |start, i| {
                const axis = if (axes) |a| a[i] else @as(i64, @intCast(i));
                const axis_usize = if (axis < 0) @as(usize, @intCast(axis + @as(i64, @intCast(self.shape.len)))) else @as(usize, @intCast(axis));
                if (axis_usize >= self.shape.len) return TensorError.InvalidSliceIndices;

                const dim_size = @as(i64, @intCast(self.shape[axis_usize]));

                // Handle negative indices and clamp to valid range
                var actual_start = if (start < 0) start + dim_size else start;
                actual_start = @max(0, @min(actual_start, dim_size));
                actual_starts[axis_usize] = actual_start;

                var actual_end = if (ends[i] < 0) ends[i] + dim_size else ends[i];
                if (steps) |s| {
                    if (s[i] < 0) {
                        // For negative steps, if end is negative, we want to include 0
                        actual_end = if (ends[i] < 0) -1 else actual_end;
                    } else {
                        actual_end = @max(0, @min(actual_end, dim_size));
                    }
                } else {
                    actual_end = @max(0, @min(actual_end, dim_size));
                }
                actual_ends[axis_usize] = actual_end;

                if (steps) |s| {
                    if (s[i] == 0) return TensorError.InvalidSliceStep;
                    actual_steps[axis_usize] = s[i];
                }
            }

            // Calculate output shape
            var output_shape = try self.allocator.alloc(usize, self.shape.len);
            errdefer self.allocator.free(output_shape);

            var total_elements: usize = 1;
            for (0..self.shape.len) |i| {
                const start = actual_starts[i];
                const end = actual_ends[i];
                const step = actual_steps[i];

                var dim_size: usize = 0;
                if (step > 0) {
                    if (end > start) {
                        dim_size = @intCast(@divTrunc((@as(i64, @intCast(end - start)) + step - 1), step));
                        std.debug.print("\nPositive step: start={}, end={}, step={}, dim_size={}", .{ start, end, step, dim_size });
                    }
                } else {
                    if (start > end) {
                        // For negative steps, we need to handle the range differently
                        // Add 1 to end because end is exclusive
                        const range = start - (end + 1);
                        const abs_step = -step;
                        dim_size = @intCast(@divTrunc(range + abs_step - 1, abs_step));
                        std.debug.print("\nNegative step: start={}, end={}, step={}, range={}, abs_step={}, dim_size={}", .{ start, end, step, range, abs_step, dim_size });
                    }
                }
                std.debug.print("\nDimension {}: dim_size={}", .{ i, dim_size });
                output_shape[i] = dim_size;
                total_elements *= dim_size;
            }

            // Allocate output data
            var output_data = try self.allocator.alloc(T, total_elements);
            errdefer self.allocator.free(output_data);

            // Helper function to convert flat index to coordinates
            var input_coords = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(input_coords);
            var output_coords = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(output_coords);

            // Copy data
            var output_idx: usize = 0;
            std.debug.print("\nTotal elements: {}", .{total_elements});
            while (output_idx < total_elements) : (output_idx += 1) {
                // Convert output_idx to coordinates
                var temp = output_idx;
                for (0..self.shape.len) |i| {
                    const dim_i = self.shape.len - 1 - i;
                    output_coords[dim_i] = temp % output_shape[dim_i];
                    temp /= output_shape[dim_i];
                }

                // Calculate input coordinates
                for (0..self.shape.len) |i| {
                    const coord = @as(i64, @intCast(output_coords[i]));
                    input_coords[i] = @intCast(actual_starts[i] + coord * actual_steps[i]);
                    std.debug.print("\noutput_coord[{}]={}, input_coord[{}]={}", .{ i, output_coords[i], i, input_coords[i] });
                }

                // Get input value
                const input_idx = try self.flatten_index(input_coords);
                output_data[output_idx] = self.data[input_idx];
                std.debug.print("\noutput_idx={}, input_idx={}, value={}", .{ output_idx, input_idx, output_data[output_idx] });
            }

            return Tensor(T){
                .data = output_data,
                .shape = output_shape,
                .size = total_elements,
                .allocator = self.allocator,
            };
        }

        /// Split a tensor into multiple tensors along a specified axis.
        /// If split_sizes is null, the tensor is split into equal parts.
        /// If split_sizes is provided, it specifies the size of each split.
        /// Negative axis values count from the back (-1 means last axis).
        /// Returns an array of tensors that must be freed by the caller.
        pub fn split(self: *@This(), axis: i64, split_sizes: ?[]const usize) ![]Tensor(T) {
            // Handle negative axis
            const positive_axis = @as(usize, @intCast(if (axis < 0) @as(i64, @intCast(self.shape.len)) + axis else axis));
            if (positive_axis >= self.shape.len) return TensorError.InvalidAxis;

            // Calculate split sizes
            const dim_size = self.shape[positive_axis];
            var sizes = std.ArrayList(usize).init(self.allocator.*);
            defer sizes.deinit();

            if (split_sizes) |s| {
                // Validate and use provided split sizes
                var total_size: usize = 0;
                for (s) |size| {
                    try sizes.append(size);
                    total_size += size;
                }
                if (total_size != dim_size) return TensorError.InvalidSplitSize;
            } else {
                // Split into equal parts
                if (dim_size == 0) return TensorError.InvalidSplitSize;
                const split_size = dim_size;
                try sizes.append(split_size);
            }

            // Create output tensors
            var output_tensors = try self.allocator.alloc(Tensor(T), sizes.items.len);
            errdefer {
                for (output_tensors) |*tensor| {
                    tensor.deinit();
                }
                self.allocator.free(output_tensors);
            }

            var offset: usize = 0;
            for (sizes.items, 0..) |split_size, i| {
                // Create shape for the split tensor
                var new_shape = try self.allocator.alloc(usize, self.shape.len);
                errdefer self.allocator.free(new_shape);
                @memcpy(new_shape, self.shape);
                new_shape[positive_axis] = split_size;

                // Calculate total size for the split tensor
                var total_size: usize = 1;
                for (new_shape) |dim| {
                    total_size *= dim;
                }

                // Allocate memory for the split tensor's data
                var new_data = try self.allocator.alloc(T, total_size);
                errdefer self.allocator.free(new_data);

                // Calculate strides
                var stride: usize = 1;
                for (positive_axis + 1..self.shape.len) |j| {
                    stride *= self.shape[j];
                }

                // Copy data to the split tensor
                const block_size = split_size * stride;
                const num_blocks = total_size / block_size;

                var block_idx: usize = 0;
                while (block_idx < num_blocks) : (block_idx += 1) {
                    const src_start = offset + block_idx * dim_size * stride;
                    const dst_start = block_idx * split_size * stride;
                    const copy_size = split_size * stride;
                    @memcpy(new_data[dst_start .. dst_start + copy_size], self.data[src_start .. src_start + copy_size]);
                }

                // Create the split tensor
                output_tensors[i] = .{
                    .data = new_data,
                    .size = total_size,
                    .shape = new_shape,
                    .allocator = self.allocator,
                };

                offset += split_size * stride;
            }

            return output_tensors;
        }
    };
}

/// Recursive function to flatten a multidimensional array
fn flattenArray(comptime T: type, arr: anytype, flatArr: []T, startIndex: usize) usize {
    var idx = startIndex;

    const arrTypeInfo = @typeInfo(@TypeOf(arr));

    if (arrTypeInfo == .Array or arrTypeInfo == .Pointer) {
        // if arr is a lice or 1d  DIRECTLY COPY
        if (@TypeOf(arr[0]) == T) {
            for (arr) |val| {
                flatArr[idx] = val;
                idx += 1;
            }
        } else {
            // iff arr is mulltidimensional array recursive call
            for (arr) |subArray| {
                idx = flattenArray(T, subArray, flatArr, idx);
            }
        }
    } else {
        @panic("The type of `arr` is not compatible with the required type.");
    }

    return idx;
}
