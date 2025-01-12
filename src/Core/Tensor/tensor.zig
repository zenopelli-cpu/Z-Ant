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

        /// Print the Tensor() in the shape of a matrix
        pub fn printMultidim(self: *@This()) void {
            const dim = self.shape.len;
            for (0..self.shape[dim - 2]) |i| {
                std.debug.print("\n[ ", .{});
                for (0..self.shape[dim - 1]) |j| {
                    std.debug.print("{} ", .{self.data[i * self.shape[dim - 1] + j]});
                }
                std.debug.print("]", .{});
            }
        }

        /// Returns a Tensor self transposed. Does not modify self.
        /// It sobstitute init(), but defer yourTensor.deinit() is still necessary.
        pub fn transpose2D(self: *@This()) !Tensor(T) {
            if (self.shape.len != 2) {
                return error.InvalidDimension; // For simplicity, let's focus on 2D for now
            }

            const allocator = self.allocator;

            // Shape of the transposed tensor
            const transposed_shape: [2]usize = [_]usize{ self.shape[1], self.shape[0] };
            const tensorShape = try allocator.alloc(usize, self.shape.len);
            @memcpy(tensorShape, &transposed_shape);

            // Allocate space for transposed data
            const transposed_data = try allocator.alloc(T, self.size);

            // Perform the transposition
            for (0..self.shape[0]) |i| {
                for (0..self.shape[1]) |j| {
                    // For 2D tensor, flatten the index and swap row/column positions
                    const old_idx = i * self.shape[1] + j;
                    const new_idx = j * self.shape[0] + i;
                    transposed_data[new_idx] = self.data[old_idx];
                }
            }

            return Tensor(T){
                .data = transposed_data,
                .size = self.size,
                .shape = tensorShape,
                .allocator = allocator,
            };
        }

        /// Returns a Tensor self transposed. Does not modify self.
        /// By default, it transposes the tensor to the reverse shape.
        pub fn transposeDefault(self: *@This()) !Tensor(T) {
            // Reverse the shape of the tensor
            const tensorShape = try self.allocator.alloc(usize, self.shape.len);
            for (0..self.shape.len) |i| {
                tensorShape[i] = self.shape.len - 1 - i;
            }

            return self.transpose(tensorShape);
        }

        /// Returns a Tensor self transposed. Does not modify self.
        pub fn transpose(self: *@This(), perms: []usize) !Tensor(T) {
            defer self.allocator.free(perms);
            const num_dims = self.shape.len;
            if (perms.len != num_dims) {
                return error.InvalidDimension;
            }

            // Check that the permutation is valid
            var bitmap = try self.allocator.alloc(bool, perms.len);
            defer self.allocator.free(bitmap);

            for (perms) |perm| {
                if (perm >= perms.len) {
                    return error.InvalidPermutation;
                }
                if (bitmap[perm] == true) {
                    return error.InvalidPermutation;
                }
                bitmap[perm] = true;
            }

            // Allocate space for the new shape
            const new_shape = try self.allocator.alloc(usize, num_dims);
            for (0..num_dims) |i| {
                new_shape[i] = self.shape[perms[i]];
            }
            defer self.allocator.free(new_shape);

            // Create the new tensor
            const new_tensor = try Tensor(T).fromShape(self.allocator, new_shape);

            // Copy data to the new tensor
            for (0..self.size) |i| {
                new_tensor.data[i] = self.data[i];
            }

            return new_tensor;
        }

        /// Performs Element-wise Multiplication of two tensors.
        pub fn mul(lhs: *@This(), rhs: *@This()) !Tensor(T) {
            if (lhs.size != rhs.size) {
                return TensorError.MismatchedShape;
            }

            const allocator = lhs.allocator;
            const result = try Tensor(T).fromShape(allocator, lhs.shape);

            for (0..lhs.size) |i| {
                result.data[i] = lhs.data[i] * rhs.data[i];
            }

            return result;
        }

        /// Returns true if the Tensor is one-hot encoded
        fn isOneHot(self: *@This()) !bool {
            const elems_row = self.shape[self.shape.len - 1];
            if (elems_row == 0) {
                return TensorError.EmptyTensor;
            }
            const numb_rows = self.size / elems_row;
            if (numb_rows == 0) {
                return TensorError.ZeroSizeTensor;
            }

            for (0..numb_rows) |row| {
                var oneHotFound = false;
                for (0..self.shape[self.shape.len - 1]) |i| {
                    if (self.data[row * elems_row + i] == 1 and !oneHotFound) {
                        if (!oneHotFound) oneHotFound = true else return TensorError.NotOneHotEncoded;
                    }
                }
            }

            return true;
        }

        /// Returns true only if all the values of shape and data are valid numbers
        pub fn isSafe(self: *@This()) !void {
            switch (@typeInfo(T)) {
                .Float => {
                    // Loop over tensor data
                    for (self.data) |*value| {
                        if (std.math.isNan(value.*)) return TensorError.NanValue;
                        if (!std.math.isFinite(value.*)) return TensorError.NotFiniteValue;
                    }

                    // Loop over tensor shape
                    for (self.shape) |*value| {
                        if (std.math.isNan(value.*)) return TensorError.NanValue;
                    }
                },
                else => {
                    // If T is not Float, skip isSafe checks
                    return;
                },
            }
        }

        /// Set all tensor values to zero.
        pub fn setToZero(self: *@This()) !void {
            if (self.size == 0) {
                return TensorError.TensorNotInitialized;
            }
            @memset(self.data, 0);
        }

        /// Method to add a top&bottom padding and a left&right padding.
        /// At the moment the function only supports 2 padding params, but the method
        /// is already set to have different left, right, top and bottom padding values.
        pub fn addPaddingAndDilation(
            self: *@This(),
            upDownPadding: usize,
            leftRightPadding: usize,
            verticalDil: usize,
            horizontalDil: usize,
        ) !void {

            //checks on padding dim (usize is alway >= 0)
            if (self.shape.len < 2) return TensorError.TooSmallToPadding;

            const upPadding = upDownPadding;
            const downPadding = upDownPadding;
            const leftPadding = leftRightPadding;
            const rightPadding = leftRightPadding;
            const dim = self.shape.len;

            const new_row_numb = self.shape[dim - 2] + upPadding + downPadding + verticalDil * (self.shape[dim - 2] - 1);
            const new_col_numb = self.shape[dim - 1] + leftPadding + rightPadding + horizontalDil * (self.shape[dim - 1] - 1);
            //std.debug.print("\n new_row_numb: {} new_col_numb:{}", .{ new_row_numb, new_col_numb });

            //compute new shape
            const new_shape = try self.allocator.alloc(usize, dim);
            @memcpy(new_shape, self.shape);
            new_shape[dim - 1] = new_col_numb;
            new_shape[dim - 2] = new_row_numb;

            //compute new size
            var new_total_size: usize = 1;
            for (new_shape) |size_i| {
                new_total_size *= size_i;
            }

            //alloc new tensor.data memory space to all zero
            const new_data = try self.allocator.alloc(T, new_total_size);
            @memset(new_data, 0);

            const new_matrix_dim = new_row_numb * new_col_numb;
            const total_number_2DMatrices = new_total_size / new_matrix_dim;
            const old_matrix_dim = self.shape[dim - 2] * self.shape[dim - 1];
            const old_total_number_2DMatrices = self.size / old_matrix_dim; //just for check assertion
            std.debug.assert(total_number_2DMatrices == old_total_number_2DMatrices);

            for (0..total_number_2DMatrices) |matix_i| {
                const num_elem_prec_new_matr = matix_i * new_matrix_dim;
                const num_elem_prec_old_matr = matix_i * old_matrix_dim;
                // for (upPadding..new_row_numb - downPadding) |i| { //do a while!!
                //     for (leftPadding..new_col_numb - rightPadding) |j| {
                //         //std.debug.print("\n i:{}, j:{} new_data[{}], self.data[{}]", .{ i, j, i * new_col_numb + j, (i - upPadding) * (self.shape[dim - 1]) + (j - leftPadding) });

                //         new_data[num_elem_prec_new_matr + i * new_col_numb + j] = self.data[num_elem_prec_old_matr + (i - upPadding) * (self.shape[dim - 1]) + (j - leftPadding)];
                //         //j += horizontalDil;
                //     }
                // }
                var i = upPadding;
                var old_row: usize = 0;
                while (i < new_row_numb - downPadding) : (i += (1 + verticalDil)) {
                    var j = leftPadding;
                    var old_col: usize = 0;
                    while (j < new_col_numb - rightPadding) : (j += (1 + horizontalDil)) {
                        const idx_new_matr = num_elem_prec_new_matr + i * new_col_numb + j;
                        const idx_old_matr = num_elem_prec_old_matr + old_row * (self.shape[dim - 1]) + old_col;
                        new_data[idx_new_matr] = self.data[idx_old_matr];
                        old_col += 1;
                    }
                    old_row += 1;
                }
            }

            //free all old attributes and setting new ones
            self.allocator.free(self.data);
            self.allocator.free(self.shape);

            self.shape = new_shape;
            self.data = new_data;
            self.size = new_total_size;
        }

        /// Helper function to flip the kernel (rotate 180 degrees horizontaly and vertically)
        /// ex:
        ///  flip( [[a, b], [c, d], [e, f]] ) = [[f, e], [d, c], [b, a]]
        pub fn flip(self: *@This()) !Tensor(T) {
            const kernel_dim = self.shape.len;
            const kernel_row = self.shape[kernel_dim - 2];
            const kernel_cols = self.shape[kernel_dim - 1];
            const matrix_dim = kernel_cols * kernel_row;

            //create and initialize the new shape
            const flipped_shape = try self.allocator.alloc(usize, self.shape.len);
            defer self.allocator.free(flipped_shape);
            @memcpy(flipped_shape, self.shape);

            var flipped_kernel = try Tensor(T).fromShape(self.allocator, flipped_shape);

            const total_number_2DMatrices = flipped_kernel.size / matrix_dim;

            for (0..total_number_2DMatrices) |matix_i| {
                for (0..kernel_row) |i| {
                    for (0..kernel_cols) |j| {
                        flipped_kernel.data[(matix_i + 1) * matrix_dim - (i * kernel_cols + j + 1)] = self.data[matix_i * matrix_dim + i * kernel_cols + j];
                    }
                }
            }

            return flipped_kernel;
        }

        /// Gather elements from the tensor along an axis using the provided indices.
        /// The axis parameter specifies the axis along which the elements will be gathered.
        /// The indices tensor must have the same number of dimensions as the input tensor, except for the axis dimension.
        /// The shape of the output tensor is the same as the shape of the indices tensor, with the axis dimension removed.
        /// The output tensor is created by copying elements from the input tensor using the indices tensor.
        pub fn gather(self: *@This(), indices: Tensor(usize), axis: usize) !@This() {
            // Validate that the axis is within the tensor's dimensions
            if (axis >= self.shape.len) {
                return TensorError.InvalidAxis;
            }

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

        /// Resize the input tensor using interpolation.
        /// Supports 'nearest', 'linear', and 'cubic' interpolation modes.
        pub fn resize(self: *@This(), comptime mode: []const u8, scales: ?[]const f32, sizes: ?[]const usize, coordinate_transformation_mode: []const u8) !@This() {
            if (scales == null and sizes == null) {
                return TensorError.InvalidInput;
            }
            if (scales != null and sizes != null) {
                return TensorError.InvalidInput;
            }

            // Calculate output dimensions
            var output_shape = try self.allocator.alloc(usize, self.shape.len);
            errdefer self.allocator.free(output_shape);

            if (scales) |s| {
                if (s.len != self.shape.len) {
                    return TensorError.InvalidInput;
                }
                for (0..self.shape.len) |i| {
                    output_shape[i] = @intFromFloat(@floor(@as(f32, @floatFromInt(self.shape[i])) * s[i]));
                }
            } else if (sizes) |sz| {
                if (sz.len != self.shape.len) {
                    return TensorError.InvalidInput;
                }
                @memcpy(output_shape, sz);
            }

            // Calculate total size of output tensor
            var total_size: usize = 1;
            for (output_shape) |dim| {
                total_size *= dim;
            }

            // Allocate memory for output data
            const output_data = try self.allocator.alloc(T, total_size);
            errdefer self.allocator.free(output_data);

            // Perform interpolation based on mode
            if (std.mem.eql(u8, mode, "nearest")) {
                try self.nearest_interpolation(output_data, output_shape, coordinate_transformation_mode);
            } else if (std.mem.eql(u8, mode, "linear")) {
                try self.linear_interpolation(output_data, output_shape, coordinate_transformation_mode);
            } else if (std.mem.eql(u8, mode, "cubic")) {
                try self.cubic_interpolation(output_data, output_shape, coordinate_transformation_mode);
            } else {
                return TensorError.UnsupportedMode;
            }

            return @This(){
                .data = output_data,
                .shape = output_shape,
                .size = total_size,
                .allocator = self.allocator,
            };
        }

        fn nearest_interpolation(self: *@This(), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
            const input_strides = try self.getStrides();
            defer self.allocator.free(input_strides);
            const output_strides = try self.allocator.alloc(usize, output_shape.len);
            defer self.allocator.free(output_strides);

            // Calculate output strides
            var stride: usize = 1;
            var idx: usize = output_shape.len;
            while (idx > 0) {
                idx -= 1;
                output_strides[idx] = stride;
                stride *= output_shape[idx];
            }

            var output_indices = try self.allocator.alloc(usize, output_shape.len);
            defer self.allocator.free(output_indices);
            @memset(output_indices, 0);

            var done = false;
            while (!done) {
                var output_idx: usize = 0;
                var input_idx: usize = 0;

                for (0..output_shape.len) |i| {
                    const scale = @as(f32, @floatFromInt(output_shape[i])) / @as(f32, @floatFromInt(self.shape[i]));
                    var input_pos: f32 = undefined;

                    if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
                        input_pos = (@as(f32, @floatFromInt(output_indices[i])) + 0.5) / scale - 0.5;
                    } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
                        input_pos = @as(f32, @floatFromInt(output_indices[i])) * @as(f32, @floatFromInt(self.shape[i] - 1)) / @as(f32, @floatFromInt(output_shape[i] - 1));
                    } else { // asymmetric
                        input_pos = @as(f32, @floatFromInt(output_indices[i])) / scale;
                    }

                    const input_idx_i = @as(i32, @intFromFloat(@round(input_pos)));
                    const clamped_idx = @min(@max(input_idx_i, 0), @as(i32, @intCast(self.shape[i] - 1)));
                    input_idx += @as(usize, @intCast(clamped_idx)) * input_strides[i];
                    output_idx += output_indices[i] * output_strides[i];
                }

                output_data[output_idx] = self.data[input_idx];

                // Increment indices
                done = true;
                for (0..output_shape.len) |i| {
                    output_indices[output_shape.len - 1 - i] += 1;
                    if (output_indices[output_shape.len - 1 - i] < output_shape[output_shape.len - 1 - i]) {
                        done = false;
                        break;
                    }
                    output_indices[output_shape.len - 1 - i] = 0;
                }
            }
        }

        fn linear_interpolation(self: *@This(), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
            // For now, implement only for 1D and 2D tensors
            if (self.shape.len > 2) return TensorError.UnsupportedDimension;

            const input_strides = try self.getStrides();
            defer self.allocator.free(input_strides);

            var output_indices = try self.allocator.alloc(usize, output_shape.len);
            defer self.allocator.free(output_indices);
            @memset(output_indices, 0);

            var done = false;
            while (!done) {
                var output_idx: usize = 0;
                if (output_shape.len == 1) {
                    output_idx = output_indices[0];
                } else {
                    output_idx = output_indices[0] * output_shape[1] + output_indices[1];
                }

                // Calculate interpolation coordinates
                var x: f32 = undefined;
                if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
                    x = (@as(f32, @floatFromInt(output_indices[0])) + 0.5) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0])) - 0.5;
                } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
                    x = @as(f32, @floatFromInt(output_indices[0])) * @as(f32, @floatFromInt(self.shape[0] - 1)) / @as(f32, @floatFromInt(output_shape[0] - 1));
                } else { // asymmetric
                    x = @as(f32, @floatFromInt(output_indices[0])) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0]));
                }

                const x_floor = @floor(x);
                const x0 = @as(usize, @intFromFloat(@max(0, x_floor)));
                const x1 = @min(x0 + 1, self.shape[0] - 1);
                const dx = x - x_floor;

                if (self.shape.len == 1) {
                    const v0 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x0]))));
                    const v1 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x1]))));
                    const interpolated = v0 * (1 - dx) + v1 * dx;
                    output_data[output_idx] = @as(T, @intFromFloat(@round(interpolated)));
                } else {
                    var y: f32 = undefined;
                    if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
                        y = (@as(f32, @floatFromInt(output_indices[1])) + 0.5) * @as(f32, @floatFromInt(self.shape[1])) / @as(f32, @floatFromInt(output_shape[1])) - 0.5;
                    } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
                        y = @as(f32, @floatFromInt(output_indices[1])) * @as(f32, @floatFromInt(self.shape[1] - 1)) / @as(f32, @floatFromInt(output_shape[1] - 1));
                    } else { // asymmetric
                        y = @as(f32, @floatFromInt(output_indices[1])) * @as(f32, @floatFromInt(self.shape[1])) / @as(f32, @floatFromInt(output_shape[1]));
                    }

                    const y_floor = @floor(y);
                    const y0 = @as(usize, @intFromFloat(@max(0, y_floor)));
                    const y1 = @min(y0 + 1, self.shape[1] - 1);
                    const dy = y - y_floor;

                    const v00 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x0 * self.shape[1] + y0]))));
                    const v01 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x0 * self.shape[1] + y1]))));
                    const v10 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x1 * self.shape[1] + y0]))));
                    const v11 = @as(f32, @floatFromInt(@as(i32, @intCast(self.data[x1 * self.shape[1] + y1]))));

                    const tmp1 = v00 * (1 - dx) * (1 - dy);
                    const tmp2 = v01 * (1 - dx) * dy;
                    const tmp3 = v10 * dx * (1 - dy);
                    const tmp4 = v11 * dx * dy;

                    const interpolated = tmp1 + tmp2 + tmp3 + tmp4;
                    output_data[output_idx] = @as(T, @intFromFloat(@round(interpolated)));
                }

                // Increment indices
                done = true;
                for (0..output_shape.len) |i| {
                    output_indices[output_shape.len - 1 - i] += 1;
                    if (output_indices[output_shape.len - 1 - i] < output_shape[output_shape.len - 1 - i]) {
                        done = false;
                        break;
                    }
                    output_indices[output_shape.len - 1 - i] = 0;
                }
            }
        }

        fn cubic_interpolation(self: *@This(), output_data: []T, output_shape: []usize, coordinate_transformation_mode: []const u8) !void {
            // For simplicity, implement only for 1D tensors initially
            if (self.shape.len != 1) return TensorError.UnsupportedDimension;

            var output_idx: usize = 0;
            while (output_idx < output_shape[0]) : (output_idx += 1) {
                var x: f32 = undefined;
                if (std.mem.eql(u8, coordinate_transformation_mode, "half_pixel")) {
                    x = (@as(f32, @floatFromInt(output_idx)) + 0.5) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0])) - 0.5;
                } else if (std.mem.eql(u8, coordinate_transformation_mode, "align_corners")) {
                    x = @as(f32, @floatFromInt(output_idx)) * @as(f32, @floatFromInt(self.shape[0] - 1)) / @as(f32, @floatFromInt(output_shape[0] - 1));
                } else { // asymmetric
                    x = @as(f32, @floatFromInt(output_idx)) * @as(f32, @floatFromInt(self.shape[0])) / @as(f32, @floatFromInt(output_shape[0]));
                }

                const x0 = @as(i32, @intFromFloat(@floor(x)));
                const dx = x - @as(f32, @floatFromInt(x0));

                var sum: f32 = 0;
                var weight_sum: f32 = 0;

                var i: i32 = -1;
                while (i < 3) : (i += 1) {
                    const idx = x0 + i;
                    if (idx >= 0 and idx < @as(i32, @intCast(self.shape[0]))) {
                        const w = cubic_weight(dx - @as(f32, @floatFromInt(i)));
                        sum += @as(f32, @floatFromInt(@as(i32, @intCast(self.data[@as(usize, @intCast(idx))])))) * w;
                        weight_sum += w;
                    }
                }

                output_data[output_idx] = @as(T, @intFromFloat(@round(sum / weight_sum)));
            }
        }

        fn cubic_weight(x: f32) f32 {
            const a = -0.75;
            const abs_x = @abs(x);
            if (abs_x <= 1) {
                return ((a + 2) * abs_x - (a + 3)) * abs_x * abs_x + 1;
            } else if (abs_x < 2) {
                return ((a * abs_x - 5 * a) * abs_x + 8 * a) * abs_x - 4 * a;
            }
            return 0;
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
