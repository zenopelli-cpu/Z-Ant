
pub const lib = @import("lib_new2.zig");
pub const name = "new2";
pub const input_shape = [4]u32{ 1, 3, 96, 96 };
pub const output_data_len = 80;
pub const input_data_type = f32;
pub const output_data_type = f32;
pub const has_inputs = true;
pub const user_tests: bool = false;
pub const user_tests_path = "generated/new2/user_tests.json";
pub const have_log: bool = false;
pub const is_dynamic: bool =true;