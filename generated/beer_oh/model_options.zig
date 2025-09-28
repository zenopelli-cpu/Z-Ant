
pub const lib = @import("lib_beer_oh.zig");
pub const name = "beer_oh";
pub const input_shape = [4]u32{ 1, 96, 96, 1 };
pub const output_data_len = 432;
pub const input_data_type = f32;
pub const output_data_type = f32;
pub const has_inputs = true;
pub const user_tests: bool = false;
pub const user_tests_path = "generated/beer_oh/user_tests.json";
pub const have_log: bool = true;
pub const is_dynamic: bool =true;