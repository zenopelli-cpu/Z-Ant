const std = @import("std");
const protobuf = @import("protobuf.zig");

//TODO errori custom più precisi

//possibili errori dati dalla parse
pub const ParseError = protobuf.Error || std.mem.Allocator.Error || error{
    TagNotAvailable,
    UnknownOperator,
    InvalidAttributeType,
    MissingRequiredField,
};
