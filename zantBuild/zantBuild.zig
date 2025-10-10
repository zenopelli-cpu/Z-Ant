const std = @import("std");

const ZantStepOptions = @import("zantStepOptions.zig").ZantStepOptions;
const ZantOptions = @import("zantOptions.zig").ZantOptions;

pub const ZantBuild = struct {
    zantOptions: ZantOptions,
    zantStepOptions: ZantStepOptions,

    pub fn init(b: *std.Build) !ZantBuild {
        const zantOptions = try ZantOptions.init(b);
        const zantStepOptions = try ZantStepOptions.init(b, zantOptions);

        return ZantBuild{
            .zantOptions = zantOptions,
            .zantStepOptions = zantStepOptions,
        };
    }
};
