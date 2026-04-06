const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build all Noor kernels as a shared library
    const lib = b.addSharedLibrary(.{
        .name = "noor_kernels",
        .root_source_file = b.path("matmul.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Add other kernel source files
    // lib.addCSourceFile(.{ .file = b.path("../c/some_kernel.c") });

    b.installArtifact(lib);

    // Also build a static library for linking into Rust
    const static_lib = b.addStaticLibrary(.{
        .name = "noor_kernels_static",
        .root_source_file = b.path("matmul.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(static_lib);
}
