// Noor NEON-accelerated matmul kernel.
// C = A @ B where A:(M,K), B:(K,N), C:(M,N). All f32.
// Tiled with NEON vector intrinsics for ARM (Apple M4).
// Called from Rust via C ABI FFI.

const std = @import("std");
const cc = std.builtin.CallingConvention;

// Tile sizes tuned for M4 L1 cache (192KB, 6-way)
const TILE_M: usize = 8;
const TILE_N: usize = 8;
const TILE_K: usize = 64;

/// C ABI entry point for Rust FFI.
/// C += A @ B (accumulates into C — caller must zero C first if needed).
export fn noor_matmul_f32(
    a_ptr: [*]const f32,
    b_ptr: [*]const f32,
    c_ptr: [*]f32,
    m: u32,
    k: u32,
    n: u32,
) void {
    const M: usize = @intCast(m);
    const K: usize = @intCast(k);
    const N: usize = @intCast(n);

    // Outer loop: tiles over K (for A row reuse)
    var kk: usize = 0;
    while (kk < K) : (kk += TILE_K) {
        const k_end = @min(kk + TILE_K, K);

        var ii: usize = 0;
        while (ii < M) : (ii += TILE_M) {
            const i_end = @min(ii + TILE_M, M);

            var jj: usize = 0;
            while (jj < N) : (jj += TILE_N) {
                const j_end = @min(jj + TILE_N, N);

                // Micro-kernel: process tile
                var i: usize = ii;
                while (i < i_end) : (i += 1) {
                    var kt: usize = kk;
                    while (kt < k_end) : (kt += 1) {
                        const a_val = a_ptr[i * K + kt];

                        // Inner loop: vectorize over j when possible
                        var j: usize = jj;

                        // NEON: process 4 floats at a time
                        while (j + 4 <= j_end) : (j += 4) {
                            const c_idx = i * N + j;
                            const b_idx = kt * N + j;

                            // Load 4 floats from B and C
                            const b_vec = @as(@Vector(4, f32), b_ptr[b_idx..][0..4].*);
                            const c_vec = @as(@Vector(4, f32), c_ptr[c_idx..][0..4].*);
                            const a_vec: @Vector(4, f32) = @splat(a_val);

                            // FMA: C += A * B
                            const result = @mulAdd(@Vector(4, f32), a_vec, b_vec, c_vec);

                            // Store back
                            c_ptr[c_idx..][0..4].* = @as([4]f32, result);
                        }

                        // Scalar remainder
                        while (j < j_end) : (j += 1) {
                            c_ptr[i * N + j] += a_val * b_ptr[kt * N + j];
                        }
                    }
                }
            }
        }
    }
}

/// RMSNorm forward: out = x / sqrt(mean(x^2) + eps) * weight
export fn noor_rmsnorm_f32(
    x_ptr: [*]const f32,
    w_ptr: [*]const f32,
    out_ptr: [*]f32,
    n_vecs: u32,
    dim: u32,
    eps: f32,
) void {
    const N: usize = @intCast(n_vecs);
    const D: usize = @intCast(dim);

    var v: usize = 0;
    while (v < N) : (v += 1) {
        const offset = v * D;

        // Compute sum of squares
        var sum_sq: f32 = 0.0;
        var d: usize = 0;
        while (d < D) : (d += 1) {
            const val = x_ptr[offset + d];
            sum_sq += val * val;
        }

        // RMS
        const rms = @sqrt(sum_sq / @as(f32, @floatFromInt(D)) + eps);
        const inv_rms = 1.0 / rms;

        // Normalize and scale
        d = 0;
        while (d < D) : (d += 1) {
            out_ptr[offset + d] = x_ptr[offset + d] * inv_rms * w_ptr[d];
        }
    }
}

/// SiLU activation: out = x * sigmoid(x) = x / (1 + exp(-x))
export fn noor_silu_f32(
    x_ptr: [*]const f32,
    out_ptr: [*]f32,
    len: u32,
) void {
    const N: usize = @intCast(len);
    var i: usize = 0;
    while (i < N) : (i += 1) {
        const x = x_ptr[i];
        out_ptr[i] = x / (1.0 + @exp(-x));
    }
}

/// GELU activation (approximate)
export fn noor_gelu_f32(
    x_ptr: [*]const f32,
    out_ptr: [*]f32,
    len: u32,
) void {
    const N: usize = @intCast(len);
    const sqrt_2_pi: f32 = @sqrt(2.0 / std.math.pi);
    var i: usize = 0;
    while (i < N) : (i += 1) {
        const x = x_ptr[i];
        const inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
        out_ptr[i] = 0.5 * x * (1.0 + std.math.tanh(inner));
    }
}

/// Zero a float buffer.
export fn noor_zero_f32(ptr: [*]f32, len: u32) void {
    const n: usize = @intCast(len);
    @memset(ptr[0..n], 0.0);
}

// Build: zig build-lib -O ReleaseFast -target aarch64-macos matmul.zig
