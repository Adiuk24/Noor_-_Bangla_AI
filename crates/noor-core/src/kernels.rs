//! FFI bridge to accelerated compute kernels.
//! Priority: CUDA (cuBLAS) > CBLAS (Accelerate on macOS, OpenBLAS on Windows/Linux) > Zig NEON > Rust fallback.

// CBLAS — Apple Accelerate on macOS, OpenBLAS on Windows/Linux.
// Both expose the identical cblas_sgemm symbol and ABI.
#[cfg(feature = "cblas")]
extern "C" {
    /// cblas_sgemm: C = alpha * A @ B + beta * C
    /// CblasRowMajor=101, CblasNoTrans=111
    fn cblas_sgemm(
        order: i32, trans_a: i32, trans_b: i32,
        m: i32, n: i32, k: i32,
        alpha: f32,
        a: *const f32, lda: i32,
        b: *const f32, ldb: i32,
        beta: f32,
        c: *mut f32, ldc: i32,
    );
}

#[cfg(feature = "zig_kernels")]
extern "C" {
    /// Tiled matmul with NEON vectorization. C += A @ B.
    pub fn noor_matmul_f32(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: u32,
        k: u32,
        n: u32,
    );

    /// RMSNorm forward.
    pub fn noor_rmsnorm_f32(
        x: *const f32,
        w: *const f32,
        out: *mut f32,
        n_vecs: u32,
        dim: u32,
        eps: f32,
    );

    /// SiLU activation: x / (1 + exp(-x))
    pub fn noor_silu_f32(x: *const f32, out: *mut f32, len: u32);

    /// GELU activation (approximate)
    pub fn noor_gelu_f32(x: *const f32, out: *mut f32, len: u32);

    /// Zero a buffer
    pub fn noor_zero_f32(ptr: *mut f32, len: u32);
}

// ── CUDA / cuBLAS backend (optional, enabled with --features cuda) ──────────
//
// cuBLAS uses column-major storage, but our tensors are row-major.
// The standard trick to compute C = A @ B with row-major data via a
// column-major BLAS:
//
//   cuBLAS sees the *transpose* of each matrix.
//   So feed it:  C^T = B^T @ A^T
//   i.e. swap A↔B and swap m↔n, keep k the same, leave both ops as OP_N.
//
// This is the same trick used by PyTorch / cuDNN / CUTLASS for row-major GEMM.
#[cfg(feature = "cuda")]
mod cuda_backend {
    use std::sync::{Arc, OnceLock};

    use cudarc::cublas::safe::{CudaBlas, GemmConfig};
    use cudarc::cublas::sys::cublasOperation_t;
    use cudarc::driver::{CudaContext, CudaStream};

    /// Lazily-initialised (device, stream, blas) triple.
    /// The stream is stored as Arc so it can be shared between the blas handle
    /// and ad-hoc htod/dtoh copies executed on the same stream.
    struct CudaState {
        stream: Arc<CudaStream>,
        blas: CudaBlas,
    }

    static CUDA_STATE: OnceLock<CudaState> = OnceLock::new();

    fn get_state() -> &'static CudaState {
        CUDA_STATE.get_or_init(|| {
            let ctx = CudaContext::new(0).expect("Failed to initialise CUDA device 0");
            let stream = ctx.default_stream();
            let blas = CudaBlas::new(stream.clone())
                .expect("Failed to create cuBLAS handle");
            CudaState { stream, blas }
        })
    }

    /// Compute C (m×n) = A (m×k) @ B (k×n), all row-major, via cuBLAS.
    pub fn matmul_cuda(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        let state = get_state();
        let stream = &state.stream;

        // Copy inputs to device memory on the same stream.
        let a_dev = stream.memcpy_stod(a).expect("htod copy of A failed");
        let b_dev = stream.memcpy_stod(b).expect("htod copy of B failed");
        let mut c_dev = stream
            .alloc_zeros::<f32>(m * n)
            .expect("CUDA alloc for C failed");

        // cuBLAS GEMM: C^T = B^T @ A^T  (swapped A/B and m/n → row-major result)
        //
        //  GemmConfig { transa, transb, m, n, k, alpha, lda, ldb, beta, ldc }
        //
        //  We call sgemm(OP_N, OP_N, n, m, k, 1.0, B, n, A, k, 0.0, C, n)
        //  which performs:  C (n×m col-major) = B (n×k col-major) @ A (k×m col-major)
        //  Interpreted as row-major: C (m×n) = A (m×k) @ B (k×n)  ✓
        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as i32,   // leading dim of B^T / C (= n in row-major speak)
            n: m as i32,   // rows of A^T (= m in row-major speak)
            k: k as i32,
            alpha: 1.0f32,
            lda: n as i32, // leading dim of B (row-major B has n cols → lda=n)
            ldb: k as i32, // leading dim of A (row-major A has k cols → ldb=k)
            beta: 0.0f32,
            ldc: n as i32, // leading dim of C (row-major C has n cols → ldc=n)
        };

        unsafe {
            use cudarc::cublas::safe::Gemm;
            state
                .blas
                .gemm(cfg, &b_dev, &a_dev, &mut c_dev)
                .expect("cuBLAS sgemm failed");
        }

        // Copy result back to host (synchronises the stream).
        let result = stream
            .memcpy_dtov(&c_dev)
            .expect("dtoh copy of C failed");
        c.copy_from_slice(&result);
    }
}

/// Dispatch matmul. Priority: CUDA (cuBLAS) > CBLAS (Accelerate/OpenBLAS) > Zig NEON > Rust tiled.
pub fn matmul_dispatch(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    #[cfg(feature = "cuda")]
    {
        cuda_backend::matmul_cuda(a, b, c, m, k, n);
        return;
    }

    #[cfg(feature = "cblas")]
    {
        // Accelerate (AMX on M4) or OpenBLAS (AVX-512 on i7-14700K) — fastest path
        unsafe {
            cblas_sgemm(
                101,  // CblasRowMajor
                111,  // CblasNoTrans
                111,  // CblasNoTrans
                m as i32, n as i32, k as i32,
                1.0,  // alpha
                a.as_ptr(), k as i32,
                b.as_ptr(), n as i32,
                0.0,  // beta (overwrite C, don't accumulate)
                c.as_mut_ptr(), n as i32,
            );
        }
        return;
    }

    #[cfg(all(feature = "zig_kernels", not(feature = "cblas")))]
    unsafe {
        noor_matmul_f32(
            a.as_ptr(),
            b.as_ptr(),
            c.as_mut_ptr(),
            m as u32,
            k as u32,
            n as u32,
        );
        return;
    }

    #[cfg(not(any(feature = "cblas", feature = "zig_kernels")))]
    {
        super::tensor::tiled_matmul_fallback(a, b, c, m, k, n);
    }
}

/// Dispatch RMSNorm to Zig kernel if available.
pub fn rmsnorm_dispatch(x: &[f32], w: &[f32], out: &mut [f32], n_vecs: usize, dim: usize, eps: f32) {
    #[cfg(feature = "zig_kernels")]
    unsafe {
        noor_rmsnorm_f32(
            x.as_ptr(),
            w.as_ptr(),
            out.as_mut_ptr(),
            n_vecs as u32,
            dim as u32,
            eps,
        );
        return;
    }

    #[cfg(not(feature = "zig_kernels"))]
    {
        // Rust fallback (inline)
        for v in 0..n_vecs {
            let offset = v * dim;
            let mut sum_sq = 0.0f64;
            for d in 0..dim {
                let val = x[offset + d] as f64;
                sum_sq += val * val;
            }
            let rms = ((sum_sq / dim as f64) + eps as f64).sqrt();
            let inv_rms = 1.0 / rms;
            for d in 0..dim {
                out[offset + d] = (x[offset + d] as f64 * inv_rms * w[d] as f64) as f32;
            }
        }
    }
}

/// Dispatch SiLU to Zig kernel if available.
pub fn silu_dispatch(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());

    #[cfg(feature = "zig_kernels")]
    unsafe {
        noor_silu_f32(x.as_ptr(), out.as_mut_ptr(), x.len() as u32);
        return;
    }

    #[cfg(not(feature = "zig_kernels"))]
    {
        for i in 0..x.len() {
            out[i] = x[i] / (1.0 + (-x[i]).exp());
        }
    }
}

/// Dispatch GELU to Zig kernel if available.
pub fn gelu_dispatch(x: &[f32], out: &mut [f32]) {
    debug_assert_eq!(x.len(), out.len());

    #[cfg(feature = "zig_kernels")]
    unsafe {
        noor_gelu_f32(x.as_ptr(), out.as_mut_ptr(), x.len() as u32);
        return;
    }

    #[cfg(not(feature = "zig_kernels"))]
    {
        let sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        for i in 0..x.len() {
            let x_val = x[i];
            let inner = sqrt_2_pi * (x_val + 0.044715 * x_val * x_val * x_val);
            out[i] = 0.5 * x_val * (1.0 + inner.tanh());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_dispatch() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0f32, 6.0, 7.0, 8.0]; // 2x2
        let mut c = vec![0.0f32; 4]; // 2x2
        matmul_dispatch(&a, &b, &mut c, 2, 2, 2);
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert!((c[0] - 19.0).abs() < 1e-4, "c[0]={}", c[0]);
        assert!((c[1] - 22.0).abs() < 1e-4, "c[1]={}", c[1]);
        assert!((c[2] - 43.0).abs() < 1e-4, "c[2]={}", c[2]);
        assert!((c[3] - 50.0).abs() < 1e-4, "c[3]={}", c[3]);
    }

    #[test]
    fn test_matmul_dispatch_large() {
        let m = 64;
        let k = 32;
        let n = 48;
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32) * 0.01).collect();
        let mut c = vec![0.0f32; m * n];
        matmul_dispatch(&a, &b, &mut c, m, k, n);

        // Spot check with naive
        let mut expected = 0.0f32;
        for ki in 0..k {
            expected += a[0 * k + ki] * b[ki * n + 0];
        }
        assert!((c[0] - expected).abs() < 0.1, "c[0]={}, expected={}", c[0], expected);
    }

    #[test]
    fn test_silu_dispatch() {
        let x = vec![0.0f32, 1.0, -1.0, 5.0];
        let mut out = vec![0.0f32; 4];
        silu_dispatch(&x, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!(out[1] > 0.7); // silu(1) ≈ 0.731
        assert!(out[3] > 4.9); // silu(5) ≈ 4.966
    }

    #[test]
    fn test_gelu_dispatch() {
        let x = vec![0.0f32, 1.0, -1.0];
        let mut out = vec![0.0f32; 3];
        gelu_dispatch(&x, &mut out);
        assert!((out[0] - 0.0).abs() < 1e-5);
        assert!(out[1] > 0.8); // gelu(1) ≈ 0.841
    }

    #[test]
    fn test_rmsnorm_dispatch() {
        let x = vec![1.0f32, 2.0, 3.0, 4.0]; // 2 vectors of dim 2
        let w = vec![1.0f32, 1.0]; // weight = ones
        let mut out = vec![0.0f32; 4];
        rmsnorm_dispatch(&x, &w, &mut out, 2, 2, 1e-6);

        // First vec [1,2]: rms = sqrt((1+4)/2) = sqrt(2.5)
        // normalized: [1/sqrt(2.5), 2/sqrt(2.5)]
        let rms1 = (2.5f32).sqrt();
        assert!((out[0] - 1.0 / rms1).abs() < 0.01);
        assert!((out[1] - 2.0 / rms1).abs() < 0.01);
    }
}
