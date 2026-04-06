use std::path::Path;

fn main() {
    // macOS: Apple Accelerate framework (vecLib BLAS — AMX-optimized on M4)
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-cfg=feature=\"cblas\"");
        println!("cargo:warning=Apple Accelerate BLAS enabled (cblas_sgemm)");
    }

    // Windows / Linux: link OpenBLAS (provides identical cblas_sgemm interface)
    #[cfg(not(target_os = "macos"))]
    {
        if let Ok(path) = std::env::var("OPENBLAS_PATH") {
            // Explicit path supplied (e.g. vcpkg install, pre-built zip)
            println!("cargo:rustc-link-search=native={}", path);
            println!("cargo:rustc-link-lib=openblas");
            println!("cargo:rustc-cfg=feature=\"cblas\"");
            println!("cargo:warning=OpenBLAS enabled from OPENBLAS_PATH={}", path);
        } else {
            // Fall back to system OpenBLAS (apt / pacman / choco / vcpkg global)
            println!("cargo:rustc-link-lib=openblas");
            println!("cargo:rustc-cfg=feature=\"cblas\"");
            println!("cargo:warning=System OpenBLAS enabled (set OPENBLAS_PATH if link fails)");
        }
    }

    // Zig NEON kernels (optional — ARM only, skip on x86_64)
    #[cfg(target_arch = "aarch64")]
    {
        let kernel_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent().unwrap()
            .parent().unwrap()
            .join("kernels/zig");

        let lib_path = kernel_dir.join("libmatmul.a");

        if lib_path.exists() {
            println!("cargo:rustc-link-search=native={}", kernel_dir.display());
            println!("cargo:rustc-link-lib=static=matmul");
            println!("cargo:rustc-cfg=feature=\"zig_kernels\"");
            println!("cargo:warning=Zig NEON kernels enabled");
        }

        println!("cargo:rerun-if-changed={}", lib_path.display());
    }
    println!("cargo:rerun-if-env-changed=OPENBLAS_PATH");

    // CUDA detection (Windows / Linux only — macOS has no NVCC toolchain).
    // The `cuda` feature is opt-in; build.rs just emits a helpful warning when
    // CUDA_PATH is present so the user knows they can enable it.
    #[cfg(not(target_os = "macos"))]
    {
        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            println!(
                "cargo:warning=CUDA detected at {} — enable with --features cuda",
                cuda_path
            );
        }
    }
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
}
