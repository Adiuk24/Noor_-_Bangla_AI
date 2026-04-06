use std::path::Path;

fn main() {
    // Apple Accelerate framework (vecLib BLAS) — always available on macOS
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-cfg=feature=\"accelerate\"");
        println!("cargo:warning=Apple Accelerate framework enabled (cblas_sgemm)");
    }

    // Zig NEON kernels (optional — for activations, norms)
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
