use std::path::Path;

fn main() {
    let kernel_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .parent().unwrap()
        .join("kernels/zig");

    let lib_path = kernel_dir.join("libmatmul.a");

    if lib_path.exists() {
        // Link the Zig NEON kernel library
        println!("cargo:rustc-link-search=native={}", kernel_dir.display());
        println!("cargo:rustc-link-lib=static=matmul");
        println!("cargo:rustc-cfg=feature=\"zig_kernels\"");
        println!("cargo:warning=Zig NEON kernels enabled: {}", lib_path.display());
    } else {
        println!("cargo:warning=Zig kernels not found at {}. Using Rust fallback.", lib_path.display());
    }

    // Rerun if the library changes
    println!("cargo:rerun-if-changed={}", lib_path.display());
}
