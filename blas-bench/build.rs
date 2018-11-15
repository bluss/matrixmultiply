
///
/// This build script emits the openblas linking directive if requested
///

fn main() {
    // Always linking openblas
    // Compiling blas just for testing is tedious -- install it on your system
    // and run this.
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rustc-link-lib={}=openblas", "dylib");
}
