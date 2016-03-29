extern crate gcc;

fn main_common() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=microkernels");
}

fn main() {
    main_common();
    gcc::compile_library("libmicrokernels_avx.a",
        &[
            "microkernels/blis_sgemm_8x8_avx.c",
            "microkernels/blis_dgemm_8x4_avx.c",
        ]);
}
