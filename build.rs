fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or(String::new());

    let ac = match autocfg::AutoCfg::new() {
        Ok(ac) => ac,
        Err(err) => {
            println!("cargo:warning={}", err);
            return;
        }
    };

    // Avoid `unexpected_cfgs` lint from 1.80+ toolchains
    if ac.probe_rustc_version(1, 80) {
        println!("cargo::rustc-check-cfg=cfg(has_avx512)");
    }

    if target_arch == "aarch64" {
        // From 1.61 aarch64 intrinsics and #[target_feature]
        if ac.probe_rustc_version(1, 61) {
            println!("cargo:rustc-cfg=has_aarch64_simd");
        }
    }
    if target_arch == "x86" || target_arch == "x86_64" {
        // From 1.89 AVX-512 intrinsics ("avx512f")
        if ac.probe_rustc_version(1, 89) {
            println!("cargo:rustc-cfg=has_avx512");
        }
    }
}
