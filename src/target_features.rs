
/// Is the target feature allowed currently?
///
/// The environment variable MMTEST_FEATURE is read at compile-time.
/// Used for testing only - if the environment variable is non-empty,
/// **only** features listed, comma-separated, are allowed to be detected,
/// all other are disabled.
///
/// This is internal only, not stable.
pub(crate) const fn allow_feature(feature: &str) -> bool {
    match option_env!("MMTEST_FEATURE") {
        None => true,
        Some(s) if s.is_empty() => true,
        Some(value) => crate::constfind::comma_separated_contains(value, feature),
    }
}



#[cfg(any(target_arch = "x86", target_arch = "x86_64", target_arch="aarch64"))]
#[test]
fn test_features() {
    // This is just a test you can run to see the effect
    // of environment variable parsing.
    let features = &["sse2", "avx", "avx2", "fma", "avx512f", "neon"];
    for feat in features {
        println!(r#"feature= {:12} allowed= {}"#, feat, allow_feature(feat));

    }
}
