#![cfg(any(target_arch="x86", target_arch="x86_64"))]

macro_rules! compile_env_enabled {
    ($($name:tt)*) => {
        !option_env!($($name)*).unwrap_or("").is_empty()
    }
}

macro_rules! is_x86_feature_detected_ {
    ($name:tt) => {
        // for testing purposes, we can disable a feature at compile time by
        // setting MMNO_avx=1 etc.
        !compile_env_enabled!(concat!("MMNO_", $name)) && is_x86_feature_detected!($name)
    }
}

