#![cfg(any(target_arch="x86", target_arch="x86_64"))]

macro_rules! compile_env_matches_or_is_empty {
    ($envvar:tt, $feature_name:tt) => {
        (match option_env!($envvar) {
            None => true,
            Some(v) => v == $feature_name
        })
    }
}

macro_rules! is_x86_feature_detected_ {
    ($name:tt) => {
        // for testing purposes, we can make sure only one specific feature
        // is enabled by setting MMTEST_FEATURE=featurename (all others
        // disabled). This does not force it to be detected, it must also be.
        compile_env_matches_or_is_empty!("MMTEST_FEATURE", $name) && is_x86_feature_detected!($name)
    }
}

