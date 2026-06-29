# This file exists to document some auxiliary commands for development;
# for most development tasks prefer cargo; i.e. cargo test; which does not need to be here.
wasmtest *args:
    CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime RUSTFLAGS="-C target-feature=+simd128,+relaxed-simd" cargo test --target wasm32-wasip1 --features=cgemm {{args}}
