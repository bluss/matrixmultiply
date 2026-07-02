
## Guidelines

As a current guiding principle, the entrance functions (i.e sgemm, dgemm etc) are non-generic by design, so that the compile time cost of the library is limited.

Threading is supported using thread-tree for its lower dispatch overhead but it's welcome to replace it by rayon - if the low dispatch overhead can be preserved.

## Test tricks

Use MMTEST_FEATURE=fma and so on to restrict target feature detection to the given feature. Note that this currently only supports a single target feature at a time.

## Benchmarks

To run benchmarks, use `./benches/benchloop.py`

## Wasm

To test and benchmark wasm, add the wasm32-wasip1 target using rustup and install wasmtime-cli.
