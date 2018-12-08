
Run BLAS benchmarks to compare with matrixmultiply.

These tests are set up to run vs a system-installed openblas (see the build.rs file),
because building all of openblas just to benchmark versus it is tedious.
So make sure openblas is installed, or other library that supports the cblas interface,
and tweak the build.rs file to suit.

Remember to consider and/or set environment variable `OPENBLAS_NUM_THREADS=` or
equivalent for the given blas library to test single- or multithreaded
scenarios!
