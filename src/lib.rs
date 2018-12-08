// Copyright 2016 - 2018 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//! 
//! General matrix multiplication for f32, f64 matrices. Operates on matrices
//! with general layout (they can use arbitrary row and column stride).
//! 
//! This crate uses the same macro/microkernel approach to matrix multiplication as
//! the [BLIS][bl] project.
//! 
//! We presently provide a few good microkernels, portable and for x86-64, and
//! only one operation: the general matrix-matrix multiplication (“gemm”).
//! 
//! [bl]: https://github.com/flame/blis
//!
//! ## Matrix Representation
//!
//! **matrixmultiply** supports matrices with general stride, so a matrix
//! is passed using a pointer and four integers:
//!
//! - `a: *const f32`, pointer to the first element in the matrix
//! - `m: usize`, number of rows
//! - `k: usize`, number of columns
//! - `rsa: isize`, row stride
//! - `csa: isize`, column stride
//!
//! In this example, A is a m by k matrix. `a` is a pointer to the element at
//! index *0, 0*.
//!
//! The *row stride* is the pointer offset (in number of elements) to the
//! element on the next row. It’s the distance from element *i, j* to *i + 1,
//! j*.
//!
//! The *column stride* is the pointer offset (in number of elements) to the
//! element in the next column. It’s the distance from element *i, j* to *i,
//! j + 1*.
//!
//! For example for a contiguous matrix, row major strides are *rsa=k,
//! csa=1* and column major strides are *rsa=1, csa=m*.
//!
//! Stides can be negative or even zero, but for a mutable matrix elements
//! may not alias each other.
//!
//! ## Portability and Performance
//!
//! - The default kernels are written in portable Rust and available
//!   on all targets. These may depend on autovectorization to perform well.
//!
//! - *x86* and *x86-64* features can be detected at runtime by default or
//!   compile time (if enabled), and the crate following kernel variants are
//!   implemented:
//!
//!   - `fma`
//!   - `avx`
//!   - `sse2`
//!
//! ## Other Notes
//!
//! The functions in this crate are thread safe, as long as the destination
//! matrix is distinct.

#![doc(html_root_url = "https://docs.rs/matrixmultiply/0.2/")]

extern crate rawpointer;

#[macro_use] mod debugmacros;
#[macro_use] mod loopmacros;
mod archparam;
mod kernel;
mod gemm;

mod util;
mod aligned_alloc;

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
#[macro_use]
mod x86;
mod sgemm_kernel;
mod dgemm_kernel;

pub use gemm::sgemm;
pub use gemm::dgemm;
