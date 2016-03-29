// Copyright 2016 bluss
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//! 
//! General matrix multiplication for f32, f64 matrices.
//! 
//! Allows arbitrary row, column strided matrices.
//! 
//! Uses the same microkernel algorithm as [BLIS][bl], but in a much simpler
//! and less featureful implementation.
//! See their [multithreading][mt] page for a very good diagram over how
//! the algorithm partitions the matrix (*Note:* this crate does not implement
//! multithreading).
//! 
//! [bl]: https://github.com/flame/blis
//! 
//! [mt]: https://github.com/flame/blis/wiki/Multithreading

#[macro_use] mod debugmacros;
#[macro_use] mod loopmacros;
mod archparam;
mod kernel;
mod gemm;
mod sgemm_kernel;
mod dgemm_kernel;
mod pointer;
mod util;

pub use gemm::sgemm;
pub use gemm::dgemm;
