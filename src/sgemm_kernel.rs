// Copyright 2016 bluss
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use kernel::GemmKernel;
use archparam;

pub enum Gemm { }

pub type T = f32;

impl GemmKernel for Gemm {
    type Elem = T;

    #[inline(always)]
    fn mr() -> usize { 4 }
    #[inline(always)]
    fn nr() -> usize { 4 }

    #[inline(always)]
    fn align_to() -> usize { 0 }

    #[inline(always)]
    fn nc() -> usize { archparam::S_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::S_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::S_MC }

    #[inline(always)]
    unsafe fn kernel(
        k: usize,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *mut T, rsc: isize, csc: isize) {
        kernel_4x4(k, alpha, a, b, beta, c, rsc, csc)
    }
}

/// 4x4 matrix multiplication kernel for f32
///
/// This does the matrix multiplication:
///
/// C ← α A B + β C
///
/// + k: length of data in a, b
/// + a, b are packed
/// + c has general strides
/// + rsc: row stride of c
/// + csc: col stride of c
/// + if `beta` is 0, then c does not need to be initialized
#[inline(always)]
pub unsafe fn kernel_4x4(k: usize, alpha: T, a: *const T, b: *const T,
                         beta: T, c: *mut T, rsc: isize, csc: isize)
{
    let mut ab = [[0.; 4]; 4];
    let mut a = a;
    let mut b = b;

    // Compute matrix multiplication into ab[i][j]
    unroll_by_8!(k, {
        let v0 = [at(a, 0), at(a, 1), at(a, 2), at(a, 3)];
        let v1 = [at(b, 0), at(b, 1), at(b, 2), at(b, 3)];
        loop4x4!(i, j, ab[i][j] += v0[i] * v1[j]);

        a = a.offset(4);
        b = b.offset(4);
    });

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // Compute C = alpha A B + beta C,
    // except we can not read C if beta is zero.
    if beta == 0. {
        loop4x4!(i, j, *c![i, j] = alpha * ab[i][j]);
    } else {
        loop4x4!(i, j, *c![i, j] = *c![i, j] * beta + alpha * ab[i][j]);
    }
}

#[inline(always)]
unsafe fn at(ptr: *const T, i: usize) -> T {
    *ptr.offset(i as isize)
}

#[test]
fn test_gemm_kernel() {
    let mut a = [1.; 16];
    let mut b = [0.; 16];
    for (i, x) in a.iter_mut().enumerate() {
        *x = i as f32;
    }
    for i in 0..4 {
        b[i + i * 4] = 1.;
    }
    let mut c = [0.; 16];
    unsafe {
        kernel_4x4(4, 1., &a[0], &b[0],
                   0., &mut c[0], 1, 4);
        // transposed C so that results line up
    }
    assert_eq!(&a, &c);

    // Test scale + add
    //
    let mut aprim = a;
    for elt in &mut aprim { *elt *= 3.; }
    unsafe {
        kernel_4x4(4, 2.5, &a[0], &b[0],
                   0.5, &mut c[0], 1, 4);
    }
    assert_eq!(&aprim, &c);
}

