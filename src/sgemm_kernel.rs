// Copyright 2016 bluss
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use kernel::GemmKernel;
use archparam;
use std::arch::x86_64::*;

pub enum Gemm { }

pub type T = f32;

const MR: usize = 4;
const NR: usize = 4;

#[cfg(sgemm_8x8)]
macro_rules! loop_m { ($i:ident, $e:expr) => { loop4!($i, $e) }; }
#[cfg(not(sgemm_8x8))]
macro_rules! loop_m { ($i:ident, $e:expr) => { loop4!($i, $e) }; }

macro_rules! loop_n { ($j:ident, $e:expr) => { loop4!($j, $e) }; }

impl GemmKernel for Gemm {
    type Elem = T;

    #[inline(always)]
    fn align_to() -> usize { 0 }

    #[inline(always)]
    fn mr() -> usize { MR }
    #[inline(always)]
    fn nr() -> usize { NR }

    #[inline(always)]
    fn always_masked() -> bool { true }

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
        kernel(k, alpha, a, b, beta, c, rsc, csc)
    }
}

/// matrix multiplication kernel
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
/// + if beta is 0, then c does not need to be initialized
#[inline(always)]
//#[target_feature(enable="sse")]
pub unsafe fn kernel(k: usize, alpha: T, a: *const T, b: *const T,
                     beta: T, c: *mut T, rsc: isize, csc: isize)
{
    // using `uninitialized` is a workaround for issue https://github.com/bluss/matrixmultiply/issues/9
    //let mut ab: [[T; NR]; MR] = ::std::mem::uninitialized();
    let mut ab0 = _mm_setzero_ps();
    let mut ab1 = _mm_setzero_ps();
    let mut ab2 = _mm_setzero_ps();
    let mut ab3 = _mm_setzero_ps();
    //let mut av;
    let mut bv;
    let (mut a, mut b) = (b, a); //NOTE transpose
    debug_assert_eq!(beta, 0.); // always masked
    //loop_m!(i, loop_n!(j, ab[i][j] = 0.));

    // Compute matrix multiplication into ab[i][j]
    unroll_by!(4 => k, {
        //av = _mm_loadu_ps(a as _);
        bv = _mm_loadu_ps(b as _);

        /*
        loop_m!(i, {
            ab[i][0] += at(a, i) * at(b, 0);
            ab[i][1] += at(a, i) * at(b, 1);
            ab[i][2] += at(a, i) * at(b, 2);
            ab[i][3] += at(a, i) * at(b, 3);
        });
        */

        let a0 = _mm_set1_ps(at(a, 0));
        ab0 = _mm_add_ps(ab0, _mm_mul_ps(a0, bv));
        let a1 = _mm_set1_ps(at(a, 1));
        ab1 = _mm_add_ps(ab1, _mm_mul_ps(a1, bv));
        let a2 = _mm_set1_ps(at(a, 2));
        ab2 = _mm_add_ps(ab2, _mm_mul_ps(a2, bv));
        let a3 = _mm_set1_ps(at(a, 3));
        ab3 = _mm_add_ps(ab3, _mm_mul_ps(a3, bv));
        //loop_m!(i, loop_n!(j, ab[i][j] += at(a, i) * at(b, j)));

        a = a.offset(MR as isize);
        b = b.offset(NR as isize);
    });
    /*
    _mm_storeu_ps(&mut ab[0][0] as *mut _ as _, ab0);
    _mm_storeu_ps(&mut ab[1][0] as *mut _ as _, ab1);
    _mm_storeu_ps(&mut ab[2][0] as *mut _ as _, ab2);
    _mm_storeu_ps(&mut ab[3][0] as *mut _ as _, ab3);

    */
    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    let alphav = _mm_set1_ps(alpha);

    // set C = α A B
    //loop_n!(j, loop_m!(i, *c![i, j] = alpha * ab[i][j]));
    /*
    loop_n!(j, {
        *c![0, j] = alpha * ab[0][j];
        *c![1, j] = alpha * ab[1][j];
        *c![2, j] = alpha * ab[2][j];
        *c![3, j] = alpha * ab[3][j];
    });
    */

    //loop_m!(i, {
        _mm_storeu_ps(c![0, 0], _mm_mul_ps(alphav, ab0));
        _mm_storeu_ps(c![0, 1], _mm_mul_ps(alphav, ab1));
        _mm_storeu_ps(c![0, 2], _mm_mul_ps(alphav, ab2));
        _mm_storeu_ps(c![0, 3], _mm_mul_ps(alphav, ab3));
    //});
}

#[inline(always)]
unsafe fn at(ptr: *const T, i: usize) -> T {
    *ptr.offset(i as isize)
}

#[test]
fn test_gemm_kernel() {
    let mut a = [1.; 16];
    let mut b = [0.; 32];
    for (i, x) in a.iter_mut().enumerate() {
        *x = i as f32;
    }

    for i in 0..4 {
        b[i + i * 8] = 1.;
    }
    let mut c = [0.; 32];
    unsafe {
        kernel(4, 1., &a[0], &b[0], 0., &mut c[0], 1, 4);
        // col major C
    }
    assert_eq!(&a, &c[..16]);
}

