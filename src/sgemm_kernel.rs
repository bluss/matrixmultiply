// Copyright 2016 bluss
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use kernel::GemmKernel;
use archparam;


#[cfg(target_arch="x86")]
use std::arch::x86::*;
#[cfg(target_arch="x86_64")]
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
    fn always_masked() -> bool { false }

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
pub unsafe fn kernel(k: usize, alpha: T, a: *const T, b: *const T,
                     beta: T, c: *mut T, rsc: isize, csc: isize)
{
    // dispatch to specific compiled versions
    #[cfg(any(target_arch="x86", target_arch="x86_64"))]
    {
        if is_x86_feature_detected!("avx") {
            return kernel_target_avx(k, alpha, a, b, beta, c, rsc, csc);
        } else if is_x86_feature_detected!("sse") {
            return kernel_target_sse(k, alpha, a, b, beta, c, rsc, csc);
        }
    }
    unimplemented!()
}

#[target_feature(enable="avx")]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
pub unsafe fn kernel_target_avx(k: usize, alpha: T, a: *const T, b: *const T,
                         beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_impl(k, alpha, a, b, beta, c, rsc, csc)
}

#[target_feature(enable="sse")]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
pub unsafe fn kernel_target_sse(k: usize, alpha: T, a: *const T, b: *const T,
                          beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_impl(k, alpha, a, b, beta, c, rsc, csc)
}

#[inline(always)]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
pub unsafe fn kernel_impl(k: usize, alpha: T, a: *const T, b: *const T,
                          beta: T, c: *mut T, rsc: isize, csc: isize)
{
    let mut ab0 = _mm_setzero_ps();
    let mut ab1 = _mm_setzero_ps();
    let mut ab2 = _mm_setzero_ps();
    let mut ab3 = _mm_setzero_ps();

    let mut bv;
    let (mut a, mut b) = (a, b);

    // Compute A B
    for _ in 0..k {
        bv = _mm_loadu_ps(b as _);

        let a0 = _mm_set1_ps(at(a, 0));
        ab0 = _mm_add_ps(ab0, _mm_mul_ps(a0, bv));
        let a1 = _mm_set1_ps(at(a, 1));
        ab1 = _mm_add_ps(ab1, _mm_mul_ps(a1, bv));
        let a2 = _mm_set1_ps(at(a, 2));
        ab2 = _mm_add_ps(ab2, _mm_mul_ps(a2, bv));
        let a3 = _mm_set1_ps(at(a, 3));
        ab3 = _mm_add_ps(ab3, _mm_mul_ps(a3, bv));

        a = a.offset(MR as isize);
        b = b.offset(NR as isize);
    }

    // Compute α (A B)
    let alphav = _mm_set1_ps(alpha);
    ab0 = _mm_mul_ps(alphav, ab0);
    ab1 = _mm_mul_ps(alphav, ab1);
    ab2 = _mm_mul_ps(alphav, ab2);
    ab3 = _mm_mul_ps(alphav, ab3);

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // C ← α A B + β C
    let mut c0;
    let mut c1;
    let mut c2;
    let mut c3;
    let betav = _mm_set1_ps(beta);
    if beta == 0. {
        c0 = _mm_setzero_ps();
        c1 = _mm_setzero_ps();
        c2 = _mm_setzero_ps();
        c3 = _mm_setzero_ps();
    } else {
        // Read C
        if csc == 1 {
            c0 = _mm_loadu_ps(c![0, 0]);
            c1 = _mm_loadu_ps(c![1, 0]);
            c2 = _mm_loadu_ps(c![2, 0]);
            c3 = _mm_loadu_ps(c![3, 0]);
        } else if rsc == 1 {
            c0 = _mm_loadu_ps(c![0, 0]);
            c1 = _mm_loadu_ps(c![0, 1]);
            c2 = _mm_loadu_ps(c![0, 2]);
            c3 = _mm_loadu_ps(c![0, 3]);
            _MM_TRANSPOSE4_PS(
                &mut c0,
                &mut c1,
                &mut c2,
                &mut c3,
                );
        } else {
            c0 = _mm_set_ps(*c![0, 3], *c![0, 2], *c![0, 1], *c![0, 0]);
            c1 = _mm_set_ps(*c![1, 3], *c![1, 2], *c![1, 1], *c![1, 0]);
            c2 = _mm_set_ps(*c![2, 3], *c![2, 2], *c![2, 1], *c![2, 0]);
            c3 = _mm_set_ps(*c![3, 3], *c![3, 2], *c![3, 1], *c![3, 0]);
        }
        // Compute β C
        c0 = _mm_mul_ps(c0, betav);
        c1 = _mm_mul_ps(c1, betav);
        c2 = _mm_mul_ps(c2, betav);
        c3 = _mm_mul_ps(c3, betav);
    }

    // Compute (α A B) + (β C)
    c0 = _mm_add_ps(c0, ab0);
    c1 = _mm_add_ps(c1, ab1);
    c2 = _mm_add_ps(c2, ab2);
    c3 = _mm_add_ps(c3, ab3);

    // Store C back to memory
    if csc == 1 {
        _mm_storeu_ps(c![0, 0], c0);
        _mm_storeu_ps(c![1, 0], c1);
        _mm_storeu_ps(c![2, 0], c2);
        _mm_storeu_ps(c![3, 0], c3);
    } else if rsc == 1 {
        _MM_TRANSPOSE4_PS(
            &mut c0,
            &mut c1,
            &mut c2,
            &mut c3,
            );
        _mm_storeu_ps(c![0, 0], c0);
        _mm_storeu_ps(c![0, 1], c1);
        _mm_storeu_ps(c![0, 2], c2);
        _mm_storeu_ps(c![0, 3], c3);
    } else {
        *c![0, 0] = _mm_cvtss_f32(c0);
        *c![1, 0] = _mm_cvtss_f32(c1);
        *c![2, 0] = _mm_cvtss_f32(c2);
        *c![3, 0] = _mm_cvtss_f32(c3);

        *c![0, 1] = _mm_cvtss_f32(_mm_shuffle_ps(c0, c0, 1));
        *c![1, 1] = _mm_cvtss_f32(_mm_shuffle_ps(c1, c1, 1));
        *c![2, 1] = _mm_cvtss_f32(_mm_shuffle_ps(c2, c2, 1));
        *c![3, 1] = _mm_cvtss_f32(_mm_shuffle_ps(c3, c3, 1));

        *c![0, 2] = _mm_cvtss_f32(_mm_shuffle_ps(c0, c0, 2));
        *c![1, 2] = _mm_cvtss_f32(_mm_shuffle_ps(c1, c1, 2));
        *c![2, 2] = _mm_cvtss_f32(_mm_shuffle_ps(c2, c2, 2));
        *c![3, 2] = _mm_cvtss_f32(_mm_shuffle_ps(c3, c3, 2));

        *c![0, 3] = _mm_cvtss_f32(_mm_shuffle_ps(c0, c0, 3));
        *c![1, 3] = _mm_cvtss_f32(_mm_shuffle_ps(c1, c1, 3));
        *c![2, 3] = _mm_cvtss_f32(_mm_shuffle_ps(c2, c2, 3));
        *c![3, 3] = _mm_cvtss_f32(_mm_shuffle_ps(c3, c3, 3));
    }
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

