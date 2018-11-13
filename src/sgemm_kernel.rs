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

macro_rules! loop_m { ($i:ident, $e:expr) => { loop4!($i, $e) }; }
macro_rules! loop_n { ($j:ident, $e:expr) => { loop4!($j, $e) }; }

impl GemmKernel for Gemm {
    type Elem = T;

    #[inline(always)]
    fn align_to() -> usize { 16 }

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
    kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc);
}

#[inline]
#[target_feature(enable="avx")]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
pub unsafe fn kernel_target_avx(k: usize, alpha: T, a: *const T, b: *const T,
                         beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_x86_sse(k, alpha, a, b, beta, c, rsc, csc)
}

#[inline]
#[target_feature(enable="sse")]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
pub unsafe fn kernel_target_sse(k: usize, alpha: T, a: *const T, b: *const T,
                          beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_x86_sse(k, alpha, a, b, beta, c, rsc, csc)
}

macro_rules! mm_transpose4 {
    ($c0:expr, $c1:expr, $c2:expr, $c3:expr) => {{
        // This is _MM_TRANSPOSE4_PS except we take variables, not references
        let tmp0 = _mm_unpacklo_ps($c0, $c1);
        let tmp2 = _mm_unpacklo_ps($c2, $c3);
        let tmp1 = _mm_unpackhi_ps($c0, $c1);
        let tmp3 = _mm_unpackhi_ps($c2, $c3);

        $c0 = _mm_movelh_ps(tmp0, tmp2);
        $c1 = _mm_movehl_ps(tmp2, tmp0);
        $c2 = _mm_movelh_ps(tmp1, tmp3);
        $c3 = _mm_movehl_ps(tmp3, tmp1);
    }}
}

#[inline(always)]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
pub unsafe fn kernel_x86_sse(k: usize, alpha: T, a: *const T, b: *const T,
                             beta: T, c: *mut T, rsc: isize, csc: isize)
{
    let mut ab = [_mm_setzero_ps(); MR];

    let mut bv;
    let (mut a, mut b) = (a, b);

    // Compute A B
    for _ in 0..k {
        bv = _mm_load_ps(b as _); // aligned due to GemmKernel::align_to

        loop_m!(i, {
            // Compute ab_i += [ai b_j+0, ai b_j+1, ai b_j+2, ai b_j+3]
            let aiv = _mm_set1_ps(at(a, i));
            ab[i] = _mm_add_ps(ab[i], _mm_mul_ps(aiv, bv));
        });

        a = a.add(MR);
        b = b.add(NR);
    }

    // Compute α (A B)
    let alphav = _mm_set1_ps(alpha);
    loop_m!(i, ab[i] = _mm_mul_ps(alphav, ab[i]));

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // C ← α A B + β C
    let mut c = [_mm_setzero_ps(); MR];
    let betav = _mm_set1_ps(beta);
    if beta != 0. {
        // Read C
        if csc == 1 {
            loop_m!(i, c[i] = _mm_loadu_ps(c![i, 0]));
        } else if rsc == 1 {
            loop_m!(i, c[i] = _mm_loadu_ps(c![0, i]));
            mm_transpose4!(c[0], c[1], c[2], c[3]);
        } else {
            loop_m!(i, c[i] = _mm_set_ps(*c![i, 3], *c![i, 2], *c![i, 1], *c![i, 0]));
        }
        // Compute β C
        loop_m!(i, c[i] = _mm_mul_ps(c[i], betav));
    }

    // Compute (α A B) + (β C)
    loop_m!(i, c[i] = _mm_add_ps(c[i], ab[i]));

    // Store C back to memory
    if csc == 1 {
        loop_m!(i, _mm_storeu_ps(c![i, 0], c[i]));
    } else if rsc == 1 {
        mm_transpose4!(c[0], c[1], c[2], c[3]);
        loop_m!(i, _mm_storeu_ps(c![0, i], c[i]));
    } else {
        // extract the nth value of a vector using _mm_cvtss_f32 (extract lowest)
        // in combination with shuffle (move nth value to first position)
        loop_m!(i, *c![i, 0] = _mm_cvtss_f32(c[i]));
        loop_m!(i, *c![i, 1] = _mm_cvtss_f32(_mm_shuffle_ps(c[i], c[i], 1)));
        loop_m!(i, *c![i, 2] = _mm_cvtss_f32(_mm_shuffle_ps(c[i], c[i], 2)));
        loop_m!(i, *c![i, 3] = _mm_cvtss_f32(_mm_shuffle_ps(c[i], c[i], 3)));
    }
}


pub unsafe fn kernel_fallback_impl(k: usize, alpha: T, a: *const T, b: *const T,
                                   beta: T, c: *mut T, rsc: isize, csc: isize)
{
    let mut ab: [[T; NR]; MR] = [[0.; NR]; MR];
    let mut a = a;
    let mut b = b;

    // Compute A B into ab[i][j]
    unroll_by!(4 => k, {
        loop_m!(i, loop_n!(j, ab[i][j] += at(a, i) * at(b, j)));

        a = a.offset(MR as isize);
        b = b.offset(NR as isize);
    });

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // set C = α A B + β C
    if beta == 0. {
        loop_n!(j, loop_m!(i, *c![i, j] = alpha * ab[i][j]));
    } else {
        loop_n!(j, loop_m!(i, *c![i, j] = *c![i, j] * beta + alpha * ab[i][j]));
    }
}

#[inline(always)]
unsafe fn at(ptr: *const T, i: usize) -> T {
    *ptr.offset(i as isize)
}

#[test]
fn test_gemm_kernel() {
    const K: usize = 4;
    let mut a = vec![1.; MR * K];
    let mut b = vec![0.; NR * K];
    for (i, x) in a.iter_mut().enumerate() {
        *x = i as f32;
    }

    for i in 0..K {
        b[i + i * NR] = 1.;
    }
    let mut c = [0.; MR * NR];
    unsafe {
        kernel(K, 1., &a[0], &b[0], 0., &mut c[0], 1, MR as isize);
        // col major C
    }
    assert_eq!(a, &c[..a.len()]);
}

#[test]
fn test_loop_m_n() {
    let mut m = [[0; NR]; MR];
    loop_m!(i, loop_n!(j, m[i][j] += 1));
    for arr in &m[..] {
        for elt in &arr[..] {
            assert_eq!(*elt, 1);
        }
    }
}
