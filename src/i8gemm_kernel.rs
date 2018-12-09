// Copyright 2016 - 2018 Ulrik Sverdrup "bluss"
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

pub type Tin  = i8;
pub type Tout = i16;

const MR: usize = 16;
const NR: usize = 32;

macro_rules! loop_m { ($i:ident, $e:expr) => { loop16!($i, $e) }; }
macro_rules! loop_n { ($j:ident, $e:expr) => { loop32!($j, $e) }; }

impl GemmKernel for Gemm {
    type ElemIn = Tin;
    type ElemOut = Tout;

    #[inline(always)]
    fn align_to() -> usize { 16 }

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
        alpha: Tout,
        a: *const Tin,
        b: *const Tin,
        beta: Tout,
        c: *mut Tout, rsc: isize, csc: isize) {
        kernel(k, alpha, a, b, beta, c, rsc, csc)
    }
}

/// Multiply two 128-bit vectors of 16 8-bit integers each,by sign-extending them to 256-bit
/// vectors of 16-bit integers, and then multiplying these temporaries.
#[inline(always)]
unsafe fn _mm256_mulepi8_epi16(a: __m128i, b: __m128i) -> __m256i
{
    let tmp0 = _mm256_cvtepi8_epi16(a);
    let tmp1 = _mm256_cvtepi8_epi16(b);

    _mm256_mullo_epi16(tmp0, tmp1)
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
#[inline(never)]
pub unsafe fn kernel(k: usize, alpha: Tout, a: *const Tin, b: *const Tin,
                     beta: Tout, c: *mut Tout, rsc: isize, csc: isize)
{
    // dispatch to specific compiled versions
    #[cfg(any(target_arch="x86", target_arch="x86_64"))]
    {
        if is_x86_feature_detected_!("avx2") {
            return kernel_target_avx2(k, alpha, a, b, beta, c, rsc, csc);
        } else if is_x86_feature_detected_!("avx") {
            return kernel_target_avx(k, alpha, a, b, beta, c, rsc, csc);
        } else if is_x86_feature_detected_!("sse2") {
            return kernel_target_sse2(k, alpha, a, b, beta, c, rsc, csc);
        }
    }
    kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc);
}

#[inline]
#[target_feature(enable="avx2")]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
unsafe fn kernel_target_avx2(k: usize, alpha: Tout, a: *const Tin, b: *const Tin,
                            beta: Tout, c: *mut Tout, rsc: isize, csc: isize)
{
    kernel_x86_avx2(k, alpha, a, b, beta, c, rsc, csc)
}

#[inline]
#[target_feature(enable="avx")]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
unsafe fn kernel_target_avx(k: usize, alpha: Tout, a: *const Tin, b: *const Tin,
                            beta: Tout, c: *mut Tout, rsc: isize, csc: isize)
{
    kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc)
}

#[inline]
#[target_feature(enable="sse2")]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
unsafe fn kernel_target_sse2(k: usize, alpha: Tout, a: *const Tin, b: *const Tin,
                             beta: Tout, c: *mut Tout, rsc: isize, csc: isize)
{
    kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc)
}


#[inline(always)]
unsafe fn kernel_fallback_impl(k: usize, alpha: Tout, a: *const Tin, b: *const Tin,
                               beta: Tout, c: *mut Tout, rsc: isize, csc: isize)
{
    let mut ab: [[Tout; NR]; MR] = [[0; NR]; MR];
    let mut a = a;
    let mut b = b;
    debug_assert_eq!(beta, 0);

    // Compute A B into ab[i][j]
    unroll_by!(4 => k, {
        loop_m!(i, loop_n!(j, {
            ab[i][j] = ab[i][j].saturating_add(
                (at(a, i) as i16)
                .saturating_mul(
                    at(b, j) as i16
        ));}));

        a = a.offset(MR as isize);
        b = b.offset(NR as isize);
    });

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // set C = α A B + β C
    loop_n!(j, loop_m!(i, *c![i, j] = alpha.wrapping_mul(ab[i][j])));
}

#[inline(always)]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
unsafe fn kernel_x86_avx2(k: usize, alpha: Tout, a: *const Tin, b: *const Tin,
                         beta: Tout, c: *mut Tout, rsc: isize, csc: isize)
{
    debug_assert_ne!(k, 0);

    let mut ab = [_mm256_setzero_si256(); NR];

    let (mut a, mut b) = (a, b);

    let mut a_col = _mm_loadu_si128(a as *const __m128i);

    // Load two rows from b at a time.
    let mut b_row = _mm256_loadu_si256(b as *const __m256i);

    // FIXME: Is this k a meaningful number in this context?
    unroll_by_with_last!(4 => k, is_last, {
        let b0_b16 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
                0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,  0x0,
            )
        );

        let b0 = _mm256_extracti128_si256(b0_b16, 0);
        let b16 = _mm256_extracti128_si256(b0_b16, 1);

        let b1_b17 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,
                0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,  0x1,
            )
        );

        let b1 = _mm256_extracti128_si256(b1_b17, 0);
        let b17 = _mm256_extracti128_si256(b1_b17, 1);

        let b2_b18 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,
                0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,  0x2,
            )
        );

        let b2 = _mm256_extracti128_si256(b2_b18, 0);
        let b18 = _mm256_extracti128_si256(b2_b18, 1);

        let b3_b19 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,
                0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,  0x3,
            )
        );

        let b3 = _mm256_extracti128_si256(b3_b19, 0);
        let b19 = _mm256_extracti128_si256(b3_b19, 1);

        let b4_b20 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,
                0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,  0x4,
            )
        );

        let b4 = _mm256_extracti128_si256(b4_b20, 0);
        let b20 = _mm256_extracti128_si256(b4_b20, 1);

        let b5_b21 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,
                0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,  0x5,
            )
        );

        let b5 = _mm256_extracti128_si256(b5_b21, 0);
        let b21 = _mm256_extracti128_si256(b5_b21, 1);

        let b6_b22 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,
                0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,  0x6,
            )
        );

        let b6 = _mm256_extracti128_si256(b6_b22, 0);
        let b22 = _mm256_extracti128_si256(b6_b22, 1);

        let b7_b23 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,
                0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,  0x7,
            )
        );

        let b7 = _mm256_extracti128_si256(b7_b23, 0);
        let b23 = _mm256_extracti128_si256(b7_b23, 1);

        let b8_b24 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,
                0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,  0x8,
            )
        );

        let b8 = _mm256_extracti128_si256(b8_b24, 0);
        let b24 = _mm256_extracti128_si256(b8_b24, 1);

        let b9_b25 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,
                0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,  0x9,
            )
        );

        let b9 = _mm256_extracti128_si256(b9_b25, 0);
        let b25 = _mm256_extracti128_si256(b9_b25, 1);

        let b10_b26 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,
                0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,  0xa,
            )
        );

        let b10 = _mm256_extracti128_si256(b10_b26, 0);
        let b26 = _mm256_extracti128_si256(b10_b26, 1);

        let b11_b27 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,
                0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,  0xb,
            )
        );

        let b11 = _mm256_extracti128_si256(b11_b27, 0);
        let b27 = _mm256_extracti128_si256(b11_b27, 1);

        let b12_b28 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,
                0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,  0xc,
            )
        );

        let b12 = _mm256_extracti128_si256(b12_b28, 0);
        let b28 = _mm256_extracti128_si256(b12_b28, 1);

        let b13_b29 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,
                0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,  0xd,
            )
        );

        let b13 = _mm256_extracti128_si256(b13_b29, 0);
        let b29 = _mm256_extracti128_si256(b13_b29, 1);

        let b14_b30 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,
                0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,  0xe,
            )
        );

        let b14 = _mm256_extracti128_si256(b14_b30, 0);
        let b30 = _mm256_extracti128_si256(b14_b30, 1);

        let b15_b31 = _mm256_shuffle_epi8(
            b_row,
            _mm256_set_epi8(
                0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,
                0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,  0xf,
            )
        );

        let b15 = _mm256_extracti128_si256(b15_b31, 0);
        let b31 = _mm256_extracti128_si256(b15_b31, 1);

        // Multiplication and addition with the first row.
        ab[0]  = _mm256_adds_epi16(ab[0],  _mm256_mulepi8_epi16(a_col, b0));
        ab[1]  = _mm256_adds_epi16(ab[1],  _mm256_mulepi8_epi16(a_col, b1));
        ab[2]  = _mm256_adds_epi16(ab[2],  _mm256_mulepi8_epi16(a_col, b2));
        ab[3]  = _mm256_adds_epi16(ab[3],  _mm256_mulepi8_epi16(a_col, b3));

        ab[4]  = _mm256_adds_epi16(ab[4],  _mm256_mulepi8_epi16(a_col, b4));
        ab[5]  = _mm256_adds_epi16(ab[5],  _mm256_mulepi8_epi16(a_col, b5));
        ab[6]  = _mm256_adds_epi16(ab[6],  _mm256_mulepi8_epi16(a_col, b6));
        ab[7]  = _mm256_adds_epi16(ab[7],  _mm256_mulepi8_epi16(a_col, b7));

        ab[8]  = _mm256_adds_epi16(ab[8],  _mm256_mulepi8_epi16(a_col, b8));
        ab[9]  = _mm256_adds_epi16(ab[9],  _mm256_mulepi8_epi16(a_col, b9));
        ab[10] = _mm256_adds_epi16(ab[10], _mm256_mulepi8_epi16(a_col, b10));
        ab[11] = _mm256_adds_epi16(ab[11], _mm256_mulepi8_epi16(a_col, b11));

        ab[12] = _mm256_adds_epi16(ab[12], _mm256_mulepi8_epi16(a_col, b12));
        ab[13] = _mm256_adds_epi16(ab[13], _mm256_mulepi8_epi16(a_col, b13));
        ab[14] = _mm256_adds_epi16(ab[14], _mm256_mulepi8_epi16(a_col, b14));
        ab[15] = _mm256_adds_epi16(ab[15], _mm256_mulepi8_epi16(a_col, b15));

        // Multiplication and addition with the second row.);
        ab[16]  = _mm256_adds_epi16(ab[0],  _mm256_mulepi8_epi16(a_col, b16));
        ab[17]  = _mm256_adds_epi16(ab[1],  _mm256_mulepi8_epi16(a_col, b17));
        ab[18]  = _mm256_adds_epi16(ab[2],  _mm256_mulepi8_epi16(a_col, b18));
        ab[19]  = _mm256_adds_epi16(ab[3],  _mm256_mulepi8_epi16(a_col, b19));

        ab[20]  = _mm256_adds_epi16(ab[4],  _mm256_mulepi8_epi16(a_col, b20));
        ab[21]  = _mm256_adds_epi16(ab[5],  _mm256_mulepi8_epi16(a_col, b21));
        ab[22]  = _mm256_adds_epi16(ab[6],  _mm256_mulepi8_epi16(a_col, b22));
        ab[23]  = _mm256_adds_epi16(ab[7],  _mm256_mulepi8_epi16(a_col, b23));

        ab[24]  = _mm256_adds_epi16(ab[8],  _mm256_mulepi8_epi16(a_col, b24));
        ab[25]  = _mm256_adds_epi16(ab[9],  _mm256_mulepi8_epi16(a_col, b25));
        ab[26]  = _mm256_adds_epi16(ab[10], _mm256_mulepi8_epi16(a_col, b26));
        ab[27]  = _mm256_adds_epi16(ab[11], _mm256_mulepi8_epi16(a_col, b27));

        ab[28]  = _mm256_adds_epi16(ab[12], _mm256_mulepi8_epi16(a_col, b28));
        ab[29]  = _mm256_adds_epi16(ab[13], _mm256_mulepi8_epi16(a_col, b29));
        ab[30]  = _mm256_adds_epi16(ab[14], _mm256_mulepi8_epi16(a_col, b30));
        ab[31]  = _mm256_adds_epi16(ab[15], _mm256_mulepi8_epi16(a_col, b31));

        if !is_last {
            a = a.add(MR);
            b = b.add(NR);

            a_col = _mm_loadu_si128(a as _);
            b_row = _mm256_loadu_si256(b as _);
        }
    });

    // Compute α (A B)
    let alpha_v = _mm256_set1_epi16(alpha);
    loop_m!(i, ab[i] = _mm256_mullo_epi16(alpha_v, ab[i]));

    macro_rules! c {
        ($i:expr, $j:expr) =>
            (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // C ← α A B + β C
    let mut cv = [_mm256_setzero_si256(); MR];

    if beta != 0 {
        let beta_v = _mm256_set1_epi16(beta);

        // Read C
        if rsc == 1 {
            loop_m!(i, cv[i] = _mm256_loadu_si256(c![0, i] as _));
        // } else if csc == 1 {
        //     loop4!(i, cv[i] = _mm256_loadu_pd(c![i, 0]));
        //     loop4!(i, cv[i+4] = _mm256_loadu_pd(c![i+4, 0]));
        } else {
            loop_m!(i, cv[i] =
                _mm256_setr_epi16(
                    *c![0, i],
                    *c![1, i],
                    *c![2, i],
                    *c![3, i],
                    *c![4, i],
                    *c![5, i],
                    *c![6, i],
                    *c![7, i],
                    *c![8, i],
                    *c![9, i],
                    *c![10, i],
                    *c![11, i],
                    *c![12, i],
                    *c![13, i],
                    *c![14, i],
                    *c![15, i],
            ));
        }
        // Compute β C
        loop_m!(i, cv[i] = _mm256_mullo_epi16(cv[i], beta_v));
    }

    // Compute (α A B) + (β C)
    loop_m!(i, cv[i] = _mm256_add_epi32(cv[i], ab[i]));

    if rsc == 1 {
        loop_m!(i, _mm256_storeu_si256(c![0, i] as _, cv[i]));
    // } else if csc == 1 {
    //     loop4!(i, _mm256_storeu_pd(c![i, 0], cv[i]));
    //     loop4!(i, _mm256_storeu_pd(c![i+4, 0], cv[i + 4]));
    } else {
        // TODO: This inner unrolled loop should be replaced by
        // `loop_n!(j, *c![i, j] = _mm256_extract_epi32(cv[i], j);`
        // However, rustc currently errors with:
        // > error: argument 2 is required to be a constant
        // Some reading:
        // + https://internals.rust-lang.org/t/pre-rfc-const-function-arguments/6709/12
        // + https://www.reddit.com/r/rust/comments/9pxuoj/simd_instructions_requiring_a_constant_parameter/
        loop_m!(i, {
            *c![i, 0] = _mm256_extract_epi16(cv[i], 0);
            *c![i, 1] = _mm256_extract_epi16(cv[i], 1);
            *c![i, 2] = _mm256_extract_epi16(cv[i], 2);
            *c![i, 3] = _mm256_extract_epi16(cv[i], 3);
            *c![i, 4] = _mm256_extract_epi16(cv[i], 4);
            *c![i, 5] = _mm256_extract_epi16(cv[i], 5);
            *c![i, 6] = _mm256_extract_epi16(cv[i], 6);
            *c![i, 7] = _mm256_extract_epi16(cv[i], 7);
        })
    }
}

#[inline(always)]
unsafe fn at(ptr: *const Tin, i: usize) -> Tin {
    *ptr.offset(i as isize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aligned_alloc::Alloc;

    fn aligned_alloc<T>(elt: T, n: usize) -> Alloc<T> where T: Copy
    {
        unsafe {
            Alloc::new(n, Gemm::align_to()).init_with(elt)
        }
    }

    use super::Tin;
    use super::Tout;
    type KernelFn = unsafe fn(usize, Tout, *const Tin, *const Tin, Tout, *mut Tout, isize, isize);

    fn test_a_kernel(_name: &str, kernel_fn: KernelFn) {
        const K: usize = 4;
        let mut a = aligned_alloc(1, MR * K);
        let mut b = aligned_alloc(0, NR * K);
        for (i, x) in a.iter_mut().enumerate() {
            *x = i as _;
        }

        for i in 0..K {
            b[i + i * NR] = 1;
        }
        let mut c = [0; MR * NR];
        unsafe {
            kernel_fn(K, 1, &a[0], &b[0], 0, &mut c[0], 1, MR as isize);
            // col major C
        }
        let a: Vec<_> = a.iter().map(|x| *x as i16).collect();
        assert_eq!(&a[..], &c[..a.len()]);
    }

    #[test]
    fn test_native_kernel() {
        test_a_kernel("kernel", kernel);
    }

    #[test]
    fn test_kernel_fallback_impl() {
        test_a_kernel("kernel", kernel_fallback_impl);
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

    mod test_arch_kernels {
        use super::test_a_kernel;
        macro_rules! test_arch_kernels_x86 {
            ($($feature_name:tt, $function_name:ident),*) => {
                $(
                #[test]
                fn $function_name() {
                    if is_x86_feature_detected_!($feature_name) {
                        test_a_kernel(stringify!($function_name), super::super::$function_name);
                    } else {
                        println!("Skipping, host does not have feature: {:?}", $feature_name);
                    }
                }
                )*
            }
        }

        #[cfg(any(target_arch="x86", target_arch="x86_64"))]
        test_arch_kernels_x86! {
            "avx2", kernel_target_avx2,
            "avx", kernel_target_avx,
            "sse2", kernel_target_sse2
        }
    }
}
