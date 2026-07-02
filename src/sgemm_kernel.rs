// Copyright 2016 - 2023 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::kernel::GemmKernel;
use crate::kernel::GemmSelect;
use crate::kernel::{U4, U8};
use crate::archparam;

#[cfg(target_arch="x86")]
use core::arch::x86::*;
#[cfg(target_arch="x86_64")]
use core::arch::x86_64::*;
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
use crate::x86::{FusedMulAdd, AvxMulAdd, SMultiplyAdd};

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
struct KernelAvx;
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
struct KernelFmaAvx2;
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
struct KernelFma;
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
struct KernelSse2;

#[cfg(target_arch="aarch64")]
#[cfg(has_aarch64_simd)]
struct KernelNeon;
#[cfg(all(target_arch="wasm32", target_feature="simd128"))]
struct KernelWasmSimd;
struct KernelFallback;

type T = f32;

/// Detect which implementation to use and select it using the selector's
/// .select(Kernel) method.
///
/// This function is called one or more times during a whole program's
/// execution, it may be called for each gemm kernel invocation or fewer times.
#[inline]
pub(crate) fn detect<G>(selector: G) where G: GemmSelect<T> {
    // dispatch to specific compiled versions
    #[cfg(any(target_arch="x86", target_arch="x86_64"))]
    {
        if is_x86_feature_detected_!("fma") {
            if is_x86_feature_detected_!("avx2") {
                return selector.select(KernelFmaAvx2);
            }
            return selector.select(KernelFma);
        } else if is_x86_feature_detected_!("avx") {
            return selector.select(KernelAvx);
        } else if is_x86_feature_detected_!("sse2") {
            return selector.select(KernelSse2);
        }
    }
    #[cfg(target_arch="aarch64")]
    #[cfg(has_aarch64_simd)]
    {
        if is_aarch64_feature_detected_!("neon") {
            return selector.select(KernelNeon);
        }
    }
    #[cfg(all(target_arch="wasm32", target_feature="simd128"))]
    {
        return selector.select(KernelWasmSimd);
    }
    #[allow(unreachable_code)]
    return selector.select(KernelFallback);
}

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
macro_rules! loop_m { ($i:ident, $e:expr) => { loop8!($i, $e) }; }
#[cfg(all(test, any(target_arch="x86", target_arch="x86_64")))]
macro_rules! loop_n { ($j:ident, $e:expr) => { loop8!($j, $e) }; }

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
impl GemmKernel for KernelAvx {
    type Elem = T;

    type MRTy = U8;
    type NRTy = U8;

    #[inline(always)]
    fn align_to() -> usize { 32 }

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
        kernel_target_avx(k, alpha, a, b, beta, c, rsc, csc)
    }
}

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
impl GemmKernel for KernelFma {
    type Elem = T;

    type MRTy = <KernelAvx as GemmKernel>::MRTy;
    type NRTy = <KernelAvx as GemmKernel>::NRTy;

    #[inline(always)]
    fn align_to() -> usize { KernelAvx::align_to() }

    #[inline(always)]
    fn always_masked() -> bool { KernelAvx::always_masked() }

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
        kernel_target_fma(k, alpha, a, b, beta, c, rsc, csc)
    }
}

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
impl GemmKernel for KernelFmaAvx2 {
    type Elem = T;

    type MRTy = <KernelAvx as GemmKernel>::MRTy;
    type NRTy = <KernelAvx as GemmKernel>::NRTy;

    #[inline(always)]
    fn align_to() -> usize { KernelAvx::align_to() }

    #[inline(always)]
    fn always_masked() -> bool { KernelAvx::always_masked() }

    #[inline(always)]
    fn nc() -> usize { archparam::S_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::S_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::S_MC }

    #[inline]
    unsafe fn pack_mr(kc: usize, mc: usize, pack: &mut [Self::Elem],
                      a: *const Self::Elem, rsa: isize, csa: isize)
    {
        // safety: Avx2 is enabled
        crate::packing::pack_avx2::<Self::MRTy, T>(kc, mc, pack, a, rsa, csa)
    }

    #[inline]
    unsafe fn pack_nr(kc: usize, mc: usize, pack: &mut [Self::Elem],
                      a: *const Self::Elem, rsa: isize, csa: isize)
    {
        // safety: Avx2 is enabled
        crate::packing::pack_avx2::<Self::NRTy, T>(kc, mc, pack, a, rsa, csa)
    }

    #[inline(always)]
    unsafe fn kernel(
        k: usize,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *mut T, rsc: isize, csc: isize) {
        kernel_target_fma(k, alpha, a, b, beta, c, rsc, csc)
    }
}

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
impl GemmKernel for KernelSse2 {
    type Elem = T;

    type MRTy = <KernelFallback as GemmKernel>::MRTy;
    type NRTy = <KernelFallback as GemmKernel>::NRTy;

    #[inline(always)]
    fn align_to() -> usize { 16 }

    #[inline(always)]
    fn always_masked() -> bool { KernelFallback::always_masked() }

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
        kernel_target_sse2(k, alpha, a, b, beta, c, rsc, csc)
    }
}


#[cfg(target_arch="aarch64")]
#[cfg(has_aarch64_simd)]
impl GemmKernel for KernelNeon {
    type Elem = T;

    type MRTy = U8;
    type NRTy = U8;

    #[inline(always)]
    fn align_to() -> usize { 32 }

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
        kernel_target_neon(k, alpha, a, b, beta, c, rsc, csc)
    }
}

impl GemmKernel for KernelFallback {
    type Elem = T;

    type MRTy = U8;
    type NRTy = U4;

    #[inline(always)]
    fn align_to() -> usize { 0 }

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
        kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc)
    }
}

#[cfg(all(target_arch="wasm32", target_feature="simd128"))]
impl GemmKernel for KernelWasmSimd {
    type Elem = T;

    type MRTy = U8;
    type NRTy = U8;

    #[inline(always)]
    fn align_to() -> usize { 16 }

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
        kernel_target_wasm_simd(k, alpha, a, b, beta, c, rsc, csc)
    }
}

// no inline for unmasked kernels
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
#[target_feature(enable="fma")]
unsafe fn kernel_target_fma(k: usize, alpha: T, a: *const T, b: *const T,
                            beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_x86_avx::<FusedMulAdd>(k, alpha, a, b, beta, c, rsc, csc)
}

// no inline for unmasked kernels
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
#[target_feature(enable="avx")]
unsafe fn kernel_target_avx(k: usize, alpha: T, a: *const T, b: *const T,
                            beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_x86_avx::<AvxMulAdd>(k, alpha, a, b, beta, c, rsc, csc)
}

#[inline]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
#[target_feature(enable="sse2")]
unsafe fn kernel_target_sse2(k: usize, alpha: T, a: *const T, b: *const T,
                             beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc)
}

#[inline(always)]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
unsafe fn kernel_x86_avx<MA>(k: usize, alpha: T, a: *const T, b: *const T,
                             beta: T, c: *mut T, rsc: isize, csc: isize)
    where MA: SMultiplyAdd,
{
    const MR: usize = KernelAvx::MR;
    const NR: usize = KernelAvx::NR;

    debug_assert_ne!(k, 0);

    let mut ab = [_mm256_setzero_ps(); MR];

    // this kernel can operate in either transposition (C = A B or C^T = B^T A^T)
    let prefer_row_major_c = rsc != 1;

    let (mut a, mut b) = if prefer_row_major_c { (a, b) } else { (b, a) };
    let (rsc, csc) = if prefer_row_major_c { (rsc, csc) } else { (csc, rsc) };

    macro_rules! permute_mask {
        ($z:expr, $y:expr, $x:expr, $w:expr) => {
            ($z << 6) | ($y << 4) | ($x << 2) | $w
        }
    }

    // Start data load before each iteration
    let mut av = _mm256_load_ps(a);
    let mut bvl = _mm256_broadcast_ps(&*(b.add(0) as *const _));
    let mut bvh = _mm256_broadcast_ps(&*(b.add(4) as *const _));

    // Compute A B
    unroll_by_with_last!(4 => k, is_last, {
        // We compute abij = ai bj
        //
        //
        //   ab0:    ab1:    ab2:    ab3:
        // ( ab00  ( ab10  ( ab20  ( ab30
        //   ab11    ab21    ab31    ab01
        //   ab22    ab32    ab02    ab12
        //   ab33    ab03    ab13    ab23
        //   ab40    ab50    ab60    ab70
        //   ab51    ab61    ab71    ab41
        //   ab62    ab72    ab42    ab52
        //   ab73 )  ab43 )  ab53 )  ab63 )
        //
        // ab1357: ab3175: ab5713: ab7531:
        // ( ab04  ( ab14  ( ab24  ( ab34
        //   ab15    ab25    ab35    ab05
        //   ab26    ab36    ab06    ab16
        //   ab37    ab07    ab17    ab27
        //   ab44    ab54    ab64    ab74
        //   ab55    ab65    ab75    ab45
        //   ab66    ab76    ab46    ab56
        //   ab77 )  ab47 )  ab57 )  ab67 )

        let a01234567 = av;
        let a12305674 = _mm256_permute_ps(av, permute_mask!(0, 3, 2, 1));
        let a23016745 = _mm256_permute_ps(av, permute_mask!(1, 0, 3, 2));
        let a30127456 = _mm256_permute_ps(av, permute_mask!(2, 1, 0, 3));

        ab[0] = MA::multiply_add(a01234567, bvl, ab[0]);
        ab[4] = MA::multiply_add(a01234567, bvh, ab[4]);

        ab[1] = MA::multiply_add(a12305674, bvl, ab[1]);
        ab[5] = MA::multiply_add(a12305674, bvh, ab[5]);

        ab[2] = MA::multiply_add(a23016745, bvl, ab[2]);
        ab[6] = MA::multiply_add(a23016745, bvh, ab[6]);

        ab[3] = MA::multiply_add(a30127456, bvl, ab[3]);
        ab[7] = MA::multiply_add(a30127456, bvh, ab[7]);

        if !is_last {
            a = a.add(MR);
            b = b.add(NR);

            bvl = _mm256_broadcast_ps(&*(b.add(0) as *const _));
            bvh = _mm256_broadcast_ps(&*(b.add(4) as *const _));
            av = _mm256_load_ps(a);
        }
    });

    let alphav = _mm256_set1_ps(alpha);

    // Permute to put the abij elements in order
    let t0 = ab[0];
    let t1 = ab[1];
    let t2 = ab[2];
    let t3 = ab[3];

    let (t0, t1, t2, t3) = (
        _mm256_blend_ps(t0, t3, 0b10101010),
        _mm256_blend_ps(t1, t0, 0b10101010),
        _mm256_blend_ps(t2, t1, 0b10101010),
        _mm256_blend_ps(t3, t2, 0b10101010),
    );

    let (t0, t1, t2, t3) = (
        _mm256_blend_ps(t0, t2, 0b11001100),
        _mm256_blend_ps(t1, t3, 0b11001100),
        _mm256_blend_ps(t2, t0, 0b11001100),
        _mm256_blend_ps(t3, t1, 0b11001100),
    );

    let t4 = ab[4];
    let t5 = ab[5];
    let t6 = ab[6];
    let t7 = ab[7];

    let (t4, t5, t6, t7) = (
        _mm256_blend_ps(t4, t7, 0b10101010),
        _mm256_blend_ps(t5, t4, 0b10101010),
        _mm256_blend_ps(t6, t5, 0b10101010),
        _mm256_blend_ps(t7, t6, 0b10101010),
    );

    let (t4, t5, t6, t7) = (
        _mm256_blend_ps(t4, t6, 0b11001100),
        _mm256_blend_ps(t5, t7, 0b11001100),
        _mm256_blend_ps(t6, t4, 0b11001100),
        _mm256_blend_ps(t7, t5, 0b11001100),
    );

    ab[0] = _mm256_permute2f128_ps(t0, t4, 0x20);
    ab[1] = _mm256_permute2f128_ps(t1, t5, 0x20);
    ab[2] = _mm256_permute2f128_ps(t2, t6, 0x20);
    ab[3] = _mm256_permute2f128_ps(t3, t7, 0x20);
    ab[4] = _mm256_permute2f128_ps(t0, t4, 0x31);
    ab[5] = _mm256_permute2f128_ps(t1, t5, 0x31);
    ab[6] = _mm256_permute2f128_ps(t2, t6, 0x31);
    ab[7] = _mm256_permute2f128_ps(t3, t7, 0x31);

    // Compute α (A B)
    // Compute here if we don't have fma, else pick up α further down
    if !MA::IS_FUSED {
        loop_m!(i, ab[i] = _mm256_mul_ps(alphav, ab[i]));
    }

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // C ← α A B + β C
    let mut cv = [_mm256_setzero_ps(); MR];
    if beta != 0. {
        let betav = _mm256_set1_ps(beta);
        // Read C
        if csc == 1 {
            loop_m!(i, cv[i] = _mm256_loadu_ps(c![i, 0]));
        } else {
            loop_m!(i, cv[i] = _mm256_setr_ps(*c![i, 0], *c![i, 1], *c![i, 2], *c![i, 3],
                                              *c![i, 4], *c![i, 5], *c![i, 6], *c![i, 7]));
        }
        // Compute β C
        loop_m!(i, cv[i] = _mm256_mul_ps(cv[i], betav));
    }

    // Compute (α A B) + (β C)
    if !MA::IS_FUSED {
        loop_m!(i, cv[i] = _mm256_add_ps(cv[i], ab[i]));
    } else {
        loop_m!(i, cv[i] = MA::multiply_add(alphav, ab[i], cv[i]));
    }

    // Store C back to memory
    if csc == 1 {
        loop_m!(i, _mm256_storeu_ps(c![i, 0], cv[i]));
    } else {
        // Permute to bring each element in the vector to the front and store
        loop_m!(i, {
            let cvlo = _mm256_extractf128_ps(cv[i], 0);
            let cvhi = _mm256_extractf128_ps(cv[i], 1);

            _mm_store_ss(c![i, 0], cvlo);
            let cperm = _mm_permute_ps(cvlo, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 1], cperm);
            let cperm = _mm_permute_ps(cperm, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 2], cperm);
            let cperm = _mm_permute_ps(cperm, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 3], cperm);

            _mm_store_ss(c![i, 4], cvhi);
            let cperm = _mm_permute_ps(cvhi, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 5], cperm);
            let cperm = _mm_permute_ps(cperm, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 6], cperm);
            let cperm = _mm_permute_ps(cperm, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 7], cperm);
        });
    }
}

#[cfg(target_arch="aarch64")]
#[cfg(has_aarch64_simd)]
#[target_feature(enable="neon")]
unsafe fn kernel_target_neon(k: usize, alpha: T, a: *const T, b: *const T,
                             beta: T, c: *mut T, rsc: isize, csc: isize)
{
    use core::arch::aarch64::*;
    const MR: usize = KernelNeon::MR;
    const NR: usize = KernelNeon::NR;

    let (mut a, mut b, rsc, csc) = if rsc == 1 { (b, a, csc, rsc) } else { (a, b, rsc, csc) };

    // Kernel 8 x 8 (a x b)
    // Four quadrants of 4 x 4
    let mut ab11 = [vmovq_n_f32(0.); 4];
    let mut ab12 = [vmovq_n_f32(0.); 4];
    let mut ab21 = [vmovq_n_f32(0.); 4];
    let mut ab22 = [vmovq_n_f32(0.); 4];

    // Compute
    // ab_ij = a_i * b_j for all i, j
    macro_rules! ab_ij_equals_ai_bj {
        ($dest:ident, $av:expr, $bv:expr) => {
            $dest[0] = vfmaq_laneq_f32($dest[0], $bv, $av, 0);
            $dest[1] = vfmaq_laneq_f32($dest[1], $bv, $av, 1);
            $dest[2] = vfmaq_laneq_f32($dest[2], $bv, $av, 2);
            $dest[3] = vfmaq_laneq_f32($dest[3], $bv, $av, 3);
        }
    }

    for _ in 0..k {
        let a1 = vld1q_f32(a);
        let b1 = vld1q_f32(b);
        let a2 = vld1q_f32(a.add(4));
        let b2 = vld1q_f32(b.add(4));

        // compute an outer product ab = a (*) b in four quadrants ab11, ab12, ab21, ab22

        // ab11: [a1 a2 a3 a4] (*) [b1 b2 b3 b4]
        // ab11: a1b1 a1b2 a1b3 a1b4
        //       a2b1 a2b2 a2b3 a2b4
        //       a3b1 a3b2 a3b3 a3b4
        //       a4b1 a4b2 a4b3 a4b4
        //  etc
        ab_ij_equals_ai_bj!(ab11, a1, b1);
        ab_ij_equals_ai_bj!(ab12, a1, b2);
        ab_ij_equals_ai_bj!(ab21, a2, b1);
        ab_ij_equals_ai_bj!(ab22, a2, b2);

        a = a.add(MR);
        b = b.add(NR);
    }

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // ab *= alpha
    loop4!(i, ab11[i] = vmulq_n_f32(ab11[i], alpha));
    loop4!(i, ab12[i] = vmulq_n_f32(ab12[i], alpha));
    loop4!(i, ab21[i] = vmulq_n_f32(ab21[i], alpha));
    loop4!(i, ab22[i] = vmulq_n_f32(ab22[i], alpha));

    // load one float32x4_t from four pointers
    macro_rules! loadq_from_pointers {
        ($p0:expr, $p1:expr, $p2:expr, $p3:expr) => (
            {
                let v = vld1q_dup_f32($p0);
                let v = vld1q_lane_f32($p1, v, 1);
                let v = vld1q_lane_f32($p2, v, 2);
                let v = vld1q_lane_f32($p3, v, 3);
                v
            }
        );
    }

    if beta != 0. {
        // load existing value in C
        let mut c11 = [vmovq_n_f32(0.); 4];
        let mut c12 = [vmovq_n_f32(0.); 4];
        let mut c21 = [vmovq_n_f32(0.); 4];
        let mut c22 = [vmovq_n_f32(0.); 4];

        if csc == 1 {
            loop4!(i, c11[i] = vld1q_f32(c![i + 0, 0]));
            loop4!(i, c12[i] = vld1q_f32(c![i + 0, 4]));
            loop4!(i, c21[i] = vld1q_f32(c![i + 4, 0]));
            loop4!(i, c22[i] = vld1q_f32(c![i + 4, 4]));
        } else {
            loop4!(i, c11[i] = loadq_from_pointers!(c![i + 0, 0], c![i + 0, 1], c![i + 0, 2], c![i + 0, 3]));
            loop4!(i, c12[i] = loadq_from_pointers!(c![i + 0, 4], c![i + 0, 5], c![i + 0, 6], c![i + 0, 7]));
            loop4!(i, c21[i] = loadq_from_pointers!(c![i + 4, 0], c![i + 4, 1], c![i + 4, 2], c![i + 4, 3]));
            loop4!(i, c22[i] = loadq_from_pointers!(c![i + 4, 4], c![i + 4, 5], c![i + 4, 6], c![i + 4, 7]));
        }

        let betav = vmovq_n_f32(beta);

        // ab += β C
        loop4!(i, ab11[i] = vfmaq_f32(ab11[i], c11[i], betav));
        loop4!(i, ab12[i] = vfmaq_f32(ab12[i], c12[i], betav));
        loop4!(i, ab21[i] = vfmaq_f32(ab21[i], c21[i], betav));
        loop4!(i, ab22[i] = vfmaq_f32(ab22[i], c22[i], betav));
    }

    // c <- ab
    // which is in full
    //   C <- α A B (+ β C)
    if csc == 1 {
        loop4!(i, vst1q_f32(c![i + 0, 0], ab11[i]));
        loop4!(i, vst1q_f32(c![i + 0, 4], ab12[i]));
        loop4!(i, vst1q_f32(c![i + 4, 0], ab21[i]));
        loop4!(i, vst1q_f32(c![i + 4, 4], ab22[i]));
    } else {
        loop4!(i, vst1q_lane_f32(c![i + 0, 0], ab11[i], 0));
        loop4!(i, vst1q_lane_f32(c![i + 0, 1], ab11[i], 1));
        loop4!(i, vst1q_lane_f32(c![i + 0, 2], ab11[i], 2));
        loop4!(i, vst1q_lane_f32(c![i + 0, 3], ab11[i], 3));

        loop4!(i, vst1q_lane_f32(c![i + 0, 4], ab12[i], 0));
        loop4!(i, vst1q_lane_f32(c![i + 0, 5], ab12[i], 1));
        loop4!(i, vst1q_lane_f32(c![i + 0, 6], ab12[i], 2));
        loop4!(i, vst1q_lane_f32(c![i + 0, 7], ab12[i], 3));

        loop4!(i, vst1q_lane_f32(c![i + 4, 0], ab21[i], 0));
        loop4!(i, vst1q_lane_f32(c![i + 4, 1], ab21[i], 1));
        loop4!(i, vst1q_lane_f32(c![i + 4, 2], ab21[i], 2));
        loop4!(i, vst1q_lane_f32(c![i + 4, 3], ab21[i], 3));

        loop4!(i, vst1q_lane_f32(c![i + 4, 4], ab22[i], 0));
        loop4!(i, vst1q_lane_f32(c![i + 4, 5], ab22[i], 1));
        loop4!(i, vst1q_lane_f32(c![i + 4, 6], ab22[i], 2));
        loop4!(i, vst1q_lane_f32(c![i + 4, 7], ab22[i], 3));
    }
}

#[cfg(all(target_arch="wasm32", target_feature="simd128"))]
unsafe fn kernel_target_wasm_simd(k: usize, alpha: T, a: *const T, b: *const T,
                                  beta: T, c: *mut T, rsc: isize, csc: isize)
{
    use core::arch::wasm32::*;
    const MR: usize = KernelWasmSimd::MR;
    const NR: usize = KernelWasmSimd::NR;

    // Use f32x4_relaxed_madd when enabled
    // by spec relaxed_madd is a fused multiply-add when possible, otherwise it is a multiply then add
    #[cfg(target_feature = "relaxed-simd")]
    #[inline(always)]
    unsafe fn muladd(a: v128, b: v128, c: v128) -> v128 {
        f32x4_relaxed_madd(a, b, c)
    }

    #[cfg(not(target_feature = "relaxed-simd"))]
    #[inline(always)]
    unsafe fn muladd(a: v128, b: v128, c: v128) -> v128 {
        f32x4_add(f32x4_mul(a, b), c)
    }

    let (mut a, mut b, rsc, csc) = if rsc == 1 { (b, a, csc, rsc) } else { (a, b, rsc, csc) };

    // Kernel 8 x 8 (a x b)
    // Four quadrants of 4 x 4
    let zero = f32x4_splat(0.);
    let mut ab11 = [zero; 4];
    let mut ab12 = [zero; 4];
    let mut ab21 = [zero; 4];
    let mut ab22 = [zero; 4];

    // ab_ij = a_i * b_j for all i, j
    macro_rules! ab_ij_equals_ai_bj {
        ($dest:ident, $av:expr, $bv:expr) => {
            $dest[0] = muladd($bv, f32x4_splat(f32x4_extract_lane::<0>($av)), $dest[0]);
            $dest[1] = muladd($bv, f32x4_splat(f32x4_extract_lane::<1>($av)), $dest[1]);
            $dest[2] = muladd($bv, f32x4_splat(f32x4_extract_lane::<2>($av)), $dest[2]);
            $dest[3] = muladd($bv, f32x4_splat(f32x4_extract_lane::<3>($av)), $dest[3]);
        }
    }

    for _ in 0..k {
        let a1 = v128_load(a as *const v128);
        let b1 = v128_load(b as *const v128);
        let a2 = v128_load(a.add(4) as *const v128);
        let b2 = v128_load(b.add(4) as *const v128);

        ab_ij_equals_ai_bj!(ab11, a1, b1);
        ab_ij_equals_ai_bj!(ab12, a1, b2);
        ab_ij_equals_ai_bj!(ab21, a2, b1);
        ab_ij_equals_ai_bj!(ab22, a2, b2);

        a = a.add(MR);
        b = b.add(NR);
    }

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // ab *= alpha
    let alphav = f32x4_splat(alpha);
    loop4!(i, ab11[i] = f32x4_mul(ab11[i], alphav));
    loop4!(i, ab12[i] = f32x4_mul(ab12[i], alphav));
    loop4!(i, ab21[i] = f32x4_mul(ab21[i], alphav));
    loop4!(i, ab22[i] = f32x4_mul(ab22[i], alphav));

    // load one v128 from four pointers
    macro_rules! loadq_from_pointers {
        ($p0:expr, $p1:expr, $p2:expr, $p3:expr) => ({
            let v = f32x4_splat(0.);
            let v = v128_load32_lane::<0>(v, $p0 as *const u32);
            let v = v128_load32_lane::<1>(v, $p1 as *const u32);
            let v = v128_load32_lane::<2>(v, $p2 as *const u32);
            let v = v128_load32_lane::<3>(v, $p3 as *const u32);
            v
        });
    }

    if beta != 0. {
        // load existing value in C
        let mut c11 = [zero; 4];
        let mut c12 = [zero; 4];
        let mut c21 = [zero; 4];
        let mut c22 = [zero; 4];

        if csc == 1 {
            loop4!(i, c11[i] = v128_load(c![i + 0, 0] as *const v128));
            loop4!(i, c12[i] = v128_load(c![i + 0, 4] as *const v128));
            loop4!(i, c21[i] = v128_load(c![i + 4, 0] as *const v128));
            loop4!(i, c22[i] = v128_load(c![i + 4, 4] as *const v128));
        } else {
            loop4!(i, c11[i] = loadq_from_pointers!(c![i + 0, 0], c![i + 0, 1], c![i + 0, 2], c![i + 0, 3]));
            loop4!(i, c12[i] = loadq_from_pointers!(c![i + 0, 4], c![i + 0, 5], c![i + 0, 6], c![i + 0, 7]));
            loop4!(i, c21[i] = loadq_from_pointers!(c![i + 4, 0], c![i + 4, 1], c![i + 4, 2], c![i + 4, 3]));
            loop4!(i, c22[i] = loadq_from_pointers!(c![i + 4, 4], c![i + 4, 5], c![i + 4, 6], c![i + 4, 7]));
        }

        let betav = f32x4_splat(beta);
        // ab += β C
        loop4!(i, ab11[i] = muladd(betav, c11[i], ab11[i]));
        loop4!(i, ab12[i] = muladd(betav, c12[i], ab12[i]));
        loop4!(i, ab21[i] = muladd(betav, c21[i], ab21[i]));
        loop4!(i, ab22[i] = muladd(betav, c22[i], ab22[i]));
    }

    // c <- ab
    // which is in full
    //   C <- α A B (+ β C)
    if csc == 1 {
        loop4!(i, v128_store(c![i + 0, 0] as *mut v128, ab11[i]));
        loop4!(i, v128_store(c![i + 0, 4] as *mut v128, ab12[i]));
        loop4!(i, v128_store(c![i + 4, 0] as *mut v128, ab21[i]));
        loop4!(i, v128_store(c![i + 4, 4] as *mut v128, ab22[i]));
    } else {
        loop4!(i, v128_store32_lane::<0>(ab11[i], c![i + 0, 0] as *mut u32));
        loop4!(i, v128_store32_lane::<1>(ab11[i], c![i + 0, 1] as *mut u32));
        loop4!(i, v128_store32_lane::<2>(ab11[i], c![i + 0, 2] as *mut u32));
        loop4!(i, v128_store32_lane::<3>(ab11[i], c![i + 0, 3] as *mut u32));

        loop4!(i, v128_store32_lane::<0>(ab12[i], c![i + 0, 4] as *mut u32));
        loop4!(i, v128_store32_lane::<1>(ab12[i], c![i + 0, 5] as *mut u32));
        loop4!(i, v128_store32_lane::<2>(ab12[i], c![i + 0, 6] as *mut u32));
        loop4!(i, v128_store32_lane::<3>(ab12[i], c![i + 0, 7] as *mut u32));

        loop4!(i, v128_store32_lane::<0>(ab21[i], c![i + 4, 0] as *mut u32));
        loop4!(i, v128_store32_lane::<1>(ab21[i], c![i + 4, 1] as *mut u32));
        loop4!(i, v128_store32_lane::<2>(ab21[i], c![i + 4, 2] as *mut u32));
        loop4!(i, v128_store32_lane::<3>(ab21[i], c![i + 4, 3] as *mut u32));

        loop4!(i, v128_store32_lane::<0>(ab22[i], c![i + 4, 4] as *mut u32));
        loop4!(i, v128_store32_lane::<1>(ab22[i], c![i + 4, 5] as *mut u32));
        loop4!(i, v128_store32_lane::<2>(ab22[i], c![i + 4, 6] as *mut u32));
        loop4!(i, v128_store32_lane::<3>(ab22[i], c![i + 4, 7] as *mut u32));
    }
}

#[inline]
unsafe fn kernel_fallback_impl(k: usize, alpha: T, a: *const T, b: *const T,
                               beta: T, c: *mut T, rsc: isize, csc: isize)
{
    const MR: usize = KernelFallback::MR;
    const NR: usize = KernelFallback::NR;
    let mut ab: [[T; NR]; MR] = [[0.; NR]; MR];
    let mut a = a;
    let mut b = b;
    debug_assert_eq!(beta, 0., "Beta must be 0 or is not masked");

    // Compute A B into ab[i][j]
    unroll_by!(4 => k, {
        loop8!(i, loop4!(j, ab[i][j] += at(a, i) * at(b, j)));

        a = a.offset(MR as isize);
        b = b.offset(NR as isize);
    });

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // set C = α A B
    loop4!(j, loop8!(i, *c![i, j] = alpha * ab[i][j]));
}

#[inline(always)]
unsafe fn at(ptr: *const T, i: usize) -> T {
    *ptr.offset(i as isize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::test::test_a_kernel;

    #[test]
    fn test_kernel_fallback_impl() {
        test_a_kernel::<KernelFallback, _>("kernel");
    }

    #[cfg(any(target_arch="x86", target_arch="x86_64"))]
    #[test]
    fn test_loop_m_n() {
        let mut m = [[0; KernelAvx::NR]; KernelAvx::MR];
        loop_m!(i, loop_n!(j, m[i][j] += 1));
        for arr in &m[..] {
            for elt in &arr[..] {
                assert_eq!(*elt, 1);
            }
        }
    }

    #[cfg(any(target_arch="aarch64"))]
    #[cfg(has_aarch64_simd)]
    mod test_kernel_aarch64 {
        use super::test_a_kernel;
        use super::super::*;
        #[cfg(feature = "std")]
        use std::println;

        macro_rules! test_arch_kernels_aarch64 {
            ($($feature_name:tt, $name:ident, $kernel_ty:ty),*) => {
                $(
                #[test]
                fn $name() {
                    if is_aarch64_feature_detected_!($feature_name) {
                        test_a_kernel::<$kernel_ty, _>(stringify!($name));
                    } else {
                        #[cfg(feature = "std")]
                        println!("Skipping, host does not have feature: {:?}", $feature_name);
                    }
                }
                )*
            }
        }

        test_arch_kernels_aarch64! {
            "neon", neon8x8, KernelNeon
        }
    }

    #[cfg(all(target_arch="wasm32", target_feature="simd128"))]
    mod test_kernel_wasm {
        use super::test_a_kernel;
        use super::super::*;

        #[test]
        fn wasm_simd_8x8() {
            test_a_kernel::<KernelWasmSimd, _>("wasm_simd_8x8");
        }
    }

    #[cfg(any(target_arch="x86", target_arch="x86_64"))]
    mod test_kernel_x86 {
        use super::test_a_kernel;
        use super::super::*;
        #[cfg(feature = "std")]
        use std::println;

        macro_rules! test_arch_kernels_x86 {
            ($($feature_name:tt, $name:ident, $kernel_ty:ty),*) => {
                $(
                #[test]
                fn $name() {
                    if is_x86_feature_detected_!($feature_name) {
                        test_a_kernel::<$kernel_ty, _>(stringify!($name));
                    } else {
                        #[cfg(feature = "std")]
                        println!("Skipping, host does not have feature: {:?}", $feature_name);
                    }
                }
                )*
            }
        }

        test_arch_kernels_x86! {
            "fma", fma, KernelFma,
            "avx", avx, KernelAvx,
            "sse2", sse2, KernelSse2
        }

        #[test]
        fn ensure_target_features_tested() {
            // If enabled, this test ensures that the requested feature actually
            // was enabled on this configuration, so that it was tested.
            let should_ensure_feature = !option_env!("MMTEST_ENSUREFEATURE")
                                                    .unwrap_or("").is_empty();
            if !should_ensure_feature {
                // skip
                return;
            }
            let feature_name = option_env!("MMTEST_FEATURE")
                                          .expect("No MMTEST_FEATURE configured!");
            let detected = match feature_name {
                "avx" => is_x86_feature_detected_!("avx"),
                "fma" => is_x86_feature_detected_!("fma"),
                "sse2" => is_x86_feature_detected_!("sse2"),
                _ => false,
            };
            assert!(detected, "Feature {:?} was not detected, so it could not be tested",
                    feature_name);
        }
    }
}
