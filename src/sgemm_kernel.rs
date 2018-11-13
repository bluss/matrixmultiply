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

const MR: usize = 8;
const NR: usize = 8;

macro_rules! loop_m { ($i:ident, $e:expr) => { loop8!($i, $e) }; }
macro_rules! loop_n { ($j:ident, $e:expr) => { loop8!($j, $e) }; }

impl GemmKernel for Gemm {
    type Elem = T;

    #[inline(always)]
    fn align_to() -> usize { 32 }

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
#[inline]
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
    kernel_x86_avx(k, alpha, a, b, beta, c, rsc, csc)
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

#[inline(always)]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
pub unsafe fn kernel_x86_avx(k: usize, alpha: T, a: *const T, b: *const T,
                             beta: T, c: *mut T, rsc: isize, csc: isize)
{
    let mut ab = [_mm256_setzero_ps(); MR];

    let mut bv;
    let (mut a, mut b) = (a, b);

    macro_rules! shuffle_mask {
        ($z:expr, $y:expr, $x:expr, $w:expr) => {
            ($z << 6) | ($y << 4) | ($x << 2) | $w
        }
    }
    macro_rules! permute_mask {
        ($z:expr, $y:expr, $x:expr, $w:expr) => {
            ($z << 6) | ($y << 4) | ($x << 2) | $w
        }
    }

    macro_rules! permute2f128_mask {
        ($y:expr, $x:expr) => {
            (($y << 4) | $x)
        }
    }

    // Compute A B
    unroll_by!(4 => k, {
        bv = _mm256_load_ps(b as _); // aligned due to GemmKernel::align_to

        // vmovsldup ymm2, ymmword ptr [rbx]
        //
        // Load and duplicate each even word:
        // ymm2 ← [b0 b0 b2 b2 b4 b4 b6 b6]
        //
        // vmovshdup ymm2, ymmword ptr [rbx]
        //
        // Load and duplicate each odd word:
        // ymm2 ← [b1 b1 b3 b3 b5 b5 b7 b7]
        //
        //
        // vpermil ymm3, ymm2, 0x4e
        // 0x4e is 0b1001110 which corresponds to selecting:
        //  2, 3, 0, 1
        // _mm256_permute_ps is a shuffle in each 128-bit lane.
        //
        // see blis kernel comments about the layout
        //
        // BLIS runs the main loop with this result parameter layout,
        // then shuffles everything in place post accumulation.
        //
	 // ymm15:  ymm13:  ymm11:  ymm9:
	 // ( ab00  ( ab02  ( ab04  ( ab06
	 //   ab10    ab12    ab14    ab16  
	 //   ab22    ab20    ab26    ab24
	 //   ab32    ab30    ab36    ab34
	 //   ab44    ab46    ab40    ab42
	 //   ab54    ab56    ab50    ab52  
	 //   ab66    ab64    ab62    ab60
	 //   ab76 )  ab74 )  ab72 )  ab70 )
	
	 // ymm14:  ymm12:  ymm10:  ymm8:
	 // ( ab01  ( ab03  ( ab05  ( ab07
	 //   ab11    ab13    ab15    ab17  
	 //   ab23    ab21    ab27    ab25
	 //   ab33    ab31    ab37    ab35
	 //   ab45    ab47    ab41    ab43
	 //   ab55    ab57    ab51    ab53  
	 //   ab67    ab65    ab63    ab61
	 //   ab77 )  ab75 )  ab73 )  ab71 )
        //
        // 

        const PERM32_2301: i32 = permute_mask!(1, 0, 3, 2);
        const PERM128_30: i32 = permute2f128_mask!(0, 3);
        // Compute ab_i += [ai b_j+0, ai b_j+1, ai b_j+2, ai b_j+3]
        let av = _mm256_load_ps(a);

        // 2 permute_ps per iteration
        // 4 permute2f128 per iteration

        let a0246 = _mm256_moveldup_ps(av); // Load: a0 a0 a2 a2 a4 a4 a6 a6
        let a2064 = _mm256_permute_ps(a0246, PERM32_2301);

        let a1357 = _mm256_movehdup_ps(av); // Load: a1 a1 a3 a3 a5 a5 a7 a7
        let a3175 = _mm256_permute_ps(a1357, PERM32_2301);

        let a4602 = _mm256_permute2f128_ps(a0246, a0246, PERM128_30);
        let a6420 = _mm256_permute2f128_ps(a2064, a2064, PERM128_30);

        let a5713 = _mm256_permute2f128_ps(a1357, a1357, PERM128_30);
        let a7531 = _mm256_permute2f128_ps(a3175, a3175, PERM128_30);

        ab[0] = _mm256_add_ps(ab[0], _mm256_mul_ps(a0246, bv));
        ab[1] = _mm256_add_ps(ab[1], _mm256_mul_ps(a2064, bv));
        ab[2] = _mm256_add_ps(ab[2], _mm256_mul_ps(a4602, bv));
        ab[3] = _mm256_add_ps(ab[3], _mm256_mul_ps(a6420, bv));

        ab[4] = _mm256_add_ps(ab[4], _mm256_mul_ps(a1357, bv));
        ab[5] = _mm256_add_ps(ab[5], _mm256_mul_ps(a3175, bv));
        ab[6] = _mm256_add_ps(ab[6], _mm256_mul_ps(a5713, bv));
        ab[7] = _mm256_add_ps(ab[7], _mm256_mul_ps(a7531, bv));

        a = a.add(MR);
        b = b.add(NR);
    });

    // Compute α (A B)
    let alphav = _mm256_set1_ps(alpha);
    loop_m!(i, ab[i] = _mm256_mul_ps(alphav, ab[i]));

    // Permute to get back to:
    // ab00 ab10  ... etc
    // ab01 ab11  
    // ab02 ab12  
    // ab03 ab13  
    // ab04 ab14  
    // ab05 ab15  
    // ab06 ab16  
    // ab07 ab17  
    //
    // They use order:
    //
    // 02
    // 20
    // 46
    // 64
    // 13
    // 31
    // 57
    // 75
    //
    //  // 22006644
    //  // 00224466
    //  //
    //  // shufps 0xe4 ->
    //  //
    //  // 22226666
    //
    // shufps 20 02, 0xe4 -> new reg
    // shufps 02 20, 0xe4 -> new reg
    //
    // shufps 64 46, 0xe4 -> new reg
    // shufps 46 64, 0xe4 -> new reg
    //
    // shufps 31 13, 0xe4 -> new reg
    // shufps 13 31, 0xe4 -> new reg
    //
    // shufps 75 57, 0xe4 -> new reg
    // shufps 57 75, 0xe4 -> new reg
    //
    // 0xe4 is 0b11100100 corresponding to 0, 1, 2, 3
    //
    // vperm2f128 0x30 -> 0b110000 is 0: DEST[127:0]←SRC1[127:0] AND 3: DEST[255:128]←SRC2[255:128]
    // vperm2f128 0x12 ->  0b10010 is 2: DEST[127:0]←SRC2[127:0] AND 1: DEST[255:128]←SRC1[255:128]
    //
    // vperm2 0x30: 00004444 and 44440000 -> 00000000 
    // vperm2 0x12: 00004444 and 44440000 -> 44444444 
    //
    
    // shuffle and permute to bring the result in the right order post main loop
    let ab0246 = ab[0];
    let ab2064 = ab[1];
    let ab4602 = ab[2];
    let ab6420 = ab[3];

    let ab1357 = ab[4];
    let ab3175 = ab[5];
    let ab5713 = ab[6];
    let ab7531 = ab[7];

    const SHUF_0123: i32 = shuffle_mask!(3, 2, 1, 0);
    debug_assert_eq!(SHUF_0123, 0xE4);

    const PERM128_03: i32 = permute2f128_mask!(3, 0); //0b110000;
    const PERM128_21: i32 = permute2f128_mask!(1, 2); //0b010010;

    let ab0044 = _mm256_shuffle_ps(ab0246, ab2064, SHUF_0123);
    let ab2266 = _mm256_shuffle_ps(ab2064, ab0246, SHUF_0123);

    let ab4400 = _mm256_shuffle_ps(ab4602, ab6420, SHUF_0123);
    let ab6622 = _mm256_shuffle_ps(ab6420, ab4602, SHUF_0123);

    let ab1155 = _mm256_shuffle_ps(ab1357, ab3175, SHUF_0123);
    let ab3377 = _mm256_shuffle_ps(ab3175, ab1357, SHUF_0123);

    let ab5511 = _mm256_shuffle_ps(ab5713, ab7531, SHUF_0123);
    let ab7733 = _mm256_shuffle_ps(ab7531, ab5713, SHUF_0123);

    let ab0000 = _mm256_permute2f128_ps(ab0044, ab4400, PERM128_03);
    let ab4444 = _mm256_permute2f128_ps(ab0044, ab4400, PERM128_21);

    let ab2222 = _mm256_permute2f128_ps(ab2266, ab6622, PERM128_03);
    let ab6666 = _mm256_permute2f128_ps(ab2266, ab6622, PERM128_21);

    let ab1111 = _mm256_permute2f128_ps(ab1155, ab5511, PERM128_03);
    let ab5555 = _mm256_permute2f128_ps(ab1155, ab5511, PERM128_21);

    let ab3333 = _mm256_permute2f128_ps(ab3377, ab7733, PERM128_03);
    let ab7777 = _mm256_permute2f128_ps(ab3377, ab7733, PERM128_21);

    ab[0] = ab0000;
    ab[1] = ab1111;
    ab[2] = ab2222;
    ab[3] = ab3333;
    ab[4] = ab4444;
    ab[5] = ab5555;
    ab[6] = ab6666;
    ab[7] = ab7777;

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // C ← α A B + β C
    let mut c = [_mm256_setzero_ps(); MR];
    let betav = _mm256_set1_ps(beta);
    if beta != 0. {
        // Read C
        if csc == 1 {
            loop_m!(i, c[i] = _mm256_loadu_ps(c![i, 0]));
        // Handle rsc == 1 case with transpose?
        } else {
            loop_m!(i, c[i] = _mm256_set_ps(*c![i, 7], *c![i, 6], *c![i, 5], *c![i, 4], *c![i, 3], *c![i, 2], *c![i, 1], *c![i, 0]));
        }
        // Compute β C
        loop_m!(i, c[i] = _mm256_mul_ps(c[i], betav));
    }

    // Compute (α A B) + (β C)
    loop_m!(i, c[i] = _mm256_add_ps(c[i], ab[i]));

    // Store C back to memory
    if csc == 1 {
        loop_m!(i, _mm256_storeu_ps(c![i, 0], c[i]));
    // Handle rsc == 1 case with transpose?
    } else {
        // Permute to bring each element in the vector to the front and store
        loop_m!(i, {
            let clo = _mm256_extractf128_ps(c[i], 0);
            let chi = _mm256_extractf128_ps(c[i], 1);

            _mm_store_ss(c![i, 0], clo);
            let cperm = _mm_permute_ps(clo, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 1], cperm);
            let cperm = _mm_permute_ps(cperm, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 2], cperm);
            let cperm = _mm_permute_ps(cperm, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 3], cperm);


            _mm_store_ss(c![i, 4], chi);
            let cperm = _mm_permute_ps(chi, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 5], cperm);
            let cperm = _mm_permute_ps(cperm, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 6], cperm);
            let cperm = _mm_permute_ps(cperm, permute_mask!(0, 3, 2, 1));
            _mm_store_ss(c![i, 7], cperm);
        });
    }
}

#[inline]
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
