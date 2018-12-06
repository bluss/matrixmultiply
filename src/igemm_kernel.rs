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

pub type T = i32;

const MR: usize = 8;
const NR: usize = 8;

macro_rules! loop_m { ($i:ident, $e:expr) => { loop8!($i, $e) }; }
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
#[inline(never)]
pub unsafe fn kernel(k: usize, alpha: T, a: *const T, b: *const T,
                     beta: T, c: *mut T, rsc: isize, csc: isize)
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
unsafe fn kernel_target_avx2(k: usize, alpha: T, a: *const T, b: *const T,
                            beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_x86_avx2(k, alpha, a, b, beta, c, rsc, csc)
}

#[inline]
#[target_feature(enable="avx")]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
unsafe fn kernel_target_avx(k: usize, alpha: T, a: *const T, b: *const T,
                            beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc)
}

#[inline]
#[target_feature(enable="sse2")]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
unsafe fn kernel_target_sse2(k: usize, alpha: T, a: *const T, b: *const T,
                             beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc)
}


#[inline(always)]
unsafe fn kernel_fallback_impl(k: usize, alpha: T, a: *const T, b: *const T,
                               beta: T, c: *mut T, rsc: isize, csc: isize)
{
    let mut ab: [[T; NR]; MR] = [[0; NR]; MR];
    let mut a = a;
    let mut b = b;
    debug_assert_eq!(beta, 0);

    // Compute A B into ab[i][j]
    unroll_by!(4 => k, {
        loop_m!(i, loop_n!(j, {
            ab[i][j] = ab[i][j].wrapping_add(at(a, i).wrapping_mul(at(b, j)));
        }));

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
unsafe fn kernel_x86_avx2(k: usize, alpha: T, a: *const T, b: *const T,
                         beta: T, c: *mut T, rsc: isize, csc: isize)
{
    debug_assert_ne!(k, 0);

    let mut ab = [_mm256_setzero_si256(); MR];

    let (mut a, mut b) = (a, b);

    println!("a_slice: {:?}", std::slice::from_raw_parts(a, 8));
    let mut a_0123_4567 = _mm256_loadu_si256(a as *const __m256i);
    println!("a_simd:  {:?}", a_0123_4567);

    println!("b_slice: {:?}", std::slice::from_raw_parts(b, 8));
    let mut b_0123_4567 = _mm256_loadu_si256(b as *const __m256i);
    println!("b_simd:  {:?}", b_0123_4567);

    // The task in the loop below is to multiply every number packed in b_0123_4567 with
    // every number in the two a_* vectors. With a_* loading a column, and b_* loading a row,
    // this is exactly equivalent to matrix multiplication defined by C_ij = ∑_k A_ik B_kj + C_ij, but
    // where you fix k, perform all multiplications in i and j for a fixed k, add the result to
    // C, and then increment k to and repeat.
    //
    // Reapeted indices are shortened, e.g. b_02_46 = b_0022_4466

    unroll_by_with_last!(4 => k, is_last, {
        // two bits select the value of one i32, so 4 bits select two adjacent i32. 4 bits can be
        // represented in hexadecimal notation. The first (from back to font) 4 pairs of bits
        // select the first 128bit lane, the last 4 pairs the second lane.
        let b_02_46 = _mm256_shuffle_epi32(
            b_0123_4567,
            0b_10_10_00_00__10_10_00_00
        );

        let b_20_64 = _mm256_shuffle_epi32(
            b_0123_4567,
            0b_00_00_10_10__00_00_10_10
        );

        let b_46_02 = _mm256_permute2x128_si256(
            b_02_46,
            b_02_46,
            0x03
        );

        let b_64_20 = _mm256_permute2x128_si256(
            b_20_64,
            b_20_64,
            0x03
        );

        let b_13_57 = _mm256_shuffle_epi32(
            b_0123_4567,
            0b_11_11_01_01__11_11_01_01
        );

        let b_31_75 = _mm256_shuffle_epi32(
            b_0123_4567,
            0b_01_01_11_11__01_01_11_11
        );

        let b_57_13 = _mm256_permute2x128_si256(
            b_13_57,
            b_13_57,
            0x03
        );

        let b_75_31 = _mm256_permute2x128_si256(
            b_31_75,
            b_31_75,
            0x03
        );

        // Add and multiply in one go
        ab[0] = _mm256_add_epi32(ab[0], _mm256_mul_epi32(a_0123_4567, b_02_46));
        ab[1] = _mm256_add_epi32(ab[1], _mm256_mul_epi32(a_0123_4567, b_20_64));
        ab[2] = _mm256_add_epi32(ab[2], _mm256_mul_epi32(a_0123_4567, b_46_02));
        ab[3] = _mm256_add_epi32(ab[3], _mm256_mul_epi32(a_0123_4567, b_64_20));

        ab[4] = _mm256_add_epi32(ab[4], _mm256_mul_epi32(a_0123_4567, b_13_57));
        ab[5] = _mm256_add_epi32(ab[5], _mm256_mul_epi32(a_0123_4567, b_31_75));
        ab[6] = _mm256_add_epi32(ab[6], _mm256_mul_epi32(a_0123_4567, b_57_13));
        ab[7] = _mm256_add_epi32(ab[7], _mm256_mul_epi32(a_0123_4567, b_75_31));

        if !is_last {
            a = a.add(MR);
            b = b.add(NR);

            a_0123_4567 = _mm256_loadu_si256(a as _);
            b_0123_4567 = _mm256_loadu_si256(b as _);
        }
    });

    let a0b0_a1b0_a2b0_a3b0_a4b4_a5b4_a6b4_a7b4 = _mm256_blend_epi32(
        ab[0],
        ab[1],
        0b_1100_1100
    );

    let a0b2_a1b2_a2b2_a3b2_a4b6_a5b6_a6b6_a7b6 = _mm256_blend_epi32(
        ab[0],
        ab[1],
        0b_0011_0011
    );

    let a0b4_a1b4_a2b4_a3b4_a4b0_a5b0_a6b0_a7b0 = _mm256_blend_epi32(
        ab[2],
        ab[3],
        0b_1100_1100
    );

    let a0b6_a1b6_a2b6_a3b6_a4b2_a5b2_a6b2_a7b2 = _mm256_blend_epi32(
        ab[2],
        ab[3],
        0b_0011_0011
    );

    // a0b0_a1b0_a2b0_a3b0_b4b0_b5b0_b6b0_b7b0
    ab[0] = _mm256_permute2f128_si256(
        a0b0_a1b0_a2b0_a3b0_a4b4_a5b4_a6b4_a7b4,
        a0b4_a1b4_a2b4_a3b4_a4b0_a5b0_a6b0_a7b0,
        0x30
    );
    // a0b4_a1b4_a2b4_a3b4_b4b4_b5b4_b6b4_b7b4
    ab[4] = _mm256_permute2f128_si256(
        a0b0_a1b0_a2b0_a3b0_a4b4_a5b4_a6b4_a7b4,
        a0b4_a1b4_a2b4_a3b4_a4b0_a5b0_a6b0_a7b0,
        0x12
    );
    // a0b2_a1b2_a2b2_a3b2_b4b2_b5b2_b6b2_b7b2
    ab[2] = _mm256_permute2f128_si256(
        a0b2_a1b2_a2b2_a3b2_a4b6_a5b6_a6b6_a7b6,
        a0b6_a1b6_a2b6_a3b6_a4b2_a5b2_a6b2_a7b2,
        0x30
    );
    // a0b6_a1b6_a2b6_a3b6_b4b6_b5b6_b6b6_b7b6
    ab[6] = _mm256_permute2f128_si256(
        a0b2_a1b2_a2b2_a3b2_a4b6_a5b6_a6b6_a7b6,
        a0b6_a1b6_a2b6_a3b6_a4b2_a5b2_a6b2_a7b2,
        0x12
    );

    let a0b1_a1b1_a2b1_a3b1_a4b5_a5b5_a6b5_a7b5 = _mm256_blend_epi32(
        ab[4],
        ab[5],
        0b_1100_1100
    );

    let a0b3_a1b3_a2b3_a3b3_a4b7_a5b7_a6b7_a7b7 = _mm256_blend_epi32(
        ab[4],
        ab[5],
        0b_0011_0011
    );

    let a0b5_a1b5_a2b5_a3b5_a4b1_a5b1_a6b1_a7b1 = _mm256_blend_epi32(
        ab[6],
        ab[7],
        0b_1100_1100
    );

    let a0b7_a1b7_a2b7_a3b7_a4b3_a5b3_a6b3_a7b3 = _mm256_blend_epi32(
        ab[6],
        ab[7],
        0b_0011_0011
    );

    // a0b1_a1b1_a2b1_a3b1_b4b1_b5b1_b6b1_b7b1
    ab[1] = _mm256_permute2f128_si256(
        a0b1_a1b1_a2b1_a3b1_a4b5_a5b5_a6b5_a7b5,
        a0b5_a1b5_a2b5_a3b5_a4b1_a5b1_a6b1_a7b1,
        0x30
    );
    // a0b5_a1b5_a2b5_a3b5_b4b5_b5b5_b6b5_b7b5
    ab[5] = _mm256_permute2f128_si256(
        a0b1_a1b1_a2b1_a3b1_a4b5_a5b5_a6b5_a7b5,
        a0b5_a1b5_a2b5_a3b5_a4b1_a5b1_a6b1_a7b1,
        0x12
    );
    // a0b3_a1b3_a2b3_a3b3_b4b3_b5b3_b6b3_b7b3
    ab[3] = _mm256_permute2f128_si256(
        a0b3_a1b3_a2b3_a3b3_a4b7_a5b7_a6b7_a7b7,
        a0b7_a1b7_a2b7_a3b7_a4b3_a5b3_a6b3_a7b3,
        0x30
    );
    // a0b7_a1b7_a2b7_a3b7_b4b7_b5b7_b6b7_b7b7
    ab[7] = _mm256_permute2f128_si256(
        a0b3_a1b3_a2b3_a3b3_a4b7_a5b7_a6b7_a7b7,
        a0b7_a1b7_a2b7_a3b7_a4b3_a5b3_a6b3_a7b3,
        0x12
    );

    // Compute α (A B)
    let alpha_v = _mm256_set1_epi32(alpha);
    loop_m!(i, ab[i] = _mm256_mul_epi32(alpha_v, ab[i]));

    macro_rules! c {
        ($i:expr, $j:expr) =>
            (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // C ← α A B + β C
    let mut cv = [_mm256_setzero_si256(); MR];

    if beta != 0 {
        let beta_v = _mm256_set1_epi32(beta);

        // Read C
        if rsc == 1 {
            loop_m!(i, cv[i] = _mm256_loadu_si256(c![0, i] as _));
        // } else if csc == 1 {
        //     loop4!(i, cv[i] = _mm256_loadu_pd(c![i, 0]));
        //     loop4!(i, cv[i+4] = _mm256_loadu_pd(c![i+4, 0]));
        } else {
            loop_m!(i, cv[i] = _mm256_setr_epi32(
                    *c![0, i],
                    *c![1, i],
                    *c![2, i],
                    *c![3, i],
                    *c![4, i],
                    *c![5, i],
                    *c![6, i],
                    *c![7, i],
            ));
        }
        // Compute β C
        loop_m!(i, cv[i] = _mm256_mul_epi32(cv[i], beta_v));
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
            *c![i, 0] = _mm256_extract_epi32(cv[i], 0);
            *c![i, 1] = _mm256_extract_epi32(cv[i], 1);
            *c![i, 2] = _mm256_extract_epi32(cv[i], 2);
            *c![i, 3] = _mm256_extract_epi32(cv[i], 3);
            *c![i, 4] = _mm256_extract_epi32(cv[i], 4);
            *c![i, 5] = _mm256_extract_epi32(cv[i], 5);
            *c![i, 6] = _mm256_extract_epi32(cv[i], 6);
            *c![i, 7] = _mm256_extract_epi32(cv[i], 7);
        })
    }
}

#[inline(always)]
unsafe fn at(ptr: *const T, i: usize) -> T {
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

    use super::T;
    type KernelFn = unsafe fn(usize, T, *const T, *const T, T, *mut T, isize, isize);

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
