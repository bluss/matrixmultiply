// Copyright 2016 - 2018 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use kernel::GemmKernel;
use archparam;

pub enum Gemm { }

pub type T = f64;

const MR: usize = 8;
const NR: usize = 4;

macro_rules! loop_m {
    ($i:ident, $e:expr) => { loop8!($i, $e) };
}
macro_rules! loop_n {
    ($j:ident, $e:expr) => { loop4!($j, $e) };
}

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
    fn nc() -> usize { archparam::D_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::D_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::D_MC }

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
        if is_x86_feature_detected_!("avx") {
            return kernel_target_avx(k, alpha, a, b, beta, c, rsc, csc);
        } else if is_x86_feature_detected_!("sse2") {
            return kernel_target_sse2(k, alpha, a, b, beta, c, rsc, csc);
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
    kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc)
}

#[inline]
#[target_feature(enable="sse2")]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
pub unsafe fn kernel_target_sse2(k: usize, alpha: T, a: *const T, b: *const T,
                                 beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc)
}

#[inline(always)]
pub unsafe fn kernel_fallback_impl(k: usize, alpha: T, a: *const T, b: *const T,
                                   beta: T, c: *mut T, rsc: isize, csc: isize)
{
    let mut ab: [[T; NR]; MR] = [[0.; NR]; MR];
    let mut a = a;
    let mut b = b;
    debug_assert_eq!(beta, 0.); // always masked

    // Compute matrix multiplication into ab[i][j]
    unroll_by!(4 => k, {
        loop_m!(i, loop_n!(j, ab[i][j] += at(a, i) * at(b, j)));

        a = a.offset(MR as isize);
        b = b.offset(NR as isize);
    });

    macro_rules! c {
        ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
    }

    // set C = α A B
    loop_m!(i, loop_n!(j, *c![i, j] = alpha * ab[i][j]));
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
        let mut a = aligned_alloc(1., MR * K);
        let mut b = aligned_alloc(0., NR * K);
        for (i, x) in a.iter_mut().enumerate() {
            *x = i as _;
        }

        for i in 0..K {
            b[i + i * NR] = 1.;
        }
        let mut c = [0.; MR * NR];
        unsafe {
            kernel_fn(K, 1., &a[0], &b[0], 0., &mut c[0], 1, MR as isize);
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
            "avx", kernel_target_avx,
            "sse2", kernel_target_sse2
        }
    }
}
