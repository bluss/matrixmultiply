// Copyright 2016 - 2021 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::kernel::GemmKernel;
use crate::kernel::GemmSelect;
use crate::kernel::{U2, U4, c64, Element, c64_mul as mul};
use crate::archparam;

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
struct KernelFma;
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
struct KernelSse2;
struct KernelFallback;

type T = c64;
type TReal = f64;

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
            return selector.select(KernelFma);
        } else if is_x86_feature_detected_!("sse2") {
            return selector.select(KernelSse2);
        }
    }
    return selector.select(KernelFallback);
}

macro_rules! loop_m { ($i:ident, $e:expr) => { loop4!($i, $e) }; }
macro_rules! loop_n { ($j:ident, $e:expr) => { loop2!($j, $e) }; }

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
impl GemmKernel for KernelFma {
    type Elem = T;

    type MRTy = <KernelFallback as GemmKernel>::MRTy;
    type NRTy = <KernelFallback as GemmKernel>::NRTy;

    #[inline(always)]
    fn align_to() -> usize { 16 }

    #[inline(always)]
    fn always_masked() -> bool { KernelFallback::always_masked() }

    #[inline(always)]
    fn nc() -> usize { archparam::Z_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::Z_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::Z_MC }

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
    fn nc() -> usize { archparam::Z_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::Z_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::Z_MC }

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

impl GemmKernel for KernelFallback {
    type Elem = T;

    type MRTy = U4;
    type NRTy = U2;

    #[inline(always)]
    fn align_to() -> usize { 0 }

    #[inline(always)]
    fn always_masked() -> bool { true }

    #[inline(always)]
    fn nc() -> usize { archparam::Z_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::Z_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::Z_MC }

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

#[cfg(any(target_arch="x86", target_arch="x86_64"))]
kernel_fallback_impl_complex! {
    // instantiate fma separately so that it's inlined here
    [inline target_feature(enable="fma")] kernel_target_fma, T, TReal, KernelFallback::MR, KernelFallback::NR, 2
}


#[inline]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
#[target_feature(enable="sse2")]
unsafe fn kernel_target_sse2(k: usize, alpha: T, a: *const T, b: *const T,
                             beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc)
}

kernel_fallback_impl_complex! { [inline] kernel_fallback_impl, T, TReal, KernelFallback::MR, KernelFallback::NR, 2 }

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
        let mut m = [[0; KernelSse2::NR]; KernelSse2::MR];
        loop_m!(i, loop_n!(j, m[i][j] += 1));
        for arr in &m[..] {
            for elt in &arr[..] {
                assert_eq!(*elt, 1);
            }
        }
    }

    #[cfg(any(target_arch="x86", target_arch="x86_64"))]
    mod test_arch_kernels {
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
            "sse2", sse2, KernelSse2
        }
    }
}
