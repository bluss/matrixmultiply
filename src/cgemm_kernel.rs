// Copyright 2016 - 2021 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::kernel::GemmKernel;
use crate::kernel::GemmSelect;
use crate::kernel::{U2, U4, c32, Element, c32_mul as mul};


#[cfg(any(target_arch="x86", target_arch="x86_64"))]
struct KernelSse2;
struct KernelFallback;

type T = c32;
type TReal = f32;

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
        if is_x86_feature_detected_!("sse2") {
            return selector.select(KernelSse2);
        }
    }
    return selector.select(KernelFallback);
}

macro_rules! loop_m { ($i:ident, $e:expr) => { loop4!($i, $e) }; }
macro_rules! loop_n { ($j:ident, $e:expr) => { loop2!($j, $e) }; }


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

#[inline]
#[cfg(any(target_arch="x86", target_arch="x86_64"))]
#[target_feature(enable="sse2")]
unsafe fn kernel_target_sse2(k: usize, alpha: T, a: *const T, b: *const T,
                             beta: T, c: *mut T, rsc: isize, csc: isize)
{
    kernel_fallback_impl(k, alpha, a, b, beta, c, rsc, csc)
}

kernel_fallback_impl_complex! { [inline(always)] kernel_fallback_impl, T, TReal, KernelFallback::MR, KernelFallback::NR, 2 }

#[inline(always)]
unsafe fn at(ptr: *const T, i: usize) -> T {
    *ptr.offset(i as isize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;
    use crate::aligned_alloc::Alloc;

    fn aligned_alloc<K>(elt: K::Elem, n: usize) -> Alloc<K::Elem>
        where K: GemmKernel,
              K::Elem: Copy,
    {
        unsafe {
            Alloc::new(n, K::align_to()).init_with(elt)
        }
    }

    use super::T;

    fn test_a_kernel<K: GemmKernel<Elem=T>>(_name: &str) {
        const K: usize = 4;
        let mr = K::MR;
        let nr = K::NR;
        let mut a = aligned_alloc::<K>(T::one(), mr * K);
        let mut b = aligned_alloc::<K>(T::zero(), nr * K);
        for (i, x) in a.iter_mut().enumerate() {
            *x = [i as _, 1.];
        }

        for i in 0..Ord::min(K, nr) {
            b[i + i * nr] = T::one(); 
        }

        let mut c = vec![T::zero(); mr * nr];
        unsafe {
            K::kernel(K, T::one(), &a[0], &b[0], T::zero(), &mut c[0], 1, mr as isize);
            // col major C
        }
        let common_len = Ord::min(a.len(), c.len());
        assert_eq!(&a[..common_len], &c[..common_len]);
    }

    #[test]
    fn test_kernel_fallback_impl() {
        test_a_kernel::<KernelFallback>("kernel");
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
                        test_a_kernel::<$kernel_ty>(stringify!($name));
                    } else {
                        #[cfg(feature = "std")]
                        println!("Skipping, host does not have feature: {:?}", $feature_name);
                    }
                }
                )*
            }
        }

        test_arch_kernels_x86! {
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
                "sse2" => is_x86_feature_detected_!("sse2"),
                _ => false,
            };
            assert!(detected, "Feature {:?} was not detected, so it could not be tested",
                    feature_name);
        }
    }
}
