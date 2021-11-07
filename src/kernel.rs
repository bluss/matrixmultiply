// Copyright 2016 - 2021 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use crate::archparam;

/// General matrix multiply kernel
pub(crate) trait GemmKernel {
    type Elem: Element;

    /// Kernel rows
    const MR: usize = Self::MRTy::VALUE;
    /// Kernel cols
    const NR: usize = Self::NRTy::VALUE;
    /// Kernel rows as const num type
    type MRTy: ConstNum;
    /// Kernel cols as const num type
    type NRTy: ConstNum;

    /// align inputs to this
    fn align_to() -> usize;

    /// Whether to always use the masked wrapper around the kernel.
    fn always_masked() -> bool;

    // These should ideally be tuned per kernel and per microarch
    #[inline(always)]
    fn nc() -> usize { archparam::S_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::S_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::S_MC }

    /// Matrix multiplication kernel
    ///
    /// This does the matrix multiplication:
    ///
    /// C ← α A B + β C
    ///
    /// + `k`: length of data in a, b
    /// + a, b are packed
    /// + c has general strides
    /// + rsc: row stride of c
    /// + csc: col stride of c
    /// + `alpha`: scaling factor for A B product
    /// + `beta`: scaling factor for c.
    ///   Note: if `beta` is `0.`, the kernel should not (and must not)
    ///   read from c, its value is to be treated as if it was zero.
    ///
    /// When masked, the kernel is always called with β=0 but α is passed
    /// as usual. (This is only useful information if you return `true` from
    /// `always_masked`.)
    unsafe fn kernel(
        k: usize,
        alpha: Self::Elem,
        a: *const Self::Elem,
        b: *const Self::Elem,
        beta: Self::Elem,
        c: *mut Self::Elem, rsc: isize, csc: isize);
}

pub(crate) trait Element : Copy + Send + Sync {
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    fn add_assign(&mut self, rhs: Self);
    fn mul_assign(&mut self, rhs: Self);
}

impl Element for f32 {
    fn zero() -> Self { 0. }
    fn one() -> Self { 1. }
    fn is_zero(&self) -> bool { *self == 0. }
    fn add_assign(&mut self, rhs: Self) { *self += rhs; }
    fn mul_assign(&mut self, rhs: Self) { *self *= rhs; }
}

impl Element for f64 {
    fn zero() -> Self { 0. }
    fn one() -> Self { 1. }
    fn is_zero(&self) -> bool { *self == 0. }
    fn add_assign(&mut self, rhs: Self) { *self += rhs; }
    fn mul_assign(&mut self, rhs: Self) { *self *= rhs; }
}

/// Kernel selector
pub(crate) trait GemmSelect<T> {
    /// Call `select` with the selected kernel for this configuration
    fn select<K>(self, kernel: K)
        where K: GemmKernel<Elem=T>,
              T: Element;
}

#[cfg(feature = "cgemm")]
#[allow(non_camel_case_types)]
pub(crate) type c32 = [f32; 2];

#[cfg(feature = "cgemm")]
#[allow(non_camel_case_types)]
pub(crate) type c64 = [f64; 2];

#[cfg(feature = "cgemm")]
impl Element for c32 {
    fn zero() -> Self { [0., 0.] }
    fn one() -> Self { [1., 0.] }
    fn is_zero(&self) -> bool { *self == [0., 0.] }

    #[inline(always)]
    fn add_assign(&mut self, y: Self) {
        self[0] += y[0];
        self[1] += y[1];
    }

    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = c32_mul(*self, rhs);
    }
}

#[cfg(feature = "cgemm")]
impl Element for c64 {
    fn zero() -> Self { [0., 0.] }
    fn one() -> Self { [1., 0.] }
    fn is_zero(&self) -> bool { *self == [0., 0.] }

    #[inline(always)]
    fn add_assign(&mut self, y: Self) {
        self[0] += y[0];
        self[1] += y[1];
    }

    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = c64_mul(*self, rhs);
    }
}

#[cfg(feature = "cgemm")]
#[inline(always)]
pub(crate) fn c32_mul(x: c32, y: c32) -> c32 {
    let [a, b] = x;
    let [c, d] = y;
    [a * c - b * d, b * c + a * d]
}

#[cfg(feature = "cgemm")]
#[inline(always)]
pub(crate) fn c64_mul(x: c64, y: c64) -> c64 {
    let [a, b] = x;
    let [c, d] = y;
    [a * c - b * d, b * c + a * d]
}


pub(crate) trait ConstNum {
    const VALUE: usize;
}

#[cfg(feature = "cgemm")]
pub(crate) struct U2;
pub(crate) struct U4;
pub(crate) struct U8;

#[cfg(feature = "cgemm")]
impl ConstNum for U2 { const VALUE: usize = 2; }
impl ConstNum for U4 { const VALUE: usize = 4; }
impl ConstNum for U8 { const VALUE: usize = 8; }
