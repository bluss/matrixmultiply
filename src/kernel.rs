// Copyright 2016 - 2018 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// General matrix multiply kernel
pub trait GemmKernel {
    type ElemIn: Element;
    type ElemOut: Element;

    /// Number of kernel rows
    const MR: usize;

    /// Number of kernel columns
    const NR: usize;

    /// align inputs to this
    fn align_to() -> usize;

    /// Whether to always use the masked wrapper around the kernel.
    ///
    /// If masked, the kernel is always called with α=1, β=0
    fn always_masked() -> bool;

    fn nc() -> usize;
    fn kc() -> usize;
    fn mc() -> usize;

    /// Matrix multiplication kernel
    ///
    /// This does the matrix multiplication:
    ///
    /// C := alpha * A * B + beta * C
    ///
    /// + `k`: length of data in a, b
    /// + a, b are packed
    /// + c has general strides
    /// + rsc: row stride of c
    /// + csc: col stride of c
    /// + if `beta` is `0.`, then c does not need to be initialized
    unsafe fn kernel(
        k: usize,
        alpha: Self::ElemOut,
        a: *const Self::ElemIn,
        b: *const Self::ElemIn,
        beta: Self::ElemOut,
        c: *mut Self::ElemOut, rsc: isize, csc: isize);
}

pub trait Element : Copy {
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    fn scale_by(&mut self, x: Self);
    fn scaled_add(&mut self, alpha: Self, a: Self);
}

// impl Element for f32 {
//     fn zero() -> Self { 0. }
//     fn one() -> Self { 1. }
//     fn is_zero(&self) -> bool { *self == 0. }
//     fn scale_by(&mut self, x: Self) {
//         *self *= x;
//     }
//     fn scaled_add(&mut self, alpha: Self, a: Self) {
//         *self += alpha * a;
//     }
// }

// impl Element for f64 {
//     fn zero() -> Self { 0. }
//     fn one() -> Self { 1. }
//     fn is_zero(&self) -> bool { *self == 0. }
//     fn scale_by(&mut self, x: Self) {
//         *self *= x;
//     }
//     fn scaled_add(&mut self, alpha: Self, a: Self) {
//         *self += alpha * a;
//     }
// }

// impl Element for i32 {
//     fn zero() -> Self { 0 }
//     fn one() -> Self { 1 }
//     fn is_zero(&self) -> bool { *self == 0 }
//     fn scale_by(&mut self, x: Self) {
//         *self = self.wrapping_mul(x);
//     }
//     fn scaled_add(&mut self, alpha: Self, a: Self) {
//         *self = self.wrapping_add(alpha.wrapping_mul(a));
//     }
// }

// impl Element for i32 {
//     fn zero() -> Self { 0 }
//     fn one() -> Self { 1 }
//     fn is_zero(&self) -> bool { *self == 0 }
//     fn scale_by(&mut self, x: Self) {
//         *self = self.wrapping_mul(x);
//     }
//     fn scaled_add(&mut self, alpha: Self, a: Self) {
//         *self = self.wrapping_add(alpha.wrapping_mul(a));
//     }
// }

macro_rules! impl_element_f {
    ($($t:ty),+) => {
        $(
        impl Element for $t {
            fn zero() -> Self { 0.0 }
            fn one() -> Self { 1.0 }
            fn is_zero(&self) -> bool { *self == 0.0 }
            fn scale_by(&mut self, x: Self) {
                // TODO: Change the semantics
                *self *= x;
            }
                // TODO: Change the semantics
            fn scaled_add(&mut self, alpha: Self, a: Self) {
                *self += alpha * a;
            }
        }
        )+
};}

macro_rules! impl_element_i {
    ($($t:ty),+) => {
        $(
        impl Element for $t {
            fn zero() -> Self { 0 }
            fn one() -> Self { 1 }
            fn is_zero(&self) -> bool { *self == 0 }
            fn scale_by(&mut self, x: Self) {
                // TODO: Change the semantics
                *self = self.saturating_mul(x);
            }
                // TODO: Change the semantics
            fn scaled_add(&mut self, alpha: Self, a: Self) {
                *self = self.saturating_add(alpha.saturating_mul(a));
            }
        }
        )+
};}

impl_element_f!(f32, f64);
impl_element_i!(i8, i16, i32);
