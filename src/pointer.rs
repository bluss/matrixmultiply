// Copyright 2016 bluss
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// Extension methods for raw pointers
pub trait PointerExt : Copy {
    unsafe fn offset(self, i: isize) -> Self;

    /// Increment by 1
    #[inline(always)]
    unsafe fn inc(&mut self) {
        *self = self.offset(1);
    }

    /// Offset by `s` multiplied by `index`.
    #[inline(always)]
    unsafe fn stride_offset(self, s: isize, index: usize) -> Self {
        self.offset(s * index as isize)
    }
}

impl<T> PointerExt for *const T {
    #[inline(always)]
    unsafe fn offset(self, i: isize) -> Self {
        self.offset(i)
    }
}

impl<T> PointerExt for *mut T {
    #[inline(always)]
    unsafe fn offset(self, i: isize) -> Self {
        self.offset(i)
    }
}
