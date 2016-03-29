// Copyright 2016 bluss
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// for debugging -- like println during debugging
#[cfg(debug_assertions)]
macro_rules! dprint {
    ($($t:tt)*) => {
        println!($($t)*)
    }
}

#[cfg(not(debug_assertions))]
macro_rules! dprint {
    ($($t:tt)*) => {
    }
}

#[cfg(debug_assertions)]
macro_rules! debug {
    ($e:expr) => {
        $e;
    }
}

#[cfg(not(debug_assertions))]
macro_rules! debug {
    ($e:expr) => {
    }
}

