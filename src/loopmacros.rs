// Copyright 2016 - 2018 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Unroll only in non-debug builds

#[cfg(not(debug_assertions))]
macro_rules! repeat {
    (1 $e:expr) => { $e; };
    (2 $e:expr) => { $e;$e; };
    (3 $e:expr) => { $e;$e; $e; };
    (4 $e:expr) => { $e;$e; $e;$e; };
    (5 $e:expr) => { $e;$e; $e;$e; $e; };
    (6 $e:expr) => { $e;$e; $e;$e; $e;$e; };
    (7 $e:expr) => { $e;$e; $e;$e; $e;$e; $e; };
    (8 $e:expr) => { $e;$e; $e;$e; $e;$e; $e;$e; };
}

macro_rules! loop4 {
    ($i:ident, $e:expr) => {{
        let $i = 0; $e;
        let $i = 1; $e;
        let $i = 2; $e;
        let $i = 3; $e;
    }}
}

#[cfg(debug_assertions)]
macro_rules! loop8 {
    ($i:ident, $e:expr) => {
        for $i in 0..8 { $e }
    }
}

#[cfg(not(debug_assertions))]
macro_rules! loop8 {
    ($i:ident, $e:expr) => {{
        let $i = 0; $e;
        let $i = 1; $e;
        let $i = 2; $e;
        let $i = 3; $e;
        let $i = 4; $e;
        let $i = 5; $e;
        let $i = 6; $e;
        let $i = 7; $e;
    }}
}

#[cfg(debug_assertions)]
macro_rules! loop16 {
    ($i:ident, $e:expr) => {
        for $i in 0..16 { $e }
    }
}

#[cfg(not(debug_assertions))]
macro_rules! loop16 {
    ($i:ident, $e:expr) => {{
        let $i = 0; $e;
        let $i = 1; $e;
        let $i = 2; $e;
        let $i = 3; $e;
        let $i = 4; $e;
        let $i = 5; $e;
        let $i = 6; $e;
        let $i = 7; $e;
        let $i = 8; $e;
        let $i = 9; $e;
        let $i = 10; $e;
        let $i = 11; $e;
        let $i = 12; $e;
        let $i = 13; $e;
        let $i = 14; $e;
        let $i = 15; $e;
    }}
}

#[cfg(debug_assertions)]
macro_rules! loop32 {
    ($i:ident, $e:expr) => {
        for $i in 0..32 { $e }
    }
}

#[cfg(not(debug_assertions))]
macro_rules! loop32 {
    ($i:ident, $e:expr) => {{
        let $i = 0; $e;
        let $i = 1; $e;
        let $i = 2; $e;
        let $i = 3; $e;
        let $i = 4; $e;
        let $i = 5; $e;
        let $i = 6; $e;
        let $i = 7; $e;
        let $i = 8; $e;
        let $i = 9; $e;
        let $i = 10; $e;
        let $i = 11; $e;
        let $i = 12; $e;
        let $i = 13; $e;
        let $i = 14; $e;
        let $i = 15; $e;
        let $i = 16; $e;
        let $i = 17; $e;
        let $i = 18; $e;
        let $i = 19; $e;
        let $i = 20; $e;
        let $i = 21; $e;
        let $i = 22; $e;
        let $i = 23; $e;
        let $i = 24; $e;
        let $i = 25; $e;
        let $i = 26; $e;
        let $i = 27; $e;
        let $i = 28; $e;
        let $i = 29; $e;
        let $i = 30; $e;
        let $i = 31; $e;
    }}
}

#[cfg(debug_assertions)]
macro_rules! unroll_by {
    ($by:tt => $ntimes:expr, $e:expr) => {
        for _ in 0..$ntimes { $e }
    }
}

#[cfg(not(debug_assertions))]
macro_rules! unroll_by {
    ($by:tt => $ntimes:expr, $e:expr) => {{
        let k = $ntimes;
        for _ in 0..k / $by {
            repeat!($by $e);
        }
        for _ in 0..k % $by {
            $e
        }
    }}
}

#[cfg(debug_assertions)]
macro_rules! unroll_by_with_last {
    ($by:tt => $ntimes:expr, $is_last:ident, $e:expr) => {{
        let k = $ntimes - 1;
        let $is_last = false;
        for _ in 0..k {
            $e;
        }
        let $is_last = true;
        #[allow(unused_assignments)]
        $e;
    }}
}

#[cfg(not(debug_assertions))]
macro_rules! unroll_by_with_last {
    ($by:tt => $ntimes:expr, $is_last:ident, $e:expr) => {{
        let k = $ntimes - 1;
        let $is_last = false;
        for _ in 0..k / $by {
            repeat!($by $e);
        }
        for _ in 0..k % $by {
            $e;
        }
        let $is_last = true;
        #[allow(unused_assignments)]
        $e;
    }}
}
