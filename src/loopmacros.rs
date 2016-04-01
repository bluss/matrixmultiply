// Copyright 2016 bluss
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! loop4 {
    ($i:ident, $e:expr) => {{
        let $i = 0; $e;
        let $i = 1; $e;
        let $i = 2; $e;
        let $i = 3; $e;
    }}
}

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

macro_rules! unroll_by_4 {
    ($ntimes:expr, $e:expr) => {{
        let k = $ntimes;
        for _ in 0..k / 4 {
            $e;$e; $e;$e;
        }
        for _ in 0..k % 4 {
            $e
        }
    }}
}

macro_rules! unroll_by_8 {
    ($ntimes:expr, $e:expr) => {{
        let k = $ntimes;
        for _ in 0..k / 8 {
            $e;$e; $e;$e;
            $e;$e; $e;$e;
        }
        for _ in 0..k % 8 {
            $e
        }
    }}
}

macro_rules! shuf {
    ($v:expr, $i:expr, $j:expr, $k:expr, $m:expr) => (
        [$v[$i], $v[$j], $v[$k], $v[$m]]
    );
}

