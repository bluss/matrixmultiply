// Copyright 2021 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// kernel fallback impl macro
// Depends on a couple of macro and function defitions to be in scope - loop_m/_n, at, etc.
macro_rules! kernel_fallback_impl_complex {
    ([$($attr:meta)*] $name:ident, $elem_ty:ty, $real_ty:ty, $mr:expr, $nr:expr, $unroll:tt) => {
    $(#[$attr])*
    unsafe fn $name(k: usize, alpha: $elem_ty, a: *const $elem_ty, b: *const $elem_ty,
                    beta: $elem_ty, c: *mut $elem_ty, rsc: isize, csc: isize)
    {
        const MR: usize = $mr;
        const NR: usize = $nr;

        debug_assert_eq!(beta, <$elem_ty>::zero(), "Beta must be 0 or is not masked");

        let mut pp  = [<$real_ty>::zero(); MR];
        let mut qq  = [<$real_ty>::zero(); MR];
        let mut rr  = [<$real_ty>::zero(); NR];
        let mut ss  = [<$real_ty>::zero(); NR];

        let mut ab: [[$elem_ty; NR]; MR] = [[<$elem_ty>::zero(); NR]; MR];
        let mut a = a;
        let mut b = b;

        // Compute A B into ab[i][j]
        unroll_by!($unroll => k, {
            // We set
            //
            // P + Q i = A
            // R + S i = B
            loop_m!(i, pp[i] = at(a, i)[0]);
            loop_m!(i, qq[i] = at(a, i)[1]);
            loop_n!(i, rr[i] = at(b, i)[0]);
            loop_n!(i, ss[i] = at(b, i)[1]);

            loop_m!(i, loop_n!(j, {
                ab[i][j][0] += pp[i] * rr[j];
            }));
            loop_m!(i, loop_n!(j, {
                ab[i][j][1] += pp[i] * ss[j];
            }));
            loop_m!(i, loop_n!(j, {
                ab[i][j][0] -= qq[i] * ss[j];
            }));
            loop_m!(i, loop_n!(j, {
                ab[i][j][1] += qq[i] * rr[j];
            }));

            a = a.offset(MR as isize);
            b = b.offset(NR as isize);
        });

        macro_rules! c {
            ($i:expr, $j:expr) => (c.offset(rsc * $i as isize + csc * $j as isize));
        }

        // set C = α A B
        loop_n!(j, loop_m!(i, *c![i, j] = mul(alpha, ab[i][j])));
    }
    };
}
