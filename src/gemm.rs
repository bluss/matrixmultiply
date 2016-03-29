
use std::cmp::min;

use util::range_chunk;
use util::round_up_to;

use kernel::GemmKernel;
use sgemm_kernel;
use dgemm_kernel;

/// General matrix multiplication (f32)
///
/// C ← α A B + β C
///
/// + m, k, n: dimensions
/// + a, b, c: pointer to the first element in the matrix
/// + A: m by k matrix
/// + B: k by n matrix
/// + C: m by n matrix
/// + rs<em>x</em>: row stride of *x*
/// + cs<em>x</em>: col stride of *x*
///
/// Strides for A and B may be arbitrary. Strides for C must not result in
/// elements that alias each other, for example they can not be zero.
///
/// If β is zero, then C does not need to be initialized.
pub unsafe fn sgemm(
    m: usize, k: usize, n: usize,
    alpha: f32,
    a: *const f32, rsa: isize, csa: isize,
    b: *const f32, rsb: isize, csb: isize,
    beta: f32,
    c: *mut f32, rsc: isize, csc: isize)
{
    gemm_loop::<sgemm_kernel::Gemm>(
        m, k, n,
        alpha,
        a, rsa, csa,
        b, rsb, csb,
        beta,
        c, rsc, csc, 0., 1.)
}

/// General matrix multiplication (f64)
///
/// C ← α A B + β C
///
/// + m, k, n: dimensions
/// + a, b, c: pointer to the first element in the matrix
/// + A: m by k matrix
/// + B: k by n matrix
/// + C: m by n matrix
/// + rs<em>x</em>: row stride of *x*
/// + cs<em>x</em>: col stride of *x*
///
/// Strides for A and B may be arbitrary. Strides for C must not result in
/// elements that alias each other, for example they can not be zero.
///
/// If β is zero, then C does not need to be initialized.
pub unsafe fn dgemm(
    m: usize, k: usize, n: usize,
    alpha: f64,
    a: *const f64, rsa: isize, csa: isize,
    b: *const f64, rsb: isize, csb: isize,
    beta: f64,
    c: *mut f64, rsc: isize, csc: isize)
{
    gemm_loop::<dgemm_kernel::Gemm>(
        m, k, n,
        alpha,
        a, rsa, csa,
        b, rsb, csb,
        beta,
        c, rsc, csc, 0., 1.)
}

unsafe fn gemm_loop<K>(
    m: usize, k: usize, n: usize,
    alpha: K::Elem,
    a: *const K::Elem, rsa: isize, csa: isize,
    b: *const K::Elem, rsb: isize, csb: isize,
    beta: K::Elem,
    c: *mut K::Elem, rsc: isize, csc: isize,
    zero: K::Elem, one: K::Elem)
    where K: GemmKernel
{
    let knc = K::nc();
    let kkc = K::kc();
    let kmc = K::mc();
    let mut apack = vec_uninit(K::kc() * K::mc(), K::kc(),
                               k, round_up_to(m, K::mr()));
    let mut bpack = vec_uninit(K::kc() * K::nc(), K::kc(),
                                   min(k, K::kc()), round_up_to(n, K::nr()));
    dprint!("pack len: {}", apack.len());

    // LOOP 5: split n into nc parts
    for (l5, nc) in range_chunk(n, knc) {
        let b = b.offset(csb * knc as isize * l5 as isize);
        let c = c.offset(csc * knc as isize * l5 as isize);
        dprint!("LOOP 5, {}, nc={}", l5, nc);

        // LOOP 4: split k in kc parts
        for (l4, kc) in range_chunk(k, kkc) {
            dprint!("LOOP 4, {}, kc={}", l4, kc);
            let b = b.offset(rsb * kkc as isize * l4 as isize);
            let a = a.offset(csa * kkc as isize * l4 as isize);
            debug!(for elt in &mut bpack { *elt = one; });

            // Pack B -> B~
            pack::<K>(kc, nc, bpack.as_mut_ptr(), b, csb, rsb, zero);

            // LOOP 3: split m into mc parts
            for (l3, mc) in range_chunk(m, kmc) {
                dprint!("LOOP 3, {}, mc={}", l3, mc);
                let a = a.offset(rsa * kmc as isize * l3 as isize);
                let c = c.offset(rsc * kmc as isize * l3 as isize);
                debug!(for elt in &mut apack { *elt = one; });

                // Pack A -> A~
                pack::<K>(kc, mc, apack.as_mut_ptr(), a, rsa, csa, zero);

                // First time writing to C, use user's `beta`, else accumulate
                let betap = if l4 == 0 { beta } else { one };

                // LOOP 2 and 1
                gemm_packed::<K>(nc, kc, mc,
                                 alpha,
                                 apack.as_ptr(), bpack.as_ptr(),
                                 betap,
                                 c, rsc, csc);
            }
        }
    }
}

/// Loops 1 and 2 around the µ-kernel
///
/// + app: packed A (A~)
/// + bpp: packed B (B~)
/// + nc: columns of packed B
/// + kc: columns of packed A / rows of packed B
/// + mc: rows of packed A
unsafe fn gemm_packed<K>(nc: usize, kc: usize, mc: usize,
                         alpha: K::Elem,
                         app: *const K::Elem, bpp: *const K::Elem,
                         beta: K::Elem,
                         c: *mut K::Elem, rsc: isize, csc: isize)
    where K: GemmKernel,
{
    let mr = K::mr();
    let nr = K::nr();

    for (l2, nr_) in range_chunk(nc, nr) {
        // LOOP 2: loop through panels for packed `b`
        let col_offset = (l2 * nr) as isize;
        let bpp = bpp.offset(col_offset * kc as isize);
        let c = c.offset(col_offset * csc);

        for (l1, mr_) in range_chunk(mc, mr) {
            // LOOP 1: through micropanels in packed `a` while `b` is constant
            let row_offset = (l1 * mr) as isize;
            let app = app.offset(row_offset * kc as isize);
            let c = c.offset(row_offset * rsc);

            if nr_ < nr || mr_ < mr {
                K::kernel_masked(kc, alpha, &*app, &*bpp,
                                 beta, &mut *c, rsc, csc,
                                 mr_, nr_);
                continue;
            }

            // GEMM KERNEL
            K::kernel(kc, alpha, app, bpp, beta, c, rsc, csc);
        }
    }
}

/// Allocate a vector of uninitialized data.
/// Round size up to multiples of KC.
unsafe fn vec_uninit<U>(maximal: usize, kc: usize, k: usize, nn: usize) -> Vec<U> {
    let kk = min(k, kc);
    // round up k, n to multiples of mr, nr
    // round up to multiple of kc
    let nelem = min(maximal, round_up_to(kk * nn, kc));
    let mut v = Vec::with_capacity(nelem);
    v.set_len(nelem);
    dprint!("vec_uninit: len={}", nelem);
    v
}

/// Pack matrix into `pack`
///
/// + kc: length of the micropanel
/// + mc: number of rows/columns in the matrix to be packed
/// + rsa: row stride
/// + csa: column stride
/// + zero: zero element to pad with
unsafe fn pack<K>(kc: usize, mc: usize, pack: *mut K::Elem,
                  a: *const K::Elem, rsa: isize, csa: isize,
                  zero: K::Elem)
    where K: GemmKernel,
{
    let mr = K::mr();
    debug_assert_eq!(K::mr(), K::nr());

    let mut pack = pack;
    for ir in 0..mc/mr {
        let row_offset = ir * mr;
        for j in 0..kc {
            for i in 0..mr {
                let i = i + row_offset;
                *pack = *a.offset(i as isize * rsa + j as isize * csa);
                pack = pack.offset(1);
            }
        }
    }

    // Pad with zeros to multiple of kernel size (uneven mc)
    let rest = mc % mr;
    if rest > 0 {
        let row_offset = (mc/mr) * mr;
        for j in 0..kc {
            for i in 0..mr {
                if i < rest {
                    let i = i + row_offset;
                    *pack = *a.offset(i as isize * rsa + j as isize * csa);
                } else {
                    *pack = zero;
                }
                pack = pack.offset(1);
            }
        }
    }
}


