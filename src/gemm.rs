// Copyright 2016 bluss
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cmp::min;
use std::mem::size_of;

use util::range_chunk;
use util::round_up_to;

use kernel::GemmKernel;
use kernel::Element;
use sgemm_kernel;
use dgemm_kernel;
use pointer::PointerExt;

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
        c, rsc, csc)
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
        c, rsc, csc)
}

unsafe fn gemm_loop<K>(
    m: usize, k: usize, n: usize,
    alpha: K::Elem,
    a: *const K::Elem, rsa: isize, csa: isize,
    b: *const K::Elem, rsb: isize, csb: isize,
    beta: K::Elem,
    c: *mut K::Elem, rsc: isize, csc: isize)
    where K: GemmKernel
{
    let knc = K::nc();
    let kkc = K::kc();
    let kmc = K::mc();
    let mut apack = vec_uninit(K::kc() * K::mc(), K::kc(),
                               k, round_up_to(m, K::mr()));
    let mut bpack = vec_uninit(K::kc() * K::nc(), K::kc(),
                               min(k, K::kc()), round_up_to(n, K::nr()));
    let app = make_aligned_vec_ptr(K::align_to(), &mut apack);
    let bpp = make_aligned_vec_ptr(K::align_to(), &mut bpack);
    dprint!("pack len: {}", apack.len());

    // LOOP 5: split n into nc parts
    for (l5, nc) in range_chunk(n, knc) {
        dprint!("LOOP 5, {}, nc={}", l5, nc);
        let b = b.stride_offset(csb, knc * l5);
        let c = c.stride_offset(csc, knc * l5);

        // LOOP 4: split k in kc parts
        for (l4, kc) in range_chunk(k, kkc) {
            dprint!("LOOP 4, {}, kc={}", l4, kc);
            let b = b.stride_offset(rsb, kkc * l4);
            let a = a.stride_offset(csa, kkc * l4);
            debug!(for elt in &mut bpack { *elt = <_>::one(); });

            // Pack B -> B~
            pack::<K>(kc, nc, bpp, b, csb, rsb);

            // LOOP 3: split m into mc parts
            for (l3, mc) in range_chunk(m, kmc) {
                dprint!("LOOP 3, {}, mc={}", l3, mc);
                let a = a.stride_offset(rsa, kmc * l3);
                let c = c.stride_offset(rsc, kmc * l3);
                debug!(for elt in &mut apack { *elt = <_>::one(); });

                // Pack A -> A~
                pack::<K>(kc, mc, app, a, rsa, csa);

                // First time writing to C, use user's `beta`, else accumulate
                let betap = if l4 == 0 { beta } else { <_>::one() };

                // LOOP 2 and 1
                gemm_packed::<K>(nc, kc, mc,
                                 alpha,
                                 app, bpp,
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
    // make a mask buffer that fits 8 x 8 f32 and 8 x 4 f64 kernels and alignment
    assert!(mr * nr * size_of::<K::Elem>() <= 256 && K::align_to() <= 32);
    let mut mask_buf = [0u8; 256 + 31];
    let mask_ptr = align_ptr(32, mask_buf.as_mut_ptr()) as *mut K::Elem;

    // LOOP 2: through micropanels in packed `b`
    for (l2, nr_) in range_chunk(nc, nr) {
        dprint!("LOOP 2, {}, nr_={}", l2, nr_);
        let bpp = bpp.stride_offset(1, kc * nr * l2);
        let c = c.stride_offset(csc, nr * l2);

        // LOOP 1: through micropanels in packed `a` while `b` is constant
        for (l1, mr_) in range_chunk(mc, mr) {
            dprint!("LOOP 1, {}, mr_={}", l1, mr_);
            let app = app.stride_offset(1, kc * mr * l1);
            let c = c.stride_offset(rsc, mr * l1);

            // GEMM KERNEL
            // NOTE: For the rust kernels, it performs better to simply
            // always use the masked kernel function!
            if K::always_masked() || nr_ < nr || mr_ < mr {
                masked_kernel::<_, K>(kc, alpha, &*app, &*bpp,
                                      beta, &mut *c, rsc, csc,
                                      mr_, nr_, mask_ptr);
                continue;
            } else {
                K::kernel(kc, alpha, app, bpp, beta, c, rsc, csc);
            }
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
    v
}

/// Align a pointer into the vec. Will reallocate to fit & shift the pointer
/// forwards if needed. This invalidates any previous pointers into the v.
unsafe fn make_aligned_vec_ptr<U>(align_to: usize, v: &mut Vec<U>) -> *mut U {
    let mut ptr = v.as_mut_ptr();
    if align_to != 0 {
        if v.as_ptr() as usize % align_to != 0 {
            let cap = v.capacity();
            v.reserve_exact(cap + align_to / size_of::<U>() - 1);
            ptr = align_ptr(align_to, v.as_mut_ptr());
        }
    }
    ptr
}

/// offset the ptr forwards to align to a specific byte count
unsafe fn align_ptr<U>(align_to: usize, mut ptr: *mut U) -> *mut U {
    if align_to != 0 {
        let cur_align = ptr as usize % align_to;
        if cur_align != 0 {
            ptr = ptr.offset(((align_to - cur_align) / size_of::<U>()) as isize);
        }
    }
    ptr
}

/// Pack matrix into `pack`
///
/// + kc: length of the micropanel
/// + mc: number of rows/columns in the matrix to be packed
/// + rsa: row stride
/// + csa: column stride
/// + zero: zero element to pad with
unsafe fn pack<K>(kc: usize, mc: usize, pack: *mut K::Elem,
                  a: *const K::Elem, rsa: isize, csa: isize)
    where K: GemmKernel,
{
    let mr = K::mr();
    let zero = <_>::zero();
    debug_assert_eq!(K::mr(), K::nr());

    let mut pack = pack;
    for ir in 0..mc/mr {
        let row_offset = ir * mr;
        for j in 0..kc {
            for i in 0..mr {
                *pack = *a.stride_offset(rsa, i + row_offset)
                          .stride_offset(csa, j);
                pack.inc();
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
                    *pack = *a.stride_offset(rsa, i + row_offset)
                              .stride_offset(csa, j);
                } else {
                    *pack = zero;
                }
                pack.inc();
            }
        }
    }
}

/// Call the GEMM kernel with a "masked" output C.
/// 
/// Simply redirect the MR by NR kernel output to the passed
/// in `mask_buf`, and copy the non masked region to the real
/// C.
///
/// + rows: rows of kernel unmasked
/// + cols: cols of kernel unmasked
#[inline(never)]
unsafe fn masked_kernel<T, K>(k: usize, alpha: T,
                              a: *const T,
                              b: *const T,
                              beta: T,
                              c: *mut T, rsc: isize, csc: isize,
                              rows: usize, cols: usize,
                              mask_buf: *mut T)
    where K: GemmKernel<Elem=T>, T: Element,
{
    let mr = K::mr();
    let nr = K::nr();
    // use column major order for `mask_buf`
    K::kernel(k, T::one(), a, b, T::zero(), mask_buf, 1, mr as isize);
    let mut ab = mask_buf;
    for j in 0..nr {
        for i in 0..mr {
            if i < rows && j < cols {
                let cptr = c.offset(rsc * i as isize + csc * j as isize);
                if beta.is_zero() {
                    *cptr = T::zero(); // initialize C
                } else {
                    (*cptr).scale_by(beta);
                }
                (*cptr).scaled_add(alpha, *ab);
            }
            ab.inc();
        }
    }
}
