extern crate itertools;
extern crate matrixmultiply;

use matrixmultiply::{
    sgemm,
    dgemm,
    i8gemm,
};

use itertools::Itertools;
use itertools::{
    cloned,
    enumerate,
    repeat_n,
};
use std::fmt::{Display, Debug};

trait GemmElement : Copy + Display + Debug + PartialEq {
    // TODO: Provide default associated types once the following RFCs are merged and implemented:
    // https://github.com/rust-lang/rfcs/pull/2532
    // https://github.com/rust-lang/rust/issues/29661
    // I.e., then we can do something like:
    // type Output = Self;
    //
    // XXX: Is it somehow possible to already provide default impls for the _out functions in terms
    // of the _in functions, where we assume that Input = Output?
    type Output: Copy + Display + Debug + PartialEq;

    fn zero_in() -> Self;
    fn one_in() -> Self;
    fn nan_in() -> Self;
    fn from_in(x: i64) -> Self;
    fn is_nan_in(Self) -> bool;

    fn zero_out() -> Self::Output;
    fn one_out() -> Self::Output;
    fn nan_out() -> Self::Output;
    fn from_out(x: i64) -> Self::Output;
    fn is_nan_out(Self::Output) -> bool;

    fn to_out(Self) -> Self::Output;
}

macro_rules! impl_gemm_element_f {
    ($($t:ty),+) => {
        $(
        impl GemmElement for $t {
            type Output = Self;

            fn zero_in() -> Self { 0. }
            fn one_in() -> Self { 1. }
            fn from_in(x: i64) -> Self { x as Self }
            fn nan_in() -> Self { 0./0. }
            fn is_nan_in(var: Self) -> bool { var.is_nan() }

            fn zero_out() -> Self::Output { 0. }
            fn one_out() -> Self::Output { 1. }
            fn from_out(x: i64) -> Self::Output { x as Self::Output }
            fn nan_out() -> Self { 0./0. }
            fn is_nan_out(var: Self::Output) -> bool { var.is_nan() }

            fn to_out(var: Self) -> Self::Output {
                var
            }
        }
        )+
    };
}

impl_gemm_element_f!(f32, f64);

impl GemmElement for i8 {
    type Output = i16;

    fn zero_in() -> Self { 0 }
    fn one_in() -> Self { 1 }
    fn from_in(x: i64) -> Self { x as Self }
    fn nan_in() -> Self { i8::min_value() } // hack
    fn is_nan_in(var: Self) -> bool { var == Self::nan_in() }

    fn zero_out() -> Self::Output { 0 }
    fn one_out() -> Self::Output { 1 }
    fn from_out(x: i64) -> Self::Output { x as Self::Output }
    fn nan_out() -> Self::Output { i16::min_value() } // hack
    fn is_nan_out(var: Self::Output) -> bool { var == Self::nan_out() }

    fn to_out(var: Self) -> Self::Output {
        var as Self::Output
    }
}

trait Gemm : Sized {
    type Output;

    unsafe fn gemm(
        m: usize, k: usize, n: usize,
        alpha: Self::Output,
        a: *const Self, rsa: isize, csa: isize,
        b: *const Self, rsb: isize, csb: isize,
        beta: Self::Output,
        c: *mut Self::Output, rsc: isize, csc: isize);
}

macro_rules! impl_gemm_f {
    ($(($t:ty, $f:ident)),+) => {
        $(
        impl Gemm for $t {
            type Output = Self;
            unsafe fn gemm(
                m: usize, k: usize, n: usize,
                alpha: Self,
                a: *const Self, rsa: isize, csa: isize,
                b: *const Self, rsb: isize, csb: isize,
                beta: Self,
                c: *mut Self, rsc: isize, csc: isize) {
                $f(
                    m, k, n,
                    alpha,
                    a, rsa, csa,
                    b, rsb, csb,
                    beta,
                    c, rsc, csc)
            }
        }
        )+
    };
}

impl_gemm_f!((f32, sgemm), (f64, dgemm));

impl Gemm for i8 {
    type Output = i16;
    unsafe fn gemm(
        m: usize, k: usize, n: usize,
        alpha: i16,
        a: *const Self, rsa: isize, csa: isize,
        b: *const Self, rsb: isize, csb: isize,
        beta: i16,
        c: *mut i16, rsc: isize, csc: isize) {
        i8gemm(
            m, k, n,
            alpha,
            a, rsa, csa,
            b, rsb, csb,
            beta,
            c, rsc, csc)
    }
}

#[test]
fn test_sgemm() {
    test_gemm::<f32, _>();
}
#[test]
fn test_dgemm() {
    test_gemm::<f64, _>();
}
#[test]
fn test_sgemm_strides() {
    test_gemm_strides::<f32, _>();
}
#[test]
fn test_dgemm_strides() {
    test_gemm_strides::<f64, _>();
}

#[test]
fn test_i8gemm_strides() {
    test_gemm_strides::<i8, _>();
}

fn test_gemm_strides<F, Tout>()
    where F: Gemm<Output=Tout> + GemmElement<Output=Tout>,
          Tout: Copy + Display + Debug + PartialEq
{
    for n in 0..10 {
        test_strides::<F, _>(n, n, n);
    }
    for n in (3..12).map(|x| x * 7) {
        test_strides::<F, _>(n, n, n);
    }

    test_strides::<F, _>(8, 12, 16);
    test_strides::<F, _>(8, 0, 10);
}

fn test_gemm<F, Tout>()
    where F: Gemm<Output=Tout> + GemmElement<Output=Tout>,
          Tout: Copy + Display + Debug + PartialEq
{
    test_mul_with_id::<F, _>(4, 4, true);
    test_mul_with_id::<F, _>(8, 8, true);
    test_mul_with_id::<F, _>(32, 32, false);
    test_mul_with_id::<F, _>(128, 128, false);
    test_mul_with_id::<F, _>(17, 128, false);
    for i in 0..12 {
        for j in 0..12 {
            test_mul_with_id::<F, _>(i, j, true);
        }
    }
    /*
    */
    test_mul_with_id::<F, _>(17, 257, false);
    test_mul_with_id::<F, _>(24, 512, false);
    for i in 0..10 {
        for j in 0..10 {
            test_mul_with_id::<F, _>(i * 4, j * 4, true);
        }
    }
    test_mul_with_id::<F, _>(266, 265, false);
    test_mul_id_with::<F, _>(4, 4, true);
    for i in 0..12 {
        for j in 0..12 {
            test_mul_id_with::<F, _>(i, j, true);
        }
    }
    test_mul_id_with::<F, _>(266, 265, false);
    test_scale::<F, Tout>(0, 4, 4, true);
    test_scale::<F, Tout>(4, 0, 4, true);
    test_scale::<F, Tout>(4, 4, 0, true);
    test_scale::<F, Tout>(4, 4, 4, true);
    test_scale::<F, Tout>(19, 20, 16, true);
    test_scale::<F, Tout>(150, 140, 128, false);

}

/// multiply a M x N matrix with an N x N id matrix
#[cfg(test)]
fn test_mul_with_id<F, Tout>(m: usize, n: usize, small: bool)
    where F: Gemm<Output=Tout> + GemmElement<Output=Tout>,
          Tout: Copy + Display + Debug + PartialEq
{
    let (m, k, n) = (m, n, n);
    let mut a = vec![F::zero_in(); m * k]; 
    let mut b = vec![F::zero_in(); k * n];
    let mut c = vec![F::zero_out(); m * n];
    println!("test matrix with id input M={}, N={}", m, n);

    for (i, elt) in a.iter_mut().enumerate() {
        *elt = F::from_in(i as i64);
    }
    for i in 0..k {
        b[i + i * k] = F::one_in();
    }

    unsafe {
        F::gemm(
            m, k, n,
            F::one_out(),
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            F::zero_out(),
            c.as_mut_ptr(), n as isize, 1,
            )
    }
    for (i, (x, y)) in a.iter().zip(&c).enumerate() {
        if F::to_out(*x) != *y {
            if k != 0 && n != 0 && small {
                for row in a.chunks(k) {
                    println!("{:?}", row);
                }
                for row in b.chunks(n) {
                    println!("{:?}", row);
                }
                for row in c.chunks(n) {
                    println!("{:?}", row);
                }
            }
            panic!("mismatch at index={}, x: {}, y: {} (matrix input M={}, N={})",
                   i, x, y, m, n);
        }
    }
    println!("passed matrix with id input M={}, N={}", m, n);
}

/// multiply a K x K id matrix with an K x N matrix
#[cfg(test)]
fn test_mul_id_with<F, Tout>(k: usize, n: usize, small: bool) 
    where F: Gemm<Output=Tout> + GemmElement<Output=Tout>,
          Tout: Copy + Display + Debug + PartialEq
{
    let (m, k, n) = (k, k, n);
    let mut a = vec![F::zero_in(); m * k]; 
    let mut b = vec![F::zero_in(); k * n];
    let mut c = vec![F::zero_out(); m * n];

    for i in 0..k {
        a[i + i * k] = F::one_in();
    }
    for (i, elt) in b.iter_mut().enumerate() {
        *elt = F::from_in(i as i64);
    }

    unsafe {
        F::gemm(
            m, k, n,
            F::one_out(),
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            F::zero_out(),
            c.as_mut_ptr(), n as isize, 1,
            )
    }
    for (i, (x, y)) in b.iter().zip(&c).enumerate() {
        if F::to_out(*x) != *y {
            if k != 0 && n != 0 && small {
                for row in a.chunks(k) {
                    println!("{:?}", row);
                }
                for row in b.chunks(n) {
                    println!("{:?}", row);
                }
                for row in c.chunks(n) {
                    println!("{:?}", row);
                }
            }
            panic!("mismatch at index={}, x: {}, y: {} (matrix input M={}, N={})",
                   i, x, y, m, n);
        }
    }
    println!("passed id with matrix input K={}, N={}", k, n);
}

#[cfg(test)]
fn test_scale<F, Tout>(m: usize, k: usize, n: usize, small: bool)
    where F: Gemm<Output=Tout> + GemmElement<Output=Tout>,
          Tout: Copy + Display + Debug + PartialEq
{
    let (m, k, n) = (m, k, n);
    let mut a = vec![F::zero_in(); m * k]; 
    let mut b = vec![F::zero_in(); k * n];
    let mut c1 = vec![F::one_out(); m * n];
    let mut c2 = vec![F::nan_out(); m * n];
    // init c2 with NaN to test the overwriting behavior when beta = 0.

    for (i, elt) in a.iter_mut().enumerate() {
        *elt = F::from_in(i as i64);
    }
    for (i, elt) in b.iter_mut().enumerate() {
        *elt = F::from_in(i as i64);
    }

    unsafe {
        // C1 = 3 A B
        F::gemm(
            m, k, n,
            F::from_out(3),
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            F::zero_out(),
            c1.as_mut_ptr(), n as isize, 1,
        );

        // C2 = A B 
        F::gemm(
            m, k, n,
            F::one_out(),
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            F::zero_out(),
            c2.as_mut_ptr(), n as isize, 1,
        );
        // C2 = A B + 2 C2
        F::gemm(
            m, k, n,
            F::one_out(),
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            F::from_out(2),
            c2.as_mut_ptr(), n as isize, 1,
        );
    }
    for (i, (x, y)) in c1.iter().zip(&c2).enumerate() {
        if x != y || F::is_nan_out(*x) || F::is_nan_out(*y) {
            if k != 0 && n != 0 && small {
                for row in a.chunks(k) {
                    println!("{:?}", row);
                }
                for row in b.chunks(n) {
                    println!("{:?}", row);
                }
                for row in c1.chunks(n) {
                    println!("{:?}", row);
                }
                for row in c2.chunks(n) {
                    println!("{:?}", row);
                }
            }
            panic!("mismatch at index={}, x: {}, y: {} (matrix input M={}, N={})",
                   i, x, y, m, n);
        }
    }
    println!("passed matrix with id input M={}, N={}", m, n);
}



//
// Custom stride tests
//

#[derive(Copy, Clone, Debug)]
enum Layout { C, F }
use self::Layout::*;

impl Layout {
    fn strides_scaled(self, m: usize, n: usize, scale: [usize; 2]) -> (isize, isize) {
        match self {
            C => ((n * scale[0] * scale[1]) as isize, scale[1] as isize),
            F => (scale[0] as isize, (m * scale[1] * scale[0]) as isize),
        }
    }
}

impl Default for Layout {
    fn default() -> Self { C }
}


#[cfg(test)]
fn test_strides<F, Tout>(m: usize, k: usize, n: usize)
    where F: Gemm<Output=Tout> + GemmElement<Output=Tout>,
          Tout: Copy + Display + Debug + PartialEq
{
    let (m, k, n) = (m, k, n);

    let stride_multipliers = vec![[1, 2], [2, 2], [2, 3], [1, 1], [2, 2], [4, 1], [3, 4]];
    let mut multipliers_iter = cloned(&stride_multipliers).cycle();

    let layout_species = [C, F];
    let layouts_iter = repeat_n(cloned(&layout_species), 4).multi_cartesian_product();

    for elt in layouts_iter {
        let layouts = [elt[0], elt[1], elt[2], elt[3]];
        let (m0, m1, m2, m3) = multipliers_iter.next_tuple().unwrap();
        test_strides_inner::<F, _>(m, k, n, [m0, m1, m2, m3], layouts);
    }
}


fn test_strides_inner<F, Tout>(m: usize, k: usize, n: usize,
                         stride_multipliers: [[usize; 2]; 4],
                         layouts: [Layout; 4])
    where F: Gemm<Output=Tout> + GemmElement<Output=Tout>,
          Tout: Copy + Display + Debug + PartialEq
{
    let (m, k, n) = (m, k, n);

    // stride multipliers
    let mstridea = stride_multipliers[0];
    let mstrideb = stride_multipliers[1];
    let mstridec = stride_multipliers[2];
    let mstridec2 = stride_multipliers[3];

    let mut a = vec![F::zero_in(); m * k * mstridea[0] * mstridea[1]]; 
    let mut b = vec![F::zero_in(); k * n * mstrideb[0] * mstrideb[1]];
    let mut c1 = vec![F::nan_out(); m * n * mstridec[0] * mstridec[1]];
    let mut c2 = vec![F::nan_out(); m * n * mstridec2[0] * mstridec2[1]];

    for (i, elt) in a.iter_mut().enumerate() {
        *elt = F::from_in(i as i64);
    }
    for (i, elt) in b.iter_mut().enumerate() {
        *elt = F::from_in(i as i64);
    }

    let la = layouts[0];
    let lb = layouts[1];
    let lc1 = layouts[2];
    let lc2 = layouts[3];
    let (rs_a, cs_a) = la.strides_scaled(m, k, mstridea);
    let (rs_b, cs_b) = lb.strides_scaled(k, n, mstrideb);
    let (rs_c1, cs_c1) = lc1.strides_scaled(m, n, mstridec);
    let (rs_c2, cs_c2) = lc2.strides_scaled(m, n, mstridec2);

    println!("Test matrix a : {} × {} layout: {:?} strides {}, {}", m, k, la, rs_a, cs_a);
    println!("Test matrix b : {} × {} layout: {:?} strides {}, {}", k, n, lb, rs_b, cs_b);
    println!("Test matrix c1: {} × {} layout: {:?} strides {}, {}", m, n, lc1, rs_c1, cs_c1);
    println!("Test matrix c2: {} × {} layout: {:?} strides {}, {}", m, n, lc2, rs_c2, cs_c2);

    macro_rules! c1 {
        ($i:expr, $j:expr) => (c1[(rs_c1 * $i as isize + cs_c1 * $j as isize) as usize]);
    }

    macro_rules! c2 {
        ($i:expr, $j:expr) => (c2[(rs_c2 * $i as isize + cs_c2 * $j as isize) as usize]);
    }

    unsafe {
        // C1 = A B
        F::gemm(
            m, k, n,
            F::from_out(1),
            a.as_ptr(), rs_a, cs_a,
            b.as_ptr(), rs_b, cs_b,
            F::zero_out(),
            c1.as_mut_ptr(), rs_c1, cs_c1,
        );

        // C1 += 2 A B
        F::gemm(
            m, k, n,
            F::from_out(2),
            a.as_ptr(), rs_a, cs_a,
            b.as_ptr(), rs_b, cs_b,
            F::from_out(1),
            c1.as_mut_ptr(), rs_c1, cs_c1,
        );

        // C2 = 3 A B 
        F::gemm(
            m, k, n,
            F::from_out(3),
            a.as_ptr(), rs_a, cs_a,
            b.as_ptr(), rs_b, cs_b,
            F::zero_out(),
            c2.as_mut_ptr(), rs_c2, cs_c2,
        );
    }
    for i in 0..m {
        for j in 0..n {
            let c1_elt = c1![i, j];
            let c2_elt = c2![i, j];
            assert_eq!(c1_elt, c2_elt,
                       "assertion failed for matrices, mismatch at {},{} \n\
                       a:: {:?}\n\
                       b:: {:?}\n\
                       c1: {:?}\n\
                       c2: {:?}\n",
                       i, j,
                       a, b,
                       c1, c2);
        }
    }
    // check we haven't overwritten the NaN values outside the passed output
    for (index, elt) in enumerate(&c1) {
        let i = index / rs_c1 as usize;
        let j = index / cs_c1 as usize;
        let irem = index % rs_c1 as usize;
        let jrem = index % cs_c1 as usize;
        if irem != 0 && jrem != 0 {
            assert!(F::is_nan_out(*elt),
                "Element at index={} ({}, {}) should be NaN, but was {}\n\
                c1: {:?}\n",
            index, i, j, elt,
            c1);
        }
    }
    println!("{}×{}×{} {:?} .. passed.", m, k, n, layouts);
}
