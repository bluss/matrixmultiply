#![feature(test)]
extern crate matrixmultiply;
pub use matrixmultiply::sgemm;
pub use matrixmultiply::dgemm;

extern crate test;
use test::Bencher;

#[bench]
fn mat_mul_128(bench: &mut Bencher) {
    let (m, k, n) = (128, 128, 128);
    let mut a = vec![0.; m * k]; 
    let mut b = vec![0.; k * n];
    let mut c = vec![0.; m * n];

    for (i, elt) in a.iter_mut().enumerate() {
        *elt = i as f32;
    }
    for i in 0..n {
        b[i + i * n] = 1.;
    }

    bench.iter(|| {
    unsafe {
        sgemm(
            m, k, n,
            1.,
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            0.,
            c.as_mut_ptr(), n as isize, 1,
            )
    }
    });
}

// Compute GFlop/s
// by flop / s = 2 M N K / time

#[bench]
fn mat_mul_512(bench: &mut Bencher) {
    let (m, k, n) = (512, 512, 512);
    let mut a = vec![0.; m * k]; 
    let mut b = vec![0.; k * n];
    let mut c = vec![0.; m * n];

    for (i, elt) in a.iter_mut().enumerate() {
        *elt = i as f32;
    }
    for i in 0..n {
        b[i + i * n] = 1.;
    }

    bench.iter(|| {
    unsafe {
        sgemm(
            m, k, n,
            1.,
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            0.,
            c.as_mut_ptr(), n as isize, 1,
            )
    }
    });
}

macro_rules! mat_mul {
    ($modname:ident, $gemm:ident, $(($name:ident, $m:expr, $n:expr, $k:expr))+) => {
        mod $modname {
            use test::{Bencher};
            use $gemm;
            $(
            #[bench]
            fn $name(bench: &mut Bencher)
            {
                let a = vec![0.; $m * $n]; 
                let b = vec![0.; $n * $k];
                let mut c = vec![0.; $m * $k];
                bench.iter(|| {
                    unsafe {
                        $gemm(
                            $m, $n, $k,
                            1.,
                            a.as_ptr(), $n, 1,
                            b.as_ptr(), $k, 1,
                            0.,
                            c.as_mut_ptr(), $k, 1,
                            )
                    }
                });
            }
            )+
        }
    };
}

mat_mul!{mat_mul_f32, sgemm,
    (m004, 4, 4, 4)
    (m007, 7, 7, 7)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
    (m127, 127, 127, 127)
    (m256, 256, 256, 256)
    (m512, 512, 512, 512)
    (mix16x4, 32, 4, 32)
    (mix32x2, 32, 2, 32)
    (mix97, 97, 97, 125)
    (mix128x10000x128, 128, 10000, 128)
}

mat_mul!{mat_mul_f64, dgemm,
    (m004, 4, 4, 4)
    (m007, 7, 7, 7)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
    (m127, 127, 127, 127)
    (m256, 256, 256, 256)
    (m512, 512, 512, 512)
    (mix16x4, 32, 4, 32)
    (mix32x2, 32, 2, 32)
    (mix97, 97, 97, 125)
    (mix128x10000x128, 128, 10000, 128)
}
