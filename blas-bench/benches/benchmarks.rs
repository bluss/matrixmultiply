extern crate blas_bench;
extern crate matrixmultiply;
pub use matrixmultiply::sgemm;
pub use matrixmultiply::dgemm;

#[macro_use]
extern crate bencher;
extern crate blas;

use std::os::raw::c_int;


#[allow(non_camel_case_types)]
type blas_index = c_int; // blas index type


// Compute GFlop/s
// by flop / s = 2 M N K / time


benchmark_main!(blas_mat_mul_f32, blas_mat_mul_f64);

macro_rules! blas_mat_mul {
    ($modname:ident, $gemm:ident, $(($name:ident, $m:expr, $n:expr, $k:expr))+) => {
        mod $modname {
            use bencher::{Bencher};
            use super::blas_index;
            $(
            pub fn $name(bench: &mut Bencher)
            {
                let a = vec![0.; $m * $n]; 
                let b = vec![0.; $n * $k];
                let mut c = vec![0.; $m * $k];
                bench.iter(|| {
                    unsafe {

                            blas::$gemm(
                            b'N',
                            b'N',
                            $m as blas_index, // m, rows of Op(a)
                            $n as blas_index, // n, cols of Op(b)
                            $k as blas_index, // k, cols of Op(a)
                            1.,
                            &a,
                            $n, // lda
                            &b,
                            $k, // ldb
                            0.,         // beta
                            &mut c,
                            $k, // ldc
                            );
                    }
                });
            }
            )+
        }
        benchmark_group!{ $modname, $($modname::$name),+ }
    };
}

blas_mat_mul!{blas_mat_mul_f32, sgemm,
    (m004, 4, 4, 4)
    (m006, 6, 6, 6)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
   (m127, 127, 127, 127)
}

blas_mat_mul!{blas_mat_mul_f64, dgemm,
    (m004, 4, 4, 4)
    (m006, 6, 6, 6)
    (m008, 8, 8, 8)
    (m012, 12, 12, 12)
    (m016, 16, 16, 16)
    (m032, 32, 32, 32)
    (m064, 64, 64, 64)
   (m127, 127, 127, 127)
}
