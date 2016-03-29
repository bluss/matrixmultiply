
use matrixmultiply_asm::*;
use kernel::GemmKernel;
use archparam;

pub enum Gemm { }

pub type T = f64;

impl GemmKernel for Gemm {
    type Elem = T;

    #[inline(always)]
    fn mr() -> usize { D_MR }
    #[inline(always)]
    fn nr() -> usize { D_NR }

    #[inline(always)]
    fn align_to() -> usize { D_ALIGN }

    #[inline(always)]
    fn always_masked() -> bool { false }

    #[inline(always)]
    fn nc() -> usize { archparam::D_NC }
    #[inline(always)]
    fn kc() -> usize { archparam::D_KC }
    #[inline(always)]
    fn mc() -> usize { archparam::D_MC }

    #[inline(always)]
    unsafe fn kernel(
        k: usize,
        alpha: T,
        a: *const T,
        b: *const T,
        beta: T,
        c: *mut T, rsc: isize, csc: isize) {
        dgemm_kernel(k, alpha, a, b, beta, c, rsc, csc)
    }
}

/// 4x4 matrix multiplication kernel for f64
///
/// This does the matrix multiplication:
///
/// C ← α A B + β C
///
/// + k: length of data in a, b
/// + a, b are packed
/// + c has general strides
/// + rsc: row stride of c
/// + csc: col stride of c
/// + if `beta` is 0, then c does not need to be initialized
#[inline(always)]
pub unsafe fn dgemm_kernel(k: usize, alpha: T, a: *const T, b: *const T,
                           beta: T, c: *mut T, rsc: isize, csc: isize)
{
    debug_assert_eq!(a as usize % 32, 0);
    debug_assert_eq!(b as usize % 32, 0);
    debug_assert_eq!(c as usize % 32, 0);
    dgemm_asm(k as isize,
              &alpha,
              a,
              b,
              &beta,
              c,
              rsc, csc);
}
