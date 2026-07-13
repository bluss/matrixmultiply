/// If the microkernel tile is a square, mr == nr,
/// then we can change the problem from C = A B to C^T = B^T A^T
///
/// Whenever rsc == 1, we switch the orientation around.
/// *Any contiguous positive stride* C will exit this function with rsc arbitrary, csc == 1.
///
/// Input row stride (rs), column stride (cs) and outcomes
/// rsc     csc     explanation
/// M       1       no change => (M, 1)
/// 1       Z       transpose => (Z, 1)
/// M1      M2      no change => (M1, M2)
///
/// Where M signifies number M != 1 and Z any number (pos, neg, or zero)
///
/// + `mr`, `nr` kernel size; must be square
/// + `a` pointer to packing buffer
/// + `b` pointer to packing buffer
/// + `rsc` row stride of c
/// + `csc` colum stride of c
#[inline(always)]
#[allow(unused)]
pub(crate) fn preferential_transpose<T>(mr: usize, nr: usize, a: *const T, b: *const T, rsc: isize, csc: isize)
    -> (*const T, *const T, isize, isize)
{
    debug_assert_eq!(mr, nr, "Transpose requires that MR == NR");
    // Compute C in whichever orientation makes the output columns contiguous
    let prefer_row_major_c = rsc != 1;
    if prefer_row_major_c { (a, b, rsc, csc) } else { (b, a, csc, rsc) }
}

/// Dereference pointer at offset `i`
#[inline(always)]
pub(crate) unsafe fn at<T: Copy>(ptr: *const T, i: usize) -> T {
    *ptr.add(i)
}

