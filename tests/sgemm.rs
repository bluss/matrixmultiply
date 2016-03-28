extern crate matrixmultiply;
use matrixmultiply::sgemm as gemm;

#[test]
fn test_sgemm() {
    test_mul_with_id(4, 4, true);
    test_mul_with_id(8, 8, true);
    test_mul_with_id(32, 32, false);
    test_mul_with_id(128, 128, false);
    test_mul_with_id(17, 128, false);
    for i in 0..12 {
        for j in 0..12 {
            test_mul_with_id(i, j, true);
        }
    }
    /*
    */
    test_mul_with_id(17, 257, false);
    test_mul_with_id(24, 512, false);
    for i in 0..10 {
        for j in 0..10 {
            test_mul_with_id(i * 4, j * 4, true);
        }
    }
    test_mul_with_id(266, 265, false);
    test_mul_id_with(4, 4, true);
    for i in 0..12 {
        for j in 0..12 {
            test_mul_id_with(i, j, true);
        }
    }
    test_mul_id_with(266, 265, false);
}

/// multiply a M x N matrix with an N x N id matrix
#[cfg(test)]
fn test_mul_with_id(m: usize, n: usize, small: bool) {
    let (m, k, n) = (m, n, n);
    let mut a = vec![0.; m * k]; 
    let mut b = vec![0.; k * n];
    let mut c = vec![0.; m * n];

    for (i, elt) in a.iter_mut().enumerate() {
        *elt = i as f32;
    }
    for i in 0..k {
        b[i + i * k] = 1.;
    }

    unsafe {
        gemm(
            m, k, n,
            1.,
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            0.,
            c.as_mut_ptr(), n as isize, 1,
            )
    }
    for (i, (x, y)) in a.iter().zip(&c).enumerate() {
        if x != y {
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
fn test_mul_id_with(k: usize, n: usize, small: bool) {
    let (m, k, n) = (k, k, n);
    let mut a = vec![0.; m * k]; 
    let mut b = vec![0.; k * n];
    let mut c = vec![0.; m * n];

    for i in 0..k {
        a[i + i * k] = 1.;
    }
    for (i, elt) in b.iter_mut().enumerate() {
        *elt = i as f32;
    }

    unsafe {
        gemm(
            m, k, n,
            1.,
            a.as_ptr(), k as isize, 1,
            b.as_ptr(), n as isize, 1,
            0.,
            c.as_mut_ptr(), n as isize, 1,
            )
    }
    for (i, (x, y)) in b.iter().zip(&c).enumerate() {
        if x != y {
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
