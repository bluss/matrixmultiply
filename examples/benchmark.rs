#![cfg(feature="std")]

//! Run this executable to benchmark sgemm and dgemm for arbitrary size matrices
//! See --help for usage examples.  Remember to run in release mode.

extern crate std;
extern crate itertools;
extern crate matrixmultiply;

use std::time::Instant;
use std::fmt::{Display, Debug};
use itertools::zip;

use matrixmultiply::{sgemm, dgemm};

trait Float : Copy + Display + Debug + PartialEq {
    fn zero() -> Self;
    fn one() -> Self;
    fn from(x: i64) -> Self;
    fn nan() -> Self;
    fn is_nan(self) -> bool;
}

impl Float for f32 {
    fn zero() -> Self { 0. }
    fn one() -> Self { 1. }
    fn from(x: i64) -> Self { x as Self }
    fn nan() -> Self { 0./0. }
    fn is_nan(self) -> bool { self.is_nan() }
}

impl Float for f64 {
    fn zero() -> Self { 0. }
    fn one() -> Self { 1. }
    fn from(x: i64) -> Self { x as Self }
    fn nan() -> Self { 0./0. }
    fn is_nan(self) -> bool { self.is_nan() }
}


trait Gemm : Sized {
    unsafe fn gemm(
        m: usize, k: usize, n: usize,
        alpha: Self,
        a: *const Self, rsa: isize, csa: isize,
        b: *const Self, rsb: isize, csb: isize,
        beta: Self,
        c: *mut Self, rsc: isize, csc: isize);
}

impl Gemm for f32 {
    unsafe fn gemm(
        m: usize, k: usize, n: usize,
        alpha: Self,
        a: *const Self, rsa: isize, csa: isize,
        b: *const Self, rsb: isize, csb: isize,
        beta: Self,
        c: *mut Self, rsc: isize, csc: isize) {
        sgemm(
            m, k, n,
            alpha,
            a, rsa, csa,
            b, rsb, csb,
            beta,
            c, rsc, csc)
    }
}

impl Gemm for f64 {
    unsafe fn gemm(
        m: usize, k: usize, n: usize,
        alpha: Self,
        a: *const Self, rsa: isize, csa: isize,
        b: *const Self, rsb: isize, csb: isize,
        beta: Self,
        c: *mut Self, rsc: isize, csc: isize) {
        dgemm(
            m, k, n,
            alpha,
            a, rsa, csa,
            b, rsb, csb,
            beta,
            c, rsc, csc)
    }
}

#[derive(Debug, Clone, Default)]
struct Options {
    m: usize,
    k: usize,
    n: usize,
    layout: [Layout; 3],
    use_f32: bool,
}

fn parse_args() -> Result<Options, String> {
    let mut opts = Options::default();
    let mut args = std::env::args();
    let _ = args.next();
    opts.m = args.next().ok_or("Expected argument".to_string())?
        .parse::<_>().unwrap();
    opts.k = args.next().ok_or("Expected argument".to_string())?
        .parse::<_>().unwrap();
    opts.n = args.next().ok_or("Expected argument".to_string())?
        .parse::<_>().unwrap();
    if let Some(arg) = args.next() {
        if arg == "f32" {
            opts.use_f32 = true;
        } else if arg == "f64" {
            //
        } else {
            Err(format!("Unknown argument {}", arg))?;
        }
        // layout
        if let Some(arg) = args.next() {
            if arg.len() != 3 || !arg.chars().all(|c| c == 'c' || c == 'f') {
                Err(format!("Unknown argument {}", arg))?;
            }
            for (elt, layout_arg) in zip(&mut opts.layout[..], arg.chars())
            {
                *elt = if layout_arg == 'c' { Layout::C } else { Layout::F };
            }
        }
    }
    Ok(opts)
}

fn main() -> Result<(), String> {
    #[cfg(debug_assertions)]
    eprintln!("Warning: running benchmark with debug assertions");

    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Usage: <command> m-size k-size n-size float-type layout-types");
            eprintln!("Example: <command> 1000 1000 1000 f64 fcf");
            eprintln!();
            return Err(format!("Error parsing arguments: {}", e));
        }
    };

    if opts.use_f32 {
        test_matrix::<f32>(opts.m, opts.k, opts.n, opts.layout)
    } else {
        test_matrix::<f64>(opts.m, opts.k, opts.n, opts.layout)
    }
    Ok(())
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


fn test_matrix<F>(m: usize, k: usize, n: usize, layouts: [Layout; 3])
    where F: Gemm + Float
{
    let (m, k, n) = (m, k, n);

    // stride multipliers
    let stride_multipliers = vec![[1, 1], [1, 1], [1, 1]];
    let mstridea = stride_multipliers[0];
    let mstrideb = stride_multipliers[1];
    let mstridec = stride_multipliers[2];

    let mut a = vec![F::zero(); m * k * mstridea[0] * mstridea[1]]; 
    let mut b = vec![F::zero(); k * n * mstrideb[0] * mstrideb[1]];
    let mut c1 = vec![F::zero(); m * n * mstridec[0] * mstridec[1]];

    for (i, elt) in a.iter_mut().enumerate() {
        *elt = F::from(i as i64);
    }

    for (i, elt) in b.iter_mut().enumerate() {
        *elt = F::from(i as i64);
    }

    let la = layouts[0];
    let lb = layouts[1];
    let lc1 = layouts[2];
    let (rs_a, cs_a) = la.strides_scaled(m, k, mstridea);
    let (rs_b, cs_b) = lb.strides_scaled(k, n, mstrideb);
    let (rs_c1, cs_c1) = lc1.strides_scaled(m, n, mstridec);

    println!("Test matrix a : {} × {} layout: {:?} strides {}, {}", m, k, la, rs_a, cs_a);
    println!("Test matrix b : {} × {} layout: {:?} strides {}, {}", k, n, lb, rs_b, cs_b);
    println!("Test matrix c : {} × {} layout: {:?} strides {}, {}", m, n, lc1, rs_c1, cs_c1);

    let statistics = measure(10, || {
        unsafe {
            // EXAMPLE: Compute the same result in C1 and C2 in two different ways.
            // We only use whole integer values in the low range of floats here,
            // so we have no loss of precision.

            // C1 = A B
            F::gemm(
                m, k, n,
                F::from(1),
                a.as_ptr(), rs_a, cs_a,
                b.as_ptr(), rs_b, cs_b,
                F::zero(),
                c1.as_mut_ptr(), rs_c1, cs_c1,
            );
        }
    });
             //std::any::type_name::<F>(), fmt_thousands_sep(elapsed_ns, ' '));
    //println!("{:#?}", statistics);
    print!("{}×{}×{} {:?} {} .. {} ns", m, k, n, layouts, std::any::type_name::<F>(),
           fmt_thousands_sep(statistics.average, " "));
    print!(" [minimum: {} ns .. median {} ns .. sample count {}]", 
           fmt_thousands_sep(statistics.minimum, " "),
           fmt_thousands_sep(statistics.median, " "),
           statistics.samples.len());
    // by flop / s = 2 M N K / time
    let gflop = (2 * m * n * k) as f64 / statistics.average as f64;
    print!("    {:.2} Gflop/s", gflop);
    println!();
}

#[derive(Default, Debug)]
struct Statistics {
    samples: Vec<u64>,
    samples_sorted: Vec<u64>,
    average: u64,
    median: u64,
    minimum: u64,
}

const OUTLIER_HIGH_PCT: usize = 25;
//const OUTLIER_LOW_PCT: usize = 10;

fn measure(max_samples: usize, mut function: impl FnMut()) -> Statistics {
    let mut statistics = Statistics::default();
    statistics.samples.reserve(max_samples);
    let mut goal_samples = max_samples;
    let start_batch = Instant::now();
    let mut print_each = false;
    while statistics.samples.len() < goal_samples {
        for _ in 0..goal_samples {
            let start = Instant::now();
            function();
            let dur = start.elapsed();
            let elapsed_ns = dur.as_secs() * 1_000_000_000 + dur.subsec_nanos() as u64;
            statistics.samples.push(elapsed_ns);
            print_each |= dur.as_secs() >= 1;
            if print_each {
                println!("    {}", fmt_thousands_sep(elapsed_ns, " "));
            }
        }
        let batch_dur = start_batch.elapsed();
        if batch_dur.as_millis() < 1000 {
            goal_samples *= 5;
        }
    }
    let nsamples = statistics.samples.len();
    let nsamples_winnow = nsamples - (nsamples * OUTLIER_HIGH_PCT) / 100;
    statistics.samples_sorted = statistics.samples.clone();
    // sort low to high
    statistics.samples_sorted.sort_unstable();
    statistics.samples_sorted.truncate(nsamples_winnow);
    statistics.average = (statistics.samples_sorted.iter().sum::<u64>() as f64 /
                          (nsamples_winnow as f64)) as u64;
    statistics.minimum = statistics.samples_sorted[0];
    statistics.median = statistics.samples_sorted[nsamples_winnow / 2];
    statistics
}

// Format a number with thousands separators
fn fmt_thousands_sep(mut n: u64, sep: &str) -> String {
    use std::fmt::Write;
    let mut output = String::new();
    let mut trailing = false;
    for &pow in &[12, 9, 6, 3, 0] {
        let base = 10_u64.pow(pow);
        if pow == 0 || trailing || n / base != 0 {
            if !trailing {
                output.write_fmt(format_args!("{}", n / base)).unwrap();
            } else {
                output.write_fmt(format_args!("{:03}", n / base)).unwrap();
            }
            if pow != 0 {
                output.push_str(sep);
            }
            trailing = true;
        }
        n %= base;
    }

    output
}

