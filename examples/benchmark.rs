//! Run this executable to benchmark sgemm and dgemm for arbitrary size matrices
//! See --help for usage examples.  Remember to run in release mode.
#![allow(non_camel_case_types)]

extern crate itertools;
extern crate matrixmultiply;

use std::cell::Cell;
use std::fmt::Debug;
use std::time::Instant;

use itertools::zip;

use matrixmultiply::{sgemm, dgemm};
#[cfg(feature="cgemm")]
use matrixmultiply::{cgemm, zgemm, CGemmOption};
use itertools::Itertools;

enum Arg {
    Flag { long: &'static str },
    Value { long: &'static str },
}

impl Arg {
    fn is_flag(&self) -> bool {
        match *self {
            Arg::Flag { .. } => true,
            Arg::Value { .. } => false,
        }
    }

    fn long(&self) -> &str {
        use Arg::*;
        match *self {
            Flag { long, .. } | Value { long, .. } => long,
        }
    }
}

struct Argparse<'a> {
    spec: &'a [&'a Arg],
    // true: this arg has already been parsed, false: unused
    used: Vec<Cell<bool>>,
    args: Vec<String>,
}

// Simple argument parser
impl<'a> Argparse<'a> {
    pub fn new(spec: &'a [&'a Arg], args: impl IntoIterator<Item=String>) -> Self {
        let strings: Vec<_> = args.into_iter().collect();
        Argparse {
            spec,
            used: vec![Cell::new(false); strings.len()],
            args: strings,
        }
    }
    
    fn get_arg(&self, long: &str) -> Option<(bool, &str)> {
        self.used[0].set(true);
        let arg_spec = self.spec.iter().find(|arg| arg.long() == long).expect("No such argument");
        for (i, arg) in self.args.iter().enumerate() {
            if self.used[i].get() {
                continue;
            }

            let arg_long = arg_spec.long();
            if arg.starts_with("--") {
                if arg[2..].starts_with(arg_long) {
                    /* has arg */
                    self.used[i].set(true);
                    if arg_spec.is_flag() {
                        return Some((false, ""));
                    }

                    if arg[2 + arg_long.len()..].is_empty() && self.args.len() > i + 1 {
                        self.used[i + 1].set(true);
                        return Some((true, &self.args[i + 1]));
                    } else {
                        return Some((true, &arg[3 + arg_long.len()..]))
                    }
                }
            }
        }
        None
    }


    pub fn get_flag(&self, long: &str) -> Option<bool> {
        self.get_arg(long).map(|_| true)
    }

    pub fn get_string(&self, long: &str) -> Option<&str> {
        self.get_arg(long).map(|(_, arg)| arg)
    }

    pub fn check_usage(&self) -> Result<(), String> {
        for (i, arg) in self.args.iter().enumerate() {
            if !self.used[i].get() && arg.starts_with("-") {
                return Err(format!("Unknown argument {:?}", arg));
            }
        }
        Ok(())
    }

    pub fn next_positional_int(&self) -> Option<u64> {
        for (i, arg) in self.args.iter().enumerate() {
            if !self.used[i].get() {
                self.used[i].set(true);
                return Some(arg.parse::<u64>().unwrap())
            }
        }
        None
    }
}


fn main() -> Result<(), String> {
    run_main(std::env::args())
}

fn run_main(args: impl IntoIterator<Item=String>) -> Result<(), String> {
    #[cfg(debug_assertions)]
    eprintln!("Warning: running benchmark with debug assertions");

    let opts = match parse_args(args) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Usage: <command> [--type <type>] [--layout <layout>] [--csv]  m-size k-size n-size");
            eprintln!();
            eprintln!("Where <type> is one of: f32, f64, c32, c64");
            eprintln!("Where <layout> is 3 letters from c, f like: ccc fcc fff");
            eprintln!();
            eprintln!("Example: <command> --type f64 --layout fcf 1000 1000 1000");
            eprintln!();
            eprintln!("csv headers: m,k,n,layout,type,average_ns,minimum_ns,median_ns,samples,gflops");
            eprintln!();
            return Err(format!("Error parsing arguments: {}", e));
        }
    };

    match opts.use_type {
        UseType::F32 => test_matrix::<f32>(opts.m, opts.k, opts.n, opts.layout, opts.use_csv, opts.use_type, &opts.extra_column),
        UseType::F64 => test_matrix::<f64>(opts.m, opts.k, opts.n, opts.layout, opts.use_csv, opts.use_type, &opts.extra_column),
        UseType::C32 => test_matrix::<c32>(opts.m, opts.k, opts.n, opts.layout, opts.use_csv, opts.use_type, &opts.extra_column),
        UseType::C64 => test_matrix::<c64>(opts.m, opts.k, opts.n, opts.layout, opts.use_csv, opts.use_type, &opts.extra_column),
    }
    Ok(())
}

#[derive(Debug, Copy, Clone)]
enum UseType {
    F32,
    F64,
    C32,
    C64,
}

impl UseType {
    fn type_name(self) -> &'static str {
        use UseType::*;
        match self {
            F32 => "f32",
            F64 => "f64",
            C32 => "c32",
            C64 => "c64",
        }
    }
    fn flop_factor(self) -> f64 {
        match self {
            // estimate one multiply and one addition
            UseType::F32 | UseType::F64 => 2.,
            // (P + Qi)(R + Si) = ..
            // estimate 8 flop (4 float multiplies and 4 additions).
            UseType::C32 | UseType::C64 => 8.,
        }
    }
}

impl Default for UseType {
    fn default() -> Self { Self::F64 }
}


trait Float : Copy + Debug + PartialEq {
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

type c32 = [f32; 2];
type c64 = [f64; 2];

impl Float for c32 {
    fn zero() -> Self { [0., 0.] }
    fn one() -> Self { [1., 0.] }
    fn from(x: i64) -> Self { [x as _, 0.] }
    fn nan() -> Self { [f32::NAN, f32::NAN] }
    fn is_nan(self) -> bool { self[0].is_nan() || self[1].is_nan() }
}

impl Float for c64 {
    fn zero() -> Self { [0., 0.] }
    fn one() -> Self { [1., 0.] }
    fn from(x: i64) -> Self { [x as _, 0.] }
    fn nan() -> Self { [f64::NAN, f64::NAN] }
    fn is_nan(self) -> bool { self[0].is_nan() || self[1].is_nan() }
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

impl Gemm for c32 {
    #[allow(unused)]
    unsafe fn gemm(
        m: usize, k: usize, n: usize,
        alpha: Self,
        a: *const Self, rsa: isize, csa: isize,
        b: *const Self, rsb: isize, csb: isize,
        beta: Self,
        c: *mut Self, rsc: isize, csc: isize) {
        #[cfg(not(feature="cgemm"))]
        {
            unimplemented!()
        }
        #[cfg(feature="cgemm")]
        {
        cgemm(
            CGemmOption::Standard,
            CGemmOption::Standard,
            m, k, n,
            alpha,
            a, rsa, csa,
            b, rsb, csb,
            beta,
            c, rsc, csc)
        }
    }
}

impl Gemm for c64 {
    #[allow(unused)]
    unsafe fn gemm(
        m: usize, k: usize, n: usize,
        alpha: Self,
        a: *const Self, rsa: isize, csa: isize,
        b: *const Self, rsb: isize, csb: isize,
        beta: Self,
        c: *mut Self, rsc: isize, csc: isize) {
        #[cfg(not(feature="cgemm"))]
        {
            unimplemented!()
        }
        #[cfg(feature="cgemm")]
        {
        zgemm(
            CGemmOption::Standard,
            CGemmOption::Standard,
            m, k, n,
            alpha,
            a, rsa, csa,
            b, rsb, csb,
            beta,
            c, rsc, csc)
        }
    }
}

#[derive(Debug, Clone, Default)]
struct Options {
    m: usize,
    k: usize,
    n: usize,
    layout: [Layout; 3],
    use_type: UseType,
    use_csv: bool,
    extra_column: Option<String>,
}

fn parse_args(args: impl IntoIterator<Item=String>) -> Result<Options, String> {
    let mut opts = Options::default();
    //./target/release/examples/benchmark 1280 1280 1280 c64 fcf
    let parse = Argparse::new(&[
        &Arg::Flag { long: "csv" },
        &Arg::Value { long: "layout" },
        &Arg::Value { long: "type" },
        &Arg::Value { long: "extra-column" },
    ], args);

    opts.use_type = match parse.get_string("type") {
        Some("f32") => UseType::F32,
        Some("f64") => UseType::F64,
        Some("c32") => UseType::C32,
        Some("c64") => UseType::C64,
        Some(_otherwise) => return Err("Unknown type".to_string()),
        None => UseType::F64,
    };
    if let Some(layout) = parse.get_string("layout") {
        if layout.len() != 3 || !layout.chars().all(|c| c == 'c' || c == 'f') {
            Err(format!("Unknown argument {}", layout))?;
        }
        for (elt, layout_arg) in zip(&mut opts.layout[..], layout.chars())
        {
            *elt = if layout_arg == 'c' { Layout::C } else { Layout::F };
        }
    }
    opts.use_csv = parse.get_flag("csv").is_some();
    opts.extra_column = parse.get_string("extra-column").map(|s| s.to_string());

    parse.check_usage()?;

    opts.m = parse.next_positional_int().ok_or("Expected argument".to_string())? as usize;
    opts.k = parse.next_positional_int().ok_or("Expected argument".to_string())? as usize;
    opts.n = parse.next_positional_int().ok_or("Expected argument".to_string())? as usize;

    Ok(opts)
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


fn test_matrix<F>(m: usize, k: usize, n: usize, layouts: [Layout; 3],
                  use_csv: bool, use_type: UseType, extra: &Option<String>)
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

    if !use_csv {
        println!("Test matrix a : {} × {} layout: {:?} strides {}, {}", m, k, la, rs_a, cs_a);
        println!("Test matrix b : {} × {} layout: {:?} strides {}, {}", k, n, lb, rs_b, cs_b);
        println!("Test matrix c : {} × {} layout: {:?} strides {}, {}", m, n, lc1, rs_c1, cs_c1);
    }

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
    let gflop = use_type.flop_factor() * (m as f64 * n as f64 * k as f64) / statistics.average as f64;
    if !use_csv {
        print!("{}×{}×{} {:?} {} .. {} ns", m, k, n, layouts, use_type.type_name(),
               fmt_thousands_sep(statistics.average, " "));
        print!(" [minimum: {} ns .. median {} ns .. sample count {}]", 
               fmt_thousands_sep(statistics.minimum, " "),
               fmt_thousands_sep(statistics.median, " "),
               statistics.samples.len());
        // by flop / s = 2 M N K / time
        print!("    {:.2} Gflop/s", gflop);
        println!();
    } else {
        print!("{},{},{},", m, k, n);
        print!("{:?},", layouts.iter().format(""));
        print!("{},", use_type.type_name());
        print!("{},{},{},{},", statistics.average, statistics.minimum, statistics.median,
               statistics.samples.len());
        print!("{}", gflop);
        if let Some(extra) = extra {
            print!(",{}", extra);
        }
        println!();
    }

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

#[test]
fn test_benchmark() {
    run_main("ignored 128 128 128 f64 fcc".split_whitespace().map(str::to_string)).unwrap();
}
