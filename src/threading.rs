///
/// Threading support functions and statics

#[cfg(feature="threading")]
use std::str::FromStr;
#[cfg(feature="threading")]
use once_cell::sync::Lazy;

#[cfg(feature="threading")]
pub use thread_tree::ThreadTree as ThreadPool;
#[cfg(feature="threading")]
pub use thread_tree::ThreadTreeCtx as ThreadPoolCtx;

use crate::kernel::GemmKernel;


/// Dummy threadpool
#[cfg(not(feature="threading"))]
pub(crate) struct ThreadPool;

#[cfg(not(feature="threading"))]
pub(crate) type ThreadPoolCtx<'a> = &'a ();

#[cfg(not(feature="threading"))]
impl ThreadPool {
    /// Get top dummy thread pool context
    pub(crate) fn top(&self) -> ThreadPoolCtx<'_> { &() }
}

pub(crate) fn get_thread_pool<'a>() -> (usize, ThreadPoolCtx<'a>) {
    let reg = &*REGISTRY;
    (reg.nthreads, reg.thread_pool().top())
}

struct Registry {
    nthreads: usize,
    #[cfg(feature="threading")]
    thread_pool: Box<ThreadPool>,
}

impl Registry {
    fn thread_pool(&self) -> &ThreadPool {
        #[cfg(feature="threading")]
        return &*REGISTRY.thread_pool;
        #[cfg(not(feature="threading"))]
        return &ThreadPool;
    }
}

#[cfg(not(feature="threading"))]
const REGISTRY: &'static Registry = &Registry { nthreads: 1 };

#[cfg(feature="threading")]
static REGISTRY: Lazy<Registry> = Lazy::new(|| {
    let var = ::std::env::var("MATMUL_NUM_THREADS").ok();
    let threads = match var {
        Some(s) if !s.is_empty() => {
            if let Ok(nt) = usize::from_str(&s) {
                1.max(nt)
            } else {
                eprintln!("Failed to parse MATMUL_NUM_THREADS");
                1
            }
        }
        _otherwise => num_cpus::get_physical(),
    };

    let tp = if threads <= 1 {
        Box::new(ThreadPool::new_level0())
    } else if threads <= 3 {
        ThreadPool::new_with_level(1)
    } else {
        ThreadPool::new_with_level(2)
    };

    Registry {
        nthreads: threads,
        thread_pool: tp,
    }
});

/// Describe how many threads we use in each loop
#[derive(Copy, Clone)]
pub(crate) struct LoopThreadConfig {
    /// Loop 3 threads
    pub(crate) loop3: u8,
    /// Loop 2 threads
    pub(crate) loop2: u8,
}

impl LoopThreadConfig {
    /// Decide how many threads to use in each loop
    pub(crate) fn new<K>(m: usize, k: usize, n: usize, max_threads: usize) -> Self
        where K: GemmKernel
    {
        let default_config = LoopThreadConfig { loop3: 1, loop2: 1 };

        #[cfg(not(feature="threading"))]
        {
            let _ = (m, k, n, max_threads); // used
            return default_config;
        }

        #[cfg(feature="threading")]
        {
            if max_threads == 1 {
                return default_config;
            }

            Self::new_impl(m, k, n, max_threads, K::mc())
        }
    }

    #[cfg(feature="threading")]
    fn new_impl(m: usize, k: usize, n: usize, max_threads: usize, kmc: usize) -> Self {
        // use a heuristic to try not to use too many threads for smaller matrices
        let size_factor = m * k + k * n;
        let thread_factor = 1 << 16;
        // pure guesswork in terms of what the default should be
        let arch_factor = if cfg!(any(target_arch="arm", target_arch="aarch64")) {
            20
        } else {
            1
        };

        // At the moment only a configuration of 1, 2, or 4 threads is supported.
        //
        // Prefer to split Loop 3 if only 2 threads are available, (because it was better in a
        // square matrix benchmark).

        let matrix_max_threads = size_factor / (thread_factor / arch_factor);
        let mut max_threads = max_threads.min(matrix_max_threads);

        let loop3 = if max_threads >= 2 && m >= 3 * (kmc / 2) {
            max_threads /= 2;
            2
        } else {
            1
        };
        let loop2 = if max_threads >= 2 { 2 } else { 1 };

        LoopThreadConfig {
            loop3,
            loop2,
        }
    }

    /// Number of packing buffers for A
    #[inline(always)]
    pub(crate) fn num_pack_a(&self) -> usize { self.loop3 as usize }
}

