// Copyright 2016 - 2018 Ulrik Sverdrup "bluss"
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use core::cmp::min;

use crate::threading::ThreadPoolCtx;

#[derive(Copy, Clone)]
pub struct RangeChunk { i: usize, n: usize, chunk: usize }

/// Create an iterator that splits `n` in chunks of size `chunk`;
/// the last item can be an uneven chunk.
pub fn range_chunk(n: usize, chunk: usize) -> RangeChunk {
    RangeChunk {
        i: 0,
        n: n,
        chunk: chunk,
    }
}

impl Iterator for RangeChunk {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.n == 0 {
            None
        } else {
            let i = self.i;
            let rem = min(self.n, self.chunk);
            self.i += 1;
            self.n -= rem;
            Some((i, rem))
        }
    }
}

#[inline]
pub fn round_up_to(x: usize, multiple_of: usize) -> usize {
    let (mut d, r) = (x / multiple_of, x % multiple_of);
    if r > 0 { d += 1; }
    d * multiple_of
}

#[cfg(feature="threading")]
/// Create an iterator that splits `n` in chunks of size `chunk`;
/// the last item can be an uneven chunk.
///
/// And splits the iterator in `total` parts and only iterates the `index`th part of it
pub fn range_chunk_part(n: usize, chunk: usize, index: usize, total: usize) -> RangeChunk {
    debug_assert_ne!(total, 0);

    // round up
    let mut nchunks = n / chunk;
    nchunks += (n % chunk != 0) as usize;

    // chunks per thread
    // round up
    let mut chunks_per = nchunks / total;
    chunks_per += (nchunks % total != 0) as usize;

    let i = chunks_per * index;
    let nn = min(n, (i + chunks_per) * chunk).saturating_sub(i * chunk);

    RangeChunk { i, n: nn, chunk }
}

impl RangeChunk {
    /// "Builder" method to create a RangeChunkParallel
    pub(crate) fn parallel(self, nthreads: u8, pool: ThreadPoolCtx) -> RangeChunkParallel<fn()> {
        fn nop() {}

        RangeChunkParallel {
            nthreads,
            pool,
            range: self,
            thread_local: nop,
        }
    }
}

/// Intermediate struct for building the parallel execution of a range chunk.
pub(crate) struct RangeChunkParallel<'a, G> {
    range: RangeChunk,
    nthreads: u8,
    pool: ThreadPoolCtx<'a>,
    thread_local: G,
}

impl<'a, G> RangeChunkParallel<'a, G> {
    #[cfg(feature="threading")]
    /// Set thread local setup function - called once per thread to setup thread local data.
    pub(crate) fn thread_local<G2, R>(self, func: G2) -> RangeChunkParallel<'a, G2>
        where G2: Fn(usize, usize) -> R + Sync
    {
        RangeChunkParallel {
            nthreads: self.nthreads,
            pool: self.pool,
            thread_local: func,
            range: self.range,
        }
    }

    #[cfg(not(feature="threading"))]
    /// Set thread local setup function - called once per thread to setup thread local data.
    pub(crate) fn thread_local<G2, R>(self, func: G2) -> RangeChunkParallel<'a, G2>
        where G2: FnOnce(usize, usize) -> R + Sync
    {
        RangeChunkParallel {
            nthreads: self.nthreads,
            pool: self.pool,
            thread_local: func,
            range: self.range,
        }
    }
}

#[cfg(not(feature="threading"))]
impl<G, R> RangeChunkParallel<'_, G>
    where G: FnOnce(usize, usize) -> R + Sync,
{
    pub(crate) fn for_each<F>(self, for_each: F)
        where F: Fn(ThreadPoolCtx<'_>, &mut R, usize, usize) + Sync,
    {
        let mut local = (self.thread_local)(0, 1);
        for (ln, chunk_size) in self.range {
            for_each(self.pool, &mut local, ln, chunk_size)
        }
    }
}


#[cfg(feature="threading")]
impl<G, R> RangeChunkParallel<'_, G>
    where G: Fn(usize, usize) -> R + Sync,
{
    /// Execute loop iterations (parallel if enabled) using the given closure.
    ///
    /// The closure gets the following arguments for each iteration:
    ///
    /// - Thread pool context (used for child threads)
    /// - Mutable reference to thread local data
    /// - index of chunk (like RangeChunk)
    /// - size of chunk (like RangeChunk)
    pub(crate) fn for_each<F>(self, for_each: F)
        where F: Fn(ThreadPoolCtx<'_>, &mut R, usize, usize) + Sync,
    {
        fn inner<F, G, R>(range: RangeChunk, index: usize, nthreads: usize, pool: ThreadPoolCtx<'_>,
                          thread_local: G, for_each: F)
            where G: Fn(usize, usize) -> R + Sync,
                  F: Fn(ThreadPoolCtx<'_>, &mut R, usize, usize) + Sync
        {
            let mut local = thread_local(index, nthreads);
            for (ln, chunk_size) in range_chunk_part(range.n, range.chunk, index, nthreads) {
                for_each(pool, &mut local, ln, chunk_size)
            }
        }

        debug_assert!(self.nthreads <= 4, "this method does not support nthreads > 4, got {}",
                      self.nthreads);
        let pool = self.pool;
        let range = self.range;
        let for_each = &for_each;
        let local = &self.thread_local;
        let nthreads = min(self.nthreads as usize, 4);
        let f = move |ctx: ThreadPoolCtx<'_>, i| inner(range, i, nthreads, ctx, local, for_each);
        if nthreads >= 4 {
            pool.join4(&f);
        } else if nthreads >= 3 {
            pool.join3l(&f);
        } else if nthreads >= 2 {
            pool.join(|ctx| f(ctx, 0), |ctx| f(ctx, 1));
        } else {
            f(pool, 0)
        }
    }

}

