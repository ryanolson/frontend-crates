// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The lib tests' allocator: it counts allocations per thread, so a test can
//! bound what a stage allocates without the tests running beside it counting.
//! One binary has one global allocator, so every lib test that counts uses
//! this one.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
    static BYTES: Cell<usize> = const { Cell::new(0) };
}

fn count(size: usize) {
    let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
    let _ = BYTES.try_with(|n| n.set(n.get() + size));
}

/// The allocations this thread has made so far.
pub(crate) fn allocations() -> usize {
    ALLOCATIONS.with(Cell::get)
}

/// The bytes this thread has allocated so far.
// Only some tests bound bytes; a crate built without them never reads it.
#[allow(dead_code)]
pub(crate) fn bytes() -> usize {
    BYTES.with(Cell::get)
}

struct Counting;

// SAFETY: every call forwards to `System` with the same arguments.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;
