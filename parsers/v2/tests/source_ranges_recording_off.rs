// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! With recording off, the scanner keeps no range beside the text it holds.
//!
//! A stream of failed DeepSeek V4.1 blocks, each followed by text, makes the
//! scanner hold one disjoint span per block for `reset`. Recording keeps one
//! range per span; with recording off the parser must keep none, so its retained
//! bytes stay at least one range per span below the recording run.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::ops::Range;

use dynamo_parsers_v2::{UnifiedParserInit, UnifiedParserOutput, create_unified_parser_for_family};

struct CountingAllocator;

thread_local! {
    static RETAINED: Cell<Option<isize>> = const { Cell::new(None) };
}

fn account(bytes: isize) {
    let _ = RETAINED.try_with(|count| {
        if let Some(current) = count.get() {
            count.set(Some(current + bytes));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            account(layout.size() as isize);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        account(-(layout.size() as isize));
        unsafe { System.dealloc(ptr, layout) };
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let new_ptr = unsafe { System.realloc(ptr, layout, size) };
        if !new_ptr.is_null() {
            account(size as isize - layout.size() as isize);
        }
        new_ptr
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const BAD: &str = concat!(
    "<｜DSML｜ calls><｜DSML｜ invoke name=\"run\">",
    "<｜DSML｜ parameter name=\"value\" string=\"false\">{\"a\":",
    "</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"
);
const SPANS: usize = 2000;

/// Bytes the parser keeps after the stream, with the output already dropped.
fn retained(record: bool) -> isize {
    let mut parser = create_unified_parser_for_family("deepseek_v41", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit::default())
        .unwrap();
    if record {
        assert!(parser.record_sources());
    }
    RETAINED.with(|count| count.set(Some(0)));
    for _ in 0..SPANS {
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(BAD, &mut output).unwrap();
        parser.parse_into("x", &mut output).unwrap();
        let mut ranges = Vec::new();
        parser.take_sources(&mut ranges);
    }
    let retained = RETAINED.with(|count| count.replace(None).unwrap());
    drop(parser);
    retained
}

#[test]
fn recording_off_keeps_no_range_beside_held_text() {
    let off = retained(false);
    let on = retained(true);
    let ranges = (SPANS * std::mem::size_of::<Range<usize>>()) as isize;
    eprintln!("retained after {SPANS} held spans: off {off} bytes, on {on} bytes");
    assert!(
        off + ranges <= on,
        "recording off retained {off} bytes, on {on}: off must keep no range per span"
    );
}
