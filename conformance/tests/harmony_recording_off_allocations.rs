// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

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

#[test]
fn recording_off_keeps_the_quoted_tail_allocation_budget() {
    let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
    parser
        .initialize_request(UnifiedParserInit::default())
        .unwrap();
    let mut output = UnifiedParserOutput::default();
    parser
        .parse_into(
            "<|start|>assistant to=functions.lookup<|channel|>commentary<|message|>{\"s\":\"<|end|>",
            &mut output,
        )
        .unwrap();
    let payload = "z ".repeat(8192);
    RETAINED.with(|count| count.set(Some(0)));
    parser.parse_into(&payload, &mut output).unwrap();
    let retained = RETAINED.with(|count| count.replace(None).unwrap());
    eprintln!(
        "recording off: retained {retained} bytes for {} input bytes",
        payload.len()
    );
    assert!(
        retained <= 128 * 1024,
        "recording off retained {retained} bytes"
    );
}
