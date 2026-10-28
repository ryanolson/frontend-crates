// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::{
    UnifiedParserEvent, UnifiedParserInit, UnifiedParserOutput, create_unified_parser_for_family,
};

#[test]
// A one-range list is the expected value, not a mistaken `Vec` of a range.
#[allow(clippy::single_range_in_vec_init)]
fn request_initialization_after_reset_preserves_input_offsets() {
    for reinitialize in [false, true] {
        let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
        parser
            .initialize_request(UnifiedParserInit::default())
            .unwrap();
        assert!(parser.record_sources());
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("A", &mut output).unwrap();
        assert_eq!(output.events, [UnifiedParserEvent::Text("A".into())]);
        let mut ranges = Vec::new();
        parser.take_sources(&mut ranges);
        assert_eq!(ranges, [0..1]);
        assert_eq!(parser.reset(), "");
        if reinitialize {
            parser
                .initialize_request(UnifiedParserInit::default())
                .unwrap();
        }
        let mut output = UnifiedParserOutput::default();
        parser.parse_into("B", &mut output).unwrap();
        assert_eq!(output.events, [UnifiedParserEvent::Text("B".into())]);
        let mut ranges = Vec::new();
        parser.take_sources(&mut ranges);
        assert_eq!(ranges, [1..2], "reinitialize={reinitialize}");
    }
}
