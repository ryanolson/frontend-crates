// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_parsers_v2::unified::harmony::encode_harmony;
use dynamo_parsers_v2::{
    UnifiedParserEvent, UnifiedParserInit, UnifiedParserOutput, create_unified_parser_for_family,
};

#[test]
fn prompt_header_recovery_has_sources_for_every_emitted_byte() {
    let init = UnifiedParserInit {
        prompt_token_ids: encode_harmony("<|start|>assistant<|channel|>final ").unwrap(),
        ..UnifiedParserInit::default()
    };
    let input = "answer";
    let mut results = Vec::new();
    for record in [false, true] {
        let mut parser = create_unified_parser_for_family("gpt_oss", &[]).unwrap();
        parser.initialize_request(init.clone()).unwrap();
        if record && !parser.record_sources() {
            return;
        }
        let mut output = UnifiedParserOutput::default();
        parser.parse_into(input, &mut output).unwrap();
        output.append(&mut parser.finish().unwrap());
        let mut ranges = Vec::new();
        parser.take_sources(&mut ranges);
        results.push((output.events, ranges));
    }
    assert_eq!(results[0].0, results[1].0);
    let text: String = results[1]
        .0
        .iter()
        .filter_map(|event| match event {
            UnifiedParserEvent::Text(text) | UnifiedParserEvent::Reasoning(text) => {
                Some(text.as_str())
            }
            _ => None,
        })
        .collect();
    assert_eq!(text, " answer");
    let rebuilt: Vec<u8> = results[1]
        .1
        .iter()
        .flat_map(|range| input.as_bytes()[range.clone()].iter().copied())
        .collect();
    assert_eq!(rebuilt, text.as_bytes(), "ranges: {:?}", results[1].1);
}
