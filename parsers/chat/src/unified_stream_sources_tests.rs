// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Source recording through the v2 stream adapter.
//!
//! The `dynamo-parsers-v2` families cannot record yet (their `record_sources`
//! is the trait default), so these tests swap in [`TagParser`], which records
//! exactly. The family names stay as they are, so the adapter's per-family
//! paths (the Qwen opener hold, GLM input capture, GPT-OSS init) still run.

use std::io::Read;
use std::ops::Range;

use dynamo_parsers::sources::{ChoiceSources, SourceSlot};
use serde_json::{Value, json};

use super::*;

/// dynamo's chunk shape plus a source slot.
#[derive(Clone, Debug, PartialEq)]
struct Slotted {
    chunk: NvCreateChatCompletionStreamResponse,
    slot: Vec<ChoiceSources>,
}

impl SourceSlot for Slotted {
    fn sources(&self) -> Option<&[ChoiceSources]> {
        Some(&self.slot)
    }

    fn sources_mut(&mut self) -> Option<&mut Vec<ChoiceSources>> {
        Some(&mut self.slot)
    }
}

impl UnifiedStreamChunk for Slotted {
    fn inner(&self) -> &CreateChatCompletionStreamResponse {
        self.chunk.inner()
    }

    fn inner_mut(&mut self) -> &mut CreateChatCompletionStreamResponse {
        self.chunk.inner_mut()
    }

    fn tool_call_completion(&self) -> &[ToolCallCompletion] {
        self.chunk.tool_call_completion()
    }

    fn tool_call_completion_mut(&mut self) -> &mut Vec<ToolCallCompletion> {
        self.chunk.tool_call_completion_mut()
    }

    fn clear_nvext(&mut self) {
        self.chunk.clear_nvext();
    }

    fn clear_llm_metrics(&mut self) {
        self.chunk.clear_llm_metrics();
    }

    fn scrub_synthetic(&mut self) {
        self.chunk.scrub_synthetic();
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Text,
    Reasoning,
    Tool,
}

const OPENERS: [&str; 2] = ["<think>", "<|channel>thought\n"];
const CLOSERS: [&str; 2] = ["</think>", "<channel|>"];
const TOOL_OPEN: &str = "<tool>";
const TOOL_CLOSE: &str = "</tool>";

/// A small grammar that records exact sources: `<think>…</think>` and
/// `<|channel>thought\n…<channel|>` are reasoning, `<tool>NAME</tool>` is a
/// call, and the rest is text. It holds a partial marker until the next input
/// decides it, and deletes markers, so a wrapper could not date its output.
struct TagParser {
    buf: String,
    /// Input offset of `buf[0]`.
    start: usize,
    mode: Mode,
    record: bool,
    sources: Vec<Range<usize>>,
    calls: usize,
    ids: Vec<String>,
    /// Report one byte too few for each run (a broken recorder).
    short: bool,
}

impl TagParser {
    fn new() -> Self {
        Self {
            buf: String::new(),
            start: 0,
            mode: Mode::Text,
            record: false,
            sources: Vec::new(),
            calls: 0,
            ids: Vec::new(),
            short: false,
        }
    }

    fn markers(&self) -> Vec<&'static str> {
        match self.mode {
            Mode::Text => OPENERS.into_iter().chain([TOOL_OPEN]).collect(),
            Mode::Reasoning => CLOSERS.to_vec(),
            Mode::Tool => vec![TOOL_CLOSE],
        }
    }

    fn emit(&mut self, n: usize, out: &mut UnifiedParserOutput) {
        if n == 0 || self.mode == Mode::Tool {
            return;
        }
        let text: String = self.buf.drain(..n).collect();
        if self.record {
            let end = self.start + n - usize::from(self.short);
            self.sources.push(self.start..end);
        }
        self.start += n;
        out.events.push(match self.mode {
            Mode::Text => UnifiedParserEvent::Text(text),
            _ => UnifiedParserEvent::Reasoning(text),
        });
    }

    fn consume(&mut self, n: usize) {
        self.buf.drain(..n);
        self.start += n;
    }

    fn drain(&mut self, out: &mut UnifiedParserOutput, at_end: bool) {
        loop {
            let found = self
                .markers()
                .into_iter()
                .filter_map(|marker| self.buf.find(marker).map(|position| (position, marker)))
                .min_by_key(|(position, _)| *position);
            if let Some((position, marker)) = found {
                if self.mode == Mode::Tool {
                    let name = self.buf[..position].to_string();
                    self.consume(position + marker.len());
                    self.ids.push(format!("call_test_{}", self.calls));
                    out.events.push(UnifiedParserEvent::ToolCall(ToolCallDelta {
                        tool_index: self.calls,
                        name: Some(name),
                        arguments: "{}".to_string(),
                        complete: true,
                    }));
                    self.calls += 1;
                    self.mode = Mode::Text;
                    continue;
                }
                self.emit(position, out);
                self.consume(marker.len());
                self.mode = match marker {
                    TOOL_OPEN => Mode::Tool,
                    _ if self.mode == Mode::Text => Mode::Reasoning,
                    _ => Mode::Text,
                };
                continue;
            }
            if self.mode == Mode::Tool {
                if at_end {
                    // A cut-off call is dropped.
                    let n = self.buf.len();
                    self.consume(n);
                }
                return;
            }
            let hold = if at_end {
                0
            } else {
                self.markers()
                    .into_iter()
                    .flat_map(|marker| (1..marker.len()).map(move |k| &marker[..k]))
                    .filter(|prefix| self.buf.ends_with(prefix))
                    .map(str::len)
                    .max()
                    .unwrap_or(0)
            };
            self.emit(self.buf.len() - hold, out);
            return;
        }
    }
}

impl UnifiedParser for TagParser {
    fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        self.buf.push_str(delta);
        self.drain(output, false);
        Ok(())
    }

    fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
        let mut output = UnifiedParserOutput::default();
        self.drain(&mut output, true);
        Ok(output)
    }

    fn reset(&mut self) -> String {
        let text = std::mem::take(&mut self.buf);
        if self.record && !text.is_empty() {
            self.sources.push(self.start..self.start + text.len());
        }
        self.start += text.len();
        self.mode = Mode::Text;
        text
    }

    fn tool_call_id(&self, tool_index: usize) -> Option<&str> {
        self.ids.get(tool_index).map(String::as_str)
    }

    /// As the real parsers do: refuse while bytes are held, and count offsets
    /// from here.
    fn record_sources(&mut self) -> bool {
        if !self.buf.is_empty() {
            return false;
        }
        self.record = true;
        self.start = 0;
        true
    }

    fn take_sources(&mut self, out: &mut Vec<Range<usize>>) {
        out.append(&mut self.sources);
    }
}

/// Install a parser factory for this thread, and remove it when dropped.
struct ParserOverride;

impl ParserOverride {
    fn install(short: bool) -> Self {
        TEST_PARSER.with(|factory| {
            *factory.borrow_mut() = Some(Box::new(move |_, _| {
                let mut parser = TagParser::new();
                parser.short = short;
                Box::new(parser)
            }));
        });
        Self
    }
}

impl Drop for ParserOverride {
    fn drop(&mut self) {
        TEST_PARSER.with(|factory| factory.borrow_mut().take());
    }
}

type Plain = Annotated<NvCreateChatCompletionStreamResponse>;
type Recorded = Annotated<Slotted>;

/// Give each input chunk a slot. Each choice index is its own caller text:
/// upstream reasoning first, then content, in chunk order.
fn with_slots(items: &[Plain]) -> (Vec<Recorded>, HashMap<u32, String>) {
    let mut texts: HashMap<u32, String> = HashMap::new();
    let items = items
        .iter()
        .map(|item| {
            let data = item.data.as_ref().map(|chunk| {
                let slot = chunk
                    .inner
                    .choices
                    .iter()
                    .map(|choice| {
                        let text = texts.entry(choice.index).or_default();
                        let mut sources = ChoiceSources {
                            index: choice.index,
                            ..ChoiceSources::default()
                        };
                        if let Some(reasoning) = &choice.delta.reasoning_content {
                            sources.reasoning = vec![text.len()..text.len() + reasoning.len()];
                            text.push_str(reasoning);
                        }
                        if let Some(ChatCompletionMessageContent::Text(content)) =
                            &choice.delta.content
                        {
                            sources.content = vec![text.len()..text.len() + content.len()];
                            text.push_str(content);
                        }
                        sources
                    })
                    .collect();
                Slotted {
                    chunk: chunk.clone(),
                    slot,
                }
            });
            Annotated {
                data,
                id: item.id.clone(),
                event: item.event.clone(),
                comment: item.comment.clone(),
                error: item.error.clone(),
            }
        })
        .collect();
    (items, texts)
}

fn without_slot(item: &Recorded) -> (Option<NvCreateChatCompletionStreamResponse>, String) {
    (
        item.data.as_ref().map(|data| data.chunk.clone()),
        format!(
            "{:?}{:?}{:?}{:?}",
            item.id, item.event, item.comment, item.error
        ),
    )
}

fn plain_key(item: &Plain) -> (Option<NvCreateChatCompletionStreamResponse>, String) {
    (
        item.data.clone(),
        format!(
            "{:?}{:?}{:?}{:?}",
            item.id, item.event, item.comment, item.error
        ),
    )
}

fn read(text: &str, ranges: &[Range<usize>]) -> Option<String> {
    ranges
        .iter()
        .map(|range| text.get(range.clone()))
        .collect::<Option<Vec<_>>>()
        .map(|parts| parts.concat())
}

/// The verbatim and tiling rules: each chunk's slot has one entry per choice,
/// each entry's ranges read back exactly that choice's text, and no input byte
/// is dated twice. Ranges need not move forward: a held byte can be released
/// after a later one. Reading text back cannot tell which occurrence of that
/// text a range names (see `the_checker_rejects_a_range_from_the_wrong_place`).
fn check_sources(output: &[Recorded], texts: &HashMap<u32, String>) -> Result<(), String> {
    let mut used: HashMap<u32, Vec<Range<usize>>> = HashMap::new();
    for (position, item) in output.iter().enumerate() {
        let Some(data) = &item.data else { continue };
        let choices = &data.chunk.inner.choices;
        if data.slot.len() != choices.len() {
            return Err(format!(
                "item {position}: {} choices, {} slot entries",
                choices.len(),
                data.slot.len()
            ));
        }
        for (choice, sources) in choices.iter().zip(&data.slot) {
            if choice.index != sources.index {
                return Err(format!("item {position}: slot index {}", sources.index));
            }
            let text = texts.get(&choice.index).map_or("", String::as_str);
            let content = match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(content)) => content.as_str(),
                _ => "",
            };
            let reasoning = choice.delta.reasoning_content.as_deref().unwrap_or("");
            for (field, expected, ranges) in [
                ("content", content, &sources.content),
                ("reasoning", reasoning, &sources.reasoning),
            ] {
                if read(text, ranges).as_deref() != Some(expected) {
                    return Err(format!(
                        "item {position}: {field} {expected:?} has ranges {ranges:?} over {text:?}"
                    ));
                }
                used.entry(choice.index)
                    .or_default()
                    .extend(ranges.iter().cloned());
            }
        }
    }
    for (index, mut ranges) in used {
        ranges.sort_by_key(|range| range.start);
        for pair in ranges.windows(2) {
            if pair[0].end > pair[1].start {
                return Err(format!("choice {index}: {pair:?} overlap"));
            }
        }
    }
    Ok(())
}

struct Request {
    family: &'static str,
    tools: bool,
    constraint: GuidedToolConstraint,
    prefill: StreamPrefill,
    guided_streaming: bool,
    policy: UnifiedRequestPolicy,
}

impl Request {
    fn native(family: &'static str) -> Self {
        Self {
            family,
            tools: true,
            constraint: GuidedToolConstraint::None,
            prefill: UnifiedParserStartingState::None.into(),
            guided_streaming: true,
            policy: UnifiedRequestPolicy::default(),
        }
    }

    fn run<I: UnifiedStreamItem>(&self, input: Vec<I>) -> Vec<I> {
        apply_stream_with_policy(
            stream::iter(input),
            self.tools.then(weather_tools),
            self.constraint.clone(),
            self.prefill,
            self.family,
            self.guided_streaming,
            self.policy,
        )
        .collect::<Vec<_>>()
        .now_or_never()
        .expect("the adapter has no await points beyond its input")
    }
}

fn text_chunk(index: u32, content: Option<&str>, reasoning: Option<&str>, finish: bool) -> Plain {
    let mut item = chunk(content.unwrap_or(""), finish);
    let choice = &mut item.data.as_mut().unwrap().inner.choices[0];
    choice.index = index;
    if content.is_none() {
        choice.delta.content = None;
    }
    choice.delta.reasoning_content = reasoning.map(str::to_string);
    item
}

fn split(text: &str, size: usize) -> Vec<Plain> {
    let chars: Vec<char> = text.chars().collect();
    let pieces: Vec<String> = chars
        .chunks(size.max(1))
        .map(|piece| piece.iter().collect())
        .collect();
    let last = pieces.len().saturating_sub(1);
    pieces
        .iter()
        .enumerate()
        .map(|(position, piece)| text_chunk(0, Some(piece), None, position == last))
        .collect()
}

/// Run the input with and without slots and require the same chunks, then
/// check the recorded sources.
fn assert_recording_is_transparent(request: &Request, input: Vec<Plain>) {
    let _parser = ParserOverride::install(false);
    let expected: Vec<_> = request.run(input.clone()).iter().map(plain_key).collect();
    let (slotted, texts) = with_slots(&input);
    let output = request.run(slotted);
    let actual: Vec<_> = output.iter().map(without_slot).collect();
    assert_eq!(actual, expected, "recording changed the chunks");
    if let Err(error) = check_sources(&output, &texts) {
        panic!("{error}");
    }
}

#[test]
fn recording_matches_the_stock_chunks_at_every_split() {
    let text = "a<think>r1</think>b<tool>get_weather</tool>c<think>r2</think>d";
    for family in ["glm47", "deepseek_v4", "qwen3", "gpt_oss", "kimi_k2"] {
        for size in 1..=text.len() {
            assert_recording_is_transparent(&Request::native(family), split(text, size));
        }
    }
}

#[test]
fn a_held_marker_is_dated_by_the_byte_that_produced_it() {
    // A `<` is held at the end of chunk 0. Chunk 1 completes the closer, which
    // is deleted, then holds a second `<`, which chunk 2 shows to be text.
    let _parser = ParserOverride::install(false);
    let (input, texts) = with_slots(&[
        text_chunk(0, Some("<think>r<"), None, false),
        text_chunk(0, Some("/think><"), None, false),
        text_chunk(0, Some("y"), None, true),
    ]);
    let output = Request::native("glm47").run(input);
    check_sources(&output, &texts).unwrap();
    let (mut content, mut reasoning) = (Vec::new(), Vec::new());
    for data in output.iter().filter_map(|item| item.data.as_ref()) {
        for sources in &data.slot {
            content.extend(sources.content.clone());
            reasoning.extend(sources.reasoning.clone());
        }
    }
    assert_eq!(reasoning, vec![7..8]);
    assert_eq!(content, vec![16..18]);
}

#[test]
fn upstream_reasoning_keeps_its_sources_in_a_mixed_chunk_and_a_detour() {
    for family in ["glm47", "qwen3"] {
        assert_recording_is_transparent(
            &Request::native(family),
            vec![
                text_chunk(0, Some("x<think>"), Some("up"), false),
                text_chunk(0, None, Some("only reasoning"), false),
                text_chunk(0, Some("r</think>y"), Some("more"), true),
            ],
        );
    }
}

#[test]
fn a_split_qwen_opener_keeps_its_sources() {
    let text = "say <tool_call> then <tool>get_weather</tool> done";
    for size in 1..=8 {
        assert_recording_is_transparent(&Request::native("qwen3"), split(text, size));
    }
    // Reasoning that arrives while the opener is held is queued with its sources.
    assert_recording_is_transparent(
        &Request::native("qwen3"),
        vec![
            text_chunk(0, Some("a <tool_"), None, false),
            text_chunk(0, None, Some("between"), false),
            text_chunk(0, Some("call> b"), None, true),
        ],
    );
}

#[test]
fn two_choices_record_separately() {
    let _parser = ParserOverride::install(false);
    let mut input = Vec::new();
    for (index, piece) in [
        (0, "a<thi"),
        (1, "b<think>"),
        (0, "nk>r</think>c"),
        (1, "s</think>d"),
    ] {
        input.push(text_chunk(index, Some(piece), None, false));
    }
    input.push(text_chunk(0, Some(""), None, true));
    input.push(text_chunk(1, Some(""), None, true));
    assert_recording_is_transparent(&Request::native("glm47"), input);
}

/// Recording starts after the prompt prefix replays, so a range counts the
/// generated text only.
#[test]
fn the_prompt_prefix_is_not_counted_as_input() {
    let _parser = ParserOverride::install(false);
    let request = Request {
        prefill: StreamPrefill {
            starting_state: UnifiedParserStartingState::Reasoning,
            prefix: Some("<|channel>thought\n"),
        },
        ..Request::native("gemma4")
    };
    let (input, texts) = with_slots(&split("R<channel|>A", 3));
    let output = request.run(input);
    check_sources(&output, &texts).unwrap();
    let (mut content, mut reasoning) = (Vec::new(), Vec::new());
    for data in output.iter().filter_map(|item| item.data.as_ref()) {
        for sources in &data.slot {
            content.extend(sources.content.clone());
            reasoning.extend(sources.reasoning.clone());
        }
    }
    assert_eq!(reasoning, vec![0..1]);
    assert_eq!(content, vec![11..12]);
}

/// A prompt prefix that leaves bytes held refuses recording when the stream
/// starts, before any generated output. The test parser holds `<|channel>thou`
/// as a partial opener; had the output disproved it, those prompt bytes would
/// have surfaced as text with no offset in the generated input.
#[test]
fn a_held_prompt_prefix_refuses_before_any_output() {
    let _parser = ParserOverride::install(false);
    let request = Request {
        prefill: StreamPrefill {
            starting_state: UnifiedParserStartingState::Reasoning,
            prefix: Some("<|channel>thou"),
        },
        ..Request::native("gemma4")
    };
    let plain = split("ght\nR<channel|>A", 3);
    let stock = request.run(plain.clone());
    assert!(
        stock.iter().all(|item| !item.is_error()),
        "recording off must not refuse"
    );
    let (input, _) = with_slots(&plain);
    let output = request.run(input);
    assert!(output.iter().all(|item| item.data.is_none()), "{output:?}");
    let error = output.iter().find_map(|item| item.error.as_deref());
    assert!(
        error.is_some_and(|error| error.contains("the gemma4 parser cannot record sources")),
        "{error:?}"
    );
}

fn ends_with_error(output: &[Recorded]) -> bool {
    output.last().is_some_and(|item| item.is_error())
}

fn no_content_passed_through(output: &[Recorded]) -> bool {
    output
        .iter()
        .filter_map(|item| item.data.as_ref())
        .all(|data| {
            data.chunk
                .inner
                .choices
                .iter()
                .all(|choice| text_len(choice.delta.content.as_ref()) == 0)
        })
}

#[test]
fn a_parser_that_cannot_record_fails_the_stream() {
    // No override: the real family parser keeps the trait default and refuses.
    let (input, _) = with_slots(&split("<think>r</think>visible text", 4));
    let output = Request::native("glm47").run(input);
    assert!(ends_with_error(&output), "{output:?}");
    assert!(no_content_passed_through(&output), "{output:?}");
    // The stock path is unchanged: the same input without slots parses.
    let plain = Request::native("glm47").run(split("<think>r</think>visible text", 4));
    assert!(!plain.iter().any(Annotated::is_error));
}

#[test]
fn guided_mode_does_not_record() {
    let _parser = ParserOverride::install(false);
    let request = Request {
        constraint: GuidedToolConstraint::GuidedJsonNamed {
            tool_name: "get_weather".to_string(),
        },
        ..Request::native("glm47")
    };
    let (input, _) = with_slots(&split(r#"{"city":"Paris"}"#, 4));
    let output = request.run(input);
    assert!(ends_with_error(&output), "{output:?}");
    assert!(no_content_passed_through(&output), "{output:?}");
}

#[test]
fn a_parser_that_misreports_its_sources_fails_the_stream() {
    let _parser = ParserOverride::install(true);
    let (input, _) = with_slots(&split("plain text", 3));
    let output = Request::native("glm47").run(input);
    assert!(ends_with_error(&output), "{output:?}");
}

#[test]
fn a_chunk_whose_sources_do_not_cover_its_text_fails_the_stream() {
    let _parser = ParserOverride::install(false);
    let (mut input, _) = with_slots(&split("plain text", 3));
    input[1].data.as_mut().unwrap().slot[0].content = vec![100..101];
    let output = Request::native("glm47").run(input);
    assert!(ends_with_error(&output), "{output:?}");
}

/// The recorded corpus as dynamo-llm's chunks.
fn corpus() -> Vec<Value> {
    let bytes = include_bytes!("../tests/data/unified_stream_parity.jsonl.gz");
    let mut text = String::new();
    flate2::read::GzDecoder::new(&bytes[..])
        .read_to_string(&mut text)
        .expect("decompress fixture");
    text.lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("fixture line"))
        .filter(|case| case["parsers_v2_selected"] == json!(false))
        .collect()
}

fn corpus_item(value: &Value) -> Plain {
    let text = |key: &str| value.get(key).and_then(Value::as_str).map(str::to_string);
    let data = value.get("data").map(|data| {
        let mut data = data.as_object().expect("data object").clone();
        let nvext = data.remove("nvext");
        let llm_metrics = data.remove("llm_metrics").and_then(|v| v.as_u64());
        let prompt_logprobs = data.remove("prompt_logprobs").and_then(|v| v.as_u64());
        let tool_call_completion = data
            .remove("tool_call_completion")
            .map(|v| serde_json::from_value(v).expect("completion evidence"))
            .unwrap_or_default();
        for (key, default) in [
            ("id", json!("chatcmpl-parity")),
            ("object", json!("chat.completion.chunk")),
            ("model", json!("parity-model")),
            ("created", json!(1)),
        ] {
            data.entry(key).or_insert(default);
        }
        NvCreateChatCompletionStreamResponse {
            inner: serde_json::from_value(Value::Object(data)).expect("chunk"),
            nvext,
            prompt_logprobs: prompt_logprobs.map(|len| std::sync::Arc::new(vec![0; len as usize])),
            llm_metrics: llm_metrics.map(|tokens| LLMMetricAnnotation {
                chunk_tokens: tokens as usize,
                ..LLMMetricAnnotation::default()
            }),
            tool_call_completion,
        }
    });
    Annotated {
        data,
        id: text("id"),
        event: text("event"),
        comment: value
            .get("comment")
            .map(|v| serde_json::from_value(v.clone()).expect("comment")),
        error: text("error"),
    }
}

fn corpus_family(name: &str) -> &'static str {
    [
        "deepseek_v4",
        "deepseek_v41",
        "glm47",
        "gpt_oss",
        "qwen3",
        "kimi_k2",
        "kimi_k3",
        "gemma4",
        "muse_glimmer",
    ]
    .into_iter()
    .find(|family| *family == name)
    .expect("known family")
}

fn corpus_request(case: &Value) -> Request {
    const HEADER: &str = "<|channel>thought\n";
    Request {
        family: corpus_family(case["family"].as_str().unwrap()),
        tools: case["tools"].as_bool().unwrap(),
        constraint: serde_json::from_value(case["constraint"].clone()).unwrap(),
        prefill: StreamPrefill {
            starting_state: match case["prefill"].as_str().unwrap() {
                "reasoning" => UnifiedParserStartingState::Reasoning,
                "response" => UnifiedParserStartingState::Response,
                _ => UnifiedParserStartingState::None,
            },
            prefix: case["prefix"]
                .as_str()
                .map(|prefix| &HEADER[..prefix.len()]),
        },
        guided_streaming: case["guided_streaming"].as_bool().unwrap(),
        policy: UnifiedRequestPolicy {
            reasoning_disabled: case["reasoning_disabled"].as_bool().unwrap(),
            structured_response: case["structured_response"].as_bool().unwrap(),
            parsers_v2_selected: false,
        },
    }
}

/// Every corpus stream with the recording test parser: with slots and without,
/// the chunks are the same, and the sources follow the rules. Guided requests
/// must refuse instead.
#[test]
fn the_corpus_records_without_changing_a_chunk() {
    let mut checked = 0;
    let mut refused = 0;
    let mut prefix_refused = 0;
    for case in corpus() {
        let request = corpus_request(&case);
        let input: Vec<Plain> = case["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(corpus_item)
            .collect();
        let name = case["name"].as_str().unwrap();
        if request.constraint.installs_guided_json() {
            let _parser = ParserOverride::install(false);
            let (slotted, _) = with_slots(&input);
            let output = request.run(slotted);
            // A stream that never builds a parser (an early error) has nothing to refuse.
            if output.iter().any(Annotated::is_error) {
                refused += 1;
            }
            continue;
        }
        let _parser = ParserOverride::install(false);
        let expected: Vec<_> = request.run(input.clone()).iter().map(plain_key).collect();
        let (slotted, texts) = with_slots(&input);
        let output = request.run(slotted);
        // A replayed prompt prefix that leaves bytes held refuses to record,
        // before any generated output: those prompt bytes could surface as
        // text, and they have no offset in the generated input. Gemma 4 does
        // this for any prefix short of the whole `<|channel>thought\n` header,
        // and so does the test parser.
        if name.starts_with("gemma_prefix/")
            && output.iter().any(|item| {
                item.error
                    .as_deref()
                    .is_some_and(|error| error.contains("cannot record sources"))
            })
        {
            assert!(
                output.iter().all(|item| item.data.is_none()),
                "{name}: the refusal came after generated output"
            );
            prefix_refused += 1;
            continue;
        }
        let actual: Vec<_> = output.iter().map(without_slot).collect();
        assert_eq!(actual, expected, "{name}: recording changed the chunks");
        if let Err(error) = check_sources(&output, &texts) {
            panic!("{name}: {error}");
        }
        checked += 1;
    }
    assert!(checked > 1000, "{checked} cases checked");
    assert!(refused > 400, "{refused} guided cases refused");
    // Every corpus prefix ends inside the opener, so the test parser holds it.
    assert_eq!(prefix_refused, 16, "{prefix_refused} prefix cases refused");
}

/// The same with the real parsers. It enforces nothing until the
/// `dynamo-parsers-v2` families implement `record_sources`.
#[test]
#[ignore = "needs record_sources in dynamo-parsers-v2"]
fn the_corpus_records_with_the_real_parsers() {
    for case in corpus() {
        let request = corpus_request(&case);
        if request.constraint.installs_guided_json() {
            continue;
        }
        let input: Vec<Plain> = case["input"]
            .as_array()
            .unwrap()
            .iter()
            .map(corpus_item)
            .collect();
        let expected: Vec<_> = request.run(input.clone()).iter().map(plain_key).collect();
        let (slotted, texts) = with_slots(&input);
        let output = request.run(slotted);
        let actual: Vec<_> = output.iter().map(without_slot).collect();
        let name = case["name"].as_str().unwrap();
        assert_eq!(actual, expected, "{name}: recording changed the chunks");
        if let Err(error) = check_sources(&output, &texts) {
            panic!("{name}: {error}");
        }
    }
}

/// A parser that fails to start is not a recording refusal. The stock path
/// passes the choice through unparsed, and so does a recording stream, with
/// the input's exact ranges, so recording does not change the chunks.
#[test]
fn a_parser_that_fails_to_start_passes_through_as_stock_does() {
    struct InitFailure(TagParser);

    impl UnifiedParser for InitFailure {
        fn initialize_request(&mut self, _init: UnifiedParserInit) -> anyhow::Result<()> {
            anyhow::bail!("injected initialization failure")
        }

        fn parse_into(&mut self, delta: &str, out: &mut UnifiedParserOutput) -> anyhow::Result<()> {
            self.0.parse_into(delta, out)
        }

        fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
            self.0.finish()
        }

        fn record_sources(&mut self) -> bool {
            self.0.record_sources()
        }
    }

    let _parser = ParserOverride::install(false);
    TEST_PARSER.with(|factory| {
        *factory.borrow_mut() = Some(Box::new(|_, _| Box::new(InitFailure(TagParser::new()))));
    });
    let plain = vec![text_chunk(
        0,
        Some("<think>private</think>answer"),
        None,
        true,
    )];
    let expected: Vec<_> = Request::native("glm47")
        .run(plain.clone())
        .iter()
        .map(plain_key)
        .collect();
    let (input, texts) = with_slots(&plain);
    let output = Request::native("glm47").run(input);
    let actual: Vec<_> = output.iter().map(without_slot).collect();
    assert_eq!(actual, expected);
    check_sources(&output, &texts).unwrap();
    assert_eq!(
        output[0].data.as_ref().unwrap().slot[0].content,
        vec![0..28]
    );
}

#[test]
fn recording_refuses_in_bounds_ranges_with_the_wrong_bytes() {
    struct ShiftedSources(TagParser);

    impl UnifiedParser for ShiftedSources {
        fn parse_into(&mut self, delta: &str, out: &mut UnifiedParserOutput) -> anyhow::Result<()> {
            self.0.parse_into(delta, out)
        }

        fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
            self.0.finish()
        }

        fn record_sources(&mut self) -> bool {
            self.0.record_sources()
        }

        fn take_sources(&mut self, out: &mut Vec<Range<usize>>) {
            self.0.take_sources(out);
            for range in out {
                if range.start > 0 {
                    range.start -= 1;
                    range.end -= 1;
                }
            }
        }
    }

    let _parser = ParserOverride::install(false);
    TEST_PARSER.with(|factory| {
        *factory.borrow_mut() = Some(Box::new(|_, _| Box::new(ShiftedSources(TagParser::new()))));
    });
    let (input, texts) = with_slots(&[
        text_chunk(0, Some("a"), None, false),
        text_chunk(0, Some("b"), None, true),
    ]);
    let output = Request::native("glm47").run(input);
    assert!(
        ends_with_error(&output),
        "recording accepted a range for the wrong byte: {:?}; {output:?}",
        check_sources(&output, &texts)
    );
    check_sources(&output, &texts).unwrap();
}

#[test]
fn usage_flush_checks_sources_before_emitting_held_text() {
    for usage_flush in [false, true] {
        let _parser = ParserOverride::install(true);
        let mut input = vec![text_chunk(0, Some("<"), None, false)];
        if usage_flush {
            let mut usage = chunk("", false);
            let data = usage.data.as_mut().unwrap();
            data.inner.choices.clear();
            data.inner.usage = Some(
                serde_json::from_value(json!({
                    "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
                }))
                .unwrap(),
            );
            input.push(usage);
        }
        let (input, texts) = with_slots(&input);
        let output = Request::native("glm47").run(input);
        assert!(
            ends_with_error(&output),
            "usage_flush={usage_flush}: {output:?}"
        );
        assert!(
            check_sources(&output, &texts).is_ok(),
            "usage_flush={usage_flush}: malformed text was emitted before the error: {:?}; {output:?}",
            check_sources(&output, &texts)
        );
        assert!(no_content_passed_through(&output), "{output:?}");
    }
}

#[test]
fn stock_qwen_restoration_allocates_only_the_choice_and_envelope() {
    let mut item = text_chunk(0, None, Some("queued"), false);
    let mut data = item.data.take().unwrap();
    let choice = data.inner.choices.remove(0);
    let mut record = ChoiceRecord::<Plain>::default();
    record
        .pending_qwen_opener_responses
        .push_back(PendingQwenReasoningResponse {
            text: "queued".to_string(),
            spans: Spans::new(),
            choice,
            data,
            annotation: (None, None, None, None),
        });

    use crate::test_allocations::{allocations, bytes};
    let before = (allocations(), bytes());
    let restored = take_qwen_reasoning_response(&mut record);
    let observed = (allocations() - before.0, bytes() - before.1);
    let expected_bytes = std::mem::size_of::<ChatChoiceStream>()
        + std::mem::size_of::<(NvCreateChatCompletionStreamResponse, AnnotationParts)>();
    assert_eq!(
        observed,
        (2, expected_bytes),
        "restoring a chunk without a source slot must not allocate a source vector"
    );
    assert_eq!(
        restored.0.inner.choices[0]
            .delta
            .reasoning_content
            .as_deref(),
        Some("queued")
    );
}

#[test]
fn reset_recovery_keeps_sources_and_matches_the_stock_chunks() {
    struct FailsAfterHold(TagParser);

    impl UnifiedParser for FailsAfterHold {
        fn parse_into(&mut self, delta: &str, out: &mut UnifiedParserOutput) -> anyhow::Result<()> {
            if delta == "!" {
                anyhow::bail!("injected failure while a marker is held")
            }
            self.0.parse_into(delta, out)
        }

        fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
            self.0.finish()
        }

        fn reset(&mut self) -> String {
            self.0.reset()
        }

        fn record_sources(&mut self) -> bool {
            self.0.record_sources()
        }

        fn take_sources(&mut self, out: &mut Vec<Range<usize>>) {
            self.0.take_sources(out)
        }
    }

    let _parser = ParserOverride::install(false);
    TEST_PARSER.with(|factory| {
        *factory.borrow_mut() = Some(Box::new(|_, _| Box::new(FailsAfterHold(TagParser::new()))));
    });
    let input = vec![
        text_chunk(0, Some("a<"), None, false),
        text_chunk(0, Some("!"), None, false),
        text_chunk(0, Some("tail"), None, true),
    ];
    let request = Request::native("glm47");
    let expected: Vec<_> = request.run(input.clone()).iter().map(plain_key).collect();
    let (input, texts) = with_slots(&input);
    let output = request.run(input);
    assert_eq!(
        output.iter().map(without_slot).collect::<Vec<_>>(),
        expected,
        "recording changed parser-error recovery"
    );
    check_sources(&output, &texts).unwrap();
    let content: Vec<_> = output
        .iter()
        .filter_map(|item| item.data.as_ref())
        .flat_map(|data| data.slot.iter())
        .flat_map(|sources| sources.content.iter().cloned())
        .collect();
    assert_eq!(content, vec![0..1, 1..2, 3..7]);
}

#[test]
fn enabling_sources_before_usage_flush_refuses_undated_held_text() {
    #[derive(Clone)]
    struct OptionalSources {
        data: Slotted,
        enabled: bool,
    }

    impl SourceSlot for OptionalSources {
        fn sources(&self) -> Option<&[ChoiceSources]> {
            self.enabled.then_some(self.data.slot.as_slice())
        }

        fn sources_mut(&mut self) -> Option<&mut Vec<ChoiceSources>> {
            self.enabled.then_some(&mut self.data.slot)
        }
    }

    impl UnifiedStreamChunk for OptionalSources {
        fn inner(&self) -> &CreateChatCompletionStreamResponse {
            self.data.inner()
        }

        fn inner_mut(&mut self) -> &mut CreateChatCompletionStreamResponse {
            self.data.inner_mut()
        }

        fn tool_call_completion(&self) -> &[ToolCallCompletion] {
            self.data.tool_call_completion()
        }

        fn tool_call_completion_mut(&mut self) -> &mut Vec<ToolCallCompletion> {
            self.data.tool_call_completion_mut()
        }

        fn clear_nvext(&mut self) {
            self.data.clear_nvext();
        }

        fn clear_llm_metrics(&mut self) {
            self.data.clear_llm_metrics();
        }

        fn scrub_synthetic(&mut self) {
            self.data.scrub_synthetic();
        }
    }

    let _parser = ParserOverride::install(false);
    let mut usage = chunk("", false);
    let data = usage.data.as_mut().unwrap();
    data.inner.choices.clear();
    data.inner.usage = Some(
        serde_json::from_value(json!({
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
        }))
        .unwrap(),
    );
    let (input, _) = with_slots(&[text_chunk(0, Some("<"), None, false), usage]);
    let input = input
        .into_iter()
        .enumerate()
        .map(|(position, item)| Annotated {
            data: item.data.map(|data| OptionalSources {
                data,
                enabled: position != 0,
            }),
            id: item.id,
            event: item.event,
            comment: item.comment,
            error: item.error,
        })
        .collect();
    let output = Request::native("glm47").run(input);
    let emitted: Vec<_> = output
        .iter()
        .filter_map(|item| item.data.as_ref())
        .map(|data| (&data.data.chunk.inner.choices, data.sources()))
        .collect();
    assert!(
        output.last().is_some_and(Annotated::is_error),
        "enabling recording emitted held text without sources: {emitted:?}"
    );
    assert!(
        output
            .iter()
            .filter_map(|item| item.data.as_ref())
            .all(|data| {
                data.inner()
                    .choices
                    .iter()
                    .all(|choice| text_len(choice.delta.content.as_ref()) == 0)
            })
    );
}

#[test]
fn reset_ranges_for_other_bytes_are_refused_before_emission() {
    struct WrongResetSources {
        parser: TagParser,
        reset: bool,
    }

    impl UnifiedParser for WrongResetSources {
        fn parse_into(&mut self, delta: &str, out: &mut UnifiedParserOutput) -> anyhow::Result<()> {
            if delta == "!" {
                anyhow::bail!("injected failure while a marker is held")
            }
            self.parser.parse_into(delta, out)
        }

        fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
            self.parser.finish()
        }

        fn reset(&mut self) -> String {
            self.reset = true;
            self.parser.reset()
        }

        fn record_sources(&mut self) -> bool {
            self.parser.record_sources()
        }

        fn take_sources(&mut self, out: &mut Vec<Range<usize>>) {
            self.parser.take_sources(out);
            if self.reset {
                for range in out {
                    range.start -= 1;
                    range.end -= 1;
                }
            }
        }
    }

    let _parser = ParserOverride::install(false);
    TEST_PARSER.with(|factory| {
        *factory.borrow_mut() = Some(Box::new(|_, _| {
            Box::new(WrongResetSources {
                parser: TagParser::new(),
                reset: false,
            })
        }));
    });
    let (input, texts) = with_slots(&[
        text_chunk(0, Some("a<"), None, false),
        text_chunk(0, Some("!"), None, false),
        text_chunk(0, Some("tail"), None, true),
    ]);
    let output = Request::native("glm47").run(input);
    assert!(
        ends_with_error(&output),
        "reset accepted 0..1 (a) for recovered <; source check: {:?}",
        check_sources(&output, &texts)
    );
    check_sources(&output, &texts).unwrap();
}

#[test]
fn disabling_sources_before_usage_flush_refuses_undated_held_text() {
    #[derive(Clone)]
    struct OptionalSources {
        data: Slotted,
        enabled: bool,
    }

    impl SourceSlot for OptionalSources {
        fn sources(&self) -> Option<&[ChoiceSources]> {
            self.enabled.then_some(self.data.slot.as_slice())
        }

        fn sources_mut(&mut self) -> Option<&mut Vec<ChoiceSources>> {
            self.enabled.then_some(&mut self.data.slot)
        }
    }

    impl UnifiedStreamChunk for OptionalSources {
        fn inner(&self) -> &CreateChatCompletionStreamResponse {
            self.data.inner()
        }

        fn inner_mut(&mut self) -> &mut CreateChatCompletionStreamResponse {
            self.data.inner_mut()
        }

        fn tool_call_completion(&self) -> &[ToolCallCompletion] {
            self.data.tool_call_completion()
        }

        fn tool_call_completion_mut(&mut self) -> &mut Vec<ToolCallCompletion> {
            self.data.tool_call_completion_mut()
        }

        fn clear_nvext(&mut self) {
            self.data.clear_nvext();
        }

        fn clear_llm_metrics(&mut self) {
            self.data.clear_llm_metrics();
        }

        fn scrub_synthetic(&mut self) {
            self.data.scrub_synthetic();
        }
    }

    let _parser = ParserOverride::install(false);
    let mut usage = chunk("", false);
    let data = usage.data.as_mut().unwrap();
    data.inner.choices.clear();
    data.inner.usage = Some(
        serde_json::from_value(json!({
            "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2
        }))
        .unwrap(),
    );
    let (input, _) = with_slots(&[text_chunk(0, Some("<"), None, false), usage]);
    let input = input
        .into_iter()
        .enumerate()
        .map(|(position, item)| Annotated {
            data: item.data.map(|data| OptionalSources {
                data,
                enabled: position == 0,
            }),
            id: item.id,
            event: item.event,
            comment: item.comment,
            error: item.error,
        })
        .collect();
    let output = Request::native("glm47").run(input);
    let emitted: Vec<_> = output
        .iter()
        .filter_map(|item| item.data.as_ref())
        .map(|data| (&data.data.chunk.inner.choices, data.sources()))
        .collect();
    assert!(
        output.last().is_some_and(Annotated::is_error),
        "disabling recording emitted held text without a slot: {emitted:?}"
    );
    assert!(
        output
            .iter()
            .filter_map(|item| item.data.as_ref())
            .all(|data| {
                data.sources().is_some()
                    && data
                        .inner()
                        .choices
                        .iter()
                        .all(|choice| text_len(choice.delta.content.as_ref()) == 0)
            })
    );
}

/// Exact ranges need not move forward through the input: the held `<` is
/// released after the input reasoning `B` that follows it.
#[test]
fn the_checker_accepts_exact_ranges_out_of_input_order() {
    let _parser = ParserOverride::install(false);
    let request = Request::native("qwen3");
    let plain = vec![
        text_chunk(0, Some("<think><"), None, false),
        text_chunk(0, Some("C</think>D"), Some("B"), true),
    ];
    let stock: Vec<_> = request.run(plain.clone()).iter().map(plain_key).collect();
    let (input, texts) = with_slots(&plain);
    let output = request.run(input);
    assert_eq!(output.iter().map(without_slot).collect::<Vec<_>>(), stock);
    assert_eq!(&texts[&0], "<think><BC</think>D");
    let entries: Vec<_> = output
        .iter()
        .filter_map(|item| item.data.as_ref())
        .flat_map(|data| &data.slot)
        .collect();
    let reasoning: Vec<_> = entries
        .iter()
        .flat_map(|entry| entry.reasoning.iter().cloned())
        .collect();
    let content: Vec<_> = entries
        .iter()
        .flat_map(|entry| entry.content.iter().cloned())
        .collect();
    assert_eq!(reasoning, vec![8..9, 7..8, 9..10]);
    assert_eq!(content, vec![18..19]);
    assert_eq!(read(&texts[&0], &reasoning).as_deref(), Some("B<C"));
    check_sources(&output, &texts).unwrap();
}

/// A range that reads back the right text from the wrong place: `think` dated
/// inside the stripped `<think>` opener.
#[test]
#[ignore = "the checker reads text back, so it cannot tell which occurrence a range names; needs a per-byte oracle"]
fn the_checker_rejects_a_range_from_the_wrong_place() {
    let _parser = ParserOverride::install(false);
    let (input, texts) = with_slots(&[text_chunk(0, Some("<think>think</think>"), None, true)]);
    let mut output = Request::native("qwen3").run(input);
    check_sources(&output, &texts).unwrap();
    assert_eq!(&texts[&0][1..6], "think");
    assert_eq!(&texts[&0][7..12], "think");
    let mut changed = 0;
    for data in output.iter_mut().filter_map(|item| item.data.as_mut()) {
        for entry in &mut data.slot {
            for range in &mut entry.reasoning {
                assert_eq!(*range, 7..12);
                *range = 1..6;
                changed += 1;
            }
        }
    }
    assert_eq!(changed, 1);
    assert!(
        check_sources(&output, &texts).is_err(),
        "accepted stripped opener 1..6 instead of reasoning 7..12"
    );
}

/// Extra allocations per chunk that recording may cost in the adapter, with
/// the recording test parser: its own range list, and the entry list and
/// range lists that each output chunk carries. It was 11.2.
const ADAPTER_EXTRA_PER_CHUNK: f64 = 8.0;

/// Recording through the adapter costs few allocations per chunk. Each event's
/// range list was a vector, the per-push event lists of ranges were vectors,
/// and the ranges kept for the choices were taken out on every chunk, so
/// recording allocated about four times per chunk in the adapter alone.
#[test]
fn recording_through_the_adapter_allocates_little_per_chunk() {
    use crate::test_allocations::allocations;
    let _parser = ParserOverride::install(false);
    let request = Request::native("glm47");
    let reply = "<think>The user asks about the weather, so I check it.</think>Let me check the weather for you, then answer.";
    let plain = split(reply, 1);
    let chunks = plain.len();
    request.run(plain.clone());
    let input = plain.clone();
    let before = allocations();
    std::hint::black_box(request.run(input));
    let off = allocations() - before;
    let (input, _) = with_slots(&plain);
    request.run(input);
    let (input, _) = with_slots(&plain);
    let before = allocations();
    std::hint::black_box(request.run(input));
    let on = allocations() - before;
    let extra = (on as f64 - off as f64) / chunks as f64;
    println!("adapter: off {off}, on {on}, {extra:.3} extra per chunk over {chunks} chunks");
    assert!(
        extra <= ADAPTER_EXTRA_PER_CHUNK,
        "recording costs {extra:.3} allocations per chunk, budget {ADAPTER_EXTRA_PER_CHUNK}"
    );
}
