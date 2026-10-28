// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! dynamo-llm's unit tests for the v2 stream adapter, moved with it.
//!
//! They run on stand-ins for dynamo's stream types, with the same field names:
//! dynamo-runtime's `Annotated` and dynamo-llm's
//! `NvCreateChatCompletionStreamResponse` and `LLMMetricAnnotation`. Tests of
//! the batch path, the route choice, and `DYN_PARSER_VERSION` stay in dynamo.

/// dynamo-runtime's `Annotated`, with a string error.
#[derive(Clone, Debug)]
struct Annotated<R> {
    data: Option<R>,
    id: Option<String>,
    event: Option<String>,
    comment: Option<Vec<String>>,
    error: Option<String>,
}

impl<R> Annotated<R> {
    fn from_data(data: R) -> Self {
        Self {
            data: Some(data),
            id: None,
            event: None,
            comment: None,
            error: None,
        }
    }

    fn from_error(error: impl Into<String>) -> Self {
        Self {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: None,
            error: Some(error.into()),
        }
    }

    fn is_error(&self) -> bool {
        self.event.as_deref() == Some("error")
    }
}

type AnnotationParts = (
    Option<String>,
    Option<String>,
    Option<Vec<String>>,
    Option<String>,
);

impl<C: UnifiedStreamChunk> UnifiedStreamItem for Annotated<C> {
    type Chunk = C;
    type Annotation = AnnotationParts;

    fn is_error(&self) -> bool {
        Annotated::is_error(self)
    }

    fn into_parts(self) -> Result<(C, AnnotationParts), Self> {
        match self.data {
            Some(data) => Ok((data, (self.id, self.event, self.comment, self.error))),
            None => Err(self),
        }
    }

    fn from_parts(data: C, (id, event, comment, error): AnnotationParts) -> Self {
        Self {
            data: Some(data),
            id,
            event,
            comment,
            error,
        }
    }

    fn from_error(message: String) -> Self {
        Annotated::from_error(message)
    }
}

/// The fields of dynamo-llm's `LLMMetricAnnotation` that these tests set.
#[derive(Clone, Debug, Default, PartialEq)]
struct LLMMetricAnnotation {
    input_tokens: usize,
    output_tokens: usize,
    chunk_tokens: usize,
    cached_tokens: Option<usize>,
}

/// dynamo-llm's `NvCreateChatCompletionStreamResponse`.
#[derive(Clone, Debug, PartialEq)]
struct NvCreateChatCompletionStreamResponse {
    inner: CreateChatCompletionStreamResponse,
    nvext: Option<serde_json::Value>,
    prompt_logprobs: Option<std::sync::Arc<Vec<u32>>>,
    llm_metrics: Option<LLMMetricAnnotation>,
    tool_call_completion: Vec<ToolCallCompletion>,
}

// dynamo's chunk type records nothing: it keeps the trait defaults.
impl dynamo_parsers::sources::SourceSlot for NvCreateChatCompletionStreamResponse {}

impl UnifiedStreamChunk for NvCreateChatCompletionStreamResponse {
    fn inner(&self) -> &CreateChatCompletionStreamResponse {
        &self.inner
    }

    fn inner_mut(&mut self) -> &mut CreateChatCompletionStreamResponse {
        &mut self.inner
    }

    fn tool_call_completion(&self) -> &[ToolCallCompletion] {
        &self.tool_call_completion
    }

    fn tool_call_completion_mut(&mut self) -> &mut Vec<ToolCallCompletion> {
        &mut self.tool_call_completion
    }

    fn clear_nvext(&mut self) {
        self.nvext = None;
    }

    fn clear_llm_metrics(&mut self) {
        self.llm_metrics = None;
    }

    fn scrub_synthetic(&mut self) {
        self.inner.usage = None;
        self.llm_metrics = None;
        self.tool_call_completion.clear();
        self.nvext = None;
        self.prompt_logprobs = None;
    }
}

use super::*;
use crate::families::{
    DEEPSEEK_V4_UNIFIED_FAMILY, GEMMA4_UNIFIED_FAMILY, KIMI_K2_UNIFIED_FAMILY,
    KIMI_K3_UNIFIED_FAMILY,
};
use dynamo_parsers_v2::{ToolCallDelta, UnifiedEvent};
use dynamo_protocols::types::{
    ChatCompletionStreamResponseDeltaFunctionCall, CreateChatCompletionStreamResponse, Role,
};
use futures::FutureExt;
use futures::stream;

/// `qwen_chunks_for_choices` builds one `FinishedStreamChunk` per parsed choice,
/// for every family. dynamo boxed the rare queued-source payload, so an ordinary
/// choice cost no more than a `ChatChoiceStream`; the move must not grow it.
#[test]
fn ordinary_fanout_keeps_the_source_adapter_allocation_size() {
    #[allow(dead_code)]
    enum SourceAdapterChunk {
        Choice(ChatChoiceStream),
        Source(Annotated<Box<NvCreateChatCompletionStreamResponse>>),
    }

    let state = ChoiceState::new(
        QWEN3_UNIFIED_FAMILY,
        &[],
        UnifiedParserStartingState::None,
        UnifiedToolOutputMode::Native,
        false,
        false,
    )
    .unwrap();
    let mut record = ChoiceRecord::<Annotated<NvCreateChatCompletionStreamResponse>>::default();
    let choices = vec![empty_choice(0)];
    let chunks = qwen_chunks_for_choices(&state, &mut record, choices, None);
    assert!(matches!(
        chunks.as_slice(),
        [FinishedStreamChunk::Choice(_)]
    ));
    let allocated = std::mem::size_of_val(chunks.as_slice());
    let source_allocated = std::mem::size_of::<SourceAdapterChunk>();
    assert!(
        allocated <= source_allocated,
        "one ordinary choice now needs {allocated} bytes; the source adapter needs {source_allocated} bytes"
    );
}

struct PartialCommitParser {
    recovered: String,
}

impl UnifiedParser for PartialCommitParser {
    fn parse_into(&mut self, _delta: &str, output: &mut UnifiedParserOutput) -> anyhow::Result<()> {
        output
            .events
            .push(UnifiedParserEvent::Text("committed".to_string()));
        anyhow::bail!("intentional failure after a committed event")
    }

    fn finish(&mut self) -> anyhow::Result<UnifiedParserOutput> {
        Ok(UnifiedParserOutput::default())
    }

    fn reset(&mut self) -> String {
        std::mem::take(&mut self.recovered)
    }
}

fn chunk(text: &str, finish: bool) -> Annotated<NvCreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let response = NvCreateChatCompletionStreamResponse {
        inner: CreateChatCompletionStreamResponse {
            id: "test".to_string(),
            choices: vec![ChatChoiceStream {
                index: 0,
                delta: ChatCompletionStreamResponseDelta {
                    role: Some(Role::Assistant),
                    content: Some(ChatCompletionMessageContent::Text(text.to_string())),
                    tool_calls: None,
                    function_call: None,
                    refusal: None,
                    reasoning_content: None,
                },
                finish_reason: finish.then_some(FinishReason::Stop),
                logprobs: None,
            }],
            created: 0,
            model: "qwen3".to_string(),
            system_fingerprint: None,
            service_tier: None,
            object: "chat.completion.chunk".to_string(),
            usage: None,
        },
        nvext: None,
        llm_metrics: None,
        prompt_logprobs: None,
        tool_call_completion: Vec::new(),
    };
    Annotated::from_data(response)
}

fn weather_tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "get_weather".to_string(),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        })),
        strict: None,
    }]
}

fn count_tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "count_items".to_string(),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"]
        })),
        strict: None,
    }]
}

fn collect_choices(
    responses: &[Annotated<NvCreateChatCompletionStreamResponse>],
) -> Vec<&ChatChoiceStream> {
    responses
        .iter()
        .filter_map(|response| response.data.as_ref())
        .flat_map(|data| data.inner.choices.iter())
        .collect()
}

fn named_choice(name: &str) -> ChatCompletionToolChoiceOption {
    serde_json::from_value(serde_json::json!({
        "type": "function",
        "function": {"name": name}
    }))
    .expect("named tool choice")
}

#[derive(Debug, PartialEq, Eq)]
enum LogicalEvent {
    Reasoning(String),
    Text(String),
    Tool {
        index: u32,
        name: Option<String>,
        arguments: String,
    },
}

fn logical_events(
    responses: &[Annotated<NvCreateChatCompletionStreamResponse>],
) -> Vec<LogicalEvent> {
    let mut events = Vec::new();
    for choice in collect_choices(responses) {
        if let Some(reasoning) = &choice.delta.reasoning_content {
            match events.last_mut() {
                Some(LogicalEvent::Reasoning(existing)) => existing.push_str(reasoning),
                _ => events.push(LogicalEvent::Reasoning(reasoning.clone())),
            }
        }
        if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
            match events.last_mut() {
                Some(LogicalEvent::Text(existing)) => existing.push_str(text),
                _ => events.push(LogicalEvent::Text(text.clone())),
            }
        }
        for call in choice.delta.tool_calls.iter().flatten() {
            let name = call.function.as_ref().and_then(|f| f.name.clone());
            let arguments = call
                .function
                .as_ref()
                .and_then(|f| f.arguments.clone())
                .unwrap_or_default();
            match events.last_mut() {
                Some(LogicalEvent::Tool {
                    index,
                    name: existing_name,
                    arguments: existing_arguments,
                }) if *index == call.index => {
                    if existing_name.is_none() {
                        *existing_name = name;
                    }
                    existing_arguments.push_str(&arguments);
                }
                _ => events.push(LogicalEvent::Tool {
                    index: call.index,
                    name,
                    arguments,
                }),
            }
        }
    }
    events
}

async fn parse_at_split(
    input: &str,
    split: usize,
    tool_choice: Option<ChatCompletionToolChoiceOption>,
) -> Vec<LogicalEvent> {
    let (first, second) = input.split_at(split);
    let responses = apply_stream(
        stream::iter([chunk(first, false), chunk(second, true)]),
        Some(weather_tools()),
        tool_choice,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    logical_events(&responses)
}

// --- selected_family / configured_family -------------------------------------

#[tokio::test]
async fn deepseek_v41_recovers_malformed_closed_arguments_as_text() {
    let text = concat!(
        "<｜DSML｜ calls><｜DSML｜ invoke name=\"get_weather\">",
        "<｜DSML｜ parameter name=\"count\" string=\"false\">not-json</｜DSML｜ parameter>",
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    );
    let responses = apply_stream(
        stream::iter([chunk(text, true)]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::Response,
        DEEPSEEK_V41_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    assert!(responses.iter().all(|response| !response.is_error()));
    assert_eq!(
        logical_events(&responses),
        vec![LogicalEvent::Text(text.to_string())]
    );
    assert_eq!(
        collect_choices(&responses).last().unwrap().finish_reason,
        Some(FinishReason::Stop)
    );
}

// --- tool_choice -> UnifiedToolOutputMode ------------------------------------

#[test]
fn maps_tool_choice_onto_the_output_mode() {
    assert_eq!(
        tool_output_mode(&GuidedToolConstraint::GuidedJsonNamed {
            tool_name: "get_weather".to_string(),
        }),
        UnifiedToolOutputMode::GuidedJson {
            named_tool: Some("get_weather".to_string())
        }
    );
    assert_eq!(
        tool_output_mode(&GuidedToolConstraint::GuidedJsonRequired),
        UnifiedToolOutputMode::GuidedJson { named_tool: None }
    );
    for native in [
        GuidedToolConstraint::None,
        GuidedToolConstraint::StructuralTag,
    ] {
        assert_eq!(
            tool_output_mode(&native),
            UnifiedToolOutputMode::Native,
            "{native:?} must stay on native markup"
        );
    }
}

#[tokio::test]
async fn every_valid_utf8_split_matches_whole_input_across_output_modes() {
    let cases = [
        (
            concat!(
                "<think>理由は東京です。</think>",
                "<tool_call>\n<function=get_weather>\n",
                "<parameter=city>東京</parameter>\n</function>\n</tool_call>"
            ),
            Some(ChatCompletionToolChoiceOption::Auto),
        ),
        (
            r#"{"city":"東</think>京"}"#,
            Some(named_choice("get_weather")),
        ),
        (
            r#"[{"name":"get_weather","parameters":{"city":"東京"}}]"#,
            Some(ChatCompletionToolChoiceOption::Required),
        ),
    ];

    for (input, tool_choice) in cases {
        let expected = parse_at_split(input, 0, tool_choice.clone()).await;
        for split in (0..=input.len()).filter(|split| input.is_char_boundary(*split)) {
            assert_eq!(
                parse_at_split(input, split, tool_choice.clone()).await,
                expected,
                "output changed at UTF-8 split {split} for {input:?}"
            );
        }
    }
}

#[tokio::test]
async fn structural_tag_keeps_required_on_native_markup() {
    // A structural tag constrains generation to Qwen's XML, so `required` must NOT
    // put the parser into guided-JSON mode; doing so would surface the whole call
    // as text.
    let output = concat!(
        "<think>reason</think>",
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let responses = apply_stream(
        stream::iter([chunk(output, true)]),
        Some(weather_tools()),
        Some(ChatCompletionToolChoiceOption::Required),
        true,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    let call = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .next()
        .expect("a native tool call");
    let function = call.function.as_ref().expect("function");
    assert_eq!(function.name.as_deref(), Some("get_weather"));
    assert_eq!(function.arguments.as_deref(), Some(r#"{"city":"Tokyo"}"#));
}

// --- prefill -----------------------------------------------------------------

#[test]
fn prompt_injected_reasoning_selects_the_reasoning_prefill() {
    assert_eq!(
        stream_prefill(QWEN3_UNIFIED_FAMILY, true),
        UnifiedParserStartingState::Reasoning
    );
    // Qwen3's template pre-opens no channel, so the model emits `<think>` itself.
    assert_eq!(
        stream_prefill(QWEN3_UNIFIED_FAMILY, false),
        UnifiedParserStartingState::None
    );
}

#[test]
fn detects_batch_prefill_from_complete_output() {
    assert_eq!(
        detect_prefill(QWEN3_UNIFIED_FAMILY, "<think>reason</think>answer").unwrap(),
        UnifiedParserStartingState::None
    );
    assert_eq!(
        detect_prefill(QWEN3_UNIFIED_FAMILY, "reason</think>answer").unwrap(),
        UnifiedParserStartingState::Reasoning
    );
    assert_eq!(
        detect_prefill(QWEN3_UNIFIED_FAMILY, "answer").unwrap(),
        UnifiedParserStartingState::Response
    );
    assert_eq!(
        detect_prefill(QWEN3_UNIFIED_FAMILY, "It's hidden</think>visible").unwrap(),
        UnifiedParserStartingState::Reasoning,
        "an apostrophe in prose must not hide a later control marker"
    );
    assert!(detect_prefill("unknown", "answer").is_err());
}

#[test]
fn detect_prefill_compares_first_marker_positions_not_mere_presence() {
    // A prompt-opened reasoning span that the model closes, followed by the model
    // opening (and closing) a SECOND thought later in the same output. The leading
    // `</think>` has no opener before it, so the prompt pre-opened reasoning: this
    // must return `Reasoning`, not `None`. Presence-only testing ("does <think>
    // exist anywhere") wrongly answers `None` because a `<think>` exists somewhere
    // in the string, even though it comes AFTER the first `</think>`.
    assert_eq!(
        detect_prefill(
            QWEN3_UNIFIED_FAMILY,
            "secret</think>answer<think>second thought</think>tail"
        )
        .unwrap(),
        UnifiedParserStartingState::Reasoning,
        "a leading closer with no prior opener means the prompt pre-opened reasoning, \
         regardless of a later model-opened thought"
    );
}

#[test]
fn apostrophe_exemption_needs_only_one_alphanumeric_side() {
    // A possessive apostrophe that ends a word (`James'`) has non-alphanumeric on
    // the RIGHT. Requiring alphanumeric on both sides treats it as a real
    // quote-open, and the later contraction apostrophe in `isn't` then closes that
    // phantom span, hiding the real `</think>` marker in between.
    assert_eq!(
        detect_prefill(
            QWEN3_UNIFIED_FAMILY,
            "James' secret </think>answer isn't final"
        )
        .unwrap(),
        UnifiedParserStartingState::Reasoning,
        "a trailing possessive apostrophe must not be treated as a real quote-open"
    );
    // Mirror case: a leading contraction (`'twas`) has non-alphanumeric on the LEFT
    // and alphanumeric only on the right — the OR-based fix must exempt this too.
    assert!(
        contains_unquoted_marker("'twas </think> hidden", "</think>"),
        "a leading-apostrophe contraction must not be treated as a real quote-open"
    );
}

// --- delta -> chunk conversion -----------------------------------------------

fn call_delta(tool_index: usize, name: Option<&str>, arguments: &str) -> UnifiedParserEvent {
    UnifiedParserEvent::ToolCall(ToolCallDelta {
        tool_index,
        name: name.map(str::to_string),
        arguments: arguments.to_string(),
        complete: true,
    })
}

fn partial_call_delta(
    tool_index: usize,
    name: Option<&str>,
    arguments: &str,
) -> UnifiedParserEvent {
    UnifiedParserEvent::ToolCall(ToolCallDelta {
        tool_index,
        name: name.map(str::to_string),
        arguments: arguments.to_string(),
        complete: false,
    })
}

fn test_state() -> ChoiceState {
    ChoiceState::new(
        QWEN3_UNIFIED_FAMILY,
        &[],
        UnifiedParserStartingState::None,
        UnifiedToolOutputMode::Native,
        true,
        false,
    )
    .expect("qwen3 unified parser")
}

#[test]
fn each_delta_becomes_its_own_chunk_in_order() {
    // The whole point of this path: a thought that followed a call stays after it.
    let mut state = test_state();
    let base = empty_choice(0);
    let choices = state.choices_for(
        &base,
        vec![
            UnifiedParserEvent::Reasoning("look it up".into()),
            call_delta(0, Some("get_weather"), r#"{"city":"Tokyo"}"#),
            UnifiedParserEvent::Reasoning("now answer".into()),
            UnifiedParserEvent::Text("It's 18C.".into()),
        ],
        true,
        Some(FinishReason::Stop),
    );

    assert_eq!(choices.len(), 4, "one chunk per delta: {choices:?}");
    assert_eq!(
        choices[0].delta.reasoning_content.as_deref(),
        Some("look it up")
    );
    assert!(choices[1].delta.tool_calls.is_some());
    assert_eq!(
        choices[2].delta.reasoning_content.as_deref(),
        Some("now answer"),
        "the second thought must stay AFTER the call, not fuse with the first"
    );
    assert_eq!(
        choices[3].delta.content,
        Some(ChatCompletionMessageContent::Text("It's 18C.".into()))
    );
    // Only the last chunk terminates, and Stop became ToolCalls.
    assert_eq!(choices[3].finish_reason, Some(FinishReason::ToolCalls));
    assert!(choices[..3].iter().all(|c| c.finish_reason.is_none()));
}

#[test]
fn mixed_raw_content_and_upstream_tool_call_do_not_reuse_tool_index() {
    let mut state = test_state();
    let mut original = empty_choice(0);
    original.delta.content = Some(ChatCompletionMessageContent::Text(
        "raw parser input".to_string(),
    ));
    original.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("upstream-call".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("upstream".to_string()),
            arguments: Some("{}".to_string()),
        }),
        ..Default::default()
    }]);

    let choices = state.choices_for(
        &original,
        vec![call_delta(0, Some("parsed"), "{}")],
        true,
        None,
    );
    let mut indices: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .map(|call| call.index)
        .collect();
    indices.sort_unstable();

    assert_eq!(indices, vec![0, 1]);

    let mut upstream_continuation = empty_choice(0);
    upstream_continuation.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: None,
        r#type: None,
        function: Some(FunctionCallStream {
            name: None,
            arguments: Some("more".to_string()),
        }),
        ..Default::default()
    }]);
    let choices = state.choices_for(&upstream_continuation, Vec::new(), true, None);
    assert_eq!(
        choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
        0,
        "the pass-through call keeps its remapped index across chunks"
    );

    let choices = state.choices_for(
        &empty_choice(0),
        vec![call_delta(0, None, "more")],
        true,
        None,
    );
    assert_eq!(
        choices[0].delta.tool_calls.as_ref().unwrap()[0].index,
        1,
        "the parser call keeps its index while the pass-through call continues"
    );
}

#[test]
fn first_tool_chunk_opens_the_call_and_later_ones_only_add_arguments() {
    let mut state = test_state();
    let base = empty_choice(7);
    let choices = state.choices_for(
        &base,
        vec![
            call_delta(0, Some("get_weather"), r#"{"city":"#),
            call_delta(0, None, r#""Tokyo"}"#),
        ],
        true,
        None,
    );

    assert_eq!(choices.len(), 2);
    let first = &choices[0].delta.tool_calls.as_ref().unwrap()[0];
    assert_eq!(choices[0].index, 7, "the choice index is preserved");
    assert_eq!(first.index, 0, "the tool index is preserved");
    assert!(first.id.is_some(), "the opening chunk mints an id");
    assert_eq!(first.r#type, Some(FunctionType::Function));
    assert_eq!(
        first.function.as_ref().unwrap().name.as_deref(),
        Some("get_weather")
    );

    let second = &choices[1].delta.tool_calls.as_ref().unwrap()[0];
    assert!(second.id.is_none(), "only the first chunk carries an id");
    assert!(second.r#type.is_none());
    assert!(second.function.as_ref().unwrap().name.is_none());
    assert_eq!(
        second.function.as_ref().unwrap().arguments.as_deref(),
        Some(r#""Tokyo"}"#)
    );
}

#[test]
fn two_calls_get_distinct_ids_and_keep_their_indices() {
    let mut state = test_state();
    let base = empty_choice(0);
    let choices = state.choices_for(
        &base,
        vec![
            call_delta(0, Some("a"), "{}"),
            call_delta(1, Some("b"), "{}"),
        ],
        true,
        None,
    );

    let ids: Vec<_> = choices
        .iter()
        .map(|choice| {
            let call = &choice.delta.tool_calls.as_ref().unwrap()[0];
            (call.index, call.id.clone().expect("id"))
        })
        .collect();
    assert_eq!(ids[0].0, 0);
    assert_eq!(ids[1].0, 1);
    assert_ne!(ids[0].1, ids[1].1, "each call gets its own id");
}

#[test]
fn adjacent_same_kind_deltas_coalesce_but_calls_never_do() {
    let merged = coalesce(vec![
        UnifiedParserEvent::Text("he".into()),
        UnifiedParserEvent::Text("llo".into()),
        call_delta(0, Some("f"), "{"),
        call_delta(0, None, "}"),
        UnifiedParserEvent::Reasoning("a".into()),
        UnifiedParserEvent::Reasoning("b".into()),
    ]);
    assert_eq!(merged.len(), 4);
    assert_eq!(merged[0], UnifiedParserEvent::Text("hello".into()));
    assert_eq!(merged[1], call_delta(0, Some("f"), "{"));
    assert_eq!(merged[2], call_delta(0, None, "}"));
    assert_eq!(merged[3], UnifiedParserEvent::Reasoning("ab".into()));
}

#[test]
fn role_rides_the_first_chunk_and_finish_reason_the_last() {
    let mut state = test_state();
    let mut base = empty_choice(0);
    base.delta.role = Some(Role::Assistant);
    let choices = state.choices_for(
        &base,
        vec![
            UnifiedParserEvent::Text("a".into()),
            call_delta(0, Some("f"), "{}"),
        ],
        true,
        Some(FinishReason::Stop),
    );

    assert_eq!(choices[0].delta.role, Some(Role::Assistant));
    assert!(choices[1].delta.role.is_none());
    assert!(choices[0].finish_reason.is_none());
    assert_eq!(choices[1].finish_reason, Some(FinishReason::ToolCalls));
}

#[test]
fn incomplete_tool_call_fragments_do_not_mark_the_choice_terminal() {
    let mut state = test_state();
    let partial = state.choices_for(
        &empty_choice(0),
        vec![partial_call_delta(
            0,
            Some("get_weather"),
            r#"{"city":"Tok"#,
        )],
        true,
        Some(FinishReason::Stop),
    );

    assert_eq!(partial.len(), 1);
    assert!(partial[0].delta.tool_calls.is_some());
    assert_eq!(
        partial[0].finish_reason,
        Some(FinishReason::Length),
        "an unfinished parser fragment must expose truncation"
    );
    assert!(state.tool_emitted());
    assert_eq!(
        state.unterminated_finish_reason(),
        Some(FinishReason::Length)
    );

    let completed = state.choices_for(
        &empty_choice(0),
        vec![call_delta(0, None, "yo\"}")],
        true,
        Some(FinishReason::Stop),
    );
    assert_eq!(completed[0].finish_reason, Some(FinishReason::ToolCalls));
    assert!(state.tool_emitted());
    assert_eq!(
        state.unterminated_finish_reason(),
        Some(FinishReason::ToolCalls)
    );
}

#[test]
fn completed_call_does_not_hide_another_incomplete_call() {
    let mut state = test_state();
    let partial = state.choices_for(
        &empty_choice(0),
        vec![partial_call_delta(0, Some("first"), r#"{"city":"Tok"#)],
        true,
        None,
    );
    assert_eq!(partial.len(), 1);

    let completed = state.choices_for(
        &empty_choice(0),
        vec![call_delta(1, Some("second"), r#"{"city":"Paris"}"#)],
        true,
        Some(FinishReason::Stop),
    );
    assert_eq!(completed[0].finish_reason, Some(FinishReason::Length));
    assert_eq!(
        state.unterminated_finish_reason(),
        Some(FinishReason::Length)
    );

    let final_fragment = state.choices_for(
        &empty_choice(0),
        vec![call_delta(0, None, "yo\"}")],
        true,
        Some(FinishReason::Stop),
    );
    assert_eq!(
        final_fragment[0].finish_reason,
        Some(FinishReason::ToolCalls)
    );
    assert_eq!(
        state.unterminated_finish_reason(),
        Some(FinishReason::ToolCalls)
    );
}

#[test]
fn separate_terminal_finish_closes_all_parsed_call_fragments() {
    let mut state = test_state();
    let mut parsed_fragments = empty_choice(0);
    parsed_fragments.delta.tool_calls = Some(vec![
        ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call-first".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("first".to_string()),
                arguments: Some(r#"{"city":"Paris"}"#.to_string()),
            }),
            ..Default::default()
        },
        ChatCompletionMessageToolCallChunk {
            index: 1,
            id: Some("call-second".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("second".to_string()),
                arguments: Some(r#"{"city":"Tokyo"}"#.to_string()),
            }),
            ..Default::default()
        },
    ]);
    let partials = state.choices_for(&parsed_fragments, Vec::new(), true, None);
    assert_eq!(partials.len(), 1);
    assert_eq!(
        state.unterminated_finish_reason(),
        Some(FinishReason::Length)
    );

    let terminal = state.choices_for(&empty_choice(0), Vec::new(), true, Some(FinishReason::Stop));

    assert_eq!(terminal.len(), 1);
    assert_eq!(terminal[0].finish_reason, Some(FinishReason::ToolCalls));
    assert_eq!(
        state.unterminated_finish_reason(),
        Some(FinishReason::ToolCalls)
    );
}

#[test]
fn an_empty_run_still_carries_the_terminating_envelope() {
    let mut state = test_state();
    let mut base = empty_choice(0);
    base.delta.role = Some(Role::Assistant);
    let choices = state.choices_for(&base, Vec::new(), true, Some(FinishReason::Length));

    assert_eq!(choices.len(), 1);
    assert_eq!(choices[0].delta.role, Some(Role::Assistant));
    assert_eq!(
        choices[0].finish_reason,
        Some(FinishReason::Length),
        "Length is not rewritten"
    );
}

#[test]
fn an_empty_run_with_no_envelope_emits_nothing() {
    let mut state = test_state();
    let base = empty_choice(0);
    assert!(state.choices_for(&base, Vec::new(), true, None).is_empty());
}

// --- end-to-end streaming ----------------------------------------------------

#[tokio::test]
async fn streams_ordered_reasoning_text_and_tool_calls() {
    let output = concat!(
        "<think>reason</think>answer ",
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    // Split into small chunks so markers straddle push() boundaries.
    let mut chunks: Vec<_> = output
        .as_bytes()
        .chunks(7)
        .map(|bytes| chunk(std::str::from_utf8(bytes).unwrap(), false))
        .collect();
    chunks.push(chunk("", true));

    let responses = apply_stream(
        stream::iter(chunks),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    let reasoning: String = choices
        .iter()
        .filter_map(|choice| choice.delta.reasoning_content.as_deref())
        .collect();
    let content: String = choices
        .iter()
        .filter_map(|choice| match &choice.delta.content {
            Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let arguments: String = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
        .collect();

    assert_eq!(reasoning, "reason");
    assert_eq!(content, "answer ");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&arguments).unwrap()["city"],
        "Tokyo"
    );
    for marker in ["<think>", "<tool_call>", "<function=", "<parameter="] {
        assert!(
            !content.contains(marker),
            "raw markup {marker:?} leaked into content: {content:?}"
        );
    }
    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .next_back(),
        Some(FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn qwen_fanout_keeps_source_metrics_on_reasoning() {
    let mut source = chunk("<think>reason</think>answer", false);
    source.data.as_mut().unwrap().llm_metrics = Some(LLMMetricAnnotation {
        chunk_tokens: 4,
        ..Default::default()
    });

    let responses = apply_stream(
        stream::iter([source]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let metric_choices: Vec<_> = responses
        .iter()
        .filter_map(|response| response.data.as_ref())
        .filter(|data| data.llm_metrics.is_some())
        .flat_map(|data| data.inner.choices.iter())
        .collect();

    assert_eq!(
        metric_choices.len(),
        1,
        "source metrics must be emitted once"
    );
    assert_eq!(
        metric_choices[0].delta.reasoning_content.as_deref(),
        Some("reason"),
        "the metrics owner must retain the source chunk's reasoning classification"
    );
}

#[tokio::test]
async fn qwen_delayed_reasoning_keeps_source_envelope_and_continuation_metadata() {
    for (opening, continuation_prefix) in [
        ("before <tool_c", "all>"),
        ("before <", "<think>raw</think><tool_call>"),
    ] {
        for reasoning_text in ["", "held thought"] {
            let mut reasoning = chunk("", false);
            let choice = &mut reasoning.data.as_mut().unwrap().inner.choices[0];
            choice.delta.content = None;
            choice.delta.reasoning_content = Some(reasoning_text.to_string());
            let source = reasoning.data.as_mut().unwrap();
            source.nvext = Some(serde_json::json!({"source": "reasoning"}));
            source.llm_metrics = Some(LLMMetricAnnotation {
                chunk_tokens: 7,
                ..Default::default()
            });
            reasoning.id = Some("reasoning-source-id".to_string());
            reasoning.event = Some("reasoning-source-event".to_string());
            reasoning.comment = Some(vec!["reasoning-source-comment".to_string()]);

            let continuation_text = format!(
                "{continuation_prefix}\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>answer"
            );
            let mut continuation = chunk(&continuation_text, true);
            let continuation_data = continuation.data.as_mut().unwrap();
            continuation_data.nvext = Some(serde_json::json!({"source": "continuation"}));
            continuation.id = Some("continuation-id".to_string());
            continuation.event = Some("continuation-event".to_string());
            continuation.comment = Some(vec!["continuation-comment".to_string()]);

            let responses = apply_stream(
                stream::iter([chunk(opening, false), reasoning, continuation]),
                Some(weather_tools()),
                None,
                false,
                UnifiedParserStartingState::None,
                QWEN3_UNIFIED_FAMILY,
            )
            .collect::<Vec<_>>()
            .await;

            let delayed_reasoning = responses
                .iter()
                .find(|response| {
                    response.data.as_ref().is_some_and(|data| {
                        data.inner.choices.iter().any(|choice| {
                            choice.delta.reasoning_content.as_deref() == Some(reasoning_text)
                        })
                    })
                })
                .expect("queued reasoning should be released when the opener resolves");
            let delayed_data = delayed_reasoning.data.as_ref().unwrap();
            assert_eq!(delayed_reasoning.id.as_deref(), Some("reasoning-source-id"));
            assert_eq!(
                delayed_reasoning.event.as_deref(),
                Some("reasoning-source-event")
            );
            assert_eq!(
                delayed_reasoning.comment.as_deref(),
                Some(["reasoning-source-comment".to_string()].as_slice())
            );
            assert_eq!(
                delayed_data.nvext,
                Some(serde_json::json!({"source": "reasoning"}))
            );
            assert_eq!(delayed_data.llm_metrics.as_ref().unwrap().chunk_tokens, 7);
            assert_eq!(delayed_data.inner.choices.len(), 1);
            assert!(delayed_data.inner.choices[0].delta.tool_calls.is_none());

            let continuation_answer = responses
            .iter()
            .find(|response| {
                response.data.as_ref().is_some_and(|data| {
                    data.inner.choices.iter().any(|choice| {
                        matches!(&choice.delta.content, Some(ChatCompletionMessageContent::Text(text)) if text == "answer")
                    })
                })
            })
            .expect("continuation answer should keep the resolving chunk envelope");
            let continuation_data = continuation_answer.data.as_ref().unwrap();
            assert_eq!(continuation_answer.id.as_deref(), Some("continuation-id"));
            assert_eq!(
                continuation_answer.event.as_deref(),
                Some("continuation-event")
            );
            assert_eq!(
                continuation_answer.comment.as_deref(),
                Some(["continuation-comment".to_string()].as_slice())
            );
            assert_eq!(
                continuation_data.nvext,
                Some(serde_json::json!({"source": "continuation"}))
            );
            let content_chunks: Vec<_> = responses
                .iter()
                .flat_map(|response| response.data.iter())
                .flat_map(|data| data.inner.choices.iter())
                .filter_map(|choice| match &choice.delta.content {
                    Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert!(
                content_chunks.iter().all(|text| !text.contains("<tool_c")),
                "a partial native opener must not leak into visible content: {content_chunks:?}"
            );
        }
    }
}

// Unified corpus input is raw model text; it cannot express upstream parsed
// reasoning or source response annotations, whose ownership this adapter adds.
#[tokio::test]
async fn qwen_queued_sources_keep_identity_beside_mixed_and_raw_reasoning() {
    for finish in [false, true] {
        let mut inputs = vec![chunk("before <", false)];
        for (source_id, tokens) in [("first-source", 3), ("second-source", 5)] {
            let mut response = chunk("", false);
            let data = response.data.as_mut().unwrap();
            data.inner.choices[0].delta.content = None;
            data.inner.choices[0].delta.reasoning_content = Some("same".to_string());
            data.nvext = Some(serde_json::json!({"source": source_id}));
            data.llm_metrics = Some(LLMMetricAnnotation {
                chunk_tokens: tokens,
                ..Default::default()
            });
            response.id = Some(source_id.to_string());
            inputs.push(response);
        }
        let mut continuation = chunk("<think>raw</think>answer", finish);
        let data = continuation.data.as_mut().unwrap();
        data.inner.choices[0].delta.reasoning_content = Some("mixed".to_string());
        data.nvext = Some(serde_json::json!({"source": "continuation"}));
        continuation.id = Some("continuation".to_string());
        inputs.push(continuation);
        let responses = apply_stream(
            stream::iter(inputs),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        for (source_id, tokens) in [("first-source", 3), ("second-source", 5)] {
            let sources: Vec<_> = responses
                .iter()
                .filter(|response| response.id.as_deref() == Some(source_id))
                .collect();
            assert_eq!(sources.len(), 1);
            let data = sources[0].data.as_ref().unwrap();
            assert_eq!(
                data.inner.choices[0].delta.reasoning_content.as_deref(),
                Some("same")
            );
            assert!(data.inner.choices[0].delta.content.is_none());
            assert_eq!(data.llm_metrics.as_ref().unwrap().chunk_tokens, tokens);
            assert_eq!(data.nvext, Some(serde_json::json!({"source": source_id})));
        }
        let reasoning: Vec<_> = collect_choices(&responses)
            .into_iter()
            .filter_map(|choice| choice.delta.reasoning_content.clone())
            .collect();
        assert_eq!(reasoning, ["mixed", "same", "same", "raw"]);
        assert_eq!(
            responses
                .iter()
                .filter(|response| response.id.as_deref() == Some("continuation"))
                .count(),
            1,
        );
        assert_eq!(
            collect_choices(&responses)
                .iter()
                .filter(|choice| choice.finish_reason.is_some())
                .count(),
            usize::from(finish),
        );
    }
}

#[tokio::test]
async fn qwen_eof_flushes_held_opener_before_source_annotated_reasoning() {
    let mut reasoning = chunk("", false);
    let choice = &mut reasoning.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("eof thought".to_string());
    let source = reasoning.data.as_mut().unwrap();
    source.nvext = Some(serde_json::json!({"source": "reasoning"}));
    source.llm_metrics = Some(LLMMetricAnnotation {
        chunk_tokens: 9,
        ..Default::default()
    });
    reasoning.id = Some("eof-reasoning-id".to_string());
    reasoning.event = Some("eof-reasoning-event".to_string());
    reasoning.comment = Some(vec!["eof-reasoning-comment".to_string()]);

    let responses = apply_stream(
        stream::iter([chunk("before <tool_c", false), reasoning]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let ordered: Vec<_> = responses
        .iter()
        .filter_map(|response| {
            let choice = response.data.as_ref()?.inner.choices.first()?;
            if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                Some(("text", text.as_str()))
            } else {
                choice
                    .delta
                    .reasoning_content
                    .as_deref()
                    .map(|text| ("reasoning", text))
            }
        })
        .collect();
    assert_eq!(
        ordered,
        vec![
            ("text", "before "),
            ("text", "<tool_c"),
            ("reasoning", "eof thought"),
        ],
        "EOF must release each held byte once, before queued reasoning"
    );

    let delayed_reasoning = responses
        .iter()
        .find(|response| {
            response.data.as_ref().is_some_and(|data| {
                data.inner
                    .choices
                    .iter()
                    .any(|choice| choice.delta.reasoning_content.as_deref() == Some("eof thought"))
            })
        })
        .expect("EOF must release the queued reasoning chunk");
    let delayed_data = delayed_reasoning.data.as_ref().unwrap();
    assert_eq!(delayed_reasoning.id.as_deref(), Some("eof-reasoning-id"));
    assert_eq!(
        delayed_reasoning.event.as_deref(),
        Some("eof-reasoning-event")
    );
    assert_eq!(
        delayed_reasoning.comment.as_deref(),
        Some(["eof-reasoning-comment".to_string()].as_slice())
    );
    assert_eq!(
        delayed_data.nvext,
        Some(serde_json::json!({"source": "reasoning"}))
    );
    assert_eq!(delayed_data.llm_metrics.as_ref().unwrap().chunk_tokens, 9);
    assert_eq!(delayed_data.inner.choices.len(), 1);
    assert!(delayed_data.inner.choices[0].delta.tool_calls.is_none());

    let held_text_count = responses
        .iter()
        .flat_map(|response| response.data.iter())
        .flat_map(|data| data.inner.choices.iter())
        .filter(|choice| {
            matches!(&choice.delta.content, Some(ChatCompletionMessageContent::Text(text)) if text == "<tool_c")
        })
        .count();
    assert_eq!(held_text_count, 1);
}

#[tokio::test]
async fn qwen_mismatched_opener_releases_source_reasoning_between_recovery_and_suffix() {
    let (input_tx, input_rx) = futures::channel::mpsc::unbounded();
    let mut output = Box::pin(apply_stream(
        input_rx,
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    ));

    input_tx
        .unbounded_send(chunk("before <tool_c", false))
        .unwrap();
    let prefix_response = output.next().await.expect("visible prefix should stream");
    let prefix_choice = &prefix_response.data.as_ref().unwrap().inner.choices[0];
    assert!(matches!(
        &prefix_choice.delta.content,
        Some(ChatCompletionMessageContent::Text(text)) if text == "before "
    ));

    let mut reasoning = chunk("", false);
    let choice = &mut reasoning.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("middle".to_string());
    let source = reasoning.data.as_mut().unwrap();
    source.nvext = Some(serde_json::json!({"source": "reasoning"}));
    source.llm_metrics = Some(LLMMetricAnnotation {
        chunk_tokens: 5,
        ..Default::default()
    });
    reasoning.id = Some("mismatch-reasoning-id".to_string());
    reasoning.event = Some("mismatch-reasoning-event".to_string());
    reasoning.comment = Some(vec!["mismatch-reasoning-comment".to_string()]);
    input_tx.unbounded_send(reasoning).unwrap();

    assert!(
        output.next().now_or_never().is_none(),
        "reasoning must remain held while the opener is ambiguous"
    );

    let mut continuation = chunk("Xtail", true);
    let continuation_data = continuation.data.as_mut().unwrap();
    continuation_data.nvext = Some(serde_json::json!({"source": "continuation"}));
    continuation.id = Some("mismatch-continuation-id".to_string());
    continuation.event = Some("mismatch-continuation-event".to_string());
    continuation.comment = Some(vec!["mismatch-continuation-comment".to_string()]);
    input_tx.unbounded_send(continuation).unwrap();
    drop(input_tx);
    let mut responses = vec![prefix_response];
    responses.extend(output.collect::<Vec<_>>().await);

    let ordered: Vec<_> = responses
        .iter()
        .filter_map(|response| {
            let choice = response.data.as_ref()?.inner.choices.first()?;
            if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                Some(("text", text.as_str()))
            } else {
                choice
                    .delta
                    .reasoning_content
                    .as_deref()
                    .map(|text| ("reasoning", text))
            }
        })
        .collect();
    assert_eq!(
        ordered,
        vec![
            ("text", "before "),
            ("text", "<tool_c"),
            ("reasoning", "middle"),
            ("text", "Xtail"),
        ]
    );

    let delayed_reasoning = responses
        .iter()
        .find(|response| {
            response.data.as_ref().is_some_and(|data| {
                data.inner
                    .choices
                    .iter()
                    .any(|choice| choice.delta.reasoning_content.as_deref() == Some("middle"))
            })
        })
        .unwrap();
    let delayed_data = delayed_reasoning.data.as_ref().unwrap();
    assert_eq!(
        delayed_reasoning.id.as_deref(),
        Some("mismatch-reasoning-id")
    );
    assert_eq!(
        delayed_reasoning.event.as_deref(),
        Some("mismatch-reasoning-event")
    );
    assert_eq!(
        delayed_reasoning.comment.as_deref(),
        Some(["mismatch-reasoning-comment".to_string()].as_slice())
    );
    assert_eq!(
        delayed_data.nvext,
        Some(serde_json::json!({"source": "reasoning"}))
    );
    assert_eq!(delayed_data.llm_metrics.as_ref().unwrap().chunk_tokens, 5);

    let suffix = responses
        .iter()
        .find(|response| {
            response.data.as_ref().is_some_and(|data| {
                data.inner.choices.iter().any(|choice| {
                    matches!(&choice.delta.content, Some(ChatCompletionMessageContent::Text(text)) if text == "Xtail")
                })
            })
        })
        .unwrap();
    assert_eq!(suffix.id.as_deref(), Some("mismatch-continuation-id"));
    assert_eq!(suffix.event.as_deref(), Some("mismatch-continuation-event"));
    assert_eq!(
        suffix.comment.as_deref(),
        Some(["mismatch-continuation-comment".to_string()].as_slice())
    );
    assert_eq!(
        suffix.data.as_ref().unwrap().nvext,
        Some(serde_json::json!({"source": "continuation"}))
    );
}

#[tokio::test]
async fn inconclusive_first_chunk_waits_before_classifying_guided_prefill() {
    let mut role_only = chunk("", false);
    role_only.data.as_mut().unwrap().inner.choices[0]
        .delta
        .content = None;

    for first in [role_only, chunk(" \n", false)] {
        let responses = apply_stream(
            stream::iter([first, chunk(r#"{"city":"Tokyo"}"#, true)]),
            Some(weather_tools()),
            Some(named_choice("get_weather")),
            false,
            UnifiedParserStartingState::Reasoning,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        let calls: Vec<_> = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .collect();

        assert_eq!(
            calls
                .iter()
                .find_map(|call| call.function.as_ref()?.name.as_deref()),
            Some("get_weather")
        );
        let arguments: String = calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        assert_eq!(arguments, r#"{"city":"Tokyo"}"#);
        assert!(
            choices
                .iter()
                .all(|choice| choice.delta.reasoning_content.is_none())
        );
    }
}

#[tokio::test]
async fn deferred_prefill_whitespace_is_reasoning_when_prose_arrives_first() {
    let responses = apply_stream(
        stream::iter([
            chunk(" \n", false),
            chunk("reason</think>{\"city\": ", false),
            chunk("\"Tokyo\"}", true),
        ]),
        Some(weather_tools()),
        Some(named_choice("get_weather")),
        false,
        UnifiedParserStartingState::Reasoning,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let reasoning: String = choices
        .iter()
        .filter_map(|choice| choice.delta.reasoning_content.as_deref())
        .collect();
    assert_eq!(reasoning, " \nreason");
    let arguments: String = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
        .collect();
    assert_eq!(arguments, r#"{"city": "Tokyo"}"#);
}

#[tokio::test]
async fn named_guided_json_becomes_a_tool_call() {
    let responses = apply_stream(
        stream::iter([
            chunk("reason</think>{\"city\": ", false),
            chunk("\"Tokyo\"}", true),
        ]),
        Some(weather_tools()),
        Some(named_choice("get_weather")),
        false,
        UnifiedParserStartingState::Reasoning,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    assert_eq!(
        choices[0].delta.reasoning_content.as_deref(),
        Some("reason")
    );
    // A named choice now STREAMS: the name rides the first delta and the argument
    // bytes follow, so collect across deltas exactly as the required case does.
    let calls: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .collect();
    assert_eq!(
        calls
            .iter()
            .find_map(|call| call.function.as_ref()?.name.as_deref()),
        Some("get_weather")
    );
    let arguments: String = calls
        .iter()
        .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
        .collect();
    assert_eq!(
        arguments, "{\"city\": \"Tokyo\"}",
        "a named choice passes the model's argument bytes through verbatim"
    );
    // The argument bytes must arrive as they are generated, not in one terminal
    // burst: the caller split the payload across two chunks, so a client that
    // renders each delta sees the call grow instead of waiting for the whole
    // object. One frame here means the wiring buffered what the parser streamed.
    let frames = calls
        .iter()
        .filter(|call| {
            call.function
                .as_ref()
                .and_then(|f| f.arguments.as_deref())
                .is_some_and(|a| !a.is_empty())
        })
        .count();
    assert!(
        frames >= 2,
        "arguments arrived in {frames} frame(s) - that is a burst, not a stream"
    );
    assert!(
        choices
            .iter()
            .any(|c| c.finish_reason == Some(FinishReason::ToolCalls)),
        "the stream must terminate with ToolCalls"
    );
}

#[tokio::test]
async fn truncated_named_guided_json_stays_pending_at_stop() {
    let responses = apply_stream(
        stream::iter([
            chunk("reason</think>{\"city\": ", false),
            chunk("\"Tokyo\"", true),
        ]),
        Some(weather_tools()),
        Some(named_choice("get_weather")),
        false,
        UnifiedParserStartingState::Reasoning,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    assert!(
        choices
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some()),
        "the guided parser may have already streamed the provisional call"
    );
    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>(),
        vec![FinishReason::Length],
        "an incomplete argument object must not be reported as a completed tool call"
    );
}

#[tokio::test]
async fn truncated_named_guided_json_downgrades_tool_calls_terminal() {
    let mut terminal = chunk("\"Tokyo\"", true);
    terminal.data.as_mut().unwrap().inner.choices[0].finish_reason = Some(FinishReason::ToolCalls);
    let responses = apply_stream(
        stream::iter([chunk("reason</think>{\"city\": ", false), terminal]),
        Some(weather_tools()),
        Some(named_choice("get_weather")),
        false,
        UnifiedParserStartingState::Reasoning,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    assert!(
        choices
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some())
    );
    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>(),
        vec![FinishReason::Length]
    );
}

#[tokio::test]
async fn named_guided_json_continues_across_reasoning_detour() {
    let mut reasoning_detour = chunk("", false);
    let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("interleaved thought".to_string());

    let responses = apply_stream(
        stream::iter([
            chunk("reason</think>{\"city\": ", false),
            reasoning_detour,
            chunk("\"Tokyo\"}", true),
        ]),
        Some(weather_tools()),
        Some(named_choice("get_weather")),
        false,
        UnifiedParserStartingState::Reasoning,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    let arguments: String = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
        .collect();
    assert_eq!(arguments, "{\"city\": \"Tokyo\"}");
    assert!(
        choices.iter().any(|choice| {
            choice.delta.reasoning_content.as_deref() == Some("interleaved thought")
        }),
        "the reasoning detour must remain visible in its original position"
    );
    assert!(
        choices
            .iter()
            .any(|choice| choice.finish_reason == Some(FinishReason::ToolCalls)),
        "the resumed argument object must complete the same guided tool call"
    );
}

#[tokio::test]
async fn native_qwen_call_continues_across_reasoning_detour() {
    let mut reasoning_detour = chunk("", false);
    let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("interleaved thought".to_string());

    let responses = apply_stream(
        stream::iter([
            chunk("<tool_call><function=get_weather><parameter=city>NY", false),
            reasoning_detour,
            chunk("C</parameter></function></tool_call>", true),
        ]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    let calls: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .collect();
    let arguments: String = calls
        .iter()
        .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
        .collect();
    assert_eq!(arguments, r#"{"city":"NYC"}"#);
    assert!(calls.iter().any(|call| call.index == 0));
    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>(),
        vec![FinishReason::ToolCalls]
    );
    let reasoning_position = choices
        .iter()
        .position(|choice| choice.delta.reasoning_content.is_some())
        .expect("the already-parsed reasoning delta must remain visible");
    assert!(
        choices[..reasoning_position]
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some()),
        "the provisional native call must be released before the reasoning detour"
    );
    assert!(
        choices[reasoning_position + 1..]
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some()),
        "the resumed native call fragment must follow the reasoning detour"
    );
}

#[tokio::test]
async fn native_qwen_partial_call_opener_survives_reasoning_detour_at_every_split() {
    let opener = "<tool_call>";
    let body_and_answer = concat!(
        "\n<function=get_weather>\n",
        "<parameter=city>Paris</parameter>\n",
        "</function>\n</tool_call>answer"
    );

    for split in 1..opener.len() {
        let mut reasoning_detour = chunk("", false);
        let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("interleaved thought".to_string());

        let continuation = format!("{}{}", &opener[split..], body_and_answer);
        let responses = apply_stream(
            stream::iter([
                chunk(&opener[..split], false),
                reasoning_detour,
                chunk(&continuation, true),
            ]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        let calls: Vec<_> = choices
            .iter()
            .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
            .collect();
        let arguments: String = calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        assert_eq!(arguments, r#"{"city":"Paris"}"#, "split {split}");
        assert_eq!(
            calls
                .iter()
                .filter_map(|call| call.function.as_ref()?.name.as_deref())
                .collect::<String>(),
            "get_weather",
            "split {split}"
        );
        let mut indices: Vec<_> = calls.iter().map(|call| call.index).collect();
        indices.sort_unstable();
        indices.dedup();
        assert_eq!(indices, vec![0], "split {split}");
        assert_eq!(
            calls
                .iter()
                .filter_map(|call| call.id.as_deref())
                .collect::<HashSet<_>>()
                .len(),
            1,
            "split {split} must keep one call ID"
        );
        assert!(choices.iter().any(|choice| {
            choice.delta.reasoning_content.as_deref() == Some("interleaved thought")
        }));
        let content: String = choices
            .iter()
            .filter_map(|choice| match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, "answer", "split {split}");
        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::ToolCalls],
            "split {split}"
        );

        let reasoning_position = choices
            .iter()
            .position(|choice| choice.delta.reasoning_content.is_some())
            .expect("reasoning detour must be emitted");
        let call_position = choices
            .iter()
            .position(|choice| choice.delta.tool_calls.is_some())
            .expect("tool call must follow the detour");
        let answer_position = choices
            .iter()
            .position(|choice| choice.delta.content.is_some())
            .expect("trailing answer must be emitted");
        assert!(
            reasoning_position < call_position && call_position < answer_position,
            "raw/parsed/raw order changed at split {split}"
        );
    }
}

#[tokio::test]
async fn native_qwen_mismatched_partial_opener_orders_recovery_before_reasoning_at_every_split() {
    let opener = "<tool_call>";

    for split in 1..opener.len() {
        let mismatch_position = opener.len() - 1;
        let continuation = format!("{}Xpost", &opener[split..mismatch_position]);
        let mut reasoning_detour = chunk("", false);
        let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("interleaved thought".to_string());

        let responses = apply_stream(
            stream::iter([
                chunk(&opener[..split], false),
                reasoning_detour,
                chunk(&continuation, true),
            ]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        let ordered: Vec<_> = choices
            .iter()
            .filter_map(|choice| {
                if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                    Some(("content", text.as_str()))
                } else {
                    choice
                        .delta
                        .reasoning_content
                        .as_deref()
                        .map(|text| ("reasoning", text))
                }
            })
            .collect();

        assert_eq!(
            ordered,
            vec![
                ("content", &opener[..split]),
                ("reasoning", "interleaved thought"),
                ("content", &continuation[..]),
            ],
            "mismatch at opener split {split}"
        );
    }
}

#[tokio::test]
async fn native_qwen_partial_opener_eof_flushes_recovered_text_before_reasoning() {
    let mut reasoning_detour = chunk("", false);
    let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("interleaved thought".to_string());

    let responses = apply_stream(
        stream::iter([chunk("before <tool_c", false), reasoning_detour]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let ordered: Vec<_> = choices
        .iter()
        .filter_map(|choice| {
            if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                Some(("content", text.as_str()))
            } else {
                choice
                    .delta
                    .reasoning_content
                    .as_deref()
                    .map(|text| ("reasoning", text))
            }
        })
        .collect();

    assert_eq!(
        ordered,
        vec![
            ("content", "before "),
            ("content", "<tool_c"),
            ("reasoning", "interleaved thought"),
        ]
    );
}

#[tokio::test]
async fn native_qwen_buffered_non_string_argument_survives_reasoning_detour() {
    let mut reasoning_detour = chunk("", false);
    let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("interleaved thought".to_string());

    let responses = apply_stream(
        stream::iter([
            chunk("<tool_call><function=count_items><parameter=count>1", false),
            reasoning_detour,
            chunk("2</parameter></function></tool_call>", true),
        ]),
        Some(count_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    let calls: Vec<_> = choices
        .iter()
        .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
        .collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0]
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("count_items")
    );
    assert_eq!(
        calls[0]
            .function
            .as_ref()
            .and_then(|function| function.arguments.as_deref()),
        Some(r#"{"count":12}"#)
    );
    assert!(choices.iter().any(|choice| {
        choice.delta.reasoning_content.as_deref() == Some("interleaved thought")
    }));
    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>(),
        vec![FinishReason::ToolCalls]
    );
}

#[tokio::test]
async fn truncated_named_guided_json_stays_pending_across_reasoning_detour_at_eof() {
    let mut reasoning_detour = chunk("", false);
    let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("interleaved thought".to_string());

    let responses = apply_stream(
        stream::iter([chunk("reason</think>{\"city\": ", false), reasoning_detour]),
        Some(weather_tools()),
        Some(named_choice("get_weather")),
        false,
        UnifiedParserStartingState::Reasoning,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    assert!(
        choices
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some()),
        "the cursor may have streamed provisional name and argument fragments"
    );
    assert!(
        choices
            .iter()
            .all(|choice| choice.finish_reason != Some(FinishReason::ToolCalls)),
        "EOF must not complete an invalid or truncated argument object"
    );
}

#[tokio::test]
async fn bare_guided_json_under_prompt_injected_reasoning_still_becomes_a_tool_call() {
    // Regression for the guided-JSON-swallowed-as-reasoning bug: a Qwen3-Thinking
    // style template appends `<think>` to the generation prompt
    // (`prompt_injected_reasoning`, modeled here by starting the parser at
    // `Reasoning`), and guided decoding forbids the model from ever emitting
    // `</think>`. Unconditionally honoring the `Reasoning` prefill would make the
    // parser wait forever for a closer that will never arrive, classifying the
    // ENTIRE guided JSON payload as `reasoning_content` with zero tool calls
    // extracted. The payload here has no `reason</think>` prefix at all — its first
    // byte is `[` — so `bare_guided_json_prefill` must reclassify this choice's
    // starting state to `None` and let the array parse as a tool call instead.
    let responses = apply_stream(
        stream::iter([chunk(
            r#"[{"name":"get_weather","parameters":{"city":"Paris"}}]"#,
            true,
        )]),
        Some(weather_tools()),
        Some(ChatCompletionToolChoiceOption::Required),
        false,
        UnifiedParserStartingState::Reasoning,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    assert!(
        choices.iter().all(|c| c.delta.reasoning_content.is_none()),
        "a bare guided JSON payload with no reasoning prefix must not be classified \
         as reasoning_content: {choices:?}"
    );
    let calls: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .collect();
    assert_eq!(
        calls
            .iter()
            .find_map(|call| call.function.as_ref()?.name.as_deref()),
        Some("get_weather"),
        "the guided JSON payload must still be recovered as a tool call: {choices:?}"
    );
    let arguments: String = calls
        .iter()
        .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
        .collect();
    assert_eq!(arguments, r#"{"city":"Paris"}"#);
}

#[tokio::test]
async fn required_guided_json_becomes_a_tool_call() {
    let responses = apply_stream(
        stream::iter([chunk(
            r#"reason</think>[{"name":"get_weather","parameters":{"city":"Tokyo"}}]"#,
            true,
        )]),
        Some(weather_tools()),
        Some(ChatCompletionToolChoiceOption::Required),
        false,
        UnifiedParserStartingState::Reasoning,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    let calls: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .collect();
    assert_eq!(
        calls
            .iter()
            .find_map(|call| call.function.as_ref()?.name.as_deref()),
        Some("get_weather")
    );
    let arguments: String = calls
        .iter()
        .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
        .collect();
    assert_eq!(arguments, r#"{"city":"Tokyo"}"#);
}

#[tokio::test]
async fn required_guided_emits_name_before_polling_call_completion() {
    let prefix = r#"reason</think>[{"name":"get_weather","arguments":{"#;
    let upstream = stream::iter([chunk(prefix, false)]).chain(stream::poll_fn(|_| {
        panic!("apply_stream polled for call-completing bytes before emitting the tool name")
    }));
    let responses = apply_stream(
        upstream,
        Some(weather_tools()),
        Some(ChatCompletionToolChoiceOption::Required),
        false,
        UnifiedParserStartingState::Reasoning,
        QWEN3_UNIFIED_FAMILY,
    );
    tokio::pin!(responses);

    let reasoning = responses.next().await.expect("reasoning delta");
    assert_eq!(
        collect_choices(&[reasoning])[0]
            .delta
            .reasoning_content
            .as_deref(),
        Some("reason")
    );

    let tool = responses.next().await.expect("tool-name delta");
    let tool_responses = [tool];
    let tool_choices = collect_choices(&tool_responses);
    let call = &tool_choices[0]
        .delta
        .tool_calls
        .as_ref()
        .expect("tool call before completion")[0];
    assert_eq!(
        call.function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("get_weather")
    );
}

/// Drive one guided-`required` stream whose payload closes only in the LAST
/// upstream chunk, and report how many emitted responses carried a tool-call
/// delta plus the assembled logical events.
///
/// Anything counted above one is an argument fragment that went out before the
/// payload closed, which is exactly the difference the guided tool-call
/// streaming rollback lever is supposed to make. Nothing here touches process
/// env, so it is safe under the default parallel test runner.
async fn guided_required_tool_chunk_count(guided_streaming: bool) -> (usize, Vec<LogicalEvent>) {
    let head = r#"reason</think>[{"name":"get_weather","arguments":{"city":"Tok"#;
    let tail = r#"yo"}}]"#;
    let responses = apply_stream_with_constraint(
        stream::iter([chunk(head, false), chunk(tail, true)]),
        Some(weather_tools()),
        GuidedToolConstraint::GuidedJsonRequired,
        UnifiedParserStartingState::Reasoning,
        QWEN3_UNIFIED_FAMILY,
        guided_streaming,
    )
    .collect::<Vec<_>>()
    .await;
    let tool_chunks = responses
        .iter()
        .filter(|response| {
            response.data.as_ref().is_some_and(|data| {
                data.inner.choices.iter().any(|choice| {
                    choice
                        .delta
                        .tool_calls
                        .as_ref()
                        .is_some_and(|calls| !calls.is_empty())
                })
            })
        })
        .count();
    (tool_chunks, logical_events(&responses))
}

#[tokio::test]
async fn guided_streaming_rollback_buffers_the_call_to_completion() {
    let expected_call = LogicalEvent::Tool {
        index: 0,
        name: Some("get_weather".to_string()),
        arguments: r#"{"city":"Tokyo"}"#.to_string(),
    };

    // Released (production default when DYN_ENABLE_GUIDED_TOOL_STREAMING is unset):
    // the call goes out in pieces as the argument object streams in.
    let (released_chunks, released_events) = guided_required_tool_chunk_count(true).await;
    assert!(
        released_chunks > 1,
        "guided streaming must emit argument fragments before the payload closes, \
         got {released_chunks} tool-call chunk(s)"
    );

    // Rolled back (DYN_ENABLE_GUIDED_TOOL_STREAMING=0): nothing may go out until the
    // payload closes, so the whole call arrives as one chunk.
    let (rolled_back_chunks, rolled_back_events) = guided_required_tool_chunk_count(false).await;
    assert_eq!(
        rolled_back_chunks, 1,
        "DYN_ENABLE_GUIDED_TOOL_STREAMING=0 must buffer the guided call to completion \
         on the unified path, but {rolled_back_chunks} tool-call chunk(s) went out"
    );

    // The lever changes WHEN the call is emitted, never WHAT is emitted.
    assert!(
        released_events.contains(&expected_call),
        "released events lost the call: {released_events:?}"
    );
    assert!(
        rolled_back_events.contains(&expected_call),
        "rolled-back events lost the call: {rolled_back_events:?}"
    );
}

#[test]
fn only_stream_best_effort_may_emit_before_the_payload_closes() {
    // The two buffering contracts are interchangeable for WHEN, but not for WHAT a
    // malformed payload becomes, so the rollback must pick `RecoverAsText` (text)
    // and not `Reject` (typed error).
    assert_eq!(
        invalid_guided_payload_policy(true),
        InvalidGuidedPayloadPolicy::StreamBestEffort
    );
    assert_eq!(
        invalid_guided_payload_policy(false),
        InvalidGuidedPayloadPolicy::RecoverAsText
    );
}

#[tokio::test]
async fn already_parsed_choices_pass_through_untouched() {
    // A chunk that already carries reasoning_content was parsed upstream; running
    // it through the parser again would double it.
    let mut pre_parsed = chunk("", false);
    let choice = &mut pre_parsed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("already".to_string());

    let responses = apply_stream(
        stream::iter([pre_parsed]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    assert_eq!(choices.len(), 1);
    assert_eq!(
        choices[0].delta.reasoning_content.as_deref(),
        Some("already")
    );
}

#[tokio::test]
async fn resumed_raw_tool_call_gets_a_new_index_after_an_already_parsed_detour() {
    let first_partial = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Paris</parameter>"
    );
    let already_parsed = {
        let mut value = chunk("", false);
        let choice = &mut value.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("already parsed".to_string());
        value
    };
    let second_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let responses = apply_stream(
        stream::iter([
            chunk(first_partial, false),
            already_parsed,
            chunk(second_call, true),
        ]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let calls: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .collect();
    assert!(calls.iter().any(|call| call.index == 0));
    assert!(calls.iter().any(|call| call.index == 1));
    let first_ids: HashSet<_> = calls
        .iter()
        .filter(|call| call.index == 0)
        .filter_map(|call| call.id.as_deref())
        .collect();
    let second_ids: HashSet<_> = calls
        .iter()
        .filter(|call| call.index == 1)
        .filter_map(|call| call.id.as_deref())
        .collect();
    assert_eq!(
        first_ids.len(),
        1,
        "the first call identity must stay stable"
    );
    assert_eq!(second_ids.len(), 1, "the resumed call needs one stable ID");
    assert_ne!(first_ids, second_ids, "separate calls must not reuse an ID");
    let names: Vec<_> = calls
        .iter()
        .filter_map(|call| call.function.as_ref()?.name.as_deref())
        .collect();
    assert_eq!(names, vec!["get_weather", "get_weather"]);
}

#[tokio::test]
async fn mixed_preparsed_reasoning_still_parses_raw_content() {
    let output = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let mut mixed = chunk(output, true);
    mixed.data.as_mut().unwrap().inner.choices[0]
        .delta
        .reasoning_content = Some("already parsed".to_string());

    let responses = apply_stream(
        stream::iter([mixed]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let visible: String = choices
        .iter()
        .filter_map(|choice| match &choice.delta.content {
            Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.delta.reasoning_content.as_deref())
            .collect::<String>(),
        "already parsed"
    );
    assert!(
        choices
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some()),
        "raw native tool markup must still become a structured call"
    );
    assert!(
        !visible.contains("<tool_call>"),
        "raw markup leaked: {visible:?}"
    );
}

#[tokio::test]
async fn mixed_preparsed_tool_call_still_parses_raw_content() {
    let output = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let mut mixed = chunk(output, true);
    let choice = &mut mixed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 7,
        id: Some("existing-call".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("already_parsed".to_string()),
            arguments: Some("{}".to_string()),
        }),
        ..Default::default()
    }]);

    let responses = apply_stream(
        stream::iter([mixed]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let visible: String = choices
        .iter()
        .filter_map(|choice| match &choice.delta.content {
            Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
            _ => None,
        })
        .collect();
    let names: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .filter_map(|call| call.function.as_ref()?.name.as_deref())
        .collect();

    assert!(names.contains(&"already_parsed"));
    assert!(names.contains(&"get_weather"));
    assert!(
        !visible.contains("<tool_call>"),
        "raw markup leaked: {visible:?}"
    );
}

#[tokio::test]
async fn already_parsed_detour_allocates_index_after_parser_index_collision() {
    let raw_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let mut upstream_call = chunk("", false);
    let choice = &mut upstream_call.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("upstream-call".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("upstream".to_string()),
            arguments: Some("{\"city\":\"Paris\"}".to_string()),
        }),
        ..Default::default()
    }]);

    let mut continuation = chunk("", false);
    let choice = &mut continuation.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: None,
        r#type: None,
        function: Some(FunctionCallStream {
            name: None,
            arguments: Some("more".to_string()),
        }),
        ..Default::default()
    }]);

    let responses = apply_stream(
        stream::iter([chunk(raw_call, false), upstream_call, continuation]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let calls: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .collect();
    let parser_call = calls
        .iter()
        .find(|call| {
            call.function
                .as_ref()
                .and_then(|function| function.name.as_deref())
                == Some("get_weather")
        })
        .expect("raw parser call must be emitted");
    let upstream_open = calls
        .iter()
        .find(|call| {
            call.function
                .as_ref()
                .and_then(|function| function.name.as_deref())
                == Some("upstream")
        })
        .expect("the first upstream call chunk must keep its name");
    assert_eq!(parser_call.index, 0);
    assert_eq!(upstream_open.index, 1);
    assert_eq!(upstream_open.id.as_deref(), Some("upstream-call"));
    let passthrough_continuation = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .find(|call| {
            call.function
                .as_ref()
                .and_then(|function| function.arguments.as_deref())
                == Some("more")
        })
        .expect("upstream continuation must be emitted");

    assert_eq!(
        passthrough_continuation.index, 1,
        "continuation must retain the upstream call's remapped index"
    );
    assert_eq!(passthrough_continuation.id, None);
    assert_eq!(passthrough_continuation.r#type, None);
    assert_eq!(
        passthrough_continuation
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        None
    );
}

#[tokio::test]
async fn preparsed_terminal_choice_does_not_drop_buffered_bytes() {
    let mut terminal = chunk("", true);
    let choice = &mut terminal.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("already".to_string());

    let responses = apply_stream(
        stream::iter([chunk("hello <tool_c", false), terminal]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let visible: String = choices
        .iter()
        .filter_map(|choice| match &choice.delta.content {
            Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(visible, "hello <tool_c");
    let terminal_position = choices
        .iter()
        .position(|choice| choice.finish_reason.is_some())
        .expect("terminal choice");
    assert!(
        choices[terminal_position + 1..]
            .iter()
            .all(|choice| choice.delta.content.is_none()),
        "buffered content appeared after the terminal choice"
    );
}

#[tokio::test]
async fn preparsed_terminal_normalizes_stop_after_prior_raw_tool_call() {
    let tool_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let mut terminal = chunk("", true);
    let choice = &mut terminal.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("already parsed".to_string());

    let responses = apply_stream(
        stream::iter([chunk(tool_call, false), terminal]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert!(
        choices
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some()),
        "the raw first chunk must emit a tool call"
    );
    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .next_back(),
        Some(FinishReason::ToolCalls),
        "an already-parsed terminal Stop must retain the earlier tool-call state"
    );
}

#[tokio::test]
async fn raw_preparsed_raw_sequence_preserves_order_and_buffered_bytes() {
    let mut pre_parsed = chunk("", false);
    let choice = &mut pre_parsed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("middle".to_string());

    let responses = apply_stream(
        stream::iter([
            chunk("before <tool_c", false),
            pre_parsed,
            chunk(" after", true),
        ]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let ordered: Vec<_> = choices
        .iter()
        .filter_map(|choice| {
            if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                Some(("content", text.as_str()))
            } else {
                choice
                    .delta
                    .reasoning_content
                    .as_deref()
                    .map(|reasoning| ("reasoning", reasoning))
            }
        })
        .collect();

    assert_eq!(
        ordered,
        vec![
            ("content", "before "),
            ("content", "<tool_c"),
            ("reasoning", "middle"),
            ("content", " after"),
        ]
    );
}

#[tokio::test]
async fn raw_preparsed_raw_resumes_at_response_not_original_prefill() {
    // Request-level prefill is `Reasoning` (prompt-injected reasoning): the first
    // raw run closes reasoning mid-run. An already-parsed chunk then detours the
    // choice, dropping its live `ChoiceState`. The next raw chunk must NOT rebuild
    // starting at `Reasoning` again — an already-parsed chunk implies the model has
    // already emitted structured output, which only happens after any reasoning
    // phase concluded, so the resumed run's text belongs to `content`.
    let mut pre_parsed = chunk("", false);
    let choice = &mut pre_parsed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("already parsed".to_string());

    let responses = apply_stream(
        stream::iter([
            chunk("secret</think>visible-before-gap ", false),
            pre_parsed,
            chunk("visible-after-gap", true),
        ]),
        None,
        None,
        false,
        UnifiedParserStartingState::Reasoning,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    let reasoning: String = choices
        .iter()
        .filter_map(|choice| choice.delta.reasoning_content.as_deref())
        .collect();
    let visible: String = choices
        .iter()
        .filter_map(|choice| match &choice.delta.content {
            Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(
        visible, "visible-before-gap visible-after-gap",
        "text after the already-parsed gap must resume as content, not reasoning"
    );
    assert_eq!(reasoning, "secretalready parsed");
}

#[tokio::test]
async fn raw_preparsed_raw_terminal_normalizes_stop_after_tool_call_before_gap() {
    let tool_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let mut pre_parsed = chunk("", false);
    let choice = &mut pre_parsed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("already parsed".to_string());

    let responses = apply_stream(
        stream::iter([
            chunk(tool_call, false),
            pre_parsed,
            chunk(" trailing text", true),
        ]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert!(
        choices
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some()),
        "the raw first chunk must emit a tool call"
    );
    let finish_reasons: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.finish_reason)
        .collect();
    assert_eq!(
        finish_reasons,
        vec![FinishReason::ToolCalls],
        "a fresh parser instance for the resumed raw run must not lose the \
         tool-call history from before the already-parsed gap, and exactly \
         one terminal must be emitted"
    );
}

#[tokio::test]
async fn two_consecutive_already_parsed_chunks_after_tool_call_normalize_once() {
    let tool_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let already_parsed_chunk = |reasoning: &str, finish: bool| {
        let mut c = chunk("", finish);
        let choice = &mut c.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some(reasoning.to_string());
        c
    };

    let responses = apply_stream(
        stream::iter([
            chunk(tool_call, false),
            already_parsed_chunk("first already-parsed", false),
            already_parsed_chunk("second already-parsed", false),
            already_parsed_chunk("third already-parsed (terminal)", true),
        ]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert!(
        choices
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some()),
        "the raw first chunk must emit a tool call"
    );
    let finish_reasons: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.finish_reason)
        .collect();
    assert_eq!(
        finish_reasons,
        vec![FinishReason::ToolCalls],
        "two already-parsed chunks in a row after the tool call (the second \
         with no live ChoiceState left to remove) must still normalize to \
         exactly one ToolCalls terminal, not Stop and not more than one terminal"
    );
}

#[tokio::test]
async fn tool_history_is_isolated_per_choice_index() {
    fn two_choice_response(
        index0: (&str, bool),
        index1: (&str, bool),
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let make = |index: u32, text: &str, finish: bool| ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(ChatCompletionMessageContent::Text(text.to_string())),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: finish.then_some(FinishReason::Stop),
            logprobs: None,
        };
        #[allow(deprecated)]
        let response = NvCreateChatCompletionStreamResponse {
            inner: CreateChatCompletionStreamResponse {
                id: "test".to_string(),
                choices: vec![make(0, index0.0, index0.1), make(1, index1.0, index1.1)],
                created: 0,
                model: "qwen3".to_string(),
                system_fingerprint: None,
                service_tier: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
            },
            nvext: None,
            llm_metrics: None,
            prompt_logprobs: None,
            tool_call_completion: Vec::new(),
        };
        Annotated::from_data(response)
    }
    let already_parsed_pair = || {
        let mut response = two_choice_response(("", false), ("", false));
        for choice in &mut response.data.as_mut().unwrap().inner.choices {
            choice.delta.content = None;
            choice.delta.reasoning_content = Some("already parsed".to_string());
        }
        response
    };

    let tool_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let responses = apply_stream(
        stream::iter([
            two_choice_response((tool_call, false), ("hello", false)),
            already_parsed_pair(),
            two_choice_response((" trailing0", true), (" trailing1", true)),
        ]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    let finish_for = |index: u32| {
        choices
            .iter()
            .filter(|choice| choice.index == index)
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        finish_for(0),
        vec![FinishReason::ToolCalls],
        "choice 0 emitted a tool call before the gap and must normalize to ToolCalls"
    );
    assert_eq!(
        finish_for(1),
        vec![FinishReason::Stop],
        "choice 1 never emitted a tool call and must not be contaminated by \
         choice 0's tool history"
    );
}

#[tokio::test]
#[allow(deprecated)]
async fn legacy_function_call_passes_through_untouched() {
    let mut legacy = chunk("", true);
    let choice = &mut legacy.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.function_call = Some(ChatCompletionStreamResponseDeltaFunctionCall {
        name: Some("get_weather".to_string()),
        arguments: Some(r#"{"city":"Tokyo"}"#.to_string()),
    });

    let responses = apply_stream(
        stream::iter([legacy]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert_eq!(choices.len(), 1);
    assert_eq!(
        choices[0]
            .delta
            .function_call
            .as_ref()
            .and_then(|call| call.name.as_deref()),
        Some("get_weather")
    );
}

#[tokio::test]
async fn a_stream_without_a_finish_reason_still_terminates_a_tool_call() {
    let output = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    // No terminating chunk at all — the stream just ends.
    let responses = apply_stream(
        stream::iter([chunk(output, false)]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .next_back(),
        Some(FinishReason::ToolCalls),
        "the backstop must synthesize a terminal reason or a strict client hangs"
    );
}

#[tokio::test]
async fn already_parsed_gap_then_immediate_eof_still_terminates_a_tool_call() {
    let tool_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let mut already_parsed = chunk("", false);
    let choice = &mut already_parsed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("already parsed".to_string());

    // Raw tool call, then an already-parsed chunk that removes the live
    // ChoiceState (no finish_reason on it), then the stream just ends — no
    // further chunks, no usage-only chunk, no explicit terminal at all.
    let responses = apply_stream(
        stream::iter([chunk(tool_call, false), already_parsed]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert!(
        choices
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some()),
        "the raw first chunk must emit a tool call"
    );
    let finish_reasons: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.finish_reason)
        .collect();
    assert_eq!(
        finish_reasons,
        vec![FinishReason::ToolCalls],
        "a choice whose live state was removed by an already-parsed gap must \
         still get exactly one terminal at end-of-stream, not zero — the EOF \
         backstop must not silently drop it"
    );
}

#[tokio::test]
async fn complete_passthrough_tool_call_without_terminal_gets_eof_terminal() {
    let mut parsed = chunk("", false);
    let choice = &mut parsed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call-complete".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("get_weather".to_string()),
            arguments: Some("{}".to_string()),
        }),
        ..Default::default()
    }]);

    let responses = apply_stream(
        stream::iter([parsed]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let finish_reasons: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.finish_reason)
        .collect();

    assert_eq!(
        finish_reasons,
        vec![FinishReason::ToolCalls],
        "a complete already-parsed call must receive a terminal reason at EOF"
    );
}

#[tokio::test]
async fn passthrough_argument_fragments_are_combined_before_eof_validation() {
    let mut first = chunk("", false);
    let first_choice = &mut first.data.as_mut().unwrap().inner.choices[0];
    first_choice.delta.content = None;
    first_choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call-fragmented".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("get_weather".to_string()),
            arguments: Some(r#"{"city":"#.to_string()),
        }),
        ..Default::default()
    }]);

    let mut second = chunk("", false);
    let second_choice = &mut second.data.as_mut().unwrap().inner.choices[0];
    second_choice.delta.content = None;
    second_choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: None,
        r#type: None,
        function: Some(FunctionCallStream {
            name: None,
            arguments: Some(r#""Tokyo"}"#.to_string()),
        }),
        ..Default::default()
    }]);

    let responses = apply_stream(
        stream::iter([first, second]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let arguments = choices
        .iter()
        .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
        .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
        .fold(String::new(), |mut all, fragment| {
            all.push_str(fragment);
            all
        });
    let finish_reasons: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.finish_reason)
        .collect();

    assert_eq!(arguments, r#"{"city":"Tokyo"}"#);
    assert_eq!(finish_reasons, vec![FinishReason::ToolCalls]);
}

#[tokio::test]
async fn valid_json_prefix_with_invalid_suffix_stays_provisional_at_eof() {
    let parsed_fragment = |arguments: &str, id: Option<&str>| {
        let mut parsed = chunk("", false);
        let choice = &mut parsed.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: id.map(str::to_string),
            r#type: id.map(|_| FunctionType::Function),
            function: Some(FunctionCallStream {
                name: id.map(|_| "get_weather".to_string()),
                arguments: Some(arguments.to_string()),
            }),
            ..Default::default()
        }]);
        parsed
    };

    let responses = apply_stream(
        stream::iter([
            parsed_fragment("{}", Some("call-suffix")),
            parsed_fragment(" trailing", None),
        ]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>()
            == vec![FinishReason::Length],
        "a valid JSON prefix followed by malformed bytes must remain provisional"
    );
}

#[tokio::test]
async fn already_parsed_gap_terminal_precedes_a_usage_only_chunk() {
    let tool_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let mut already_parsed = chunk("", false);
    let choice = &mut already_parsed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("already parsed".to_string());
    let mut usage_only = chunk("unused", false);
    usage_only.data.as_mut().unwrap().inner.choices.clear();
    usage_only.data.as_mut().unwrap().inner.usage = Some(
        serde_json::from_value(
            serde_json::json!({"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}),
        )
        .unwrap(),
    );

    let responses = apply_stream(
        stream::iter([chunk(tool_call, false), already_parsed, usage_only]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let usage_position = responses
        .iter()
        .position(|response| {
            response
                .data
                .as_ref()
                .is_some_and(|data| data.inner.choices.is_empty())
        })
        .expect("the usage-only chunk must still be forwarded");
    let terminal_position = responses
        .iter()
        .position(|response| {
            response.data.as_ref().is_some_and(|data| {
                data.inner
                    .choices
                    .iter()
                    .any(|choice| choice.finish_reason == Some(FinishReason::ToolCalls))
            })
        })
        .expect("a history-only choice must still get exactly one ToolCalls terminal");
    assert!(
        terminal_position < usage_position,
        "the terminal must precede the usage-only chunk, per OpenAI stream ordering"
    );
}

#[tokio::test]
async fn non_tool_choice_gets_no_synthetic_terminal_after_a_gap() {
    let mut already_parsed = chunk("", false);
    let choice = &mut already_parsed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("already parsed".to_string());

    // No tool call anywhere: raw plain text, then a gap, then immediate EOF.
    let responses = apply_stream(
        stream::iter([chunk("plain text, no tool call", false), already_parsed]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert!(
        choices.iter().all(|choice| choice.finish_reason.is_none()),
        "a choice that never called a tool must not get a synthesized terminal \
         after a gap, matching the text-only, no-signal contract a live \
         ChoiceState already has"
    );
}

#[tokio::test]
async fn late_tool_metadata_and_fragmented_name_survive_index_open() {
    fn call_chunk(
        id: Option<&str>,
        call_type: Option<FunctionType>,
        name: Option<&str>,
        arguments: Option<&str>,
        finish: Option<FinishReason>,
    ) -> Annotated<NvCreateChatCompletionStreamResponse> {
        let mut response = chunk("", false);
        let choice = &mut response.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: id.map(str::to_string),
            r#type: call_type,
            function: Some(FunctionCallStream {
                name: name.map(str::to_string),
                arguments: arguments.map(str::to_string),
            }),
            ..Default::default()
        }]);
        choice.finish_reason = finish;
        response
    }

    let responses = apply_stream(
        stream::iter([
            call_chunk(None, None, None, Some(""), None),
            call_chunk(
                Some("call-late"),
                Some(FunctionType::Function),
                Some("get_"),
                None,
                None,
            ),
            call_chunk(
                None,
                None,
                Some("weather"),
                Some("{}"),
                Some(FinishReason::ToolCalls),
            ),
        ]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let calls: Vec<_> = choices
        .iter()
        .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
        .collect();

    assert_eq!(
        calls
            .iter()
            .filter_map(|call| call.id.as_deref())
            .collect::<Vec<_>>(),
        vec!["call-late"],
        "an ID supplied after its tool index opens must be forwarded once"
    );
    assert_eq!(
        calls
            .iter()
            .filter_map(|call| call.r#type.clone())
            .collect::<Vec<_>>(),
        vec![FunctionType::Function],
        "a type supplied after its tool index opens must be forwarded once"
    );
    assert_eq!(
        calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.name.as_deref())
            .collect::<Vec<_>>(),
        vec!["get_", "weather"],
        "name fragments must remain separate deltas so clients can reassemble them"
    );
    assert_eq!(
        calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect::<Vec<_>>(),
        vec!["", "{}"]
    );
}

#[tokio::test]
async fn late_name_fragments_survive_raw_content_and_reasoning_detour() {
    let mut first = chunk("visible", false);
    first.data.as_mut().unwrap().inner.choices[0]
        .delta
        .tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call-once".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("get_".to_string()),
            arguments: Some(r#"{"city":"Pa"#.to_string()),
        }),
        ..Default::default()
    }]);
    let mut gap = chunk("", false);
    let gap_choice = &mut gap.data.as_mut().unwrap().inner.choices[0];
    gap_choice.delta.content = None;
    gap_choice.delta.reasoning_content = Some("reasoning gap".to_string());

    let mut resumed = chunk("", true);
    let resumed_choice = &mut resumed.data.as_mut().unwrap().inner.choices[0];
    resumed_choice.delta.content = None;
    resumed_choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("duplicate-id".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("weather".to_string()),
            arguments: Some("ris\"}".to_string()),
        }),
        ..Default::default()
    }]);
    resumed_choice.finish_reason = Some(FinishReason::ToolCalls);

    let responses = apply_stream(
        stream::iter([first, gap, resumed]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let calls: Vec<_> = choices
        .iter()
        .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
        .collect();

    assert_eq!(
        calls
            .iter()
            .filter_map(|call| call.id.as_deref())
            .collect::<Vec<_>>(),
        vec!["call-once"],
        "an already emitted ID must not be repeated after a detour"
    );
    assert_eq!(
        calls
            .iter()
            .filter_map(|call| call.r#type.clone())
            .collect::<Vec<_>>(),
        vec![FunctionType::Function],
        "an already emitted type must not be repeated after a detour"
    );
    assert_eq!(
        calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.name.as_deref())
            .collect::<Vec<_>>(),
        vec!["get_", "weather"],
        "a later name fragment must survive the parsed detour"
    );
    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>(),
        vec![FinishReason::ToolCalls]
    );
}

#[tokio::test]
async fn incomplete_upstream_call_stays_pending_across_reasoning_detour_at_eof() {
    let mut first = chunk("visible", false);
    first.data.as_mut().unwrap().inner.choices[0]
        .delta
        .tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call-pending".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("get_weather".to_string()),
            arguments: Some(r#"{"city":"Pa"#.to_string()),
        }),
        ..Default::default()
    }]);
    let mut gap = chunk("", false);
    let gap_choice = &mut gap.data.as_mut().unwrap().inner.choices[0];
    gap_choice.delta.content = None;
    gap_choice.delta.reasoning_content = Some("reasoning gap".to_string());

    let responses = apply_stream(
        stream::iter([first, gap]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert!(
        choices.iter().any(|choice| choice
            .delta
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())),
        "the partial upstream tool-call delta must remain visible"
    );
    assert!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>()
            == vec![FinishReason::Length],
        "EOF must not claim tool-call completion when upstream arguments remain partial"
    );
}

#[tokio::test]
#[allow(deprecated)]
async fn legacy_function_call_choice_gets_no_synthetic_terminal_after_a_gap() {
    // The legacy `function_call` field is a distinct signal from `tool_calls`
    // (it has its own `FinishReason::FunctionCall`, never `ToolCalls`) and, like
    // plain text, is never treated as a completed tool call anywhere in this file
    // (only complete parser events and terminal parsed deltas are). An
    // already-parsed chunk that only carries `function_call` must therefore get
    // the same "no synthetic terminal" treatment after a gap as plain text does,
    // matching the pre-existing live-`ChoiceState` precedent, not a new gap.
    let mut already_parsed = chunk("", false);
    let choice = &mut already_parsed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.function_call = Some(ChatCompletionStreamResponseDeltaFunctionCall {
        name: Some("get_weather".to_string()),
        arguments: Some(r#"{"city": "Tokyo"}"#.to_string()),
    });

    let responses = apply_stream(
        stream::iter([chunk("plain text, no tool call", false), already_parsed]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert!(
        choices.iter().all(|choice| choice.finish_reason.is_none()),
        "a choice whose only already-parsed signal is the legacy function_call \
         field must not get a synthesized ToolCalls terminal after a gap"
    );
}

#[tokio::test]
async fn already_parsed_terminal_carrying_its_own_tool_call_normalizes_stop() {
    // The live ChoiceState never emits a tool call itself (raw reasoning-only
    // content), but the already-parsed TERMINAL chunk that follows carries its
    // own `delta.tool_calls` together with `finish_reason: Stop` in the same
    // chunk. The state's own completed-call state is false, but the record must still
    // pick up this chunk's own tool call and normalize Stop -> ToolCalls.
    let mut terminal = chunk("", true);
    let choice = &mut terminal.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call-terminal".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("get_weather".to_string()),
            arguments: Some("{}".to_string()),
        }),
        ..Default::default()
    }]);

    let responses = apply_stream(
        stream::iter([chunk("reasoning only, no tool call", false), terminal]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>(),
        vec![FinishReason::ToolCalls],
        "an already-parsed terminal chunk that itself carries a tool call must \
         normalize Stop to ToolCalls even when the live state never saw one"
    );
}

#[tokio::test]
async fn terminal_already_parsed_call_keeps_index_separate_from_raw_call() {
    let raw_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let mut terminal = chunk("", true);
    let choice = &mut terminal.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call-complete".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("upstream".to_string()),
            arguments: Some("{\"city\":\"Paris\"}".to_string()),
        }),
        ..Default::default()
    }]);

    let responses = apply_stream(
        stream::iter([chunk(raw_call, false), terminal]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let tool_call_chunks: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.delta.tool_calls.as_ref())
        .flatten()
        .collect();
    assert_eq!(tool_call_chunks.len(), 2);
    assert_eq!(tool_call_chunks[0].index, 0);
    assert!(tool_call_chunks[0].id.is_some());
    assert_eq!(tool_call_chunks[1].index, 1);
    assert_eq!(tool_call_chunks[1].id.as_deref(), Some("call-complete"));
    assert_eq!(
        tool_call_chunks[1]
            .function
            .as_ref()
            .and_then(|function| function.name.as_deref()),
        Some("upstream")
    );
    let finish_reasons: Vec<_> = choices
        .iter()
        .filter_map(|choice| choice.finish_reason)
        .collect();
    assert_eq!(
        finish_reasons,
        vec![FinishReason::ToolCalls],
        "the already-parsed completion must clear the matching raw provisional call"
    );
}

#[tokio::test]
async fn finish_recovery_during_reasoning_detour_clears_pending_tool_call() {
    let complete_json_without_call_end = concat!(
        "<|tool_calls_section_begin|><|tool_call_begin|>",
        "functions.get_weather:0<|tool_call_argument_begin|>",
        r#"{"city":"Paris"}"#
    );
    for terminal in [false, true] {
        let mut parsed_reasoning = chunk("", terminal);
        let choice = &mut parsed_reasoning.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("already parsed reasoning".to_string());

        let responses = apply_stream(
            stream::iter([
                chunk(complete_json_without_call_end, false),
                parsed_reasoning,
            ]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            "kimi_k2",
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);

        assert!(
            choices
                .iter()
                .any(|choice| choice.delta.tool_calls.is_some()),
            "finish must recover the valid JSON body when Kimi omits the call-end marker"
        );
        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::ToolCalls],
            "finish-recovered calls must clear the pending index before EOF (terminal={terminal})"
        );
    }
}

#[tokio::test]
async fn raw_provisional_call_downgrades_already_parsed_tool_calls_terminal() {
    let mut parsed_terminal = chunk("", true);
    let choice = &mut parsed_terminal.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.reasoning_content = Some("already parsed reasoning".to_string());
    choice.finish_reason = Some(FinishReason::ToolCalls);

    let responses = apply_stream(
        stream::iter([
            chunk(
                "<tool_call><function=get_weather><parameter=city>Paris",
                false,
            ),
            parsed_terminal,
        ]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::Response,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert!(
        choices
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some())
    );
    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>(),
        vec![FinishReason::Length]
    );
}

#[tokio::test]
async fn qwen_call_start_marker_inside_parameter_value_survives_reasoning_detour_at_every_split() {
    async fn run(
        split: usize,
        continuation: &str,
    ) -> (
        String,
        String,
        bool,
        String,
        Vec<FinishReason>,
        Vec<u32>,
        usize,
        bool,
    ) {
        let mut reasoning_detour = chunk("", false);
        let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("interleaved thought".to_string());

        let responses = apply_stream(
            stream::iter([
                chunk("<tool_call><function=get_weather><parameter=city>Pa", false),
                reasoning_detour,
                chunk(&continuation[..split], false),
                chunk(&continuation[split..], true),
            ]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        let calls: Vec<_> = choices
            .iter()
            .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
            .collect();
        let arguments = calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        let name = calls
            .iter()
            .filter_map(|call| call.function.as_ref()?.name.as_deref())
            .collect();
        let reasoning = choices
            .iter()
            .any(|choice| choice.delta.reasoning_content.as_deref() == Some("interleaved thought"));
        let content = choices
            .iter()
            .filter_map(|choice| match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let finish_reasons = choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect();
        let mut indices: Vec<_> = calls.iter().map(|call| call.index).collect();
        indices.sort_unstable();
        indices.dedup();
        let call_ids: HashSet<_> = calls.iter().filter_map(|call| call.id.as_deref()).collect();
        let reasoning_position = choices
            .iter()
            .position(|choice| choice.delta.reasoning_content.is_some())
            .expect("the reasoning detour must be emitted");
        let streamed_before_reasoning = choices[..reasoning_position].iter().any(|choice| {
            choice.delta.tool_calls.as_ref().is_some_and(|calls| {
                calls.iter().any(|call| {
                    call.function
                        .as_ref()
                        .and_then(|function| function.arguments.as_deref())
                        .is_some_and(|arguments| !arguments.is_empty())
                })
            })
        });
        (
            arguments,
            name,
            reasoning,
            content,
            finish_reasons,
            indices,
            call_ids.len(),
            streamed_before_reasoning,
        )
    }

    let continuation = "<tool_call>after</parameter></function></tool_call>";
    let whole = run(continuation.len(), continuation).await;
    assert_eq!(whole.0, r#"{"city":"Pa<tool_call>after"}"#);
    assert_eq!(whole.1, "get_weather");
    assert!(whole.2, "the reasoning detour must stay visible");
    assert_eq!(whole.3, "", "native markup must not leak into content");
    assert_eq!(whole.4, vec![FinishReason::ToolCalls]);
    assert_eq!(whole.5, vec![0]);
    assert_eq!(whole.6, 1, "exactly one call ID must be emitted");
    assert!(whole.7, "argument bytes must stream before the detour");

    for split in 0..=continuation.len() {
        assert_eq!(
            run(split, continuation).await,
            whole,
            "splitting the continuation at byte {split} changed the result"
        );
    }
}

#[tokio::test]
async fn kimi_call_start_marker_inside_json_string_survives_reasoning_detour_at_every_split() {
    async fn run(split: usize, continuation: &str) -> (String, bool, String, Vec<FinishReason>) {
        let mut reasoning_detour = chunk("", false);
        let choice = &mut reasoning_detour.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("interleaved thought".to_string());

        let responses = apply_stream(
            stream::iter([
                chunk(
                    "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\":\"Pa",
                    false,
                ),
                reasoning_detour,
                chunk(&continuation[..split], false),
                chunk(&continuation[split..], true),
            ]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            "kimi_k2",
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        let arguments = choices
            .iter()
            .filter_map(|choice| choice.delta.tool_calls.as_ref())
            .flatten()
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        let reasoning = choices
            .iter()
            .any(|choice| choice.delta.reasoning_content.as_deref() == Some("interleaved thought"));
        let content = choices
            .iter()
            .filter_map(|choice| match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let finish_reasons = choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect();
        (arguments, reasoning, content, finish_reasons)
    }

    let continuation = "<|tool_call_begin|>after\"}<|tool_call_end|><|tool_calls_section_end|>";
    let whole = run(continuation.len(), continuation).await;
    assert_eq!(whole.0, "{\"city\":\"Pa<|tool_call_begin|>after\"}");
    assert!(whole.1, "the reasoning detour must stay visible");
    assert_eq!(whole.2, "", "no native markup may leak into content");
    assert_eq!(whole.3, vec![FinishReason::ToolCalls]);

    for split in 0..=continuation.len() {
        assert_eq!(
            run(split, continuation).await,
            whole,
            "splitting the continuation at byte {split} changed the result"
        );
    }
}

#[tokio::test]
async fn nonterminal_already_parsed_tool_fragment_does_not_complete_pending_call() {
    let first_partial = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Paris"
    );
    let mut parsed_fragment = chunk("", false);
    let choice = &mut parsed_fragment.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call-parsed".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("get_weather".to_string()),
            arguments: Some(r#"{"city":"Paris""#.to_string()),
        }),
        ..Default::default()
    }]);

    let responses = apply_stream(
        stream::iter([chunk(first_partial, false), parsed_fragment]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert!(
        choices
            .iter()
            .any(|choice| choice.delta.tool_calls.is_some()),
        "both the raw and already-parsed partial fragments should be emitted"
    );
    assert!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>()
            == vec![FinishReason::Length],
        "nonterminal parsed argument fragments must not synthesize a completed-call terminal"
    );
}

#[tokio::test]
async fn qwen_queued_reasoning_is_emitted_once_when_parsed_tool_call_interrupts() {
    #[derive(Debug, PartialEq)]
    enum OutputEvent {
        Content(String),
        Reasoning(String),
        ToolCall {
            index: u32,
            id: Option<String>,
            name: Option<String>,
            arguments: Option<String>,
        },
        Finish(FinishReason),
    }

    for terminal in [false, true] {
        let mut reasoning = chunk("", false);
        let choice = &mut reasoning.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("held thought".to_string());
        let source = reasoning.data.as_mut().unwrap();
        source.nvext = Some(serde_json::json!({"source": "reasoning"}));
        source.llm_metrics = Some(LLMMetricAnnotation {
            chunk_tokens: 7,
            ..Default::default()
        });
        reasoning.id = Some("reasoning-source-id".to_string());
        reasoning.event = Some("reasoning-source-event".to_string());
        reasoning.comment = Some(vec!["reasoning-source-comment".to_string()]);

        let mut parsed_call = chunk("", terminal);
        let choice = &mut parsed_call.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("parsed-call".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"NY"}"#.to_string()),
            }),
            ..Default::default()
        }]);

        let responses = apply_stream(
            stream::iter([chunk("before <tool_c", false), reasoning, parsed_call]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        let observed: Vec<_> = choices
            .iter()
            .flat_map(|choice| {
                let mut events = Vec::new();
                if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                    events.push(OutputEvent::Content(text.clone()));
                }
                if let Some(reasoning) = &choice.delta.reasoning_content {
                    events.push(OutputEvent::Reasoning(reasoning.clone()));
                }
                if let Some(calls) = &choice.delta.tool_calls {
                    events.extend(calls.iter().map(|call| {
                        let function = call.function.as_ref();
                        OutputEvent::ToolCall {
                            index: call.index,
                            id: call.id.clone(),
                            name: function.and_then(|function| function.name.clone()),
                            arguments: function.and_then(|function| function.arguments.clone()),
                        }
                    }));
                }
                if let Some(reason) = &choice.finish_reason {
                    events.push(OutputEvent::Finish(*reason));
                }
                events
            })
            .collect();
        let mut expected = vec![
            OutputEvent::Content("before ".to_string()),
            OutputEvent::Content("<tool_c".to_string()),
            OutputEvent::Reasoning("held thought".to_string()),
            OutputEvent::ToolCall {
                index: 0,
                id: Some("parsed-call".to_string()),
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"NY"}"#.to_string()),
            },
        ];
        // The explicit upstream terminal and a valid complete JSON call at EOF
        // must expose the same terminal event to stream consumers.
        expected.push(OutputEvent::Finish(FinishReason::ToolCalls));
        assert_eq!(
            observed, expected,
            "content, held reasoning, the parsed call, and terminal metadata must stay ordered and appear once"
        );
        let source_reasoning: Vec<_> = responses
            .iter()
            .filter(|response| {
                response.data.as_ref().is_some_and(|data| {
                    data.inner.choices.iter().any(|choice| {
                        choice.delta.reasoning_content.as_deref() == Some("held thought")
                    })
                })
            })
            .collect();
        assert_eq!(source_reasoning.len(), 1);
        assert_eq!(
            source_reasoning[0].id.as_deref(),
            Some("reasoning-source-id")
        );
        assert_eq!(
            source_reasoning[0].event.as_deref(),
            Some("reasoning-source-event")
        );
        assert_eq!(
            source_reasoning[0].comment.as_deref(),
            Some(["reasoning-source-comment".to_string()].as_slice())
        );
        let source_data = source_reasoning[0].data.as_ref().unwrap();
        assert_eq!(
            source_data.nvext,
            Some(serde_json::json!({"source": "reasoning"}))
        );
        assert_eq!(source_data.llm_metrics.as_ref().unwrap().chunk_tokens, 7);
    }
}

#[tokio::test]
async fn qwen_multi_choice_queued_reasoning_emits_source_metadata_once() {
    fn choice(index: u32, content: Option<&str>, reasoning: Option<&str>) -> ChatChoiceStream {
        ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: content.map(|text| ChatCompletionMessageContent::Text(text.to_string())),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: reasoning.map(str::to_string),
            },
            finish_reason: None,
            logprobs: None,
        }
    }

    for (reasoning_zero, reasoning_one) in [
        ("thought zero", "thought one"),
        ("", ""),
        ("", "thought one"),
        ("thought zero", ""),
    ] {
        let mut opening = chunk("", false);
        opening.data.as_mut().unwrap().inner.choices = vec![
            choice(0, Some("before <tool_c"), None),
            choice(1, Some("before <tool_c"), None),
        ];
        let mut reasoning = chunk("", false);
        let data = reasoning.data.as_mut().unwrap();
        data.inner.choices = vec![
            choice(0, None, Some(reasoning_zero)),
            choice(1, None, Some(reasoning_one)),
        ];
        data.nvext = Some(serde_json::json!({"source": "reasoning"}));
        data.llm_metrics = Some(LLMMetricAnnotation {
            chunk_tokens: 11,
            ..Default::default()
        });
        reasoning.id = Some("shared-reasoning-id".to_string());
        reasoning.event = Some("shared-reasoning-event".to_string());
        reasoning.comment = Some(vec!["shared-reasoning-comment".to_string()]);

        let mut continuation = chunk("", true);
        continuation.data.as_mut().unwrap().inner.choices = vec![
            choice(
                0,
                Some(
                    "all>\n<function=get_weather>\n<parameter=city>NY</parameter>\n</function>\n</tool_call>",
                ),
                None,
            ),
            choice(
                1,
                Some(
                    "all>\n<function=get_weather>\n<parameter=city>SF</parameter>\n</function>\n</tool_call>",
                ),
                None,
            ),
        ];
        let responses = apply_stream(
            stream::iter([opening, reasoning, continuation]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;

        let choices = collect_choices(&responses);
        for (index, expected) in [(0, reasoning_zero), (1, reasoning_one)] {
            assert_eq!(
                choices
                    .iter()
                    .filter(|choice| {
                        choice.index == index
                            && choice.delta.reasoning_content.as_deref() == Some(expected)
                    })
                    .count(),
                1,
                "choice {index} reasoning must remain attached to the correct choice once"
            );
        }
        assert_eq!(
            responses
                .iter()
                .filter(|response| response.id.as_deref() == Some("shared-reasoning-id"))
                .count(),
            1,
            "the source response id must be emitted once across the two queued choices"
        );
        assert_eq!(
            responses
                .iter()
                .filter(|response| response.event.as_deref() == Some("shared-reasoning-event"))
                .count(),
            1
        );
        assert_eq!(
            responses
                .iter()
                .filter(|response| response.comment.is_some())
                .count(),
            1
        );
        assert_eq!(
            responses
                .iter()
                .filter_map(|response| response.data.as_ref())
                .filter(|data| data.nvext.is_some())
                .count(),
            1
        );
        assert_eq!(
            responses
                .iter()
                .filter_map(|response| response.data.as_ref())
                .filter(|data| data.llm_metrics.is_some())
                .count(),
            1
        );
    }
}

#[tokio::test]
async fn separate_terminal_chunk_keeps_malformed_parsed_fragments_provisional() {
    for tool_calls_finish in [false, true] {
        let mut parsed_fragments = chunk("", false);
        let choice = &mut parsed_fragments.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.tool_calls = Some(vec![
            ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some("call-first".to_string()),
                r#type: Some(FunctionType::Function),
                function: Some(FunctionCallStream {
                    name: Some("get_weather".to_string()),
                    arguments: Some(r#"{"city":"Par"#.to_string()),
                }),
                ..Default::default()
            },
            ChatCompletionMessageToolCallChunk {
                index: 1,
                id: Some("call-second".to_string()),
                r#type: Some(FunctionType::Function),
                function: Some(FunctionCallStream {
                    name: Some("get_weather".to_string()),
                    arguments: Some(r#"{"city":"Tok"#.to_string()),
                }),
                ..Default::default()
            },
        ]);

        let mut terminal = chunk("", false);
        let choice = &mut terminal.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some("terminal reasoning".to_string());
        choice.finish_reason = Some(if tool_calls_finish {
            FinishReason::ToolCalls
        } else {
            FinishReason::Stop
        });

        let responses = apply_stream(
            stream::iter([parsed_fragments, terminal]),
            Some(weather_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        assert_eq!(
            choices
                .iter()
                .filter_map(|choice| choice.finish_reason)
                .collect::<Vec<_>>(),
            vec![FinishReason::Length],
            "a terminal reason must not complete malformed parsed fragments (tool_calls_finish={tool_calls_finish})"
        );
        let emitted_arguments: Vec<_> = choices
            .iter()
            .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
            .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
            .collect();
        assert_eq!(
            emitted_arguments,
            vec![r#"{"city":"Par"#, r#"{"city":"Tok"#],
            "malformed fragments remain identifiable but receive no successful terminal"
        );
    }
}

#[tokio::test]
async fn terminal_argument_fragments_are_validated_after_assembly_per_choice() {
    let fragments = chunk("raw prelude", false);
    let mut partials = chunk("", false);
    let first = &mut partials.data.as_mut().unwrap().inner.choices[0];
    first.delta.content = None;
    first.delta.tool_calls = Some(vec![
        ChatCompletionMessageToolCallChunk {
            index: 0,
            id: Some("call-invalid".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"#.to_string()),
            }),
            ..Default::default()
        },
        ChatCompletionMessageToolCallChunk {
            index: 1,
            id: Some("call-valid".to_string()),
            r#type: Some(FunctionType::Function),
            function: Some(FunctionCallStream {
                name: Some("get_weather".to_string()),
                arguments: Some(r#"{"city":"#.to_string()),
            }),
            ..Default::default()
        },
    ]);
    let mut second = first.clone();
    second.index = 1;
    second.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call-parameterless".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("get_weather".to_string()),
            arguments: Some(String::new()),
        }),
        ..Default::default()
    }]);
    partials.data.as_mut().unwrap().inner.choices.push(second);

    let mut terminal = chunk("", false);
    let first = &mut terminal.data.as_mut().unwrap().inner.choices[0];
    first.delta.content = None;
    first.finish_reason = Some(FinishReason::ToolCalls);
    first.delta.tool_calls = Some(vec![
        ChatCompletionMessageToolCallChunk {
            index: 0,
            id: None,
            r#type: None,
            function: Some(FunctionCallStream {
                name: None,
                arguments: Some("not-json".to_string()),
            }),
            ..Default::default()
        },
        ChatCompletionMessageToolCallChunk {
            index: 1,
            id: None,
            r#type: None,
            function: Some(FunctionCallStream {
                name: None,
                arguments: Some(r#""Tokyo"}"#.to_string()),
            }),
            ..Default::default()
        },
    ]);
    let mut second = first.clone();
    second.index = 1;
    second.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: None,
        r#type: None,
        function: Some(FunctionCallStream {
            name: None,
            arguments: Some(String::new()),
        }),
        ..Default::default()
    }]);
    terminal.data.as_mut().unwrap().inner.choices.push(second);

    let responses = apply_stream(
        stream::iter([fragments, partials, terminal]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);
    let finishes: HashMap<_, _> = choices
        .iter()
        .filter_map(|choice| choice.finish_reason.map(|reason| (choice.index, reason)))
        .collect();

    assert_eq!(finishes.get(&0), Some(&FinishReason::Length));
    assert_eq!(finishes.get(&1), Some(&FinishReason::ToolCalls));
    let calls: HashMap<_, _> = choices
        .iter()
        .flat_map(|choice| {
            choice
                .delta
                .tool_calls
                .iter()
                .flatten()
                .filter_map(move |call| {
                    call.id
                        .as_deref()
                        .map(|id| ((choice.index, call.index as usize), id))
                })
        })
        .collect();

    assert_eq!(calls.get(&(0, 0)), Some(&"call-invalid"));
    assert_eq!(calls.get(&(0, 1)), Some(&"call-valid"));
    assert_eq!(calls.get(&(1, 0)), Some(&"call-parameterless"));
}

#[tokio::test]
async fn already_parsed_first_terminal_chunk_normalizes_stop() {
    let mut already_parsed = chunk("", true);
    let choice = &mut already_parsed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call-preexisting".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("get_weather".to_string()),
            arguments: Some("{}".to_string()),
        }),
        ..Default::default()
    }]);

    // This choice index never has a raw chunk at all — no ChoiceState is ever
    // constructed for it — but the parsed chunk carries the terminal Stop signal.
    let responses = apply_stream(
        stream::iter([already_parsed]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>(),
        vec![FinishReason::ToolCalls],
        "an already-parsed tool call on the terminal chunk must normalize Stop to ToolCalls"
    );
}

#[tokio::test]
async fn multi_choice_interleave_finishes_each_unfinished_index_once() {
    fn choice(index: u32, text: &str, finish: bool) -> ChatChoiceStream {
        ChatChoiceStream {
            index,
            delta: ChatCompletionStreamResponseDelta {
                role: Some(Role::Assistant),
                content: Some(ChatCompletionMessageContent::Text(text.to_string())),
                tool_calls: None,
                function_call: None,
                refusal: None,
                reasoning_content: None,
            },
            finish_reason: finish.then_some(FinishReason::Stop),
            logprobs: None,
        }
    }
    fn response(choices: Vec<ChatChoiceStream>) -> Annotated<NvCreateChatCompletionStreamResponse> {
        #[allow(deprecated)]
        let response = NvCreateChatCompletionStreamResponse {
            inner: CreateChatCompletionStreamResponse {
                id: "test".to_string(),
                choices,
                created: 0,
                model: "qwen3".to_string(),
                system_fingerprint: None,
                service_tier: None,
                object: "chat.completion.chunk".to_string(),
                usage: None,
            },
            nvext: None,
            llm_metrics: None,
            prompt_logprobs: None,
            tool_call_completion: Vec::new(),
        };
        Annotated::from_data(response)
    }
    let tool_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    // Index 0: live the whole time, never removed, never explicitly finished —
    // the ordinary backstop case. Index 1: raw tool call, then a gap that removes
    // its state and is never touched again — history-only at EOF. Index 2: raw
    // tool call, explicitly finished right away, never touched again.
    let first = response(vec![
        choice(0, tool_call, false),
        choice(1, tool_call, false),
        choice(2, tool_call, false),
    ]);
    let mut second = response(vec![choice(0, " more", false), choice(2, " done", true)]);
    {
        let mut gap = choice(1, "", false);
        gap.delta.content = None;
        gap.delta.reasoning_content = Some("already parsed".to_string());
        second.data.as_mut().unwrap().inner.choices.push(gap);
    }
    // Index 1 gets no further chunks after this — it must stay history-only
    // through to EOF for this to actually exercise the bug.

    let responses = apply_stream(
        stream::iter([first, second]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    let finish_for = |index: u32| {
        choices
            .iter()
            .filter(|choice| choice.index == index)
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>()
    };
    assert_eq!(
        finish_for(0),
        vec![FinishReason::ToolCalls],
        "the ordinary live choice must get exactly one terminal via the backstop"
    );
    assert_eq!(
        finish_for(1),
        vec![FinishReason::ToolCalls],
        "the history-only choice must get exactly one terminal, not zero"
    );
    assert_eq!(
        finish_for(2),
        vec![FinishReason::ToolCalls],
        "the already-explicitly-finished choice must not be finished a second \
         time by the EOF backstop"
    );
}

#[tokio::test]
async fn finished_choices_keep_other_choices_and_late_metadata() {
    use LLMMetricAnnotation;

    let mut first = chunk("A", true);
    let mut active = empty_choice(1);
    active.delta.content = Some(ChatCompletionMessageContent::Text("B".into()));
    first.data.as_mut().unwrap().inner.choices.push(active);

    let mut mixed = chunk("<think>late</think>", false);
    let mut terminal = empty_choice(1);
    terminal.delta.content = Some(ChatCompletionMessageContent::Text("C".into()));
    terminal.delta.reasoning_content = Some("r".into());
    terminal.finish_reason = Some(FinishReason::Stop);
    mixed.data.as_mut().unwrap().inner.choices.push(terminal);

    let mut late = chunk("late raw payload", false);
    late.id = Some("late-metadata".into());
    let data = late.data.as_mut().unwrap();
    let mut parsed = empty_choice(1);
    parsed.delta.reasoning_content = Some("late parsed payload".into());
    parsed.finish_reason = Some(FinishReason::Stop);
    data.inner.choices.push(parsed);
    data.inner.usage = Some(
        serde_json::from_value(serde_json::json!({
            "prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3
        }))
        .unwrap(),
    );
    data.nvext = Some(serde_json::json!({"completion_token_ids": [42]}));
    data.llm_metrics = Some(LLMMetricAnnotation {
        chunk_tokens: 2,
        ..Default::default()
    });

    let responses: Vec<_> = apply_stream(
        stream::iter([first, mixed, late]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect()
    .await;
    let payloads: Vec<_> = collect_choices(&responses)
        .into_iter()
        .map(|choice| {
            let text = match &choice.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            };
            (
                choice.index,
                text,
                choice.delta.reasoning_content.as_deref(),
                choice.finish_reason,
            )
        })
        .collect();
    assert_eq!(
        payloads,
        [
            (0, Some("A"), None, Some(FinishReason::Stop)),
            (1, Some("B"), None, None),
            (1, Some("C"), Some("r"), Some(FinishReason::Stop)),
        ]
    );
    assert_eq!(
        responses
            .iter()
            .filter_map(|response| response.data.as_ref())
            .filter(|data| data.inner.usage.is_some())
            .count(),
        1
    );
    let tail = responses.last().unwrap();
    assert_eq!(tail.id.as_deref(), Some("late-metadata"));
    let metadata = tail.data.as_ref().unwrap();
    assert!(metadata.inner.choices.is_empty());
    assert_eq!(metadata.inner.usage.as_ref().unwrap().total_tokens, 3);
    assert_eq!(
        metadata.nvext,
        Some(serde_json::json!({"completion_token_ids": [42]}))
    );
    assert_eq!(metadata.llm_metrics.as_ref().unwrap().chunk_tokens, 2);
}

#[tokio::test]
async fn three_consecutive_already_parsed_gaps_then_eof_finishes_once() {
    let tool_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let already_parsed_chunk = |reasoning: &str| {
        let mut c = chunk("", false);
        let choice = &mut c.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some(reasoning.to_string());
        c
    };

    // Three already-parsed chunks in a row, no raw chunk between them, no
    // explicit terminal ever, then the stream just ends.
    let responses = apply_stream(
        stream::iter([
            chunk(tool_call, false),
            already_parsed_chunk("first"),
            already_parsed_chunk("second"),
            already_parsed_chunk("third"),
        ]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>(),
        vec![FinishReason::ToolCalls],
        "N consecutive already-parsed gaps with no explicit terminal must still \
         produce exactly one ToolCalls terminal at EOF, not zero and not N"
    );
}

#[tokio::test]
async fn raw_gap_resume_second_gap_then_eof_finishes_once_in_order() {
    // Distinct from `three_consecutive_already_parsed_gaps_then_eof_finishes_once`
    // (which never resumes with a raw chunk in between): this drives raw -> gap ->
    // raw-resumed (a fresh `ChoiceState`, completed-call state reseeded from the
    // `ChoiceRecord`) -> a SECOND gap on that resumed instance -> EOF, so the
    // fold-into-record step at removal runs twice on two different `ChoiceState`
    // instances for the same choice index.
    let tool_call = concat!(
        "<tool_call>\n<function=get_weather>\n",
        "<parameter=city>Tokyo</parameter>\n</function>\n</tool_call>"
    );
    let already_parsed_chunk = |reasoning: &str| {
        let mut c = chunk("", false);
        let choice = &mut c.data.as_mut().unwrap().inner.choices[0];
        choice.delta.content = None;
        choice.delta.reasoning_content = Some(reasoning.to_string());
        c
    };

    let responses = apply_stream(
        stream::iter([
            chunk(tool_call, false),
            already_parsed_chunk("first gap"),
            chunk(" resumed text", false),
            already_parsed_chunk("second gap"),
        ]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    let ordered: Vec<_> = choices
        .iter()
        .filter_map(|choice| {
            if choice.delta.tool_calls.is_some() {
                Some("tool_call")
            } else if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                if text.is_empty() {
                    None
                } else {
                    Some("content")
                }
            } else {
                choice
                    .delta
                    .reasoning_content
                    .as_deref()
                    .map(|_| "reasoning")
            }
        })
        .collect();
    assert_eq!(
        ordered,
        vec!["tool_call", "reasoning", "content", "reasoning"],
        "delivery order across the raw/gap/resume/gap sequence must be preserved: \
         tool call, first gap's reasoning, resumed content, second gap's reasoning"
    );

    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>(),
        vec![FinishReason::ToolCalls],
        "a second gap on a resumed (fresh) ChoiceState must still fold into the \
         same choice's record and produce exactly one ToolCalls terminal at EOF, \
         not zero and not two"
    );
}

#[tokio::test]
async fn text_only_stream_gets_no_synthetic_finish_reason() {
    let responses = apply_stream(
        stream::iter([chunk("hello world", false)]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;

    let choices = collect_choices(&responses);
    assert!(
        choices.iter().all(|choice| choice.finish_reason.is_none()),
        "there is no signal to synthesize a finish_reason from"
    );
}

// --- error recovery ----------------------------------------------------------

#[test]
fn a_failed_parser_surfaces_buffered_text_and_stops_parsing() {
    let mut state = test_state();
    // The parser releases the settled text immediately and holds back only the
    // bytes that could still turn out to be a `<tool_call>` opener.
    assert_eq!(
        state.push("hello <tool_c"),
        vec![UnifiedParserEvent::Text("hello ".into())]
    );
    // Now force the failure path with those held-back bytes still buffered.
    let recovered = state.give_up("");
    assert_eq!(
        recovered,
        vec![UnifiedParserEvent::Text("<tool_c".into())],
        "buffered bytes must be surfaced, not silently dropped"
    );
    assert!(state.failed);
    // Every later chunk now passes through as plain text.
    assert_eq!(
        state.push("more"),
        vec![UnifiedParserEvent::Text("more".into())]
    );
    assert!(state.finish().is_empty());
}

#[test]
fn a_failed_push_keeps_committed_events_before_recovered_bytes() {
    let mut state = ChoiceState {
        glm_native_input: None,
        incomplete_native_call: false,
        family: "test".to_string(),
        parsers_v2_selected: false,
        recording: None,
        parser: Box::new(PartialCommitParser {
            recovered: "<broken>".to_string(),
        }),
        tool_output_mode: UnifiedToolOutputMode::Native,
        qwen_native_lexical_state: QwenNativeLexicalState::default(),
        pending_qwen_opener_text: String::new(),
        pending_qwen_opener_reasoning: Vec::new(),
        qwen_reasoning_event_positions: HashSet::new(),
        qwen_reasoning_choice_positions: HashSet::new(),
        opened_calls: HashSet::new(),
        tool_ids_emitted: HashSet::new(),
        tool_types_emitted: HashSet::new(),
        parser_tool_indices: HashMap::new(),
        passthrough_tool_indices: HashMap::new(),
        tool_index_offset: 0,
        next_tool_index: 0,
        pending_tool_calls: HashSet::new(),
        parser_call_completion: HashMap::new(),
        pending_parsed_tool_calls: HashMap::new(),
        tool_emitted: false,
        failed: false,
    };

    assert_eq!(
        state.push("ignored"),
        vec![
            UnifiedParserEvent::Text("committed".to_string()),
            UnifiedParserEvent::Text("<broken>".to_string()),
        ]
    );
}

#[tokio::test]
async fn terminal_error_suppresses_eof_recovery() {
    let responses = apply_stream(
        stream::iter([
            chunk("hello <tool_c", false),
            Annotated::from_error("backend exploded"),
        ]),
        None,
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let error_position = responses
        .iter()
        .position(Annotated::is_error)
        .expect("terminal error");

    assert!(
        responses[error_position + 1..]
            .iter()
            .all(|response| response.data.is_none()),
        "no parser data may be emitted after a terminal error"
    );
}

#[test]
fn give_up_falls_back_to_the_chunk_when_nothing_was_buffered() {
    let mut state = test_state();
    assert_eq!(
        state.give_up("the chunk that broke it"),
        vec![UnifiedParserEvent::Text("the chunk that broke it".into())]
    );
}

// --- batch -------------------------------------------------------------------

#[test]
fn repeated_quoted_markers_are_excluded_from_prefill_classification() {
    let literal = "The literal \"<think>private</think>\" stays visible. ".repeat(1024);
    let mut shield = QuotedControlMarkerProbe::new(QWEN3_UNIFIED_FAMILY);
    let mut masked = shield.push(&literal);
    masked.push_str(&shield.finish());
    assert!(!masked.contains("</think>"));
    assert_eq!(masked.len(), literal.len());
}

#[test]
fn contains_unquoted_marker_finds_real_marker_past_many_unmatched_escaped_quotes() {
    // Every `"` here is immediately preceded by an escaping backslash, so none of
    // them closes any other: each is individually a "does an unescaped close exist
    // later" candidate. `later_unescaped_quote_closes` answers all of these in one
    // O(n) pass; a naive per-candidate rescan (the pre-fix `has_unescaped_close`
    // called fresh per candidate) would instead be O(n^2) on input shaped like
    // this. This case is a deterministic correctness check, not a timing
    // assertion — see the stage evidence ledger for measured old-vs-new behavior.
    let unit = "a".repeat(50) + "\\\"";
    let filler: String = unit.repeat(4000);
    let content = format!("{filler}</think>visible");

    assert!(
        contains_unquoted_marker(&content, "</think>"),
        "the real marker after {} chars of adversarial unmatched-escaped-quote \
         filler must still be found",
        content.len()
    );
}

#[tokio::test]
async fn qwen_reasoning_detours_preserve_native_grammar_at_every_utf8_split() {
    async fn observe(
        input: Vec<Annotated<NvCreateChatCompletionStreamResponse>>,
        tools: Vec<ToolDefinition>,
    ) -> (
        String,
        Vec<(u32, String, String)>,
        Vec<FinishReason>,
        String,
    ) {
        let responses = apply_stream(
            stream::iter(input),
            Some(tools),
            None,
            false,
            UnifiedParserStartingState::None,
            QWEN3_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        let mut calls = std::collections::BTreeMap::<u32, (String, String)>::new();
        let mut text = String::new();
        let mut reasoning = String::new();
        let mut finishes = Vec::new();
        for choice in choices {
            if let Some(ChatCompletionMessageContent::Text(value)) = &choice.delta.content {
                text.push_str(value);
            }
            if let Some(value) = &choice.delta.reasoning_content {
                reasoning.push_str(value);
            }
            if let Some(value) = choice.finish_reason {
                finishes.push(value);
            }
            for call in choice.delta.tool_calls.iter().flatten() {
                let entry = calls.entry(call.index).or_default();
                if let Some(function) = &call.function {
                    if let Some(name) = &function.name {
                        entry.0.push_str(name);
                    }
                    if let Some(arguments) = &function.arguments {
                        entry.1.push_str(arguments);
                    }
                }
            }
        }
        (
            text,
            calls
                .into_iter()
                .map(|(index, (name, arguments))| (index, name, arguments))
                .collect(),
            finishes,
            reasoning,
        )
    }
    let cases = [
        (
            serde_json::json!({"type":"string"}),
            "prefix </function><tool_call> literal",
        ),
        (
            serde_json::json!({"type":"string"}),
            "quote \\\" é <tool_call> literal",
        ),
        (
            serde_json::json!({"type":"object"}),
            r#"{"a":"é <tool_call> literal"}"#,
        ),
        (
            serde_json::json!({"type":"array"}),
            r#"["é <tool_call>", {"a":true}]"#,
        ),
        (serde_json::json!({"type":"integer"}), "42"),
        (serde_json::json!({"$ref":"#/$defs/value"}), r#"{"a":"é"}"#),
        (
            serde_json::json!({"type":["object","null"]}),
            r#"{"a":"é"}"#,
        ),
        (serde_json::json!({}), r#"{"a":"é"}"#),
        // Qwen closes parameters literally even inside JSON-looking text.
        // Pin its whole-input recovery rather than inventing a JSON boundary.
        (
            serde_json::json!({"type":"object"}),
            r#"{"a":"</parameter></tool_call><tool_call>literal"}"#,
        ),
    ];
    for (schema, value) in cases {
        let tools = vec![ToolDefinition {
            name: "f".into(),
            parameters: Some(
                serde_json::json!({"type":"object","properties":{"x":schema},"$defs":{"value":{"type":"object"}}}),
            ),
            strict: None,
        }];
        let input = format!(
            "<tool_call><function=f><parameter=x>{value}</parameter></function></tool_call>"
        );
        let whole = observe(vec![chunk(&input, true)], tools.clone()).await;
        assert_eq!(
            whole.1.len(),
            1,
            "whole-input control must emit f: {value:?}"
        );
        assert_eq!(whole.1[0].1, "f");
        if value.contains("</parameter>") {
            assert_eq!(whole.1[0].2, r#"{"x":"{\"a\":\""}"#);
        }
        for split in (1..input.len()).filter(|at| input.is_char_boundary(*at)) {
            let mut detour = chunk("", false);
            let delta = &mut detour.data.as_mut().unwrap().inner.choices[0].delta;
            delta.content = None;
            delta.reasoning_content = Some("detour".into());
            let actual = observe(
                vec![
                    chunk(&input[..split], false),
                    detour,
                    chunk(&input[split..], true),
                ],
                tools.clone(),
            )
            .await;
            assert_eq!(
                (&actual.0, &actual.1, &actual.2),
                (&whole.0, &whole.1, &whole.2),
                "value={value:?} split={split}"
            );
            assert_eq!(
                actual.3, "detour",
                "reasoning must survive once at split={split}"
            );
        }
    }
}

#[tokio::test]
async fn qwen_queued_reasoning_preserves_choice_envelope_on_resolution_and_eof() {
    for suffix in [
        Some("all><function=get_weather><parameter=city>Paris</parameter></function></tool_call>"),
        Some("at literal"),
        None,
    ] {
        for refusal in [None, Some("declined")] {
            let mut opening = chunk("<tool_c", false);
            opening.data.as_mut().unwrap().inner.choices[0].delta.role = None;
            let mut source = chunk("", false);
            let original = &mut source.data.as_mut().unwrap().inner.choices[0];
            original.delta.content = None;
            original.delta.reasoning_content = Some("detour".into());
            original.delta.refusal = refusal.map(str::to_string);
            original.logprobs = Some(serde_json::from_value(serde_json::json!({"content":[{"token":"detour","logprob":-0.5,"bytes":[100],"top_logprobs":[]}],"refusal":null})).unwrap());
            let expected = original.clone();
            let mut records = HashMap::from([(
                0,
                ChoiceRecord {
                    pending_qwen_opener_responses: VecDeque::from([
                        PendingQwenReasoningResponse::<
                            Annotated<NvCreateChatCompletionStreamResponse>,
                        > {
                            text: "detour".into(),
                            spans: Spans::new(),
                            choice: expected.clone(),
                            data: source.data.clone().unwrap(),
                            annotation: (
                                source.id.clone(),
                                source.event.clone(),
                                source.comment.clone(),
                                source.error.clone(),
                            ),
                        },
                    ]),
                    ..ChoiceRecord::default()
                },
            )]);
            let fallback = finish_unterminated_choices(
                &mut HashMap::new(),
                &mut records,
                QWEN3_UNIFIED_FAMILY,
                false,
                false,
            );
            assert_eq!(fallback.len(), 1);
            let FinishedStreamChunk::Source(fallback) = &fallback[0] else {
                panic!("queued source must retain its source response");
            };
            assert_eq!(fallback.0.inner.choices, std::slice::from_ref(&expected));
            let mut input = vec![opening, source];
            if let Some(suffix) = suffix {
                let mut closing = chunk(suffix, true);
                closing.data.as_mut().unwrap().inner.choices[0].delta.role = None;
                input.push(closing);
            }
            let responses = apply_stream(
                stream::iter(input),
                Some(weather_tools()),
                None,
                false,
                UnifiedParserStartingState::None,
                QWEN3_UNIFIED_FAMILY,
            )
            .collect::<Vec<_>>()
            .await;
            let choices = collect_choices(&responses);
            let reasoning: Vec<_> = choices
                .iter()
                .filter(|choice| choice.delta.reasoning_content.as_deref() == Some("detour"))
                .collect();
            assert_eq!(reasoning.len(), 1, "suffix={suffix:?} refusal={refusal:?}");
            assert_eq!(
                *reasoning[0], &expected,
                "queued or immediate reasoning must retain its untouched choice envelope"
            );
            assert_eq!(
                choices
                    .iter()
                    .filter(|choice| choice.delta.role.is_some())
                    .count(),
                1
            );
            assert_eq!(
                choices
                    .iter()
                    .filter(|choice| choice.logprobs.is_some())
                    .count(),
                1
            );
        }
    }
}
#[tokio::test]
#[ignore = "Parser 0.7.17 missing-parameter-close recovery delays text across external reasoning; run explicitly to measure ordering"]
async fn qwen_recovered_parameter_close_restores_opener_ordering_at_every_split() {
    for first in [
        "<tool_call><function=get_weather><parameter=city>Paris</function></tool_call>",
        "<tool_call><function=get_weather><parameter=city</function></tool_call>",
    ] {
        for split in 1..first.len() {
            let mut reasoning = chunk("", false);
            let delta = &mut reasoning.data.as_mut().unwrap().inner.choices[0].delta;
            delta.content = None;
            delta.reasoning_content = Some("thought".into());
            let responses = apply_stream(
                stream::iter([
                    chunk(&first[..split], false),
                    chunk(&first[split..], false),
                    chunk("before <tool_c", false),
                    reasoning,
                    chunk("XYZanswer", true),
                ]),
                Some(weather_tools()),
                None,
                false,
                UnifiedParserStartingState::None,
                QWEN3_UNIFIED_FAMILY,
            )
            .collect::<Vec<_>>()
            .await;
            let choices = collect_choices(&responses);
            let mut sides = [String::new(), String::new()];
            let mut seen_reasoning = false;
            for choice in choices {
                if let Some(ChatCompletionMessageContent::Text(text)) = &choice.delta.content {
                    sides[usize::from(seen_reasoning)].push_str(text);
                }
                if choice.delta.reasoning_content.as_deref() == Some("thought") {
                    seen_reasoning = true;
                }
            }
            assert!(seen_reasoning);
            assert_eq!(
                sides,
                ["before <tool_c".to_string(), "XYZanswer".to_string()],
                "split={split}"
            );
        }
    }
}

pub(crate) fn eight_family_calls(value: &str) -> Vec<(&'static str, String)> {
    native_write_file_calls(&[
        ("count", serde_json::json!(2)),
        ("content", serde_json::json!(value)),
    ])
}

pub(crate) fn ordered_arguments(fields: &[(&str, serde_json::Value)]) -> String {
    format!(
        "{{{}}}",
        fields
            .iter()
            .map(|(key, value)| format!("{}:{value}", serde_json::to_string(key).unwrap()))
            .collect::<Vec<_>>()
            .join(",")
    )
}

pub(crate) fn native_write_file_calls(
    fields: &[(&str, serde_json::Value)],
) -> Vec<(&'static str, String)> {
    [QWEN3_UNIFIED_FAMILY, KIMI_K2_UNIFIED_FAMILY, KIMI_K3_UNIFIED_FAMILY, DEEPSEEK_V4_UNIFIED_FAMILY, DEEPSEEK_V41_UNIFIED_FAMILY, GEMMA4_UNIFIED_FAMILY, GLM47_UNIFIED_FAMILY, MUSE_UNIFIED_FAMILY]
        .into_iter().map(|family| {
            let parameters: String = fields.iter().map(|(key, value)| {
                let raw = value.as_str().map(str::to_owned).unwrap_or_else(|| value.to_string());
                match family {
                    QWEN3_UNIFIED_FAMILY => format!("<parameter={key}>{raw}</parameter>"),
                    KIMI_K3_UNIFIED_FAMILY => format!("<|open|>argument key=\"{key}\" type=\"{}\"<|sep|>{raw}<|close|>argument<|sep|>", if value.is_string() {"string"} else {"number"}),
                    DEEPSEEK_V4_UNIFIED_FAMILY | DEEPSEEK_V41_UNIFIED_FAMILY => {
                        let gap = if family == DEEPSEEK_V41_UNIFIED_FAMILY {" "} else {""};
                        format!("<｜DSML｜{gap}parameter name=\"{key}\" string=\"{}\">{raw}</｜DSML｜{gap}parameter>", value.is_string())
                    },
                    GLM47_UNIFIED_FAMILY => format!("<arg_key>{key}</arg_key><arg_value>{raw}</arg_value>"),
                    MUSE_UNIFIED_FAMILY => format!("<atem:parameter name=\"{key}\">{raw}</atem:parameter>"),
                    _ => String::new(),
                }
            }).collect();
            let input = match family {
                QWEN3_UNIFIED_FAMILY => format!("<tool_call><function=write_file>{parameters}</function></tool_call>"),
                KIMI_K2_UNIFIED_FAMILY => format!("<|tool_calls_section_begin|><|tool_call_begin|>functions.write_file:0<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>", ordered_arguments(fields)),
                KIMI_K3_UNIFIED_FAMILY => format!("<|open|>tools<|sep|><|open|>call tool=\"write_file\" index=\"1\"<|sep|>{parameters}<|close|>call<|sep|><|close|>tools<|sep|>"),
                DEEPSEEK_V4_UNIFIED_FAMILY => format!("<｜DSML｜tool_calls><｜DSML｜invoke name=\"write_file\">{parameters}</｜DSML｜invoke></｜DSML｜tool_calls>"),
                DEEPSEEK_V41_UNIFIED_FAMILY => format!("<｜DSML｜ calls><｜DSML｜ invoke name=\"write_file\">{parameters}</｜DSML｜ invoke></｜DSML｜ calls>"),
                GEMMA4_UNIFIED_FAMILY => {
                    let parameters = fields.iter().map(|(key,value)| if let Some(raw)=value.as_str() {format!("{key}:<|\"|>{raw}<|\"|>")} else {format!("{key}:{value}")}).collect::<Vec<_>>().join(",");
                    format!("<|tool_call>call:write_file{{{parameters}}}<tool_call|>")
                },
                GLM47_UNIFIED_FAMILY => format!("<tool_call>write_file{parameters}</tool_call>"),
                MUSE_UNIFIED_FAMILY => format!("<|start|>assistant to=write_file<|message|><atem:invoke name=\"write_file\">{parameters}</atem:invoke><|eom|>"),
                _ => unreachable!(),
            };
            (family, input)
        }).collect()
}

pub(crate) fn write_file_progress_cases()
-> Vec<(&'static str, Vec<(&'static str, serde_json::Value)>)> {
    [
        "count_first",
        "path_first",
        "content_first",
        "quoted_markers",
    ]
    .into_iter()
    .map(|shape| {
        let value = format!(
            "é {} STAGE_TWO {} END",
            "first content ".repeat(64),
            "second content ".repeat(64)
        );
        let value = if shape == "quoted_markers" {
            format!("literal \"<think>\" \"</think>\" {value}")
        } else {
            value
        };
        let fields = match shape {
            "path_first" => vec![
                ("path", serde_json::json!("a.rs")),
                ("content", serde_json::json!(value)),
            ],
            "content_first" => vec![
                ("content", serde_json::json!(value)),
                ("count", serde_json::json!(2)),
            ],
            _ => vec![
                ("count", serde_json::json!(2)),
                ("content", serde_json::json!(value)),
            ],
        };
        (shape, fields)
    })
    .collect()
}

fn argument_progress_matrix(guided: bool) {
    let mut failures = Vec::new();
    for (shape, fields) in write_file_progress_cases() {
        let expected: serde_json::Value =
            serde_json::from_str(&ordered_arguments(&fields)).unwrap();
        for (family, native) in native_write_file_calls(&fields) {
            for named in if guided {
                vec![false, true]
            } else {
                vec![false]
            } {
                let mode = if guided {
                    UnifiedToolOutputMode::GuidedJson {
                        named_tool: named.then(|| "write_file".into()),
                    }
                } else {
                    UnifiedToolOutputMode::Native
                };
                let input = if guided {
                    let args = ordered_arguments(&fields);
                    if named {
                        args
                    } else {
                        format!("[{{\"name\":\"write_file\",\"arguments\":{args}}}]")
                    }
                } else {
                    native.clone()
                };
                let tools = to_v2_tools(Some(&write_file_tools()));
                let mut parser = create_unified_parser_for_family(family, &tools).unwrap();
                parser
                    .initialize_request(UnifiedParserInit {
                        prompt_token_ids: vec![],
                        starting_state: UnifiedParserStartingState::None,
                        tool_output_mode: mode.clone(),
                        invalid_guided_payload: invalid_guided_payload_policy(true),
                    })
                    .unwrap();
                let mut adapter = ChoiceState::new(
                    family,
                    &tools,
                    UnifiedParserStartingState::None,
                    mode,
                    true,
                    false,
                )
                .unwrap();
                let mut library = Vec::new();
                let mut wire = Vec::new();
                let mut start = 0;
                let mut progress = Vec::new();
                for marker in [" STAGE_TWO ", " END"] {
                    let end = input.find(marker).unwrap();
                    let mut out = UnifiedParserOutput::default();
                    parser.parse_into(&input[start..end], &mut out).unwrap();
                    library.extend(out.events);
                    let expected_fragment = if marker == " STAGE_TWO " {
                        "first content"
                    } else {
                        "second content"
                    };
                    let lib_name = library.iter().any(|event| matches!(event, UnifiedParserEvent::ToolCall(call) if call.name.as_deref() == Some("write_file")));
                    let lib_arguments: String = library
                        .iter()
                        .filter_map(|event| match event {
                            UnifiedParserEvent::ToolCall(call) => Some(call.arguments.as_str()),
                            _ => None,
                        })
                        .collect();
                    let lib_progress = lib_name && lib_arguments.contains(expected_fragment);
                    let events = adapter.push(&input[start..end]);
                    wire.extend(adapter.choices_for(&empty_choice(0), events, true, None));
                    let wire_calls: Vec<_> = wire
                        .iter()
                        .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
                        .collect();
                    let wire_name: String = wire_calls
                        .iter()
                        .filter_map(|call| call.function.as_ref()?.name.as_deref())
                        .collect();
                    let wire_arguments: String = wire_calls
                        .iter()
                        .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
                        .collect();
                    let wire_progress =
                        wire_name == "write_file" && wire_arguments.contains(expected_fragment);
                    progress.push((lib_progress, wire_progress));
                    start = end;
                }
                let mut out = UnifiedParserOutput::default();
                parser.parse_into(&input[start..], &mut out).unwrap();
                library.extend(out.events);
                library.extend(parser.finish().unwrap().events);
                let mut events = adapter.push(&input[start..]);
                events.extend(adapter.finish());
                wire.extend(adapter.choices_for(
                    &empty_choice(0),
                    events,
                    true,
                    Some(FinishReason::Stop),
                ));
                assert_eq!(
                    dynamo_parsers_v2::assemble(&library),
                    vec![UnifiedEvent::ToolCall {
                        name: "write_file".into(),
                        arguments: expected.clone()
                    }],
                    "library {family}/{shape}/{guided}/{named}"
                );
                let calls: Vec<_> = wire
                    .iter()
                    .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
                    .collect();
                assert!(calls.iter().all(|call| call.index == 0));
                assert_eq!(calls.iter().filter(|call| call.id.is_some()).count(), 1);
                assert_eq!(
                    calls
                        .iter()
                        .filter_map(|call| call.function.as_ref()?.name.as_deref())
                        .collect::<String>(),
                    "write_file"
                );
                let args: String = calls
                    .iter()
                    .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
                    .collect();
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&args).unwrap(),
                    expected,
                    "adapter {family}/{shape}/{guided}/{named}"
                );
                eprintln!(
                    "progress {family}/{shape}/guided={guided}/named={named}: {progress:?} final=true"
                );
                if progress != [(true, true), (true, true)] {
                    failures.push(format!("{family}/{shape}/{named}: {progress:?}"));
                }
            }
        }
    }
    assert!(
        failures.is_empty(),
        "library/adapter progress failures: {failures:?}"
    );
}

#[test]
fn eight_family_guided_argument_progress_matrix() {
    argument_progress_matrix(true);
}

#[test]
#[ignore = "Run explicitly to measure upstream native argument progress"]
fn eight_family_native_argument_progress_matrix() {
    argument_progress_matrix(false);
}

fn write_file_tools() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        name: "write_file".into(),
        parameters: Some(
            serde_json::json!({"type":"object", "properties":{"count":{"type":"integer"},"path":{"type":"string"},"content":{"type":"string"}}}),
        ),
        strict: None,
    }]
}

#[tokio::test]
async fn glm_truncation_survives_terminal_detour_and_eof() {
    for terminal_detour in [false, true] {
        let mut input = vec![chunk("<tool_call>", false)];
        if terminal_detour {
            let mut terminal = chunk("", true);
            let delta = &mut terminal.data.as_mut().unwrap().inner.choices[0].delta;
            delta.content = None;
            delta.reasoning_content = Some("done".into());
            input.push(terminal);
        }
        let responses = apply_stream(
            stream::iter(input),
            Some(write_file_tools()),
            None,
            false,
            UnifiedParserStartingState::None,
            GLM47_UNIFIED_FAMILY,
        )
        .collect::<Vec<_>>()
        .await;
        let choices = collect_choices(&responses);
        assert_eq!(
            choices.last().and_then(|choice| choice.finish_reason),
            Some(FinishReason::Length),
            "terminal_detour={terminal_detour}"
        );
        assert!(
            choices
                .iter()
                .all(|choice| choice.delta.tool_calls.is_none())
        );
    }
}

fn all_family_native_calls() -> [(&'static str, &'static str); 8] {
    [
        (
            "deepseek_v4",
            r#"<｜DSML｜tool_calls><｜DSML｜invoke name="get_weather"><｜DSML｜parameter name="city" string="true">Tokyo</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>"#,
        ),
        (
            "deepseek_v41",
            r#"<｜DSML｜ calls><｜DSML｜ invoke name="get_weather"><｜DSML｜ parameter name="city" string="true">Tokyo</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>"#,
        ),
        (
            "gemma4",
            "<|tool_call>call:get_weather{city:<|\"|>Tokyo<|\"|>}<tool_call|>",
        ),
        (
            "glm47",
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>",
        ),
        (
            "kimi_k2",
            r#"<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"city":"Tokyo"}<|tool_call_end|><|tool_calls_section_end|>"#,
        ),
        (
            "kimi_k3",
            r#"<|open|>tools<|sep|><|open|>call tool="get_weather" index="0"<|sep|><|open|>argument key="city" type="string"<|sep|>Tokyo<|close|>argument<|sep|><|close|>call<|sep|><|close|>tools<|sep|>"#,
        ),
        (
            "muse_glimmer",
            r#"<|start|>assistant to=get_weather<|message|><atem:function_calls><atem:invoke name="get_weather"><atem:parameter name="city">Tokyo</atem:parameter></atem:invoke></atem:function_calls><|eom|>"#,
        ),
        (
            "qwen3",
            "<tool_call><function=get_weather><parameter=city>Tokyo</parameter></function></tool_call>",
        ),
    ]
}

#[tokio::test]
async fn all_family_guided_streaming_rollback_preserves_complete_calls() {
    for (family, _) in all_family_native_calls() {
        let constraint = GuidedToolConstraint::GuidedJsonRequired;
        let input = r#"[{"name":"get_weather","arguments":{"city":"Tokyo"}}]"#;
        let released = apply_stream_with_constraint(
            stream::iter([chunk(&input[..35], false), chunk(&input[35..], true)]),
            Some(weather_tools()),
            constraint.clone(),
            UnifiedParserStartingState::None,
            family,
            true,
        )
        .collect::<Vec<_>>()
        .await;
        let buffered = apply_stream_with_constraint(
            stream::iter([chunk(&input[..35], false), chunk(&input[35..], true)]),
            Some(weather_tools()),
            constraint,
            UnifiedParserStartingState::None,
            family,
            false,
        )
        .collect::<Vec<_>>()
        .await;
        assert_eq!(
            logical_events(&released),
            logical_events(&buffered),
            "{family}"
        );
    }
}

#[tokio::test]
async fn kimi_named_guided_call_does_not_leak_native_end_marker() {
    let tools = vec![ToolDefinition {
        name: "send_email".to_string(),
        parameters: Some(
            serde_json::json!({"type":"object", "properties":{"to":{"type":"string"},"subject":{"type":"string"},"body":{"type":"string"}}}),
        ),
        strict: None,
    }];
    let input = r#"{"to":"user@example.com","subject":"Hello","body":"Hi"}<|tool_call_end|>"#;
    for by_character in [false, true] {
        for finish_reason in [FinishReason::Stop, FinishReason::Length] {
            let mut chunks: Vec<_> = if by_character {
                input
                    .char_indices()
                    .map(|(start, c)| chunk(&input[start..start + c.len_utf8()], false))
                    .collect()
            } else {
                vec![chunk(input, false)]
            };
            let mut terminal = chunk("", true);
            terminal.data.as_mut().unwrap().inner.choices[0].finish_reason = Some(finish_reason);
            chunks.push(terminal);
            let output = apply_stream_with_constraint(
                stream::iter(chunks),
                Some(tools.clone()),
                GuidedToolConstraint::GuidedJsonNamed {
                    tool_name: "send_email".to_string(),
                },
                UnifiedParserStartingState::None,
                "kimi_k2",
                true,
            )
            .collect::<Vec<_>>()
            .await;
            let events = logical_events(&output);
            let text: String = events
                .iter()
                .filter_map(|event| match event {
                    LogicalEvent::Text(t) => Some(t.as_str()),
                    _ => None,
                })
                .collect();
            let names: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    LogicalEvent::Tool { name: Some(n), .. } => Some(n.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                names,
                vec!["send_email"],
                "{by_character} {finish_reason:?}"
            );
            assert!(
                text.is_empty(),
                "{by_character} {finish_reason:?}: {text:?}"
            );
        }
    }
}

#[test]
fn spaced_kimi_think_prefill_and_response_closer_are_distinct() {
    assert_eq!(
        detect_prefill(
            "kimi_k3",
            "hidden<|close|> think <|sep|><|open|> response <|sep|>answer<|close|> response <|sep|>"
        )
        .unwrap(),
        UnifiedParserStartingState::Reasoning
    );
    assert_eq!(
        detect_prefill("kimi_k3", "answer<|close|> response <|sep|>").unwrap(),
        UnifiedParserStartingState::None
    );
}

#[tokio::test]
async fn guided_prefill_whitespace_is_decided_before_json_and_flushed_before_detours() {
    let constraint = GuidedToolConstraint::GuidedJsonNamed {
        tool_name: "get_weather".to_string(),
    };
    for (family, _) in all_family_native_calls() {
        let output = apply_stream_with_constraint(
            stream::iter([
                chunk("", false),
                chunk("  ", false),
                chunk(r#"{"city":"Tokyo"}"#, true),
            ]),
            Some(weather_tools()),
            constraint.clone(),
            UnifiedParserStartingState::Reasoning,
            family,
            true,
        )
        .collect::<Vec<_>>()
        .await;
        assert!(logical_events(&output).iter().any(|event| matches!(event, LogicalEvent::Tool { name: Some(name), .. } if name == "get_weather")), "{family}");
    }
    let mut terminal = chunk("", true);
    terminal.data.as_mut().unwrap().inner.choices[0]
        .delta
        .content = None;
    terminal.data.as_mut().unwrap().inner.choices[0]
        .delta
        .reasoning_content = Some("done".to_string());
    let output = apply_stream_with_constraint(
        stream::iter([chunk(" ", false), terminal]),
        Some(weather_tools()),
        constraint,
        UnifiedParserStartingState::Reasoning,
        "qwen3",
        true,
    )
    .collect::<Vec<_>>()
    .await;
    assert_eq!(
        logical_events(&output),
        vec![LogicalEvent::Reasoning(" done".to_string())]
    );
    let terminal_position = output
        .iter()
        .position(|r| {
            r.data
                .as_ref()
                .is_some_and(|d| d.inner.choices.iter().any(|c| c.finish_reason.is_some()))
        })
        .unwrap();
    assert!(
        !output[terminal_position + 1..]
            .iter()
            .any(|r| r.data.as_ref().is_some_and(|d| d
                .inner
                .choices
                .iter()
                .any(|c| c.delta.reasoning_content.is_some())))
    );
}

#[test]
fn apostrophe_handling_preserves_quotes_contractions_and_possessives() {
    // A possessive apostrophe that ends a word (`James'`) has non-alphanumeric on
    // the RIGHT. Requiring alphanumeric on both sides treats it as a real
    // quote-open, and the later contraction apostrophe in `isn't` then closes that
    // phantom span, hiding the real `</think>` marker in between.
    assert_eq!(
        detect_prefill(
            QWEN3_UNIFIED_FAMILY,
            "James' secret </think>answer isn't final"
        )
        .unwrap(),
        UnifiedParserStartingState::Reasoning,
        "a trailing possessive apostrophe must not be treated as a real quote-open"
    );
    // Mirror case: a leading contraction (`'twas`) has non-alphanumeric on the LEFT
    // and alphanumeric only on the right — the OR-based fix must exempt this too.
    assert!(
        contains_unquoted_marker("'twas </think> hidden", "</think>"),
        "a leading-apostrophe contraction must not be treated as a real quote-open"
    );
    assert!(
        contains_unquoted_marker("'twas didn't </think> hidden", "</think>"),
        "an apostrophe inside a later contraction must not close a leading contraction"
    );
    assert!(
        !contains_unquoted_marker("'The literal </think> marker' stays visible", "</think>"),
        "a leading quote before a word must shield markers through its closing quote"
    );
    assert!(
        !contains_unquoted_marker("'The word doesn't contain </think> as syntax'", "</think>"),
        "an internal contraction must not close a quoted span"
    );
}

#[tokio::test]
async fn already_parsed_first_chunk_with_no_prior_raw_still_terminates_correctly() {
    let mut already_parsed = chunk("", false);
    let choice = &mut already_parsed.data.as_mut().unwrap().inner.choices[0];
    choice.delta.content = None;
    choice.delta.tool_calls = Some(vec![ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("call-preexisting".to_string()),
        r#type: Some(FunctionType::Function),
        function: Some(FunctionCallStream {
            name: Some("get_weather".to_string()),
            arguments: Some("{}".to_string()),
        }),
        ..Default::default()
    }]);

    // This choice index never has a raw chunk at all — no ChoiceState is ever
    // constructed for it — then the stream ends immediately.
    let responses = apply_stream(
        stream::iter([already_parsed]),
        Some(weather_tools()),
        None,
        false,
        UnifiedParserStartingState::None,
        QWEN3_UNIFIED_FAMILY,
    )
    .collect::<Vec<_>>()
    .await;
    let choices = collect_choices(&responses);

    assert_eq!(
        choices
            .iter()
            .filter_map(|choice| choice.finish_reason)
            .collect::<Vec<_>>(),
        vec![FinishReason::ToolCalls],
        "a choice whose only-ever chunk was already-parsed and carried a tool \
         call must still get exactly one ToolCalls terminal at EOF"
    );
}

#[test]
fn quoted_control_marker_probe_masks_closed_quotes_and_flushes_unclosed_quotes() {
    for family in [
        "deepseek_v4",
        "deepseek_v41",
        "gemma4",
        "glm47",
        "kimi_k2",
        "kimi_k3",
        "muse_glimmer",
        "qwen3",
    ] {
        let marker = unified_parser_control_markers(family)
            .into_iter()
            .next()
            .unwrap();
        for quote in ['"', '\'', '`'] {
            let literal = format!("The literal {quote}{marker}{quote} stays visible.");
            for split in literal
                .char_indices()
                .map(|(offset, _)| offset)
                .chain(std::iter::once(literal.len()))
            {
                let mut shield = QuotedControlMarkerProbe::new(family);
                let mut masked = shield.push(&literal[..split]);
                masked.push_str(&shield.push(&literal[split..]));
                masked.push_str(&shield.finish());
                assert!(
                    !masked.contains(&marker),
                    "{family} quote {quote} split {split}"
                );
                assert_eq!(masked.len(), literal.len(), "{family} split {split}");
            }
        }

        let literal = format!("The literal \"{marker}");
        let mut unfinished = QuotedControlMarkerProbe::new(family);
        let mut masked = unfinished.push(&literal);
        masked.push_str(&unfinished.finish());
        assert!(masked.contains(&marker), "{family} EOF");
        assert_eq!(masked, literal, "{family} EOF");
    }

    let literal = r#"The literal "escaped \" closer </think>" remains visible."#;
    for split in literal
        .char_indices()
        .map(|(offset, _)| offset)
        .chain(std::iter::once(literal.len()))
    {
        let mut shield = QuotedControlMarkerProbe::new(QWEN3_UNIFIED_FAMILY);
        let mut masked = shield.push(&literal[..split]);
        masked.push_str(&shield.push(&literal[split..]));
        masked.push_str(&shield.finish());
        assert!(!masked.contains("</think>"), "split {split}");
        assert_eq!(masked.len(), literal.len(), "split {split}");
    }
}

#[test]
fn single_quoted_prose_shields_a_marker_after_its_opening_word() {
    for family in [
        "deepseek_v4",
        "deepseek_v41",
        "gemma4",
        "glm47",
        "kimi_k2",
        "kimi_k3",
        "muse_glimmer",
        "qwen3",
    ] {
        for marker in unified_parser_control_markers(family) {
            for (literal, should_be_shielded) in [
                (
                    format!("'The literal {marker} marker' stays visible."),
                    true,
                ),
                (
                    format!("'The word doesn't contain {marker} as syntax'"),
                    true,
                ),
                (format!("'twas {marker} must remain unshielded"), false),
            ] {
                for split in literal
                    .char_indices()
                    .map(|(offset, _)| offset)
                    .chain(std::iter::once(literal.len()))
                {
                    let mut shield = QuotedControlMarkerProbe::new(family);
                    let mut masked = shield.push(&literal[..split]);
                    let second_output = shield.push(&literal[split..]);
                    masked.push_str(&second_output);
                    masked.push_str(&shield.finish());
                    assert_eq!(
                        masked.contains(&marker),
                        !should_be_shielded,
                        "{family} marker {marker:?} split {split}, literal {literal:?}, masked {masked:?}"
                    );
                    assert_eq!(masked.len(), literal.len(), "{family} split {split}");
                }
            }
        }
    }
}

#[tokio::test]
async fn structured_unopened_channel_prefix_flush_and_progress() {
    let policy = UnifiedRequestPolicy {
        structured_response: true,
        ..Default::default()
    };
    for raw in [" \n", "\"unterminated <think>", "{", "[", "42"] {
        for ending in 0..4 {
            let mut input = vec![chunk(raw, false)];
            match ending {
                0 => {} // Raw EOF.
                1 => input.push(chunk("", true)),
                2 => {
                    let mut usage = chunk("", false);
                    usage.data.as_mut().unwrap().inner.choices.clear();
                    input.push(usage);
                }
                _ => {
                    let mut parsed = chunk("", true);
                    let delta = &mut parsed.data.as_mut().unwrap().inner.choices[0].delta;
                    delta.content = None;
                    delta.reasoning_content = Some("upstream".into());
                    input.push(parsed);
                }
            }
            let output = apply_stream_with_policy(
                futures::stream::iter(input),
                None,
                GuidedToolConstraint::None,
                UnifiedParserStartingState::None,
                "qwen3",
                true,
                policy,
            )
            .collect::<Vec<_>>()
            .await;
            let choices = collect_choices(&output);
            let text: String = choices
                .iter()
                .filter_map(|c| match &c.delta.content {
                    Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            let reasoning: String = choices
                .iter()
                .filter_map(|c| c.delta.reasoning_content.as_deref())
                .collect();
            assert_eq!(text, raw, "ending {ending}");
            assert_eq!(reasoning, if ending == 3 { "upstream" } else { "" });
        }
    }
    for raw in ["{", "[", "\"partial", "4", "t", "n", "f"] {
        let output = apply_stream_with_policy(
            futures::stream::iter([chunk(raw, false), chunk("", true)]),
            None,
            GuidedToolConstraint::None,
            UnifiedParserStartingState::None,
            "qwen3",
            true,
            policy,
        )
        .take(1)
        .collect::<Vec<_>>()
        .await;
        let text: String = collect_choices(&output)
            .iter()
            .filter_map(|c| match &c.delta.content {
                Some(ChatCompletionMessageContent::Text(text)) => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(text, raw, "JSON prefix must progress before terminal");
    }
}

#[test]
fn scalar_probe_inspects_each_byte_once_and_resolves_all_release_paths() {
    let raw = format!("\"{}\"", "x".repeat(65536));
    let mut probe = JsonPrefillProbe::default();
    for byte in raw.bytes() {
        probe.push(std::str::from_utf8(&[byte]).unwrap());
        assert!(probe.decision(false).is_none());
    }
    assert_eq!(probe.inspected_bytes, raw.len());
    assert_eq!(probe.decision(true), Some(UnifiedParserStartingState::None));
    let policy = UnifiedRequestPolicy {
        structured_response: true,
        ..Default::default()
    };
    let resolved =
        resolve_guided_prefix(7, probe, policy, UnifiedParserStartingState::Reasoning).unwrap();
    assert_eq!(
        resolved.delta.content,
        Some(ChatCompletionMessageContent::Text(raw))
    );
    assert!(resolved.delta.reasoning_content.is_none());
    for raw in ["42", "true", "null", "\"answer\""] {
        let mut probe = JsonPrefillProbe::default();
        probe.push(raw);
        assert!(probe.decision(false).is_none());
        let resolved =
            resolve_guided_prefix(0, probe, policy, UnifiedParserStartingState::Reasoning).unwrap();
        assert_eq!(
            resolved.delta.content,
            Some(ChatCompletionMessageContent::Text(raw.into()))
        );
    }
    let mut thought = JsonPrefillProbe::default();
    thought.push("42");
    assert!(thought.decision(false).is_none());
    thought.push(" reasons");
    assert_eq!(
        thought.decision(false),
        Some(UnifiedParserStartingState::Reasoning)
    );
    assert_eq!(thought.inspected_bytes, 10);
}

#[test]
fn json_prefill_marker_detection_has_linear_work_for_quoted_controls() {
    for payload in ["|", "<", ">", "<think>", "</think>"] {
        let raw = format!("[\"{}\"]", payload.repeat(4096));
        let mut probe = JsonPrefillProbe::default();
        for character in raw.chars() {
            let text = character.to_string();
            probe.push(&text);
            assert_ne!(
                probe
                    .decision_for_family(QWEN3_UNIFIED_FAMILY, false, &text)
                    .unwrap(),
                Some(UnifiedParserStartingState::Reasoning),
            );
        }
        assert_eq!(
            probe
                .decision_for_family(QWEN3_UNIFIED_FAMILY, true, "")
                .unwrap(),
            Some(UnifiedParserStartingState::None),
        );
        let shield_work = probe.marker_probe.as_ref().unwrap().inspected_bytes;
        assert!(
            probe.inspected_bytes + probe.marker_inspected_bytes + shield_work <= raw.len() * 64,
            "payload {payload:?}: JSON {}, marker {}, shield {}, input {}",
            probe.inspected_bytes,
            probe.marker_inspected_bytes,
            shield_work,
            raw.len(),
        );
    }
}

#[test]
fn structured_scalar_prefix_keeps_real_reasoning_before_json() {
    for thought in ["42 reasons", "\"a thought\"", "true", "null hypothesis"] {
        let full = format!("{thought}</think>\"answer\"");
        for split in 0..=full.len() {
            let prefix = &full[..split];
            if prefix.contains("</think>") {
                assert_eq!(
                    structured_json_prefill(QWEN3_UNIFIED_FAMILY, prefix, false).unwrap(),
                    Some(UnifiedParserStartingState::Reasoning),
                    "split {split}"
                );
            }
        }
        assert_eq!(
            structured_json_prefill(QWEN3_UNIFIED_FAMILY, &full, true).unwrap(),
            Some(UnifiedParserStartingState::Reasoning)
        );
    }
}

#[test]
fn review_completion_is_monotonic_after_whitespace_delta() {
    let mut state = test_state();
    state.delta_to_choice(0, call_delta(0, Some("get_weather"), "{}"));
    state.delta_to_choice(0, partial_call_delta(0, None, " "));
    assert_eq!(state.parser_call_completion.get(&0), Some(&true));
    assert!(state.pending_tool_calls.is_empty());
    let mut record = ChoiceRecord::<Annotated<NvCreateChatCompletionStreamResponse>>::default();
    record.remember_state(&state);
    assert_eq!(record.parser_call_completion.get(&0), Some(&true));
    assert_eq!(
        record.normalize_finish_reason(Some(FinishReason::Stop)),
        Some(FinishReason::ToolCalls)
    );
    assert_eq!(
        state.normalize_finish_reason(Some(FinishReason::Stop)),
        Some(FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn review_named_guided_whitespace_preserves_completion_at_every_split() {
    let raw = r#"{"city":"Tokyo"}"#;
    for (family, _) in all_family_native_calls() {
        for split in 0..=raw.len() {
            let output = apply_stream_with_constraint(
                stream::iter([
                    chunk(&raw[..split], false),
                    chunk(&raw[split..], false),
                    chunk(" ", false),
                    chunk("\n", true),
                ]),
                Some(weather_tools()),
                GuidedToolConstraint::GuidedJsonNamed {
                    tool_name: "get_weather".into(),
                },
                UnifiedParserStartingState::None,
                family,
                true,
            )
            .collect::<Vec<_>>()
            .await;
            let choices = collect_choices(&output);
            assert_eq!(
                choices.last().unwrap().finish_reason,
                Some(FinishReason::ToolCalls),
                "{family} split {split}"
            );
            let arguments = choices
                .iter()
                .flat_map(|choice| choice.delta.tool_calls.iter().flatten())
                .filter_map(|call| call.function.as_ref()?.arguments.as_deref())
                .collect::<String>();
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&arguments).unwrap(),
                serde_json::json!({"city":"Tokyo"}),
                "{family} split {split}"
            );
        }
    }
}

#[test]
fn review_provisional_sibling_exposes_truncation_in_live_and_history_states() {
    for completed_sibling in [false, true] {
        let mut state = test_state();
        if completed_sibling {
            state.delta_to_choice(0, call_delta(0, Some("first"), "{}"));
        }
        state.delta_to_choice(0, partial_call_delta(1, Some("unfinished"), "{"));
        let mut record = ChoiceRecord::<Annotated<NvCreateChatCompletionStreamResponse>>::default();
        record.remember_state(&state);
        for finish in [
            None,
            Some(FinishReason::Stop),
            Some(FinishReason::ToolCalls),
            Some(FinishReason::Length),
            Some(FinishReason::ContentFilter),
        ] {
            let expected = match finish {
                Some(FinishReason::Stop | FinishReason::ToolCalls) => Some(FinishReason::Length),
                other => other,
            };
            assert_eq!(state.normalize_finish_reason(finish), expected);
            assert_eq!(record.normalize_finish_reason(finish), expected);
        }
        assert_eq!(
            state.unterminated_finish_reason(),
            Some(FinishReason::Length)
        );
        let mut records = HashMap::from([(0, record)]);
        let chunks = finish_unterminated_choices(
            &mut HashMap::new(),
            &mut records,
            QWEN3_UNIFIED_FAMILY,
            false,
            false,
        );
        assert!(
            matches!(&chunks[..], [FinishedStreamChunk::Choice(choice)] if choice.finish_reason == Some(FinishReason::Length))
        );
    }
}

#[test]
fn review_glm_long_unfinished_argument_retains_truncation() {
    let (family, input) = eight_family_calls(&"x".repeat(8000))
        .into_iter()
        .find(|(family, _)| *family == GLM47_UNIFIED_FAMILY)
        .unwrap();
    let end = input.rfind('x').unwrap() + 1;
    let mut state = ChoiceState::new(
        family,
        &to_v2_tools(Some(&write_file_tools())),
        UnifiedParserStartingState::None,
        UnifiedToolOutputMode::Native,
        true,
        false,
    )
    .unwrap();
    let started = std::time::Instant::now();
    for character in input[..end].chars() {
        state.push(&character.to_string());
    }
    eprintln!(
        "GLM 8000-byte unfinished call, character cadence: {:?}",
        started.elapsed()
    );
    let events = state.finish();
    state.choices_for(&empty_choice(0), events, true, None);
    assert_eq!(
        state.unterminated_finish_reason(),
        Some(FinishReason::Length)
    );
}

#[test]
fn review_glm_pending_markers_keep_prose_quotes_and_native_payload_ownership() {
    let complete = r#"<tool_call>write_file<arg_key>content</arg_key><arg_value>say "hi</arg_value></tool_call>"#;
    let incomplete = r#"<tool_call>write_file<arg_key>content</arg_key><arg_value>"unfinished"#;
    for input in [
        format!("{complete}{incomplete}"),
        format!("{complete} The literal \"<tool_call>\" stays visible. {incomplete}"),
        "<tool_call>write_file<arg_key>content</arg_key><arg_value>literal </tool_call> text"
            .to_string(),
    ] {
        let expected_start = if input.starts_with(complete) {
            input.rfind(incomplete).unwrap()
        } else {
            0
        };
        assert_eq!(glm_unfinished_call(&input).unwrap(), Some(expected_start));
        for split in input
            .char_indices()
            .map(|(at, _)| at)
            .chain(std::iter::once(input.len()))
        {
            let mut state = ChoiceState::new(
                GLM47_UNIFIED_FAMILY,
                &to_v2_tools(Some(&write_file_tools())),
                UnifiedParserStartingState::None,
                UnifiedToolOutputMode::Native,
                true,
                false,
            )
            .unwrap();
            state.push(&input[..split]);
            state.push(&input[split..]);
            state.finish();
            assert!(state.incomplete_native_call, "split {split}: {input}");
        }
    }
    for input in [
        r#"The literal "<tool_call>" stays visible."#.to_string(),
        format!("{complete} The literal \"<tool_call>\" stays visible."),
    ] {
        assert_eq!(glm_unfinished_call(&input).unwrap(), None);
    }
}

// Added with the move: dynamo-llm calls these three helpers from its batch path
// and its v1 jail recovery, so their tests stayed there.
#[test]
fn public_marker_helpers_skip_quoted_markers() {
    let call =
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
    let answer = format!("Sure. {call}");
    assert_eq!(
        first_unquoted_native_tool_call_marker(&answer, "glm47"),
        Some(6)
    );
    assert_eq!(
        first_unquoted_structural_tool_call_marker(&answer, "glm47"),
        Some(6)
    );
    assert_eq!(
        unquoted_native_tool_call_marker_or_prefix_start(&answer, "glm47"),
        Some(6)
    );

    let quoted = "Write \"<tool_call>\" to open a call.";
    assert_eq!(
        first_unquoted_native_tool_call_marker(quoted, "glm47"),
        None
    );
    assert_eq!(
        first_unquoted_structural_tool_call_marker(quoted, "glm47"),
        None
    );
    assert_eq!(
        unquoted_native_tool_call_marker_or_prefix_start(quoted, "glm47"),
        None
    );

    // A partial marker at the end must stay buffered for the next chunk.
    assert_eq!(
        unquoted_native_tool_call_marker_or_prefix_start("Sure. <tool_", "glm47"),
        Some(6)
    );
    assert_eq!(
        first_unquoted_native_tool_call_marker("Sure. <tool_", "glm47"),
        None
    );

    // phi4 opens a call with a plain word, which is not structural.
    assert_eq!(
        first_unquoted_native_tool_call_marker("import functools", "phi4"),
        Some(7)
    );
    assert_eq!(
        first_unquoted_structural_tool_call_marker("import functools", "phi4"),
        None
    );

    assert_eq!(
        first_unquoted_native_tool_call_marker(call, "no_such_parser"),
        None
    );
}

#[path = "unified_stream_sources_tests.rs"]
#[allow(clippy::single_range_in_vec_init)]
mod sources;
