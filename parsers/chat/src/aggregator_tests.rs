// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! dynamo-llm's unit tests for the chat aggregator, moved with it.
//!
//! They run on the stand-ins for dynamo's stream types in
//! `unified_stream_tests.rs`, and on stand-ins here for dynamo-llm's
//! `ParsingOptions` constructors and its `apply` loop. Two tests stay in dynamo:
//! the `nvext` merge and the typed backend error, because dynamo keeps that
//! code. Tests that set `DYN_PARSER_VERSION` in a child process pass the
//! version in the options instead.

use futures::{Stream, StreamExt, stream};

use super::*;
use crate::unified_stream::tests::{Annotated, NvCreateChatCompletionStreamResponse};

/// dynamo-llm's `NvCreateChatCompletionResponse`, less `nvext` and prompt
/// logprobs, which dynamo folds itself.
pub(crate) struct NvCreateChatCompletionResponse {
    pub(crate) inner: CreateChatCompletionResponse,
}

async fn fold(
    stream: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
) -> Result<DeltaAggregator, String> {
    let mut stream = std::pin::pin!(stream);
    let mut aggregator = DeltaAggregator::new();
    while let Some(item) = stream.next().await {
        // dynamo-runtime's `Annotated::into_data`.
        if item.is_error() {
            return Err(item.error.unwrap_or_else(|| {
                item.comment
                    .filter(|comments| !comments.is_empty())
                    .map(|comments| comments.join(", "))
                    .unwrap_or_else(|| "unknown error".to_string())
            }));
        }
        if let Some(delta) = item.data {
            aggregator.push(delta.inner, &delta.tool_call_completion);
        }
    }
    Ok(aggregator)
}

/// dynamo-llm's `DeltaAggregator::apply`: fold the stream, then finish with no
/// explicit tool-call terminal required. An error item ends the reply.
pub(crate) async fn apply(
    stream: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
    parsing_options: AggregatorOptions,
) -> Result<NvCreateChatCompletionResponse, String> {
    let inner = fold(stream)
        .await?
        .finish_with_tool_terminal_policy(&parsing_options, false)
        .await
        .map_err(|error| error.to_string())?;
    Ok(NvCreateChatCompletionResponse { inner })
}

/// dynamo-llm's `DeltaAggregator::apply_with_tool_terminal_policy`.
pub(crate) async fn apply_with_tool_terminal_policy(
    stream: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
    parsing_options: AggregatorOptions,
    require_explicit_tool_terminal: bool,
) -> Result<NvCreateChatCompletionResponse, String> {
    let inner = fold(stream)
        .await?
        .finish_with_tool_terminal_policy(&parsing_options, require_explicit_tool_terminal)
        .await
        .map_err(|error| error.to_string())?;
    Ok(NvCreateChatCompletionResponse { inner })
}

/// dynamo-runtime's `DataStream`.
type DataStream<T> = std::pin::Pin<Box<dyn Stream<Item = T> + Send>>;

/// dynamo-llm's `NvCreateChatCompletionResponse::from_annotated_stream`, the
/// entry point its chat handler calls.
pub(crate) async fn from_annotated_stream(
    stream: impl Stream<Item = Annotated<NvCreateChatCompletionStreamResponse>>,
    parsing_options: AggregatorOptions,
) -> Result<NvCreateChatCompletionResponse, String> {
    let inner = fold(stream)
        .await?
        .finish(&parsing_options)
        .await
        .map_err(|error| error.to_string())?;
    Ok(NvCreateChatCompletionResponse { inner })
}

/// dynamo-llm's `ParsingOptions` constructors, over [`AggregatorOptions`].
pub(crate) struct ParsingOptions;

impl ParsingOptions {
    #[allow(clippy::new_ret_no_self)]
    pub(crate) fn new(
        tool_call_parser: Option<String>,
        reasoning_parser: Option<String>,
    ) -> AggregatorOptions {
        AggregatorOptions {
            tool_call_parser,
            reasoning_parser,
            ..AggregatorOptions::default()
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub(crate) fn default() -> AggregatorOptions {
        AggregatorOptions::default()
    }
}

/// dynamo-llm's `ParsingOptions` setters that these tests call.
pub(crate) trait ParsingOptionsSetters {
    fn with_tools(self, tools: Vec<ToolDefinition>) -> Self;
    fn with_parallel_tool_calls(self, parallel_tool_calls: Option<bool>) -> Self;
    fn with_move_reasoning_to_content_when_empty(self, enabled: bool) -> Self;
    fn with_parser_version(self, parser_version: ParserVersion) -> Self;
}

impl ParsingOptionsSetters for AggregatorOptions {
    fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    fn with_parallel_tool_calls(mut self, parallel_tool_calls: Option<bool>) -> Self {
        self.parallel_tool_calls = parallel_tool_calls;
        self
    }

    fn with_move_reasoning_to_content_when_empty(mut self, enabled: bool) -> Self {
        self.move_reasoning_to_content_when_empty = enabled;
        self
    }

    fn with_parser_version(mut self, parser_version: ParserVersion) -> Self {
        self.parser_version = parser_version;
        self
    }
}

/// dynamo-llm's `token_to_utf8_bytes`.
fn token_to_utf8_bytes(token: &str) -> Option<Vec<u8>> {
    if token.is_empty() {
        None
    } else {
        Some(token.as_bytes().to_vec())
    }
}

#[tokio::test]
async fn v2_batch_parser_uses_request_tool_schema() {
    let tools = vec![dynamo_parsers::tool_calling::ToolDefinition {
        name: "set_state".to_string(),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"}
            }
        })),
        strict: None,
    }];
    let content = concat!(
        "<tool_call>\n<function=set_state>\n",
        "<parameter=count>42</parameter>\n",
        "<parameter=enabled>true</parameter>\n",
        "</function>\n</tool_call>"
    );
    let (calls, _) = parse_complete_tool_output(
        content,
        "qwen3_coder",
        &GuidedToolConstraint::None,
        &tools,
        ParserVersion::V2,
    )
    .await
    .unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments).unwrap(),
        serde_json::json!({"count": 42, "enabled": true})
    );
}

#[allow(deprecated)]
fn create_test_delta(
    index: u32,
    text: &str,
    role: Option<dynamo_protocols::types::Role>,
    finish_reason: Option<dynamo_protocols::types::FinishReason>,
    logprob: Option<f32>,
    tool_calls: Option<&str>,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    // ALLOW: function_call is deprecated

    let tool_calls: Option<serde_json::Value> =
        tool_calls.map(|tool_calls| serde_json::from_str(tool_calls).unwrap());

    let tool_call_chunks = tool_calls.map(|tool_calls| {
        vec![
            dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
                index: 0,
                id: Some("test_id".to_string()),
                r#type: Some(dynamo_protocols::types::FunctionType::Function),
                function: Some(dynamo_protocols::types::FunctionCallStream {
                    name: tool_calls["name"].as_str().map(|s| s.to_string()),
                    arguments: Some(serde_json::to_string(&tool_calls["arguments"]).unwrap()),
                }),
                custom: None,
            },
        ]
    });

    let delta = dynamo_protocols::types::ChatCompletionStreamResponseDelta {
        content: Some(ChatCompletionMessageContent::Text(text.to_string())),
        function_call: None,
        tool_calls: tool_call_chunks,
        role,
        refusal: None,
        reasoning_content: None,
    };
    let logprobs = logprob.map(|lp| {
        let token = text.to_string();
        dynamo_protocols::types::ChatChoiceLogprobs {
            content: Some(vec![dynamo_protocols::types::ChatCompletionTokenLogprob {
                token: token.clone(),
                logprob: lp,
                token_id: None,
                bytes: token_to_utf8_bytes(&token),
                top_logprobs: vec![],
            }]),
            refusal: None,
        }
    });
    let choice = dynamo_protocols::types::ChatChoiceStream {
        index,
        delta,
        finish_reason,
        logprobs,
    };

    let data = NvCreateChatCompletionStreamResponse {
        inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "meta/llama-3.1-8b-instruct".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![choice],
            object: "chat.completion".to_string(),
        },
        nvext: None,
        prompt_logprobs: None,
        llm_metrics: None,
        tool_call_completion: Vec::new(),
    };

    Annotated {
        data: Some(data),
        id: Some("test_id".to_string()),
        event: None,
        comment: None,
        error: None,
    }
}

/// Build a stream delta with `reasoning_content` set and optional (possibly
/// empty) text content — mirrors what the reasoning parser emits after
/// splitting think tags. Used by the force_nonempty_content test.
fn create_reasoning_delta(
    index: u32,
    content: &str,
    reasoning_content: &str,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    // ALLOW: function_call is deprecated
    #[allow(deprecated)]
    let delta = dynamo_protocols::types::ChatCompletionStreamResponseDelta {
        content: Some(ChatCompletionMessageContent::Text(content.to_string())),
        function_call: None,
        tool_calls: None,
        role: Some(dynamo_protocols::types::Role::Assistant),
        refusal: None,
        reasoning_content: Some(reasoning_content.to_string()),
    };
    let choice = dynamo_protocols::types::ChatChoiceStream {
        index,
        delta,
        finish_reason: Some(dynamo_protocols::types::FinishReason::Stop),
        logprobs: None,
    };
    let data = NvCreateChatCompletionStreamResponse {
        inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "nvidia/nvidia-nemotron-3-ultra".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![choice],
            object: "chat.completion".to_string(),
        },
        nvext: None,
        prompt_logprobs: None,
        llm_metrics: None,
        tool_call_completion: Vec::new(),
    };
    Annotated {
        data: Some(data),
        id: Some("test_id".to_string()),
        event: None,
        comment: None,
        error: None,
    }
}

/// with Nemotron `force_nonempty_content=true`, a non-streaming
/// reasoning-only turn must surface its reasoning as `content` (the chat
/// template promised non-empty content), but only when no content was
/// generated. Gated by `ParsingOptions::move_reasoning_to_content_when_empty`.
#[tokio::test]
async fn test_move_reasoning_to_content_when_empty() {
    let reasoning_only = || {
        Box::pin(stream::iter(vec![create_reasoning_delta(
            0,
            "",
            "Let me think.",
        )]))
    };

    // Repro: without the flag, a reasoning-only turn leaves content empty
    // (None) even though force_nonempty_content promised non-empty content.
    let resp = apply(reasoning_only(), ParsingOptions::default())
        .await
        .unwrap();
    let msg = &resp.inner.choices[0].message;
    assert!(
        msg.content.is_none(),
        "repro: content empty without the flag"
    );
    assert_eq!(msg.reasoning_content.as_deref(), Some("Let me think."));

    // Fixed: with the flag, reasoning is moved into content and cleared.
    let opts = ParsingOptions::default().with_move_reasoning_to_content_when_empty(true);
    let resp = apply(reasoning_only(), opts.clone()).await.unwrap();
    let msg = &resp.inner.choices[0].message;
    assert_eq!(
        msg.content.as_ref().unwrap(),
        &ChatCompletionMessageContent::Text("Let me think.".to_string()),
    );
    assert_eq!(msg.reasoning_content, None);

    // Whitespace-only content counts as empty (a trailing newline after
    // `</think>` from the incremental parser) — matches vLLM's
    // `not final_content.strip()` check; the move must still fire.
    let whitespace_content = Box::pin(stream::iter(vec![create_reasoning_delta(
        0,
        "\n",
        "Let me think.",
    )]));
    let resp = apply(whitespace_content, opts.clone()).await.unwrap();
    let msg = &resp.inner.choices[0].message;
    assert_eq!(
        msg.content.as_ref().unwrap(),
        &ChatCompletionMessageContent::Text("Let me think.".to_string()),
    );
    assert_eq!(msg.reasoning_content, None);

    // When content WAS generated, reasoning stays in reasoning_content.
    let with_content = Box::pin(stream::iter(vec![create_reasoning_delta(
        0,
        "The answer is 42.",
        "Let me think.",
    )]));
    let resp = apply(with_content, opts).await.unwrap();
    let msg = &resp.inner.choices[0].message;
    assert_eq!(
        msg.content.as_ref().unwrap(),
        &ChatCompletionMessageContent::Text("The answer is 42.".to_string()),
    );
    assert_eq!(msg.reasoning_content.as_deref(), Some("Let me think."));
}

/// Multimodal content lives in `content_parts`, and `From<DeltaChoice>` prefers
/// parts over `text` — so reasoning must NOT be moved when parts are present
/// (it would be silently dropped). Guards the `content_parts` half of the
/// no-content check.
#[tokio::test]
async fn test_move_reasoning_skips_when_content_parts_present() {
    #[allow(deprecated)]
    let delta = dynamo_protocols::types::ChatCompletionStreamResponseDelta {
        content: Some(ChatCompletionMessageContent::Parts(vec![
            dynamo_protocols::types::ChatCompletionResponseContentPart::Text(
                dynamo_protocols::types::ChatCompletionResponseContentPartText {
                    text: "an image".to_string(),
                },
            ),
        ])),
        function_call: None,
        tool_calls: None,
        role: Some(dynamo_protocols::types::Role::Assistant),
        refusal: None,
        reasoning_content: Some("thinking".to_string()),
    };
    let choice = dynamo_protocols::types::ChatChoiceStream {
        index: 0,
        delta,
        finish_reason: Some(dynamo_protocols::types::FinishReason::Stop),
        logprobs: None,
    };
    let data = NvCreateChatCompletionStreamResponse {
        inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "m".to_string(),
            created: 1,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![choice],
            object: "chat.completion".to_string(),
        },
        nvext: None,
        prompt_logprobs: None,
        llm_metrics: None,
        tool_call_completion: Vec::new(),
    };
    let annotated = Annotated {
        data: Some(data),
        id: Some("test_id".to_string()),
        event: None,
        comment: None,
        error: None,
    };
    let opts = ParsingOptions::default().with_move_reasoning_to_content_when_empty(true);
    let resp = apply(Box::pin(stream::iter(vec![annotated])), opts)
        .await
        .unwrap();
    let msg = &resp.inner.choices[0].message;
    assert!(matches!(
        msg.content.as_ref().unwrap(),
        ChatCompletionMessageContent::Parts(_)
    ));
    assert_eq!(msg.reasoning_content.as_deref(), Some("thinking"));
}

/// Build a stream delta carrying a raw list of tool-call chunks (no content).
/// Used by multi-chunk tests that mimic vLLM hermes' streaming emission:
/// the first chunk carries `id` + `function.name` only, subsequent chunks
/// carry `function.arguments` fragments with neither `id` nor `name`.
fn create_test_delta_with_tool_chunks(
    index: u32,
    tool_chunks: Vec<dynamo_protocols::types::ChatCompletionMessageToolCallChunk>,
    finish_reason: Option<dynamo_protocols::types::FinishReason>,
    role: Option<dynamo_protocols::types::Role>,
) -> Annotated<NvCreateChatCompletionStreamResponse> {
    #[allow(deprecated)]
    let delta = dynamo_protocols::types::ChatCompletionStreamResponseDelta {
        content: None,
        function_call: None,
        tool_calls: Some(tool_chunks),
        role,
        refusal: None,
        reasoning_content: None,
    };
    let choice = dynamo_protocols::types::ChatChoiceStream {
        index,
        delta,
        finish_reason,
        logprobs: None,
    };
    let data = NvCreateChatCompletionStreamResponse {
        inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "meta/llama-3.1-8b-instruct".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![choice],
            object: "chat.completion".to_string(),
        },
        nvext: None,
        prompt_logprobs: None,
        llm_metrics: None,
        tool_call_completion: Vec::new(),
    };
    Annotated {
        data: Some(data),
        id: Some("test_id".to_string()),
        event: None,
        comment: None,
        error: None,
    }
}

/// Repro for [#8640](https://github.com/ai-dynamo/dynamo/issues/8640):
/// vLLM hermes (and any spec-compliant OpenAI tool-call streaming producer)
/// splits a single tool call across multiple deltas:
///   delta 1: `{index: 0, id: "tc1", type: function, function: {name: "get_weather"}}`
///   delta 2: `{index: 0, function: {arguments: "{\"city\":"}}`
///   delta 3: `{index: 0, function: {arguments: "\"Tokyo\"}"}}`
/// The aggregated non-stream response must reconstruct
/// `arguments = "{\"city\":\"Tokyo\"}"`. Before the fix,
/// `convert_tool_chunk_to_message_tool_call` requires id/name/arguments all
/// set on the *same* chunk and drops the argument-fragment chunks — so the
/// client sees `arguments: ""`.
#[tokio::test]
async fn test_issue_8640_split_tool_call_arguments_reconstructed() {
    let name_chunk = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("tc1".to_string()),
        r#type: Some(dynamo_protocols::types::FunctionType::Function),
        function: Some(dynamo_protocols::types::FunctionCallStream {
            name: Some("get_weather".to_string()),
            arguments: None,
        }),
        custom: None,
    };
    let args_chunk_1 = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
        index: 0,
        id: None,
        r#type: None,
        function: Some(dynamo_protocols::types::FunctionCallStream {
            name: None,
            arguments: Some("{\"city\":".to_string()),
        }),
        custom: None,
    };
    let args_chunk_2 = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
        index: 0,
        id: None,
        r#type: None,
        function: Some(dynamo_protocols::types::FunctionCallStream {
            name: None,
            arguments: Some("\"Tokyo\"}".to_string()),
        }),
        custom: None,
    };

    let deltas = vec![
        create_test_delta_with_tool_chunks(
            0,
            vec![name_chunk],
            None,
            Some(dynamo_protocols::types::Role::Assistant),
        ),
        create_test_delta_with_tool_chunks(0, vec![args_chunk_1], None, None),
        create_test_delta_with_tool_chunks(
            0,
            vec![args_chunk_2],
            Some(dynamo_protocols::types::FinishReason::ToolCalls),
            None,
        ),
    ];
    let stream = Box::pin(stream::iter(deltas));

    let result = apply(stream, ParsingOptions::default()).await;
    assert!(result.is_ok(), "aggregation should not error");
    let response = result.unwrap();
    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];

    let tool_calls = choice
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls should be Some after aggregation");
    assert_eq!(
        tool_calls.len(),
        1,
        "must produce exactly one aggregated tool_call, got {}",
        tool_calls.len()
    );
    let tc = &tool_calls[0];
    assert_eq!(tc.id, "tc1");
    assert_eq!(tc.function.name, "get_weather");
    assert_eq!(
        tc.function.arguments, "{\"city\":\"Tokyo\"}",
        "#8640: arguments must be reconstructed from split fragments, \
         not dropped (got {:?})",
        tc.function.arguments
    );
}

/// Two parallel tool calls (index=0 and index=1), their chunks interleaved
/// in emission order. Exercises that the per-index accumulator correctly
/// keeps the two calls separate — not just that split args get merged
/// within one call (which [`test_issue_8640_split_tool_call_arguments_reconstructed`]
/// already covers). Related: [#8636](https://github.com/ai-dynamo/dynamo/issues/8636)
/// is about the streaming path dropping the second call; the non-stream
/// aggregator now handles the parallel case too, and this test pins it.
#[tokio::test]
async fn test_parallel_tool_calls_interleaved_chunks_aggregate_independently() {
    let make_name = |idx: u32, id: &str, name: &str| {
        dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: idx,
            id: Some(id.to_string()),
            r#type: Some(dynamo_protocols::types::FunctionType::Function),
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: Some(name.to_string()),
                arguments: None,
            }),
            custom: None,
        }
    };
    let make_args =
        |idx: u32, fragment: &str| dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: idx,
            id: None,
            r#type: None,
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: None,
                arguments: Some(fragment.to_string()),
            }),
            custom: None,
        };

    // Emission order mimics a hermes-style parser:
    //   open call 0 → open call 1 → args-frag-0a → args-frag-1a →
    //   args-frag-0b → args-frag-1b → finish
    let deltas = vec![
        create_test_delta_with_tool_chunks(
            0,
            vec![make_name(0, "tc0", "get_weather")],
            None,
            Some(dynamo_protocols::types::Role::Assistant),
        ),
        create_test_delta_with_tool_chunks(0, vec![make_name(1, "tc1", "get_time")], None, None),
        create_test_delta_with_tool_chunks(0, vec![make_args(0, "{\"city\":")], None, None),
        create_test_delta_with_tool_chunks(0, vec![make_args(1, "{\"tz\":")], None, None),
        create_test_delta_with_tool_chunks(0, vec![make_args(0, "\"Tokyo\"}")], None, None),
        create_test_delta_with_tool_chunks(
            0,
            vec![make_args(1, "\"JST\"}")],
            Some(dynamo_protocols::types::FinishReason::ToolCalls),
            None,
        ),
    ];
    let stream = Box::pin(stream::iter(deltas));

    let response = apply(stream, ParsingOptions::default())
        .await
        .expect("aggregation should succeed");
    assert_eq!(response.inner.choices.len(), 1);
    let tool_calls = response.inner.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls should be Some");
    assert_eq!(tool_calls.len(), 2, "must produce both parallel tool calls");

    // BTreeMap iteration is index-ordered, so [0] is tc0, [1] is tc1.
    assert_eq!(tool_calls[0].id, "tc0");
    assert_eq!(tool_calls[0].function.name, "get_weather");
    assert_eq!(tool_calls[0].function.arguments, "{\"city\":\"Tokyo\"}");
    assert_eq!(tool_calls[1].id, "tc1");
    assert_eq!(tool_calls[1].function.name, "get_time");
    assert_eq!(tool_calls[1].function.arguments, "{\"tz\":\"JST\"}");
}

/// When parallel_tool_calls == false, the aggregator limits the
/// response to the first tool call, even if the model emitted multiple calls.
#[tokio::test]
async fn test_parallel_tool_calls_false_caps_to_first_call() {
    let make_name = |idx: u32, id: &str, name: &str| {
        dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: idx,
            id: Some(id.to_string()),
            r#type: Some(dynamo_protocols::types::FunctionType::Function),
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: Some(name.to_string()),
                arguments: None,
            }),
            custom: None,
        }
    };
    let make_args =
        |idx: u32, fragment: &str| dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: idx,
            id: None,
            r#type: None,
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: None,
                arguments: Some(fragment.to_string()),
            }),
            custom: None,
        };

    let deltas = vec![
        create_test_delta_with_tool_chunks(
            0,
            vec![make_name(0, "tc0", "get_weather")],
            None,
            Some(dynamo_protocols::types::Role::Assistant),
        ),
        create_test_delta_with_tool_chunks(0, vec![make_name(1, "tc1", "get_time")], None, None),
        create_test_delta_with_tool_chunks(
            0,
            vec![make_args(0, "{\"city\":\"Tokyo\"}")],
            None,
            None,
        ),
        create_test_delta_with_tool_chunks(
            0,
            vec![make_args(1, "{\"tz\":\"JST\"}")],
            Some(dynamo_protocols::types::FinishReason::ToolCalls),
            None,
        ),
    ];
    let stream = Box::pin(stream::iter(deltas));

    let response = apply(
        stream,
        ParsingOptions::default().with_parallel_tool_calls(Some(false)),
    )
    .await
    .expect("aggregation should succeed");

    assert_eq!(response.inner.choices.len(), 1);
    let tool_calls = response.inner.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("tool_calls should be Some");
    assert_eq!(
        tool_calls.len(),
        1,
        "parallel_tool_calls=false must cap to a single tool call"
    );
    // The first-emitted call (index 0) is the one retained.
    assert_eq!(tool_calls[0].id, "tc0");
    assert_eq!(tool_calls[0].function.name, "get_weather");
}

/// When fragment-only chunks arrive but no id/name ever establishes the
/// call opener (producer bug), `finalize_merged_tool_chunk` drops the
/// chunk with a warn! log instead of emitting a malformed tool call.
/// This test just pins the "no panic, no phantom tool call" half — the
/// warn! is observable via tracing subscriber in prod, not asserted here.
#[tokio::test]
async fn test_fragment_only_chunks_without_opener_drop_cleanly() {
    let args_only = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
        index: 0,
        id: None,
        r#type: None,
        function: Some(dynamo_protocols::types::FunctionCallStream {
            name: None,
            arguments: Some("{\"orphaned\":true}".to_string()),
        }),
        custom: None,
    };
    let deltas = vec![create_test_delta_with_tool_chunks(
        0,
        vec![args_only],
        Some(dynamo_protocols::types::FinishReason::Stop),
        Some(dynamo_protocols::types::Role::Assistant),
    )];
    let stream = Box::pin(stream::iter(deltas));

    let response = apply(stream, ParsingOptions::default())
        .await
        .expect("aggregation should succeed even with dropped chunk");
    // Finalization only assigns `tool_calls` when the finalized vec is
    // non-empty, so the strict post-condition here is `None`. Tight
    // assertion catches a regression that flips to `Some(vec![])`.
    assert!(
        response.inner.choices[0].message.tool_calls.is_none(),
        "orphaned fragment must not produce a tool call (got {:?})",
        response.inner.choices[0].message.tool_calls,
    );
}

#[tokio::test]
async fn test_empty_stream() {
    // Create an empty stream
    let stream: DataStream<Annotated<NvCreateChatCompletionStreamResponse>> =
        Box::pin(stream::empty());

    // Call DeltaAggregator::apply
    let result = apply(stream, ParsingOptions::default()).await;

    // Check the result
    assert!(result.is_ok());
    let response = result.unwrap();

    // Verify that the response is empty and has default values
    assert_eq!(response.inner.id, "");
    assert_eq!(response.inner.model, "");
    assert_eq!(response.inner.created, 0);
    assert!(response.inner.usage.is_none());
    assert!(response.inner.system_fingerprint.is_none());
    assert_eq!(response.inner.choices.len(), 0);
    assert!(response.inner.service_tier.is_none());
}

#[tokio::test]
async fn test_refusal_fragments_survive_aggregation() {
    let chunks = ["I cannot ", "help with that."].into_iter().map(|text| {
        let mut chunk = create_test_delta(0, "", None, None, None, None);
        chunk.data.as_mut().unwrap().inner.choices[0].delta.refusal = Some(text.to_string());
        chunk
    });
    let response = apply(Box::pin(stream::iter(chunks)), ParsingOptions::default())
        .await
        .unwrap();
    assert_eq!(
        response.inner.choices[0].message.refusal.as_deref(),
        Some("I cannot help with that.")
    );
    assert!(response.inner.choices[0].message.content.is_none());
}

#[tokio::test]
async fn test_single_delta() {
    // Create a sample delta
    let annotated_delta = create_test_delta(
        0,
        "Hello,",
        Some(dynamo_protocols::types::Role::User),
        None,
        None,
        None,
    );

    // Create a stream
    let stream = Box::pin(stream::iter(vec![annotated_delta]));

    // Call DeltaAggregator::apply
    let result = apply(stream, ParsingOptions::default()).await;

    // Check the result
    assert!(result.is_ok());
    let response = result.unwrap();

    // Verify the response fields
    assert_eq!(response.inner.id, "test_id");
    assert_eq!(response.inner.model, "meta/llama-3.1-8b-instruct");
    assert_eq!(response.inner.created, 1234567890);
    assert!(response.inner.usage.is_none());
    assert!(response.inner.system_fingerprint.is_none());
    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];
    assert_eq!(choice.index, 0);
    assert_eq!(
        choice.message.content.as_ref().unwrap(),
        &ChatCompletionMessageContent::Text("Hello,".to_string())
    );
    assert!(choice.message.refusal.is_none());
    assert!(choice.finish_reason.is_none());
    assert_eq!(choice.message.role, dynamo_protocols::types::Role::User);
    assert!(response.inner.service_tier.is_none());
}

#[tokio::test]
async fn test_multiple_deltas_same_choice() {
    // Create multiple deltas with the same choice index
    // One will have a MessageRole and no FinishReason,
    // the other will have a FinishReason and no MessageRole
    let annotated_delta1 = create_test_delta(
        0,
        "Hello,",
        Some(dynamo_protocols::types::Role::User),
        None,
        Some(-0.1),
        None,
    );
    let annotated_delta2 = create_test_delta(
        0,
        " world!",
        None,
        Some(dynamo_protocols::types::FinishReason::Stop),
        Some(-0.2),
        None,
    );

    // Create a stream
    let annotated_deltas = vec![annotated_delta1, annotated_delta2];
    let stream = Box::pin(stream::iter(annotated_deltas));

    // Call DeltaAggregator::apply
    let result = apply(stream, ParsingOptions::default()).await;

    // Check the result
    assert!(result.is_ok());
    let response = result.unwrap();

    // Verify the response fields
    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];
    assert_eq!(choice.index, 0);
    assert_eq!(
        choice.message.content.as_ref().unwrap(),
        &ChatCompletionMessageContent::Text("Hello, world!".to_string())
    );
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Stop)
    );
    assert_eq!(choice.message.role, dynamo_protocols::types::Role::User);
    assert_eq!(
        choice
            .logprobs
            .as_ref()
            .unwrap()
            .content
            .as_ref()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        choice.logprobs.as_ref().unwrap().content.as_ref().unwrap()[0].logprob,
        -0.1
    );
    assert_eq!(
        choice.logprobs.as_ref().unwrap().content.as_ref().unwrap()[1].logprob,
        -0.2
    );
}

#[tokio::test]
async fn test_missing_stream_role_defaults_to_assistant_without_panic() {
    let deltas = vec![
        create_test_delta(0, "Hello,", None, None, None, None),
        create_test_delta(
            0,
            " world!",
            None,
            Some(dynamo_protocols::types::FinishReason::Stop),
            None,
            None,
        ),
    ];
    let stream = Box::pin(stream::iter(deltas));

    let response = apply(stream, ParsingOptions::default())
        .await
        .expect("aggregation should not panic or error when stream role is missing");

    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];
    assert_eq!(
        choice.message.role,
        dynamo_protocols::types::Role::Assistant
    );
    assert_eq!(
        choice.message.content.as_ref().unwrap(),
        &ChatCompletionMessageContent::Text("Hello, world!".to_string())
    );
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Stop)
    );
}

#[tokio::test]
async fn test_preserves_intermediate_whitespace_chunks() {
    // This validates behavior before/after removing trim_end():
    // If a whitespace-only chunk (" ") arrives between tokens, it must be preserved.
    // With trim_end(), that chunk was dropped, yielding "Helloworld" instead of "Hello world".

    let annotated_delta1 = create_test_delta(
        0,
        "Hello",
        Some(dynamo_protocols::types::Role::User),
        None,
        None,
        None,
    );
    // A whitespace-only chunk
    let annotated_delta2 = create_test_delta(0, " ", None, None, None, None);
    let annotated_delta3 = create_test_delta(
        0,
        "world",
        None,
        Some(dynamo_protocols::types::FinishReason::Stop),
        None,
        None,
    );

    let stream = Box::pin(stream::iter(vec![
        annotated_delta1,
        annotated_delta2,
        annotated_delta3,
    ]));

    let result = apply(stream, ParsingOptions::default()).await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];

    assert_eq!(choice.index, 0);
    assert_eq!(
        choice.message.content.as_ref(),
        Some(&ChatCompletionMessageContent::Text(
            "Hello world".to_string()
        ))
    );
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Stop)
    );
    assert_eq!(choice.message.role, dynamo_protocols::types::Role::User);
}

#[allow(deprecated)]
#[tokio::test]
async fn test_multiple_choices() {
    // Create a delta with multiple choices
    // ALLOW: function_call is deprecated
    let data = NvCreateChatCompletionStreamResponse {
        inner: dynamo_protocols::types::CreateChatCompletionStreamResponse {
            id: "test_id".to_string(),
            model: "test_model".to_string(),
            created: 1234567890,
            service_tier: None,
            usage: None,
            system_fingerprint: None,
            choices: vec![
                dynamo_protocols::types::ChatChoiceStream {
                    index: 0,
                    delta: dynamo_protocols::types::ChatCompletionStreamResponseDelta {
                        role: Some(dynamo_protocols::types::Role::Assistant),
                        content: Some(ChatCompletionMessageContent::Text("Choice 0".to_string())),
                        function_call: None,
                        tool_calls: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some(dynamo_protocols::types::FinishReason::Stop),
                    logprobs: None,
                },
                dynamo_protocols::types::ChatChoiceStream {
                    index: 1,
                    delta: dynamo_protocols::types::ChatCompletionStreamResponseDelta {
                        role: Some(dynamo_protocols::types::Role::Assistant),
                        content: Some(ChatCompletionMessageContent::Text("Choice 1".to_string())),
                        function_call: None,
                        tool_calls: None,
                        refusal: None,
                        reasoning_content: None,
                    },
                    finish_reason: Some(dynamo_protocols::types::FinishReason::Stop),
                    logprobs: None,
                },
            ],
            object: "chat.completion".to_string(),
        },
        nvext: None,
        prompt_logprobs: None,
        llm_metrics: None,
        tool_call_completion: Vec::new(),
    };

    // Wrap it in Annotated and create a stream
    let annotated_delta = Annotated {
        data: Some(data),
        id: Some("test_id".to_string()),
        event: None,
        comment: None,
        error: None,
    };
    let stream = Box::pin(stream::iter(vec![annotated_delta]));

    // Call DeltaAggregator::apply
    let result = apply(stream, ParsingOptions::default()).await;

    // Check the result
    assert!(result.is_ok());
    let mut response = result.unwrap();

    // Verify the response fields
    assert_eq!(response.inner.choices.len(), 2);
    response.inner.choices.sort_by_key(|a| a.index); // Ensure the choices are ordered
    let choice0 = &response.inner.choices[0];
    assert_eq!(choice0.index, 0);
    assert_eq!(
        choice0.message.content.as_ref().unwrap(),
        &ChatCompletionMessageContent::Text("Choice 0".to_string())
    );
    assert_eq!(
        choice0.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Stop)
    );
    assert_eq!(
        choice0.message.role,
        dynamo_protocols::types::Role::Assistant
    );

    let choice1 = &response.inner.choices[1];
    assert_eq!(choice1.index, 1);
    assert_eq!(
        choice1.message.content.as_ref().unwrap(),
        &ChatCompletionMessageContent::Text("Choice 1".to_string())
    );
    assert_eq!(
        choice1.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Stop)
    );
    assert_eq!(
        choice1.message.role,
        dynamo_protocols::types::Role::Assistant
    );
}

#[tokio::test]
async fn test_tool_calling_finish_reason_override_from_stop() {
    // Test that when tool calls are present but finish reason is Stop, it gets overridden to ToolCalls
    let tool_call_json =
        r#"{"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}}"#;

    let annotated_delta = create_test_delta(
        0,
        "I'll check the weather for you.",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Stop), // Original finish reason is Stop
        None,
        Some(tool_call_json),
    );

    let data = annotated_delta.data.unwrap();
    let annotated_delta = Annotated {
        data: Some(data),
        id: Some("test_id".to_string()),
        event: None,
        comment: None,
        error: None,
    };
    let stream = Box::pin(stream::iter(vec![annotated_delta]));

    let result = apply(stream, ParsingOptions::default()).await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];

    // Verify tool calls are present
    assert!(choice.message.tool_calls.is_some());
    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].r#type,
        dynamo_protocols::types::FunctionType::Function
    );

    // Most importantly, verify that finish reason was overridden to ToolCalls despite original being Stop
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ToolCalls)
    );
}

/// `Length` must survive a tool call, because it is the only thing
/// telling the caller that generation was cut off before the call finished.
///
/// This test previously asserted the opposite. That made it contradict
/// `tool_choice_finish_reasons.rs::test_required_tool_choice_preserves_length_finish_reason`,
/// which feeds the same truncated `required` payload through the streaming path and
/// requires `Length` to be preserved. Both passed, because they exercised different
/// stages of the same request.
#[tokio::test]
async fn test_tool_calling_finish_reason_preserves_length() {
    // Tool calls are present AND generation hit the token limit: `Length` wins.
    let tool_call_json = r#"{"name": "search", "arguments": {"query": "rust programming"}}"#;

    let annotated_delta = create_test_delta(
        0,
        "Let me search for that.",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Length), // Original finish reason is Length
        None,
        Some(tool_call_json),
    );

    let data = annotated_delta.data.unwrap();
    let annotated_delta = Annotated {
        data: Some(data),
        id: Some("test_id".to_string()),
        event: None,
        comment: None,
        error: None,
    };
    let stream = Box::pin(stream::iter(vec![annotated_delta]));

    let result = apply(stream, ParsingOptions::default()).await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];

    // Verify tool calls are present
    assert!(choice.message.tool_calls.is_some());
    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].r#type,
        dynamo_protocols::types::FunctionType::Function
    );

    // The terminal reason explains why generation stopped and is not replaced.
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Length)
    );
}

/// The guard for the rule above: `ContentFilter` is also a terminal reason that
/// explains itself, so it must survive a tool call for the same reason `Length` does.
#[tokio::test]
async fn test_tool_calling_finish_reason_preserves_content_filter() {
    let tool_call_json = r#"{"name": "search", "arguments": {"query": "rust programming"}}"#;

    let annotated_delta = create_test_delta(
        0,
        "Let me search for that.",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::ContentFilter),
        None,
        Some(tool_call_json),
    );

    let stream = Box::pin(stream::iter(vec![annotated_delta]));
    let response = apply(stream, ParsingOptions::default())
        .await
        .expect("aggregation should succeed");

    let choice = &response.inner.choices[0];
    assert!(choice.message.tool_calls.is_some());
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ContentFilter)
    );
}

#[tokio::test]
async fn batch_completed_calls_survive_each_terminal_at_every_split() {
    use crate::unified_stream as unified_parser;
    use dynamo_protocols::types::FinishReason;

    let tools = vec![dynamo_parsers::tool_calling::ToolDefinition {
        name: "write_file".into(),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {"count": {"type": "integer"}, "content": {"type": "string"}},
            "required": ["count", "content"]
        })),
        strict: None,
    }];
    for (family, raw) in unified_parser::tests::eight_family_calls("café") {
        let (tool, reasoning) = match family {
            "qwen3" => ("qwen3_coder", "qwen3"),
            "kimi_k2" => ("kimi_k2", "kimi_k25"),
            "glm47" => ("glm47", "glm45"),
            _ => (family, family),
        };
        if crate::families::selected_family(Some(tool), Some(reasoning), ParserVersion::Auto)
            .is_none()
        {
            continue;
        }
        for finish in [
            None,
            Some(FinishReason::Stop),
            Some(FinishReason::ToolCalls),
            Some(FinishReason::Length),
            Some(FinishReason::ContentFilter),
        ] {
            for split in (0..=raw.len()).filter(|split| raw.is_char_boundary(*split)) {
                let deltas = [
                    create_test_delta(
                        0,
                        &raw[..split],
                        Some(dynamo_protocols::types::Role::Assistant),
                        None,
                        None,
                        None,
                    ),
                    create_test_delta(0, &raw[split..], None, finish, None, None),
                ];
                let result = from_annotated_stream(
                    stream::iter(deltas),
                    ParsingOptions::new(Some(tool.into()), Some(reasoning.into()))
                        .with_tools(tools.clone()),
                )
                .await
                .unwrap();
                let choice = &result.inner.choices[0];
                let calls = choice
                    .message
                    .tool_calls
                    .as_ref()
                    .expect("completed batch call");
                assert_eq!(calls.len(), 1, "{family}/{finish:?}/{split}");
                assert_eq!(calls[0].function.name, "write_file");
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments)
                        .unwrap(),
                    serde_json::json!({"count": 2, "content": "café"})
                );
                assert!(choice.message.content.is_none(), "markup {family}/{split}");
                assert_eq!(
                    choice.finish_reason,
                    Some(match finish {
                        Some(FinishReason::Length) => FinishReason::Length,
                        Some(FinishReason::ContentFilter) => FinishReason::ContentFilter,
                        _ => FinishReason::ToolCalls,
                    })
                );
            }
        }
    }
}

#[tokio::test]
async fn terminal_validation_preserves_legacy_structured_calls() {
    use dynamo_protocols::types::FinishReason;

    // dynamo ran this at whatever `DYN_PARSER_VERSION` the process had.
    for (version, (tool, reasoning)) in [ParserVersion::Auto, ParserVersion::V1, ParserVersion::V2]
        .into_iter()
        .flat_map(|version| {
            [
                (None, None),
                (Some("hermes"), None),
                (Some("qwen3_coder"), Some("qwen3")),
                (Some("deepseek_v41"), Some("deepseek_v41")),
            ]
            .map(|pair| (version, pair))
        })
    {
        let unified = crate::families::selected_family(tool, reasoning, version).is_some()
            && version == ParserVersion::V2;
        for finish in [
            None,
            Some(FinishReason::Stop),
            Some(FinishReason::ToolCalls),
            Some(FinishReason::Length),
            Some(FinishReason::ContentFilter),
        ] {
            for args in [r#"{"city":"Paris"}"#, r#"{"city":"Paris""#, ""] {
                let mut delta = create_test_delta(
                    0,
                    "",
                    Some(dynamo_protocols::types::Role::Assistant),
                    finish,
                    None,
                    Some(r#"{"name":"search","arguments":{}}"#),
                );
                delta.data.as_mut().unwrap().inner.choices[0]
                    .delta
                    .tool_calls
                    .as_mut()
                    .unwrap()[0]
                    .function
                    .as_mut()
                    .unwrap()
                    .arguments = Some(args.into());
                let result = from_annotated_stream(
                    stream::iter([delta]),
                    ParsingOptions::new(tool.map(str::to_string), reasoning.map(str::to_string))
                        .with_parser_version(version),
                )
                .await;
                let unsupported = match version {
                    ParserVersion::V2 => tool == Some("hermes"),
                    ParserVersion::V1 => tool == Some("deepseek_v41"),
                    ParserVersion::Auto => false,
                };
                if unsupported {
                    assert!(
                        result.is_err(),
                        "explicit parser generation rejects unsupported configured families"
                    );
                    continue;
                }
                let result = result.unwrap();
                let choice = &result.inner.choices[0];
                let valid = serde_json::from_str::<serde_json::Value>(args).is_ok();
                let retained = if unified {
                    valid || (finish == Some(FinishReason::ToolCalls) && args.is_empty())
                } else {
                    finish != Some(FinishReason::Length) || valid
                };
                assert_eq!(
                    choice.message.tool_calls.is_some(),
                    retained,
                    "{tool:?}/{finish:?}/{args}"
                );
                if retained {
                    assert_eq!(
                        choice.message.tool_calls.as_ref().unwrap()[0]
                            .function
                            .arguments,
                        args
                    );
                }
                assert!(choice.message.content.is_none());
            }
        }
    }
}

#[tokio::test]
async fn test_content_filter_drops_malformed_tool_call_arguments() {
    let mut annotated_delta = create_test_delta(
        0,
        "",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::ContentFilter),
        None,
        Some(r#"{"name":"search","arguments":{}}"#),
    );
    annotated_delta
        .data
        .as_mut()
        .expect("test delta data")
        .inner
        .choices[0]
        .delta
        .tool_calls
        .as_mut()
        .expect("structured tool call")[0]
        .function
        .as_mut()
        .expect("tool function")
        .arguments = Some(r#"{"query":"Par"#.to_string());

    let response = apply_with_tool_terminal_policy(
        Box::pin(stream::iter(vec![annotated_delta])),
        ParsingOptions::new(Some("deepseek_v41".into()), Some("deepseek_v41".into())),
        true,
    )
    .await
    .expect("aggregation should succeed");
    let choice = &response.inner.choices[0];

    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ContentFilter)
    );
    assert!(choice.message.tool_calls.is_none());
    assert!(!serde_json::to_string(choice).unwrap().contains("Par"));
}

#[tokio::test]
async fn explicit_terminal_policy_rejects_provisional_evidence_for_each_choice() {
    let delta = |index, finish_reason| {
        let mut input = create_test_delta(
            index,
            "",
            Some(dynamo_protocols::types::Role::Assistant),
            finish_reason,
            None,
            Some(r#"{"name":"search","arguments":{"query":"Paris"}}"#),
        );
        input
            .data
            .as_mut()
            .unwrap()
            .tool_call_completion
            .push(ToolCallCompletion {
                choice_index: index,
                tool_index: 0,
                complete: false,
            });
        input
    };
    let incomplete_stop = delta(0, Some(dynamo_protocols::types::FinishReason::Stop));
    let complete = create_test_delta(
        1,
        "",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::ToolCalls),
        None,
        Some(r#"{"name":"search","arguments":{"query":"Tokyo"}}"#),
    );
    let incomplete_length = delta(2, Some(dynamo_protocols::types::FinishReason::Length));
    let incomplete_filtered = delta(
        3,
        Some(dynamo_protocols::types::FinishReason::ContentFilter),
    );
    let unterminated = delta(4, None);

    let response = apply_with_tool_terminal_policy(
        Box::pin(stream::iter(vec![
            incomplete_stop,
            complete,
            incomplete_length,
            incomplete_filtered,
            unterminated,
        ])),
        ParsingOptions::new(
            Some("deepseek_v41".to_string()),
            Some("deepseek_v41".to_string()),
        ),
        true,
    )
    .await
    .expect("aggregation should succeed");
    let incomplete_choices = [
        &response.inner.choices[0],
        &response.inner.choices[2],
        &response.inner.choices[3],
        &response.inner.choices[4],
    ];
    let complete_choice = &response.inner.choices[1];

    for choice in incomplete_choices {
        assert!(
            choice.message.tool_calls.is_none(),
            "choice {}",
            choice.index
        );
    }
    assert_eq!(
        response.inner.choices[0].finish_reason,
        Some(dynamo_protocols::types::FinishReason::Stop)
    );
    assert_eq!(
        response.inner.choices[2].finish_reason,
        Some(dynamo_protocols::types::FinishReason::Length)
    );
    assert_eq!(
        response.inner.choices[3].finish_reason,
        Some(dynamo_protocols::types::FinishReason::ContentFilter)
    );
    assert_eq!(response.inner.choices[4].finish_reason, None);
    assert_eq!(
        complete_choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ToolCalls)
    );
    assert_eq!(
        complete_choice.message.tool_calls.as_ref().unwrap().len(),
        1
    );
    assert_eq!(
        complete_choice.message.tool_calls.as_ref().unwrap()[0]
            .function
            .arguments,
        r#"{"query":"Tokyo"}"#
    );
}

#[tokio::test]
async fn qwen_incomplete_native_wrapper_is_not_dispatchable_after_aggregation() {
    let tools = vec![dynamo_parsers::tool_calling::ToolDefinition {
        name: "get_weather".to_string(),
        parameters: Some(serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        })),
        strict: None,
    }];
    let raw = create_test_delta(
        0,
        "<tool_call><function=get_weather><parameter=city>Paris</parameter>",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Stop),
        None,
        None,
    );
    let adapted = crate::unified_stream::apply_stream(
        stream::iter(vec![raw]),
        Some(tools),
        None,
        false,
        dynamo_parsers_v2::UnifiedParserStartingState::Response,
        "qwen3",
    );

    let response = from_annotated_stream(
        adapted,
        ParsingOptions::new(Some("qwen3_coder".to_string()), Some("qwen3".to_string())),
    )
    .await
    .expect("aggregation should succeed");

    assert_eq!(
        response.inner.choices[0].finish_reason,
        Some(dynamo_protocols::types::FinishReason::Length)
    );
    assert!(response.inner.choices[0].message.tool_calls.is_none());
}

#[tokio::test]
async fn test_length_drops_incomplete_structured_tool_call_without_leaking_content() {
    let mut annotated_delta = create_test_delta(
        0,
        "",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Length),
        None,
        Some(r#"{"name":"search","arguments":{}}"#),
    );
    annotated_delta
        .data
        .as_mut()
        .expect("test delta data")
        .inner
        .choices[0]
        .delta
        .tool_calls
        .as_mut()
        .expect("structured tool call")[0]
        .function
        .as_mut()
        .expect("tool function")
        .arguments = Some(r#"{"query":"Par"#.to_string());
    let response = apply(
        Box::pin(stream::iter(vec![annotated_delta])),
        ParsingOptions::default(),
    )
    .await
    .expect("aggregation should succeed");

    let choice = &response.inner.choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Length)
    );
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(choice.message.content, None);
    assert!(!serde_json::to_string(choice).unwrap().contains("Par"));
}

#[tokio::test]
async fn terminal_normalization_drops_incomplete_call_but_keeps_closed_json() {
    let make_chunk = |id: Option<&str>, name: Option<&str>, arguments: Option<&str>| {
        dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: 0,
            id: id.map(str::to_string),
            r#type: id.map(|_| dynamo_protocols::types::FunctionType::Function),
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: name.map(str::to_string),
                arguments: arguments.map(str::to_string),
            }),
            custom: None,
        }
    };

    for finish_reason in [
        Some(dynamo_protocols::types::FinishReason::Stop),
        Some(dynamo_protocols::types::FinishReason::ToolCalls),
        None,
    ] {
        let chunks = vec![
            create_test_delta_with_tool_chunks(
                0,
                vec![make_chunk(
                    Some("call-1"),
                    Some("get_weather"),
                    Some(r#"{"city": "#),
                )],
                None,
                Some(dynamo_protocols::types::Role::Assistant),
            ),
            create_test_delta_with_tool_chunks(
                0,
                vec![make_chunk(None, None, Some(r#""Tokyo""#))],
                finish_reason,
                None,
            ),
        ];
        let result = apply_with_tool_terminal_policy(
            Box::pin(stream::iter(chunks)),
            ParsingOptions::new(Some("deepseek_v41".into()), Some("deepseek_v41".into())),
            true,
        )
        .await
        .unwrap();

        let choice = &result.inner.choices[0];
        assert_eq!(
            choice.finish_reason,
            match finish_reason {
                Some(dynamo_protocols::types::FinishReason::ToolCalls) => {
                    Some(dynamo_protocols::types::FinishReason::Stop)
                }
                other => other,
            }
        );
        assert!(
            choice.message.tool_calls.is_none(),
            "incomplete JSON must not become a final tool call"
        );
    }

    for arguments in [None, Some("")] {
        let result = apply_with_tool_terminal_policy(
            Box::pin(stream::iter(vec![create_test_delta_with_tool_chunks(
                0,
                vec![make_chunk(Some("call-1"), Some("get_weather"), arguments)],
                None,
                Some(dynamo_protocols::types::Role::Assistant),
            )])),
            ParsingOptions::new(Some("deepseek_v41".into()), Some("deepseek_v41".into())),
            true,
        )
        .await
        .unwrap();

        let choice = &result.inner.choices[0];
        assert_eq!(choice.finish_reason, None);
        assert!(
            choice.message.tool_calls.is_none(),
            "a name-only EOF fragment must remain provisional"
        );
    }

    let result = apply(
        Box::pin(stream::iter(vec![
            create_test_delta_with_tool_chunks(
                0,
                vec![make_chunk(Some("call-1"), Some("get_weather"), None)],
                None,
                Some(dynamo_protocols::types::Role::Assistant),
            ),
            create_test_delta_with_tool_chunks(
                0,
                vec![make_chunk(None, None, Some(r#"{"city":"Tokyo"}"#))],
                Some(dynamo_protocols::types::FinishReason::Stop),
                None,
            ),
        ])),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    let choice = &result.inner.choices[0];
    let tool_call = &choice.message.tool_calls.as_ref().unwrap()[0];
    assert_eq!(tool_call.function.name, "get_weather");
    assert_eq!(tool_call.function.arguments, r#"{"city":"Tokyo"}"#);
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn terminal_normalization_keeps_valid_calls_when_sibling_is_malformed() {
    let make_chunk = |index, id: Option<&str>, name: Option<&str>, arguments: &str| {
        dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index,
            id: id.map(str::to_string),
            r#type: id.map(|_| dynamo_protocols::types::FunctionType::Function),
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: name.map(str::to_string),
                arguments: Some(arguments.to_string()),
            }),
            custom: None,
        }
    };

    let chunks = vec![
        create_test_delta_with_tool_chunks(
            0,
            vec![
                make_chunk(0, Some("call-invalid"), Some("get_weather"), r#"{"city":"#),
                make_chunk(1, Some("call-valid"), Some("get_time"), r#"{"tz":"#),
            ],
            None,
            Some(dynamo_protocols::types::Role::Assistant),
        ),
        create_test_delta_with_tool_chunks(
            0,
            vec![
                make_chunk(0, None, None, "not-json"),
                make_chunk(1, None, None, r#""UTC"}"#),
            ],
            Some(dynamo_protocols::types::FinishReason::ToolCalls),
            None,
        ),
    ];
    let result = apply_with_tool_terminal_policy(
        Box::pin(stream::iter(chunks)),
        ParsingOptions::new(Some("deepseek_v41".into()), Some("deepseek_v41".into())),
        true,
    )
    .await
    .unwrap();

    let choice = &result.inner.choices[0];
    let calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id, "call-valid");
    assert_eq!(calls[0].function.name, "get_time");
    assert_eq!(calls[0].function.arguments, r#"{"tz":"UTC"}"#);
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn test_length_keeps_prose_before_incomplete_native_marker() {
    let text = "I can help. <tool_call>get_weather<arg_key>city</arg_key><arg_value>Par";
    let delta = create_test_delta(
        0,
        text,
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Length),
        None,
        None,
    );
    let result = apply(
        Box::pin(stream::iter(vec![delta])),
        ParsingOptions::new(Some("glm47".to_string()), None),
    )
    .await
    .unwrap();

    assert_eq!(
        result.inner.choices[0].message.content,
        Some(ChatCompletionMessageContent::Text(
            "I can help. ".to_string()
        ))
    );
}

#[tokio::test]
async fn test_length_drops_a_nonfinal_invalid_structured_tool_call() {
    let make_name = |idx: u32, id: &str, name: &str| {
        dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: idx,
            id: Some(id.to_string()),
            r#type: Some(dynamo_protocols::types::FunctionType::Function),
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: Some(name.to_string()),
                arguments: None,
            }),
            custom: None,
        }
    };
    let make_args =
        |idx: u32, fragment: &str| dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: idx,
            id: None,
            r#type: None,
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: None,
                arguments: Some(fragment.to_string()),
            }),
            custom: None,
        };
    let first = create_test_delta_with_tool_chunks(
        0,
        vec![make_name(0, "first", "get_weather")],
        None,
        Some(dynamo_protocols::types::Role::Assistant),
    );
    let first_args =
        create_test_delta_with_tool_chunks(0, vec![make_args(0, "{\"city\":")], None, None);
    let mut second = create_test_delta_with_tool_chunks(
        0,
        vec![
            make_name(1, "second", "get_time"),
            make_args(1, "{\"tz\":\"UTC\"}"),
        ],
        Some(dynamo_protocols::types::FinishReason::Length),
        None,
    );
    second.data.as_mut().unwrap().tool_call_completion = vec![
        ToolCallCompletion {
            choice_index: 0,
            tool_index: 0,
            complete: true,
        },
        ToolCallCompletion {
            choice_index: 0,
            tool_index: 1,
            complete: true,
        },
    ];
    for unified in [false, true] {
        let options = if unified {
            ParsingOptions::new(Some("deepseek_v41".into()), Some("deepseek_v41".into()))
        } else {
            ParsingOptions::default()
        };
        let result = apply_with_tool_terminal_policy(
            stream::iter(vec![first.clone(), first_args.clone(), second.clone()]),
            options,
            unified,
        )
        .await
        .unwrap();
        let tool_calls = result.inner.choices[0]
            .message
            .tool_calls
            .as_ref()
            .expect("the later complete call must remain visible");
        assert_eq!(tool_calls.len(), if unified { 1 } else { 2 });
        if !unified {
            assert_eq!(tool_calls[0].function.arguments, "{\"city\":");
        }
        let complete = tool_calls.last().unwrap();
        assert_eq!(complete.id, "second");
        assert_eq!(complete.function.name, "get_time");
        assert_eq!(complete.function.arguments, "{\"tz\":\"UTC\"}");
        assert_eq!(
            result.inner.choices[0].finish_reason,
            Some(dynamo_protocols::types::FinishReason::Length)
        );
    }
}

#[tokio::test]
async fn test_length_drops_the_final_invalid_structured_tool_call() {
    let make_name = |idx: u32, id: &str, name: &str| {
        dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: idx,
            id: Some(id.to_string()),
            r#type: Some(dynamo_protocols::types::FunctionType::Function),
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: Some(name.to_string()),
                arguments: None,
            }),
            custom: None,
        }
    };
    let make_args =
        |idx: u32, fragment: &str| dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
            index: idx,
            id: None,
            r#type: None,
            function: Some(dynamo_protocols::types::FunctionCallStream {
                name: None,
                arguments: Some(fragment.to_string()),
            }),
            custom: None,
        };
    let first = create_test_delta_with_tool_chunks(
        0,
        vec![
            make_name(0, "first", "get_weather"),
            make_args(0, "{\"city\":\"Paris\"}"),
        ],
        None,
        Some(dynamo_protocols::types::Role::Assistant),
    );
    let final_invalid = create_test_delta_with_tool_chunks(
        0,
        vec![make_name(1, "second", "get_time"), make_args(1, "{\"tz\":")],
        Some(dynamo_protocols::types::FinishReason::Length),
        None,
    );
    let result = apply(
        Box::pin(stream::iter(vec![first, final_invalid])),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    let tool_calls = result.inner.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("the complete first call must remain visible");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_weather");
    assert_eq!(tool_calls[0].function.arguments, "{\"city\":\"Paris\"}");
}

/// The limit can land after the function name and before the first argument
/// token. That leaves no argument bytes, which is the most truncated shape, not a
/// parameterless call, so it must not reach the client as executable.
#[tokio::test]
async fn test_length_drops_a_final_tool_call_with_no_argument_bytes() {
    let call = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("first".to_string()),
        r#type: Some(dynamo_protocols::types::FunctionType::Function),
        function: Some(dynamo_protocols::types::FunctionCallStream {
            name: Some("get_weather".to_string()),
            arguments: None,
        }),
        custom: None,
    };
    let result = apply(
        Box::pin(stream::iter(vec![create_test_delta_with_tool_chunks(
            0,
            vec![call],
            Some(dynamo_protocols::types::FinishReason::Length),
            Some(dynamo_protocols::types::Role::Assistant),
        )])),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    let choice = &result.inner.choices[0];
    assert!(
        choice.message.tool_calls.is_none(),
        "a call cut off before its arguments must not be executable"
    );
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Length)
    );
}

#[tokio::test]
async fn test_finished_tool_call_with_no_argument_bytes_remains_valid() {
    let call = dynamo_protocols::types::ChatCompletionMessageToolCallChunk {
        index: 0,
        id: Some("first".to_string()),
        r#type: Some(dynamo_protocols::types::FunctionType::Function),
        function: Some(dynamo_protocols::types::FunctionCallStream {
            name: Some("get_weather".to_string()),
            arguments: None,
        }),
        custom: None,
    };
    let result = apply(
        Box::pin(stream::iter(vec![create_test_delta_with_tool_chunks(
            0,
            vec![call],
            Some(dynamo_protocols::types::FinishReason::ToolCalls),
            Some(dynamo_protocols::types::Role::Assistant),
        )])),
        ParsingOptions::default(),
    )
    .await
    .unwrap();

    let tool_calls = result.inner.choices[0]
        .message
        .tool_calls
        .as_ref()
        .expect("a finished parameterless call must remain valid");
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_weather");
    assert_eq!(tool_calls[0].function.arguments, "");
    assert_eq!(
        result.inner.choices[0].finish_reason,
        Some(dynamo_protocols::types::FinishReason::ToolCalls)
    );
}

/// `phi4` opens a call with the bare word `functools`, so cutting content at every
/// configured marker would truncate an ordinary answer about that Python module.
#[tokio::test]
async fn test_length_keeps_prose_containing_a_word_shaped_marker() {
    let text = "Use functools.lru_cache to memoize the call, and it will";
    let result = apply(
        Box::pin(stream::iter(vec![create_test_delta(
            0,
            text,
            Some(dynamo_protocols::types::Role::Assistant),
            Some(dynamo_protocols::types::FinishReason::Length),
            None,
            None,
        )])),
        ParsingOptions::new(Some("phi4".to_string()), None),
    )
    .await
    .unwrap();

    assert_eq!(
        result.inner.choices[0].message.content,
        Some(ChatCompletionMessageContent::Text(text.to_string()))
    );
}

#[tokio::test]
async fn test_tool_calling_finish_reason_override_from_none() {
    // Test that when tool calls are present but finish reason is None, it gets set to ToolCalls
    let tool_call_json = r#"{"name": "calculate", "arguments": {"expression": "2+2"}}"#;

    let annotated_delta = create_test_delta(
        0,
        "I'll calculate that for you.",
        Some(dynamo_protocols::types::Role::Assistant),
        None, // Original finish reason is None
        None,
        Some(tool_call_json),
    );

    let data = annotated_delta.data.unwrap();
    let annotated_delta = Annotated {
        data: Some(data),
        id: Some("test_id".to_string()),
        event: None,
        comment: None,
        error: None,
    };
    let stream = Box::pin(stream::iter(vec![annotated_delta]));

    let result = apply(stream, ParsingOptions::default()).await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];

    // Verify tool calls are present
    assert!(choice.message.tool_calls.is_some());
    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);

    // Verify that finish reason was set to ToolCalls despite original being None
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn test_no_tool_calling_preserves_original_finish_reason() {
    // Test that when no tool calls are present, the original finish reason is preserved
    let annotated_delta = create_test_delta(
        0,
        "This is a regular response without tool calls.",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Stop),
        None,
        None, // No tool calls
    );

    let data = annotated_delta.data.unwrap();
    let annotated_delta = Annotated {
        data: Some(data),
        id: Some("test_id".to_string()),
        event: None,
        comment: None,
        error: None,
    };
    let stream = Box::pin(stream::iter(vec![annotated_delta]));

    let result = apply(stream, ParsingOptions::default()).await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];

    // Verify no tool calls are present
    assert!(choice.message.tool_calls.is_none());

    // Verify that original finish reason (Stop) is preserved
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Stop)
    );
}

#[tokio::test]
async fn test_empty_tool_calls_preserves_original_finish_reason() {
    // Test that when tool calls array is empty, the original finish reason is preserved
    // Create a delta with empty tool calls by modifying the create_test_delta output
    let mut annotated_delta = create_test_delta(
        0,
        "Response with empty tool calls array.",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Length),
        None,
        None,
    );

    // Manually set empty tool calls array
    if let Some(ref mut data) = annotated_delta.data {
        data.inner.choices[0].delta.tool_calls = Some(vec![]); // Empty tool calls array
    }

    let data = annotated_delta.data.unwrap();
    let annotated_delta = Annotated {
        data: Some(data),
        id: Some("test_id".to_string()),
        event: None,
        comment: None,
        error: None,
    };
    let stream = Box::pin(stream::iter(vec![annotated_delta]));

    let result = apply(stream, ParsingOptions::default()).await;

    assert!(result.is_ok());
    let response = result.unwrap();
    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];

    // Verify tool calls array is empty
    assert!(choice.message.tool_calls.is_none());

    // Verify that original finish reason (Length) is preserved since tool calls are empty
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Length)
    );
}

#[tokio::test]
async fn test_tool_calling_output() {
    // Simulate a delta with a tool call in the content
    let tool_call_json = r#"{"name": "get_weather", "arguments": {"location": "San Francisco, CA", "unit": "fahrenheit"}}"#;

    // Use create_test_delta to generate the annotated delta, then extract the inner delta for the test
    let annotated_delta = create_test_delta(
        0,
        "Hey Dude ! What's the weather in San Francisco in Fahrenheit?",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::ToolCalls),
        None,
        Some(tool_call_json),
    );
    let data = annotated_delta.data.unwrap();

    // Wrap it in Annotated and create a stream
    let annotated_delta = Annotated {
        data: Some(data),
        id: Some("test_id".to_string()),
        event: None,
        comment: None,
        error: None,
    };
    let stream = Box::pin(stream::iter(vec![annotated_delta]));

    // Call DeltaAggregator::apply
    let result = apply(stream, ParsingOptions::default()).await;

    // Check the result
    assert!(result.is_ok());
    let response = result.unwrap();

    // There should be one choice
    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];

    // The tool_calls field should be present and parsed
    assert!(choice.message.tool_calls.is_some());
    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);

    let tool_call = &tool_calls[0];
    assert_eq!(tool_call.function.name, "get_weather");
    // The arguments should be a JSON string containing the expected keys
    let args: serde_json::Value = serde_json::from_str(&tool_call.function.arguments).unwrap();
    assert_eq!(args["location"], "San Francisco, CA");
    assert_eq!(args["unit"], "fahrenheit");

    assert_eq!(
        choice.message.content.as_ref().unwrap(),
        &ChatCompletionMessageContent::Text(
            "Hey Dude ! What's the weather in San Francisco in Fahrenheit?".to_string()
        )
    );

    // The finish_reason should be ToolCalls
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ToolCalls)
    );
    assert_eq!(
        choice.message.role,
        dynamo_protocols::types::Role::Assistant
    );
}

#[tokio::test]
async fn test_tool_calling_finish_reason_override_from_stop_alternative() {
    // Test that when tool calls are present but finish reason is Stop, it gets overridden to ToolCalls
    let tool_call_json =
        r#"{"name": "get_weather", "arguments": {"location": "New York", "unit": "celsius"}}"#;

    let annotated_delta = create_test_delta(
        0,
        "Getting weather for New York",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Stop), // This should be overridden
        None,
        Some(tool_call_json),
    );

    let stream = Box::pin(stream::iter(vec![annotated_delta]));

    // Call DeltaAggregator::apply
    let result = apply(stream, ParsingOptions::default()).await;

    // Check the result
    assert!(result.is_ok());
    let response = result.unwrap();

    // There should be one choice
    assert_eq!(response.inner.choices.len(), 1);
    let choice = &response.inner.choices[0];

    // The finish_reason should be ToolCalls, not Stop, because tool calls are present
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ToolCalls)
    );

    // Verify tool calls are present
    assert!(choice.message.tool_calls.is_some());
    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(tool_calls[0].function.name, "get_weather");
}

#[tokio::test]
async fn test_parses_aggregated_tool_call_text_into_tool_calls() {
    let annotated_delta = create_test_delta(
        0,
        "<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"SF\"}}\n</tool_call>",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Stop),
        None,
        None,
    );

    let stream = Box::pin(stream::iter(vec![annotated_delta]));
    let result = apply(
        stream,
        ParsingOptions::new(Some("hermes".to_string()), None),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    let choice = &response.inner.choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ToolCalls)
    );
    assert_eq!(choice.message.content, None);
    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(tool_calls.len(), 1);
    assert_eq!(
        tool_calls[0].r#type,
        dynamo_protocols::types::FunctionType::Function
    );
    assert_eq!(tool_calls[0].function.name, "get_weather");
    assert_eq!(tool_calls[0].function.arguments, "{\"location\":\"SF\"}");
}

#[tokio::test]
async fn deepseek_v41_batch_recovers_malformed_closed_arguments_as_text() {
    let text = concat!(
        "<｜DSML｜ calls><｜DSML｜ invoke name=\"get_weather\">",
        "<｜DSML｜ parameter name=\"count\" string=\"false\">not-json</｜DSML｜ parameter>",
        "</｜DSML｜ invoke></｜DSML｜ calls>"
    );
    let annotated_delta = create_test_delta(
        0,
        text,
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Stop),
        None,
        None,
    );
    let stream = Box::pin(stream::iter(vec![annotated_delta]));

    let result = apply(
        stream,
        ParsingOptions::new(
            Some("deepseek_v41".to_string()),
            Some("deepseek_v41".to_string()),
        ),
    )
    .await;

    let response = result.expect("malformed parser output should fall back to text");
    let choice = &response.inner.choices[0];
    assert_eq!(
        choice.message.content,
        Some(ChatCompletionMessageContent::Text(text.to_string()))
    );
    assert!(choice.message.tool_calls.is_none());
}

#[tokio::test]
async fn test_disabled_tool_parsing_preserves_structured_content_with_name() {
    let json = r#"{"name":"Science Fair","date":"Friday","participants":["Alice","Bob"]}"#;
    let annotated_delta = create_test_delta(
        0,
        json,
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Stop),
        None,
        None,
    );

    let stream = Box::pin(stream::iter(vec![annotated_delta]));
    let response = apply(
        stream,
        ParsingOptions::new(Some("hermes".to_string()), None).with_tool_call_parsing_enabled(false),
    )
    .await
    .expect("aggregation should preserve assistant content");
    let choice = &response.inner.choices[0];

    assert_eq!(
        choice.message.content,
        Some(ChatCompletionMessageContent::Text(json.to_string()))
    );
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Stop)
    );
}

#[tokio::test]
async fn test_preserves_non_tool_content_when_parsing_aggregated_tool_calls() {
    let annotated_delta = create_test_delta(
        0,
        "hello\n<tool_call>\n{\"name\":\"get_weather\",\"arguments\":{\"location\":\"SF\"}}\n</tool_call>",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Stop),
        None,
        None,
    );

    let stream = Box::pin(stream::iter(vec![annotated_delta]));
    let result = apply(
        stream,
        ParsingOptions::new(Some("hermes".to_string()), None),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    let choice = &response.inner.choices[0];
    assert_eq!(
        choice.message.content,
        Some(ChatCompletionMessageContent::Text("hello".to_string()))
    );
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ToolCalls)
    );
    assert_eq!(
        choice.message.tool_calls.as_ref().unwrap()[0].r#type,
        dynamo_protocols::types::FunctionType::Function
    );
}

#[tokio::test]
async fn test_disabled_harmony_aggregate_drops_internal_analysis() {
    let annotated_delta = create_test_delta(
        0,
        r#"<|channel|>analysis<|message|>Need current weather.<|end|><|start|>assistant<|channel|>commentary to=functions.get_current_weather <|constrain|>json<|message|>{"location":"Hidden City"}"#,
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Stop),
        None,
        None,
    );

    let stream = Box::pin(stream::iter(vec![annotated_delta]));
    let result = apply(
        stream,
        ParsingOptions::new(Some("harmony".to_string()), None)
            .with_tool_call_parsing_enabled(false),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    let choice = &response.inner.choices[0];
    assert_eq!(choice.message.content, None);
    assert!(choice.message.tool_calls.is_none());
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Stop)
    );
}

#[tokio::test]
async fn test_disabled_harmony_aggregate_suppresses_parsed_tool_call() {
    let annotated_delta = create_test_delta(
        0,
        r#"<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"location":"Paris"}<|call|>"#,
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::ToolCalls),
        None,
        None,
    );

    let response = apply(
        Box::pin(stream::iter(vec![annotated_delta])),
        ParsingOptions::new(Some("harmony".to_string()), None)
            .with_tool_call_parsing_enabled(false),
    )
    .await
    .expect("Harmony decoding should succeed");
    let choice = &response.inner.choices[0];

    assert!(choice.message.tool_calls.is_none());
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Stop)
    );
}

#[tokio::test]
async fn test_harmony_aggregate_plain_text_without_markers_stays_plain_text() {
    let annotated_delta = create_test_delta(
        0,
        "plain response",
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Stop),
        None,
        None,
    );

    let stream = Box::pin(stream::iter(vec![annotated_delta]));
    let result = apply(
        stream,
        ParsingOptions::new(Some("harmony".to_string()), None),
    )
    .await;

    assert!(result.is_ok());
    let response = result.unwrap();
    let choice = &response.inner.choices[0];
    assert_eq!(
        choice.message.content,
        Some(ChatCompletionMessageContent::Text(
            "plain response".to_string()
        ))
    );
    assert!(choice.message.tool_calls.is_none());
}

#[tokio::test]
async fn test_disabled_kimi_k3_aggregate_decodes_tool_free_response() {
    let raw = concat!(
        "<|open|>response<|sep|>",
        "323",
        "<|close|>response<|sep|>",
        "<|close|>message<|sep|>",
        "<|end_of_msg|>"
    );

    for parser in ["kimi_k3", "kimi-k3"] {
        let annotated_delta = create_test_delta(
            0,
            raw,
            Some(dynamo_protocols::types::Role::Assistant),
            Some(dynamo_protocols::types::FinishReason::Stop),
            None,
            None,
        );
        let response = apply(
            Box::pin(stream::iter(vec![annotated_delta])),
            ParsingOptions::new(Some(parser.to_string()), Some("kimi_k3".to_string()))
                .with_tool_call_parsing_enabled(false),
        )
        .await
        .expect("Kimi K3 response decoding should succeed");
        let choice = &response.inner.choices[0];

        assert_eq!(
            choice.message.content,
            Some(ChatCompletionMessageContent::Text("323".to_string())),
            "parser alias {parser} must strip K3 XTML wrappers"
        );
        assert!(choice.message.tool_calls.is_none());
        assert_eq!(
            choice.finish_reason,
            Some(dynamo_protocols::types::FinishReason::Stop)
        );
    }
}

#[test]
fn test_reasoning_only_response_serializes_content_key_as_null() {
    // When a response carries reasoning_content but no text or
    // content parts, the `content` key must still be present in the
    // serialized JSON (as `null`) so clients can rely on it alongside
    // `reasoning_content`. Fixed by removing skip_serializing_if from
    // ChatCompletionResponseMessage.content.
    let delta = DeltaChoice {
        index: 0,
        text: String::new(),
        role: Some(dynamo_protocols::types::Role::Assistant),
        finish_reason: Some(dynamo_protocols::types::FinishReason::Stop),
        logprobs: None,
        tool_call_chunks: BTreeMap::new(),
        parser_call_completion: HashMap::new(),
        tool_calls: None,
        reasoning_content: Some("Analyzing the question.".to_string()),
        refusal: None,
        content_parts: vec![],
    };

    let choice: dynamo_protocols::types::ChatChoice = delta.into();

    assert!(choice.message.content.is_none());
    assert_eq!(
        choice.message.reasoning_content.as_deref(),
        Some("Analyzing the question.")
    );

    let json = serde_json::to_value(&choice.message).unwrap();
    assert_eq!(
        json.get("content"),
        Some(&serde_json::Value::Null),
        "content key must be serialized as null when absent"
    );
    assert!(json.get("reasoning_content").is_some());
}

// --- glm47 truncation recovery tests ---
//
// Reproduces the drop confirmed on dynamo-parsers 5.1.3:
//   in:  <tool_call>get_weather<arg_key>city</arg_key><arg_value>Bos   (finish_reason=length)
//   out: finish=length  tool_calls=None  content=""
// The parser sees an incomplete XML block and returns no tool call and no
// content. The non-streaming contract keeps the finish signal and suppresses
// the incomplete structured candidate rather than leaking parser markup.

#[tokio::test]
async fn test_glm47_single_truncated_call_is_suppressed() {
    let text = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Bos";
    let delta = create_test_delta(
        0,
        text,
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Length),
        None,
        None,
    );
    let stream = Box::pin(stream::iter(vec![delta]));
    let result = apply(stream, ParsingOptions::new(Some("glm47".to_string()), None))
        .await
        .unwrap();

    let choice = &result.inner.choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::Length)
    );
    assert!(
        choice.message.tool_calls.is_none(),
        "incomplete call must not produce a structured tool_call"
    );
    assert_eq!(choice.message.content, None);
    assert!(!serde_json::to_string(choice).unwrap().contains(text));
}

#[tokio::test]
async fn test_glm47_complete_call_then_truncated_second_is_suppressed() {
    // First call complete, second truncated mid-argument.
    let text = "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Boston</arg_value></tool_call><tool_call>get_time<arg_key>tz</arg_key><arg_value>US/E";
    let delta = create_test_delta(
        0,
        text,
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Length),
        None,
        None,
    );
    let stream = Box::pin(stream::iter(vec![delta]));
    let result = apply(stream, ParsingOptions::new(Some("glm47".to_string()), None))
        .await
        .unwrap();

    let choice = &result.inner.choices[0];
    // First complete call is parsed into tool_calls and remains available.
    let tool_calls = choice.message.tool_calls.as_ref().unwrap();
    assert_eq!(
        tool_calls.len(),
        1,
        "first complete call must be structured"
    );
    assert_eq!(tool_calls[0].function.name, "get_weather");
    assert_eq!(choice.message.content, None);
    assert!(
        !serde_json::to_string(choice)
            .unwrap()
            .contains("<tool_call>get_time"),
        "truncated second call markup must not leak into the response"
    );
}

#[tokio::test]
async fn test_glm47_quoted_marker_prose_is_preserved_on_length_finish() {
    let text = r#"The literal "<tool_call>" marker is part of the explanation."#;
    let delta = create_test_delta(
        0,
        text,
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Length),
        None,
        None,
    );
    let result = apply(
        Box::pin(stream::iter(vec![delta])),
        ParsingOptions::new(Some("glm47".to_string()), None),
    )
    .await
    .unwrap();

    let choice = &result.inner.choices[0];
    assert_eq!(
        choice.message.content,
        Some(ChatCompletionMessageContent::Text(text.to_string()))
    );
    assert!(choice.message.tool_calls.is_none());
}

#[tokio::test]
async fn test_glm47_complete_call_stop_no_recovery() {
    // Complete call with finish_reason=Stop: no recovery needed.
    let text =
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Boston</arg_value></tool_call>";
    let delta = create_test_delta(
        0,
        text,
        Some(dynamo_protocols::types::Role::Assistant),
        Some(dynamo_protocols::types::FinishReason::Stop),
        None,
        None,
    );
    let stream = Box::pin(stream::iter(vec![delta]));
    let result = apply(stream, ParsingOptions::new(Some("glm47".to_string()), None))
        .await
        .unwrap();

    let choice = &result.inner.choices[0];
    assert_eq!(
        choice.finish_reason,
        Some(dynamo_protocols::types::FinishReason::ToolCalls)
    );
    assert!(choice.message.tool_calls.is_some());
    assert!(
        choice.message.content.is_none(),
        "no content for a clean complete call"
    );
}

#[tokio::test]
async fn review_structured_complete_calls_survive_every_finish_without_evidence() {
    for finish in [
        None,
        Some(dynamo_protocols::types::FinishReason::Stop),
        Some(dynamo_protocols::types::FinishReason::Length),
        Some(dynamo_protocols::types::FinishReason::ToolCalls),
    ] {
        let input = create_test_delta(
            0,
            "",
            None,
            finish,
            None,
            Some(r#"{"name":"write_file","arguments":{"count":1,"content":"data"}}"#),
        );
        let result =
            apply_with_tool_terminal_policy(stream::iter([input]), ParsingOptions::default(), true)
                .await
                .unwrap();
        assert_eq!(
            result.inner.choices[0]
                .message
                .tool_calls
                .as_ref()
                .map(Vec::len),
            Some(1),
            "{finish:?}"
        );
    }
}

#[tokio::test]
async fn review_backend_completion_evidence_remains_authoritative() {
    for complete in [None, Some(false), Some(true)] {
        for finish in [
            None,
            Some(dynamo_protocols::types::FinishReason::Stop),
            Some(dynamo_protocols::types::FinishReason::Length),
            Some(dynamo_protocols::types::FinishReason::ToolCalls),
        ] {
            let mut input = create_test_delta(
                0,
                "",
                None,
                finish,
                None,
                Some(r#"{"name":"write_file","arguments":{"count":1,"content":"data"}}"#),
            );
            if let Some(complete) = complete {
                input
                    .data
                    .as_mut()
                    .unwrap()
                    .tool_call_completion
                    .push(ToolCallCompletion {
                        choice_index: 0,
                        tool_index: 0,
                        complete,
                    });
            }
            let response = apply_with_tool_terminal_policy(
                stream::iter([input]),
                ParsingOptions::default(),
                true,
            )
            .await
            .unwrap();
            let calls = &response.inner.choices[0].message.tool_calls;
            if complete == Some(false) {
                assert!(calls.is_none(), "{finish:?}");
            } else {
                let calls = calls.as_ref().unwrap();
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].function.name, "write_file");
                assert_eq!(
                    serde_json::from_str::<serde_json::Value>(&calls[0].function.arguments)
                        .unwrap(),
                    serde_json::json!({"count":1,"content":"data"})
                );
            }
        }
    }
}
