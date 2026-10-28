// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Parity between `dynamo_renderer::request_rules` and the Dynamo frontend
//! code it was moved from.
//!
//! `oracle` is a frozen copy of that code. Each test runs the oracle and the
//! moved code on the same generated inputs and requires the same output. The
//! oracle calls the code the way Dynamo's `OpenAIPreprocessor` does: the
//! thinking rules as in `generate` (`lib/llm/src/preprocessor.rs:7796-7803`),
//! and rendering through `apply_template`.

mod oracle;

use std::collections::HashMap;
use std::sync::Arc;

use dynamo_renderer::request_rules::{
    ChatPromptOptions, RequestApi, ThinkingRules, has_minimax_m3_constrained_generation,
    reject_unsupported_reasoning_effort, render_chat_prompt, should_normalize_tool_call_arguments,
};
use dynamo_renderer::{
    ChatTemplate, ContextMixins, OAIPromptFormatter, PromptFormatter, RenderedPrompt,
    RenderedSegment,
};
use oracle::{
    CommonExtProvider, InvalidArgument, NvCreateChatCompletionRequest, NvExtProvider,
    OpenAIPreprocessor, has_explicit_guided_decoding,
};
use serde_json::{Value, json};

const REASONING_PARSERS: &[Option<&str>] = &[
    None,
    Some("qwen3"),
    Some("kimi_k25"),
    Some("kimi_k3"),
    Some("kimi-k3"),
    Some("minimax_m3"),
    Some("minimax-m3"),
    Some("deepseek_v3_1"),
    Some("gpt_oss"),
];

const TOOL_CALL_PARSERS: &[Option<&str>] = &[
    None,
    Some("minimax_m3"),
    Some("minimax-m3-nom"),
    Some("glm47"),
];

const DEFAULT_MODES: &[Option<&str>] = &[None, Some("enabled"), Some("disabled"), Some("adaptive")];

/// Every value an alias takes in the matrix; `None` leaves the key out.
fn alias_values() -> Vec<Option<Value>> {
    vec![
        None,
        Some(json!(true)),
        Some(json!(false)),
        Some(json!(0)),
        Some(json!(0.5)),
        Some(Value::Null),
        Some(json!("Enabled")),
        Some(json!("DISABLED")),
        Some(json!("adaptive")),
        Some(json!("ADAPTIVE")),
        Some(json!("off")),
        Some(json!(" ")),
        Some(json!({"type": "adaptive"})),
    ]
}

/// Chat-template argument maps over the three aliases and `reasoning_effort`.
/// `None` is the request with no `chat_template_args` at all.
fn argument_maps() -> Vec<Option<HashMap<String, Value>>> {
    let values = alias_values();
    let mut maps = vec![None];
    for thinking in &values {
        for enable_thinking in &values {
            for thinking_mode in &values {
                // Only the presence of `reasoning_effort` matters to these rules.
                for reasoning_effort in [None, Some(json!("high"))] {
                    let mut map = HashMap::new();
                    for (key, value) in [
                        ("thinking", thinking),
                        ("enable_thinking", enable_thinking),
                        ("thinking_mode", thinking_mode),
                        ("reasoning_effort", &reasoning_effort),
                    ] {
                        if let Some(value) = value {
                            map.insert(key.to_string(), value.clone());
                        }
                    }
                    maps.push(Some(map));
                }
            }
        }
    }
    maps
}

/// The request shapes that change the thinking rules: output constraints and
/// a top-level `thinking` field.
fn request_shapes() -> Vec<Value> {
    let tools =
        json!([{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]);
    let constraints = [
        json!({}),
        json!({"response_format": {"type": "text"}}),
        json!({"response_format": {"type": "json_object"}}),
        json!({"response_format": {"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object"}}}}),
        json!({"tools": tools, "tool_choice": "auto"}),
        json!({"tools": tools, "tool_choice": "none"}),
        json!({"tools": tools, "tool_choice": "required"}),
        json!({"tools": tools, "tool_choice": {"type": "function", "function": {"name": "f"}}}),
        json!({"guided_json": {"type": "object"}}),
        json!({"guided_regex": "a|b"}),
        json!({"guided_choice": []}),
        json!({"guided_choice": ["a"]}),
        json!({"guided_grammar": "root ::= \"a\""}),
    ];
    let mut shapes = Vec::new();
    for constraint in constraints {
        for thinking in [None, Some(json!({"type": "adaptive"}))] {
            let mut body = json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}]
            });
            for (key, value) in constraint.as_object().unwrap() {
                body[key] = value.clone();
            }
            if let Some(thinking) = thinking {
                body["thinking"] = thinking;
            }
            shapes.push(body);
        }
    }
    shapes
}

/// Dynamo's `generate` order: client intent first, then the default, then
/// the normalization (`lib/llm/src/preprocessor.rs:7796-7803`, with
/// `apply_default_thinking_mode` inlined so the default is a plain value).
fn oracle_thinking(
    request: &mut NvCreateChatCompletionRequest,
    reasoning_parser: Option<&str>,
    tool_call_parser: Option<&str>,
    default_thinking_mode: Option<&str>,
) {
    let thinking_control_from_client =
        OpenAIPreprocessor::request_has_client_thinking_control(request);
    OpenAIPreprocessor::apply_default_thinking_mode_value(
        default_thinking_mode,
        &mut request.chat_template_args,
        thinking_control_from_client,
    );
    OpenAIPreprocessor::normalize_thinking_arg_with_source(
        request,
        reasoning_parser,
        tool_call_parser,
        thinking_control_from_client,
    );
}

/// The moved rules, called the way Dynamo will call them.
fn moved_thinking(
    request: &mut NvCreateChatCompletionRequest,
    reasoning_parser: Option<&str>,
    tool_call_parser: Option<&str>,
    default_thinking_mode: Option<&str>,
) {
    let constrained_output =
        has_minimax_m3_constrained_generation(&*request, has_explicit_guided_decoding(request));
    ThinkingRules {
        reasoning_parser,
        tool_call_parser,
        default_thinking_mode,
    }
    .apply(
        &mut request.chat_template_args,
        request.thinking.is_some(),
        constrained_output,
    );
}

/// Run both on every combination of argument map, parsers, default mode, and
/// request shape, and count the cases.
fn assert_thinking_parity(shapes: &[Value], maps: &[Option<HashMap<String, Value>>]) -> usize {
    let mut cases = 0;
    for shape in shapes {
        // One request per shape; each case only swaps its argument map.
        let mut request: NvCreateChatCompletionRequest =
            serde_json::from_value(shape.clone()).unwrap();
        for args in maps {
            for &reasoning_parser in REASONING_PARSERS {
                for &tool_call_parser in TOOL_CALL_PARSERS {
                    for &default_mode in DEFAULT_MODES {
                        request.chat_template_args = args.clone();
                        oracle_thinking(
                            &mut request,
                            reasoning_parser,
                            tool_call_parser,
                            default_mode,
                        );
                        let old = request.chat_template_args.take();
                        request.chat_template_args = args.clone();
                        moved_thinking(
                            &mut request,
                            reasoning_parser,
                            tool_call_parser,
                            default_mode,
                        );
                        let new = request.chat_template_args.take();
                        assert_eq!(
                            new, old,
                            "shape={shape} args={args:?} reasoning={reasoning_parser:?} \
                             tool={tool_call_parser:?} default={default_mode:?}"
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    cases
}

/// Every argument map, on the plain request and on one constrained request
/// with and without a top-level `thinking` field.
#[test]
fn thinking_rules_match_dynamo_on_every_argument_map() {
    let shapes: Vec<Value> = request_shapes()
        .into_iter()
        .filter(|shape| {
            shape
                .get("response_format")
                .is_none_or(|format| format["type"] == "json_schema")
                && shape.get("tools").is_none()
                && shape.get("guided_json").is_none()
                && shape.get("guided_regex").is_none()
                && shape.get("guided_choice").is_none()
                && shape.get("guided_grammar").is_none()
        })
        .collect();
    assert_eq!(shapes.len(), 4);
    let cases = assert_thinking_parity(&shapes, &argument_maps());
    eprintln!("thinking parity, every argument map: {cases} cases");
}

/// Every request shape, on a smaller set of argument maps.
#[test]
fn thinking_rules_match_dynamo_on_every_request_shape() {
    let maps: Vec<Option<HashMap<String, Value>>> = alias_values()
        .into_iter()
        .flat_map(|value| {
            ["thinking", "enable_thinking", "thinking_mode"].map(|key| {
                value
                    .clone()
                    .map(|value| HashMap::from([(key.to_string(), value)]))
            })
        })
        .chain([Some(HashMap::from([(
            "reasoning_effort".to_string(),
            json!("low"),
        )]))])
        .collect();
    let cases = assert_thinking_parity(&request_shapes(), &maps);
    eprintln!("thinking parity, every request shape: {cases} cases");
}

#[test]
fn constrained_generation_matches_dynamo() {
    for shape in request_shapes() {
        let request: NvCreateChatCompletionRequest = serde_json::from_value(shape.clone()).unwrap();
        assert_eq!(
            has_minimax_m3_constrained_generation(&request, has_explicit_guided_decoding(&request)),
            OpenAIPreprocessor::has_minimax_m3_constrained_generation(&request),
            "{shape}"
        );
    }
}

#[test]
fn reasoning_effort_rule_matches_dynamo() {
    let efforts = [
        None,
        Some(json!("none")),
        Some(json!("None")),
        Some(json!("low")),
        Some(json!("high")),
        Some(Value::Null),
        Some(json!(0)),
    ];
    let mut cases = 0;
    for family in [None, Some("gpt_oss"), Some("deepseek_v4"), Some("glm47")] {
        for effort in &efforts {
            let request: NvCreateChatCompletionRequest = serde_json::from_value(json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "chat_template_kwargs": effort.as_ref().map(|effort| json!({"reasoning_effort": effort}))
            }))
            .unwrap();
            let preprocessor = OpenAIPreprocessor {
                formatter: Arc::new(dynamo_renderer::NoOpFormatter),
                normalize_tool_call_args: false,
                selected_family: family,
            };
            let old = preprocessor
                .validate_model_request(&request)
                .map_err(|error| error.downcast::<InvalidArgument>().unwrap().0);
            let new =
                reject_unsupported_reasoning_effort(family, request.chat_template_args.as_ref())
                    .map_err(|error| error.to_string());
            assert_eq!(new, old, "family={family:?} effort={effort:?}");
            cases += 1;
        }
    }
    eprintln!("reasoning_effort parity: {cases} cases");
}

#[test]
fn tool_argument_rule_matches_dynamo() {
    // dynamo lib/llm/src/preprocessor.rs:2658-2660, with the format setting
    // as a bool.
    for parser in [None, Some("glm47"), Some("glm45"), Some("hermes")] {
        for json_object in [false, true] {
            let old = json_object || parser == Some("glm47");
            assert_eq!(
                should_normalize_tool_call_arguments(parser, json_object),
                old
            );
        }
    }
}

fn hf_formatter(template: &str) -> Arc<dyn OAIPromptFormatter> {
    let config: ChatTemplate = serde_json::from_value(json!({
        "chat_template": template,
        "bos_token": "<s>",
        "eos_token": "</s>"
    }))
    .unwrap();
    let PromptFormatter::OAI(formatter) =
        PromptFormatter::from_parts(config, ContextMixins::default(), false).unwrap();
    formatter
}

/// Templates that expose what the rules change: the last turn, its closing
/// tokens, a rewritten last turn, argument maps in key order, and the whole
/// message JSON.
fn formatters() -> Vec<(&'static str, Arc<dyn OAIPromptFormatter>)> {
    vec![
        (
            "llama",
            hf_formatter(
                "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim %}{% if loop.first %}{% set content = bos_token + content %}{% endif %}{% if not loop.last %}{% set content = content + '<|eot_id|>'%}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}",
            ),
        ),
        (
            "qwen",
            hf_formatter(
                "{%- for message in messages -%}{{ '<|im_start|>' + message.role + '\n' }}{%- if message.content is string -%}{{ message.content }}{%- else -%}{%- for part in message.content -%}{{ part.text }}{%- endfor -%}{%- endif -%}{{ '<|im_end|>\n' }}{%- endfor -%}{%- if add_generation_prompt -%}{{ '<|im_start|>assistant\n' }}{%- endif -%}",
            ),
        ),
        (
            "upper-last",
            hf_formatter(
                "{%- for message in messages -%}{%- if loop.last -%}{{ message.content | upper }}|{%- else -%}{{ message.content }}|{%- endif -%}{%- endfor -%}closed",
            ),
        ),
        (
            "glm-arguments",
            hf_formatter(
                "{%- for m in messages -%}{{ m.role }}:{%- for c in m.tool_calls or [] -%}{{ c.function.name }}({%- if c.function.arguments is mapping -%}{%- for k, v in c.function.arguments.items() -%}{{ k }}={{ v }};{%- endfor -%}{%- else -%}{{ c.function.arguments }}{%- endif -%}){%- endfor -%}{{ m.content }}|{%- endfor -%}{%- if add_generation_prompt -%}<gen>{%- endif -%}",
            ),
        ),
        (
            "tojson",
            hf_formatter("{{ messages | tojson }}{% if add_generation_prompt %}<gen>{% endif %}"),
        ),
        ("no-op", Arc::new(dynamo_renderer::NoOpFormatter)),
        (
            "deepseek-v4.1",
            Arc::new(dynamo_renderer::deepseek::v41::DeepSeekV41Formatter),
        ),
        (
            "kimi-k3",
            Arc::new(dynamo_renderer::kimi_k3::KimiK3Formatter::new(false)),
        ),
    ]
}

fn message_sets() -> Vec<Value> {
    let image = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    let call = |arguments: &str| {
        json!({"role": "assistant", "content": "prefix", "tool_calls": [{
            "id": "c1", "type": "function",
            "function": {"name": "lookup", "arguments": arguments}
        }]})
    };
    vec![
        json!([{"role": "user", "content": "hi"}]),
        json!([
            {"role": "user", "content": "Continue this sentence"},
            {"role": "assistant", "content": "LLM-Native Interaction"}
        ]),
        json!([{"role": "user", "content": "q"}, call(r#"{"zeta": 1, "alpha": "x", "mid": [1, 2]}"#)]),
        json!([{"role": "user", "content": "q"}, call(r#"{"id": 123456789012345678901234567890, "f": 1.5}"#)]),
        json!([{"role": "user", "content": "q"}, call("")]),
        json!([{"role": "user", "content": "q"}, call("not json{")]),
        json!([{"role": "user", "content": "q"}, call("[1, 2]")]),
        json!([
            {"role": "user", "content": "q"},
            call(r#"{"b": 2, "a": 1}"#),
            {"role": "tool", "tool_call_id": "c1", "content": "result"},
            {"role": "assistant", "content": "The answer is"}
        ]),
        json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "last"},
                {"type": "text", "text": ""}
            ]}
        ]),
        json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]),
        json!([{"role": "user", "content": "hello"}, {"role": "assistant", "content": "   "}]),
        json!([{"role": "developer", "content": "Be brief."}]),
        json!([
            {"role": "user", "content": "same"},
            {"role": "user", "content": "previous"},
            {"role": "assistant", "content": "same"}
        ]),
        json!([
            {"role":"assistant","tool_calls":[
                {"id":"a","type":"function","function":{"name":"image","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"image","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"b","content":[
                {"type":"image_url","image_url":{"url":image}}, {"type":"text","text":"Result B"}
            ]},
            {"role":"tool","tool_call_id":"a","content":[
                {"type":"image_url","image_url":{"url":image}}, {"type":"text","text":"Result A"}
            ]}
        ]),
        json!([]),
    ]
}

/// A render result in a comparable form: the prompt with its segments, or the
/// client error message. A dynamo error that is not a 400 is marked as such.
#[derive(Debug, PartialEq)]
enum Rendered {
    None,
    Prompt(String, Option<Vec<RenderedSegment>>),
    InvalidRequest(String),
    ServerError(String),
}

fn rendered(prompt: Option<RenderedPrompt>) -> Rendered {
    match prompt {
        None => Rendered::None,
        Some(prompt) => Rendered::Prompt(
            prompt.as_str().to_string(),
            prompt.segments().map(<[RenderedSegment]>::to_vec),
        ),
    }
}

#[test]
fn rendering_matches_dynamo() {
    let mut cases = 0;
    let mut errors = 0;
    for (name, formatter) in formatters() {
        for messages in message_sets() {
            for add_generation_prompt in [None, Some(false), Some(true)] {
                for continue_final_message in [None, Some(false), Some(true)] {
                    for use_raw_prompt in [None, Some(true)] {
                        for normalize_tool_call_args in [false, true] {
                            let request: NvCreateChatCompletionRequest =
                                serde_json::from_value(json!({
                                    "model": "m",
                                    "messages": messages,
                                    "add_generation_prompt": add_generation_prompt,
                                    "continue_final_message": continue_final_message,
                                    "nvext": {"use_raw_prompt": use_raw_prompt}
                                }))
                                .unwrap();
                            let preprocessor = OpenAIPreprocessor {
                                formatter: formatter.clone(),
                                normalize_tool_call_args,
                                selected_family: None,
                            };
                            let old = match preprocessor.apply_template(&request) {
                                Ok(prompt) => rendered(prompt),
                                Err(error) => match error.downcast::<InvalidArgument>() {
                                    Ok(error) => Rendered::InvalidRequest(error.0),
                                    Err(error) => Rendered::ServerError(format!("{error:#}")),
                                },
                            };
                            let options = ChatPromptOptions {
                                normalize_tool_call_args,
                                continue_final_message: request.get_continue_final_message()
                                    == Some(true),
                                use_raw_prompt: request
                                    .nvext()
                                    .is_some_and(|ext| ext.use_raw_prompt.unwrap_or(false)),
                            };
                            let new = match render_chat_prompt(
                                formatter.as_ref(),
                                &request,
                                options,
                                || request.raw_prompt(),
                            ) {
                                Ok(prompt) => rendered(prompt),
                                Err(error) => Rendered::InvalidRequest(error.to_string()),
                            };
                            assert_eq!(
                                new, old,
                                "formatter={name} messages={messages} options={options:?} \
                                 add_generation_prompt={add_generation_prompt:?}"
                            );
                            if matches!(old, Rendered::InvalidRequest(_)) {
                                errors += 1;
                            }
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    eprintln!("render parity: {cases} cases, {errors} of them client errors");
}

/// A raw prompt replaces the template, so the messages are never read, as
/// in Dynamo. Normalization must not copy them first.
#[test]
fn raw_prompt_skips_message_materialization() {
    use dynamo_renderer::{OAIChatLikeRequest, TextInput};
    use std::cell::Cell;

    struct RawRequest {
        inner: NvCreateChatCompletionRequest,
        message_reads: Cell<usize>,
    }

    impl OAIChatLikeRequest for RawRequest {
        fn model(&self) -> String {
            self.inner.model()
        }

        fn messages(&self) -> minijinja::Value {
            self.message_reads.set(self.message_reads.get() + 1);
            self.inner.messages()
        }

        fn typed_messages(
            &self,
        ) -> Option<&[dynamo_protocols::types::ChatCompletionRequestMessage]> {
            self.message_reads.set(self.message_reads.get() + 1);
            self.inner.typed_messages()
        }

        fn should_add_generation_prompt(&self) -> bool {
            self.inner.should_add_generation_prompt()
        }

        fn extract_text(&self) -> Option<TextInput> {
            self.inner.extract_text()
        }
    }

    impl NvExtProvider for RawRequest {
        fn nvext(&self) -> Option<&oracle::NvExt> {
            self.inner.nvext()
        }

        fn raw_prompt(&self) -> Option<String> {
            Some("raw text".to_string())
        }
    }

    impl CommonExtProvider for RawRequest {
        fn get_continue_final_message(&self) -> Option<bool> {
            None
        }
    }

    let request = RawRequest {
        inner: serde_json::from_value(json!({
            "model": "m",
            "messages": [{"role": "user", "content": "x".repeat(4 * 1024 * 1024)}],
            "nvext": {"use_raw_prompt": true}
        }))
        .unwrap(),
        message_reads: Cell::new(0),
    };
    let preprocessor = OpenAIPreprocessor {
        formatter: Arc::new(dynamo_renderer::NoOpFormatter),
        normalize_tool_call_args: true,
        selected_family: None,
    };
    let old = preprocessor.apply_template(&request).unwrap().unwrap();
    assert_eq!(old.as_str(), "raw text");
    assert_eq!(
        request.message_reads.get(),
        0,
        "Dynamo bypasses the messages"
    );

    let new = render_chat_prompt(
        preprocessor.formatter.as_ref(),
        &request,
        ChatPromptOptions {
            normalize_tool_call_args: true,
            use_raw_prompt: true,
            continue_final_message: false,
        },
        || request.raw_prompt(),
    )
    .unwrap()
    .unwrap();
    assert_eq!(new.as_str(), old.as_str());
    assert_eq!(
        request.message_reads.get(),
        0,
        "raw prompts must bypass message materialization"
    );
}

/// Top-level `thinking` values: the three modes in both spellings, and the
/// shapes the rule refuses.
fn thinking_values() -> Vec<Option<Value>> {
    vec![
        None,
        Some(json!(true)),
        Some(json!(false)),
        Some(json!({"type": "enabled"})),
        Some(json!({"type": "disabled"})),
        Some(json!({"type": "adaptive"})),
        Some(json!({"type": "enabled", "budget_tokens": 1024})),
        Some(json!({"type": "sometimes"})),
        Some(json!({})),
        Some(json!("on")),
        Some(Value::Null),
    ]
}

const REASONING_EFFORTS: &[Option<&str>] = &[
    None,
    Some("none"),
    Some("minimal"),
    Some("low"),
    Some("medium"),
    Some("high"),
    Some("xhigh"),
    Some("max"),
];

/// Argument maps over the thinking keys and a nested `reasoning_effort`,
/// smaller than [`argument_maps`]: this rule reads each key's value.
fn reasoning_argument_maps() -> Vec<Option<HashMap<String, Value>>> {
    let values = [
        None,
        Some(json!(true)),
        Some(json!(false)),
        Some(json!("Enabled")),
        Some(json!("disabled")),
        Some(json!("ADAPTIVE")),
        Some(Value::Null),
    ];
    let efforts = [
        None,
        Some(json!("low")),
        Some(json!("none")),
        Some(Value::Null),
        Some(json!(7)),
    ];
    let mut maps = vec![None, Some(HashMap::new())];
    for thinking in &values {
        for thinking_mode in &values {
            for effort in &efforts {
                let mut map = HashMap::new();
                for (key, value) in [
                    ("thinking", thinking),
                    ("thinking_mode", thinking_mode),
                    ("reasoning_effort", effort),
                ] {
                    if let Some(value) = value {
                        map.insert(key.to_string(), value.clone());
                    }
                }
                map.insert("custom".to_string(), json!(1));
                maps.push(Some(map));
            }
        }
    }
    maps
}

fn chat_request(
    thinking: &Option<Value>,
    effort: Option<&str>,
    args: &Option<HashMap<String, Value>>,
) -> NvCreateChatCompletionRequest {
    let mut body = json!({"model": "m", "messages": [{"role": "user", "content": "hi"}]});
    if let Some(effort) = effort {
        body["reasoning_effort"] = json!(effort);
    }
    let mut request: NvCreateChatCompletionRequest =
        serde_json::from_value(body).expect("a chat request");
    request.thinking = thinking.clone();
    request.chat_template_args = args.clone();
    request
}

/// The Chat Completions and Responses handlers' rule, on every combination
/// of `thinking`, `reasoning_effort`, and template arguments: the same
/// arguments, or the same error.
#[test]
fn reasoning_controls_match_dynamo() {
    let mut cases = 0;
    for thinking in thinking_values() {
        for effort in REASONING_EFFORTS {
            for args in reasoning_argument_maps() {
                let mut stock = chat_request(&thinking, *effort, &args);
                let stock_result = stock.normalize_reasoning_template_args();
                let request = chat_request(&thinking, *effort, &args);
                let mut moved = request.chat_template_args.clone();
                let moved_result = dynamo_renderer::request_rules::apply_handler_rules(
                    RequestApi::ChatCompletions,
                    &request.inner,
                    request.thinking.as_ref(),
                    true,
                    false,
                    &mut moved,
                );
                match (&stock_result, &moved_result) {
                    (Ok(()), Ok(())) => {
                        assert_eq!(
                            moved, stock.chat_template_args,
                            "thinking {thinking:?}, effort {effort:?}, args {args:?}"
                        );
                        assert!(stock.thinking.is_none(), "the rule drops `thinking`");
                    }
                    (Err(stock), Err(moved)) => assert_eq!(stock.to_string(), moved.to_string()),
                    _ => panic!(
                        "thinking {thinking:?}, effort {effort:?}, args {args:?}: \
                         stock {stock_result:?}, moved {moved_result:?}"
                    ),
                }
                cases += 1;
            }
        }
    }
    assert!(cases > 3000, "control: the matrix is not empty ({cases})");
}

/// Responses runs the same rule as Chat Completions, and Messages does not.
#[test]
fn only_chat_and_responses_fold_reasoning_controls() {
    let request = chat_request(&Some(json!({"type": "disabled"})), Some("high"), &None);
    for (api, folded) in [
        (RequestApi::ChatCompletions, true),
        (RequestApi::Responses, true),
        (RequestApi::Messages, false),
    ] {
        let mut args = None;
        dynamo_renderer::request_rules::apply_handler_rules(
            api,
            &request.inner,
            request.thinking.as_ref(),
            false,
            false,
            &mut args,
        )
        .expect("valid");
        assert_eq!(args.is_some(), folded, "{api:?}");
    }
}

/// The Messages handler's thinking default, after the Messages conversion
/// (`lib/llm/src/protocols/anthropic/types.rs:208-219`: `enabled` and
/// `disabled` set `enable_thinking`, `adaptive` sets nothing).
#[test]
fn messages_thinking_default_matches_dynamo() {
    let conversions: [(Option<&str>, Option<bool>); 4] = [
        (None, None),
        (Some("enabled"), Some(true)),
        (Some("disabled"), Some(false)),
        (Some("adaptive"), None),
    ];
    let mut cases = 0;
    for reasoning_parser in REASONING_PARSERS {
        for (thinking_type, enable) in conversions {
            for extra in [None, Some(("truncate_history_thinking", json!(true)))] {
                let mut args: Option<HashMap<String, Value>> = enable
                    .map(|enable| HashMap::from([("enable_thinking".to_string(), json!(enable))]));
                if let Some((key, value)) = &extra {
                    args.get_or_insert_default()
                        .insert(key.to_string(), value.clone());
                }
                let disabled = thinking_type == Some("disabled");
                let mut stock = chat_request(&None, None, &args);
                oracle::messages_thinking_default(*reasoning_parser, disabled, &mut stock);
                let request = chat_request(&None, None, &args);
                let mut moved = args.clone();
                dynamo_renderer::request_rules::apply_handler_rules(
                    RequestApi::Messages,
                    &request.inner,
                    None,
                    reasoning_parser.is_some(),
                    disabled,
                    &mut moved,
                )
                .expect("Messages never fails");
                assert_eq!(
                    moved, stock.chat_template_args,
                    "parser {reasoning_parser:?}, thinking {thinking_type:?}, extra {extra:?}"
                );
                cases += 1;
            }
        }
    }
    assert_eq!(cases, REASONING_PARSERS.len() * 8);
}
