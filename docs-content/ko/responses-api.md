# OpenAI Responses API 부분 지원 (`/v1/responses`)

`mlxcel serve`와 `mlxcel-server`는 OpenAI Responses API의 Phase-1 부분집합을 제공합니다.
`base_url`이 mlxcel 서버를 가리키도록 설정하면 OpenAI Python SDK로 기본적인
`client.responses.create(...)` 및 스트리밍 흐름을 사용할 수 있지만,
OpenAI Responses의 모든 기능을 완전히 구현한 것은 아닙니다.

구현 소스 맵:

| 모듈 | 역할 |
|--------|----------------|
| `src/server/types/responses_request.rs` | 요청 타입. |
| `src/server/types/responses_response.rs` | 응답 타입. |
| `src/server/types/responses_stream.rs` | SSE 이벤트 enum. |
| `src/server/responses_translator.rs` | Responses ↔ chat-completions 변환. |
| `src/server/responses_store.rs` | 인메모리 응답 저장소. |
| `src/server/conversation_store.rs` | 인메모리 대화 기록 저장소. |
| `src/server/routes/responses.rs` | 라우트 핸들러. |

## 구현된 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|-------------|
| POST | `/v1/responses` | 응답을 생성합니다(비스트리밍 또는 스트리밍). |
| GET | `/v1/responses/{id}` | 저장된 응답을 조회합니다. |
| DELETE | `/v1/responses/{id}` | 저장된 응답을 삭제합니다. |
| POST | `/v1/responses/{id}/cancel` | 최선 노력(best-effort) 방식의 취소 / 취소 표시. |

구현된 동일한 라우트에 대해 `/v1`이 없는 별칭도 마운트됩니다:
`/responses`, `/responses/{id}`, `/responses/{id}/cancel`.

다음 OpenAI 스타일 인터페이스는 이 구현에서 **마운트되지 않습니다**.

- `GET /v1/responses/{id}/input_items`
- `POST /v1/responses/compact`
- `POST /v1/responses/input_tokens`

## 빠른 시작

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="sk-local")

resp = client.responses.create(
    model="qwen3-0.6b-4bit",
    input="Reply with: hello",
    max_output_tokens=64,
)
print(resp.status)
print(resp.output_text)

with client.responses.stream(
    model="qwen3-0.6b-4bit",
    input="Count to 5.",
    max_output_tokens=64,
) as stream:
    for event in stream:
        print(event.type, getattr(event, "delta", ""))
    final = stream.get_final_response()
    print(final.usage)
```

## 지원하는 요청 필드

| 필드 | 상태 | 참고 |
|-------|--------|-------|
| `model` | 필수 | 서버가 허용하는, 로드된 모델의 별칭/경로와 일치해야 합니다. |
| `input` | 지원 | 문자열 또는 타입이 지정된 입력 항목 배열. |
| `instructions` | 지원 | system 스타일 메시지로 앞에 추가됩니다. `previous_response_id`를 통해 상속되지 않습니다. |
| `tools` | function 전용 | `{"type":"function", ...}`만 허용됩니다. |
| `tool_choice` | 지원 | `auto`, `none`, `required` 또는 이름이 지정된 함수. `required`와 이름 지정 형식은 단순히 허용되는 것이 아니라 강제됩니다. [Tool choice 강제](#tool-choice-enforcement)를 참조하십시오. |
| `parallel_tool_calls` | 허용 | 기존 tool-call 처리로 전달됩니다. |
| `text.format` | 부분집합 지원 | `text`와 `json_schema` 형태는 기존 구조화 출력 코드를 통해 처리됩니다. |
| `reasoning` | 부분집합 지원 | `reasoning.effort`는 변경 없이 그대로 반환되며 chat-completions `reasoning_effort`와 동일한 템플릿 제어에 반영됩니다. `none`, `off`, `disabled`, `false`, `0`(앞뒤 공백 제거 후 대소문자 구분 없음)은 `enable_thinking`을 비활성화하고, 그 외 값은 thinking을 활성화하며 `reasoning_effort`로 그대로 전달되거나, 로드된 템플릿이 읽는 식별자가 `reasoning_strength`이면 그것으로 전달됩니다. 도출된 제어는 서버 전체 템플릿 기본값을 키별로 재정의합니다. `summary`는 계속 모델/런타임에 따라 달라집니다. |
| `conversation` | 지원 | 문자열 id 또는 `{ "id": "..." }`. 인메모리 대화 저장소를 사용합니다. |
| `previous_response_id` | 지원 | 저장된 이전 입력/출력 항목을 복원합니다. `conversation`과 함께 사용할 수 없습니다. |
| `store` | 지원 | 기본값은 `true`이며, `false`이면 저장을 건너뜁니다. |
| `stream` | 지원 | 타입이 지정된 SSE 이벤트를 스트리밍합니다. |
| `max_output_tokens` | 지원 | 0보다 커야 합니다. |
| `max_tool_calls` | 지원 | 생성되는 function-call 항목 수의 소프트 상한. |
| `temperature`, `top_p`, `top_logprobs` | 부분집합 지원 | chat-completions 샘플링 필드로 매핑됩니다. |
| `metadata` | 지원 | 최대 16개 항목. |
| `prompt_cache_key` | 허용 | 프롬프트 캐시 처리 경로로 전달됩니다. |
| `user`, `safety_identifier` | 허용 | 둘 다 있으면 `user`가 사용되고, `safety_identifier`는 대체값으로 사용됩니다. |
| `background` | `true`이면 거부 | 비동기 폴링은 구현되어 있지 않습니다. |
| `truncation` | `disabled`만 가능 | `auto`를 포함한 다른 값은 400을 반환합니다. |
| `service_tier` | 허용 | 그대로 반환되거나 무시됩니다. 스케줄링 티어는 구현되어 있지 않습니다. |

## 입력 항목

Phase 1은 다음과 같은 타입 지정 항목을 지원합니다.

```jsonc
[
  {"type":"message", "role":"user", "content":"hello"},
  {"type":"message", "role":"system", "content":[{"type":"input_text", "text":"sys"}]},
  {"type":"message", "role":"user", "content":[{"type":"input_image", "image_url":"data:image/png;base64,...", "detail":"high"}]},
  {"type":"function_call", "call_id":"call_abc", "name":"f", "arguments":"{}"},
  {"type":"function_call_output", "call_id":"call_abc", "output":"ok"},
  {"type":"function_call_output", "call_id":"call_img", "output":[{"type":"input_text", "text":"captured"}, {"type":"input_image", "image_url":"data:image/png;base64,..."}]},
  {"type":"reasoning", "content":[{"type":"reasoning_text", "text":"..."}]}
]
```

`developer` 역할은 `system`과 동일하게 취급됩니다. Reasoning 입력 항목은 허용되어 전달됩니다. 텍스트 내용은 버퍼링되어 뒤따르는 assistant 턴의 병렬 `reasoning` 필드에 첨부됩니다. `message.get('reasoning')`을 렌더링하는 채팅 템플릿(예: Gemma 4)은 그 위치에서 이를 받습니다. `preserve_thinking` kwarg는 이 필드가 rolling-checkpoint 제거 과정에서 살아남는지 여부를 제어합니다. `false`(프롬프트 캐시가 켜져 있지 않은 경우의 기본값)이면 이전 턴의 reasoning을 인라인 `<think>` 블록과 함께 제거하고, `true`이면 유지합니다.

메시지 항목과 배열 값의 `function_call_output.output` 필드는 mlxcel의 기존 chat-completions 표기와 함께 Responses 고유의 part 표기도 허용합니다.

| Part | 필드 | 동작 |
|------|--------|----------|
| `input_text` | `text: string` | chat-completions `text` part로 매핑됩니다. |
| `input_image` | `image_url: string`, 선택적 `detail: "auto" \| "low" \| "high"` | `image_url` part로 매핑됩니다. URL과 data URI는 기존 미디어 파이프라인을 사용합니다. |
| `file_id`만 있는 `input_image` | `file_id: string` | 이 서버에는 업로드 저장소가 없으므로 400으로 거부됩니다. 대신 `image_url`을 제공하십시오. |
| `input_file` | 모든 Responses 파일 필드 | 이름이 명시된 400으로 거부됩니다. PDF 및 문서 수집은 구현되어 있지 않습니다. |
| `text`, `image_url`, `video_url`, `input_audio` | 기존 chat-completions 형태 | 변경 없이 허용됩니다. 실행 여부는 여전히 로드된 모델의 미디어 지원에 달려 있습니다. |

문자열 값의 function output은 변경되지 않습니다. 배열 값의 function output에서는 텍스트 part는 tool 메시지에서 줄바꿈으로 결합되고, 텍스트도 이미지도 아닌 part는 JSON 텍스트로 유지되며, 이미지 part는 순서대로 바로 뒤따르는 user 이미지 턴으로 내보내집니다. 이러한 후속 턴이 있으면 tool 메시지는 `[Image output attached in the next message]`로 끝납니다. user 메시지 항목의 이미지는 제자리에 유지되며, assistant, system, developer 메시지 항목의 이미지도 지원되는 채팅 템플릿이 이미지 플레이스홀더를 user 턴에서 렌더링하기 때문에 마찬가지로 바로 뒤따르는 user 턴으로 이동됩니다.

유효한 입력이 없는 요청은 모델 디스패치 이전에 400으로 거부됩니다.
빈 `input` 배열, 공백뿐인 문자열 또는 `text` part, 그리고 변환된 대화 전체(`previous_response_id` / `conversation`을 통해 가져온 이력 포함)에
이미지/비디오/오디오, tool call, reasoning 내용이 전혀 없는 경우 모두 동일한
`invalid_request_error`가 발생합니다. 비어 있는 `input`이라도 비어 있지 않은
`instructions`와 함께 오면 통과하는데, `instructions`가 렌더링된 대화에서 실제 system
메시지가 되기 때문입니다. 동일한 검사가 `/v1/chat/completions`에도 적용됩니다. `/v1/completions`(raw 프롬프트 레거시 엔드포인트)는
의도적인 예외입니다. 공백뿐인 프롬프트는 동일한 400
`invalid_request_error`로 거부하지만, 완전히 빈 프롬프트는 통과시킵니다.
BOS에서의 무조건 생성은 base 모델의 정당한 사용 사례이며, 이 라우트에는 비어 있을 수 있는 chat-template 골격이 없기 때문입니다(issue #806).

## 응답 형태

응답은 OpenAI와 유사한 객체 형태를 사용합니다.

```jsonc
{
  "id": "resp_...",
  "object": "response",
  "created_at": 1234.0,
  "completed_at": 1235.0,
  "status": "completed",
  "model": "...",
  "output": [
    {"type":"reasoning", "id":"rs_...", "status":"completed", "content":[...]},
    {"type":"function_call", "id":"fc_...", "call_id":"call_...", "name":"...", "arguments":"{}", "status":"completed"},
    {"type":"message", "id":"msg_...", "role":"assistant", "status":"completed", "content":[...]}
  ],
  "output_text": "...",
  "usage": {
    "input_tokens": 12,
    "output_tokens": 34,
    "total_tokens": 46,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}
  }
}
```

여러 요청 필드는 존재할 경우 그대로 반환됩니다. 이는 호환성을 위한 표면일 뿐이며,
반환되는 모든 필드가 런타임 동작을 바꾼다는 증거로 받아들여서는 안 됩니다.

## mlxcel 확장 필드

mlxcel은 응답 본문에 OpenAI에 없는 소수의 필드를 추가합니다. 각 필드는 선택 사항이며 로드된 모델이 생성하지 않으면 생략됩니다(`skip_serializing_if`). 이 관례는 Chat Completions의 `reasoning_content`(vLLM을 따름)에서 시작되었으며, 이제 OpenRouter 스타일 클라이언트를 위해 기본적으로 동일한 `reasoning` 별칭도 함께 제공합니다. 이 별칭은 Chat Completions에 한정되며 `--reasoning-alias-field none` 또는 `MLXCEL_REASONING_ALIAS_FIELD=none`으로 비활성화할 수 있습니다. Responses API의 reasoning은 계속 고유의 `response.reasoning_text.delta` 이벤트를 사용하며 변경되지 않습니다. 이 확장 관례에는 다음도 포함됩니다.

- 비스트리밍 `POST /v1/chat/completions`의 `choices[0].message.florence2_result`
  (issue #1073): Florence-2 작업 응답의 구조화된 형태로, 로드된 모델이 Florence-2일 때만 존재합니다.
  `message.content`에는 CLI가 출력하는 것과 같은 사람이 읽을 수 있는 텍스트 응답이 담기고,
  이 필드에는 파싱된 좌표가 JSON으로 담겨 클라이언트가 포맷된 문자열을 다시 파싱할 필요가 없습니다.
  객체는 `{"task": "<OD>", "kind": "bboxes" | "text" | "quad_boxes" | "polygons" |
  "bboxes_or_polygons", ...}` 형태이며, 좌표 배열은 업스트림
  mlx-vlm / HuggingFace의 키 이름(`bboxes`, `quad_boxes`, `polygons`,
  `labels`, `bboxes_labels`, `polygons_labels`) 아래에 있고, 좌표는 원본 이미지 범위의 픽셀 단위입니다.
  스트리밍 채팅 응답과 `/v1/responses`는 렌더링된 텍스트만 반환하며,
  구조화된 필드는 비스트리밍 chat-completions 전용입니다.

## 스트리밍 이벤트

SSE 프레임은 타입이 지정되어 있으며 응답별로 단조 증가하는 `sequence_number`를 포함합니다.
Phase 1은 다음과 같은 이벤트를 내보냅니다.

- `response.created`
- `response.in_progress`
- `response.output_item.added`
- `response.content_part.added`
- `response.output_text.delta`
- `response.output_text.done`
- `response.content_part.done`
- `response.output_item.done`
- `response.function_call_arguments.delta`
- `response.function_call_arguments.done`
- `response.reasoning_text.delta`
- `response.reasoning_text.done`
- `response.completed`
- 오류 경로의 실패/미완료/오류 이벤트

Reasoning은 message 항목보다 앞선 `reasoning` 출력 항목 안에서 `response.reasoning_text.delta` 이벤트로 스트리밍되며, `output_text`로는 절대 전송되지 않습니다. 이는 채팅 템플릿이 열린 thinking 블록을 미리 시작하는 경우(예: `<think>\n`으로 끝나는 generation 프롬프트)에도 마찬가지입니다. 이 경우 모델은 닫는 마커만 작성하며, 비스트리밍 분리와 마찬가지로 스트림은 reasoning 항목 안에서 시작됩니다.

## 응답 및 대화 저장소

저장소는 인메모리이며 항목 수, 대략적인 보존 JSON 바이트 수, TTL로 제한됩니다.

| 플래그 | 기본값 | 환경 변수 | 참고 |
|------|---------|---------|-------|
| `--responses-store-max-entries` | `1024` | `LLAMA_ARG_RESPONSES_STORE_MAX_ENTRIES` | `0`이면 응답 저장을 비활성화합니다. |
| `--responses-store-max-bytes` | `268435456` | `MLXCEL_RESPONSES_STORE_MAX_BYTES` | 대략적인 보존 바이트 예산. `0`이면 라우트는 활성 상태로 유지되지만 저장된 응답을 즉시 축출합니다. |
| `--responses-store-ttl-secs` | `3600` | `LLAMA_ARG_RESPONSES_STORE_TTL_SECS` | `0`이면 TTL을 비활성화합니다. |
| `--conversation-store-max-entries` | `256` | `LLAMA_ARG_CONVERSATION_STORE_MAX_ENTRIES` | `0`이면 대화를 비활성화합니다. |
| `--conversation-store-max-bytes` | `67108864` | `MLXCEL_CONVERSATION_STORE_MAX_BYTES` | 대략적인 보존 바이트 예산. `0`이면 라우트는 활성 상태로 유지되지만 기록을 즉시 축출합니다. |
| `--conversation-store-ttl-secs` | `3600` | `LLAMA_ARG_CONVERSATION_STORE_TTL_SECS` | `0`이면 TTL을 비활성화합니다. |

응답 저장이 비활성화되면 id 기반 조회/삭제/취소와
`previous_response_id` 체이닝은 오류를 반환합니다. 대화 저장이
비활성화되면 `conversation`을 사용하는 요청은 오류를 반환합니다.

## 체이닝 의미론

- `previous_response_id`는 저장된 응답의 입력 항목과 출력 항목을 이전 대화 이력으로 로드한 다음, 새 입력을 추가합니다.
- `conversation`은 id로 인메모리 기록을 로드하고 여기에 추가합니다.
- 두 필드는 함께 사용할 수 없습니다.
- 참조된 이전 응답의 `instructions`는 이어받지 않습니다.

## Tool choice 강제

`tool_choice`는 chat-completions 파이프라인으로 변경 없이 전달되므로,
네 가지 모드는 `/v1/chat/completions`에서와 정확히 동일하게 동작합니다.

| `tool_choice` | 렌더링되는 도구 | 프롬프트 지시문 | 제약 | 다음 경우 400 |
|---|---|---|---|---|
| 생략 / `"auto"` | 전체 | 없음 | 없음 | 해당 없음 |
| `"none"` | 없음 | 없음 | 없음 | 해당 없음 |
| `"required"` | 전체 | "You must call one or more of the available functions ..." | 문법 지원 형식에서 호출 강제 | `tools`가 없거나 비어 있음 |
| `{"type":"function","function":{"name":"f"}}` | `f`만 | "You must call the 'f' function ..." | 문법 지원 형식에서 `f` 호출 강제 | `f`가 선언되지 않음, `type`이 `function`이 아님, 또는 이름이 비어 있음 |

지시문은 첫 번째 system 메시지에 추가되며(`instructions` 필드도 포함되는데, 이 필드는 선두 system 메시지가 됩니다),
없으면 마지막 user 메시지에 추가되고, 그것도 없으면 선두의 새 system 메시지로 삽입됩니다.
저장된 요청과 반환되는 필드는 수정되지 않습니다.

문법 지원 형식이란, 출력되는 호출이 로드된 채팅 템플릿에서 읽을 수 있는 고정 래퍼 안의 JSON 객체인 형식을 말합니다:
Hermes / Qwen(`<tool_call>`), Mistral Nemo(`[TOOL_CALLS]`), Llama 3(`<|python_tag|>`
또는 `"parameters":`). 이 경우 호출은 도구 스키마로 만든 구조화 출력 문법을 통해 강제되며,
응답에는 `finish_reason: "tool_calls"`와 함께 선언된 함수에 대한 호출(`function_call` 출력 항목)이 담깁니다.
JSON 와이어 형식이 없는 템플릿(ATEM, Gemma 4, Qwen3-Coder 및 GLM 같은 XML 방언, Kimi K2, pythonic)은
지시문과 좁혀진 도구 목록만 받으며, 강제된 선택이 호출 없이 끝나면 `warn` 수준으로 기록됩니다.
문법 엔진이 표현할 수 없는 도구 스키마는 요청을 실패시키는 대신 동일한 지시문 전용 경로로 폴백합니다.

강제된 `tool_choice`는 `json_schema` 타입의 `text.format`과 함께 사용할 수 없습니다.
두 제약이 각각 생성 전체를 차지하려 하기 때문에 요청은 400을 반환합니다.

## 지원되지 않는 도구 타입

function 도구만 허용됩니다. `web_search`, `file_search`, `computer_use_preview`, `code_interpreter`,
`image_generation`, `mcp`, `custom`, `apply_patch`, `function_shell` 같은 내장/외부 도구 타입은
400 응답을 반환합니다. `mlxcel`은 Responses API에서 외부 도구를 실행하지 않습니다.

## OpenAI 전체 API와의 차이점

주요 공백:

- 백그라운드 작업 모드 없음;
- input-items 페이지네이션 엔드포인트 없음;
- 서버 측 compaction 엔드포인트 없음;
- 토큰 수 계산 엔드포인트 없음;
- 내장 도구 또는 MCP 커넥터 실행 없음;
- 디스크에 영속화되는 응답 저장소 없음;
- 업로드 저장소를 통한 `input_file` 수집 또는 `file_id` 해석 없음.
