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
| `src/server/responses_compaction.rs` | Compaction capsule, replay, 선택, 요약 핵심 로직. |
| `src/server/routes/responses_compact.rs` | `POST /v1/responses/compact` 핸들러. |


## 구현된 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|-------------|
| POST | `/v1/responses` | 응답을 생성합니다(비스트리밍 또는 스트리밍). |
| GET | `/v1/responses/{id}` | 저장된 응답을 조회합니다. |
| DELETE | `/v1/responses/{id}` | 저장된 응답을 삭제합니다. |
| POST | `/v1/responses/{id}/cancel` | 진행 중인 스트리밍 응답에 대한 최선 노력(best-effort) 방식의 취소. 저장된(완료된) 응답은 저장된 결과를 바꾸지 않고 `400 invalid_request_error`를 반환합니다. |
| POST | `/v1/responses/compact` | 오래된 턴을 `compaction` 항목으로 요약합니다. [대화 compaction](#대화-compaction)을 참조하십시오. |
| POST | `/v1/responses/input_tokens` | 본문이 prefill할 프롬프트 토큰 수를 집계합니다. `docs/llama-server-compat.md`를 참조하십시오. |

구현된 동일한 라우트에 대해 `/v1`이 없는 별칭도 마운트됩니다:
`/responses`, `/responses/{id}`, `/responses/{id}/cancel`,
`/responses/compact`, `/responses/input_tokens`.

다음 OpenAI 스타일 인터페이스는 이 구현에서 **마운트되지 않습니다**.

- `GET /v1/responses/{id}/input_items`


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
| `tool_choice` | 지원 | `auto`, `none`, `required` 또는 이름이 지정된 함수. `required`와 이름 지정 형식은 단순히 허용되는 것이 아니라 강제됩니다. [Tool choice 강제](#tool-choice-강제)를 참조하십시오. |
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
| `context_management` | 부분집합 지원 | `[{"type":"compaction","compact_threshold":N}]`은 자동 compaction을 활성화합니다. [자동 compaction](#자동-compaction)을 참조하십시오. `[]` 또는 `null`이면 비활성화합니다. |


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
  {"type":"reasoning", "content":[{"type":"reasoning_text", "text":"..."}]},
  {"type":"compaction", "id":"cmp_...", "encrypted_content":"..."}
]
```

`compaction` 항목은 `POST /v1/responses/compact`가 반환하는 capsule입니다. replay 방식은 [대화 compaction](#대화-compaction)을 참조하십시오.



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
BOS에서의 무조건 생성은 base 모델의 정당한 사용 사례이며, 빈 입력을 둘러쌀 chat-template 골격이 없는 라우트이기 때문입니다(issue #806).

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
- `mlx.compaction.started`, `mlx.compaction.completed` (MLX 확장. 자동 compaction 이후에만 발생. [자동 compaction](#자동-compaction) 참조)
- 오류 경로의 실패/미완료/오류 이벤트


Reasoning은 message 항목보다 앞선 `reasoning` 출력 항목 안에서 `response.reasoning_text.delta` 이벤트로 스트리밍되며, `output_text`로는 절대 전송되지 않습니다. 이는 채팅 템플릿이 열린 thinking 블록을 미리 시작하는 경우(예: `<think>\n`으로 끝나는 generation 프롬프트)에도 마찬가지입니다. 이 경우 모델은 닫는 마커만 작성하며, 비스트리밍 분리와 마찬가지로 스트림은 reasoning 항목 안에서 시작됩니다.

Harmony(GPT-OSS) 응답은 tool call 유무와 관계없이 `analysis` 채널을 동일한 `reasoning` 출력 항목으로 드러내며, `usage.output_tokens_details.reasoning_tokens`는 그 reasoning 텍스트의 토큰 수로 스트리밍과 비스트리밍 요청에서 동일합니다. 여러 analysis 또는 `<think>` 구간은 두 응답 모드 모두에서 같은 스트리밍 필터를 통해 누적되며, 파서가 추가하는 구분자나 누락된 뒤쪽 구간이 없습니다. reasoning이 없는 응답에서는 0입니다.

Function call은 `/v1/chat/completions`가 점진적으로 스트리밍하는 형식(Hermes / Qwen JSON, Mistral `[ARGS]` 및 JSON 리스트, Llama 3 JSON, Qwen3-Coder XML. `docs/server-features.md` 참조)에 대해 생성되는 동안 스트리밍됩니다. 호출의 이름이 완성되면 열려 있는 reasoning 또는 message 항목이 닫히고 `status: "in_progress"`와 빈 `arguments`를 가진 `function_call` 항목이 추가되며, 이후 `response.function_call_arguments.delta` 이벤트가 인자 조각을 실어 나릅니다. 생성이 끝나면 `response.function_call_arguments.done`과 `response.output_item.done`이 신뢰할 수 있는 파싱된 인자를 담고, 호출은 완료된 응답에서도 스트리밍된 `call_id`를 유지합니다. tool call을 파싱하는 스트리밍 요청에서 `response.completed`(그리고 `previous_response_id`를 위해 저장되는 사본)는 스트림이 보낸 항목들로 만들어집니다. 같은 항목 id와 `output_index` 순서, 각 message 구간은 자신만의 `message` 항목(호출 앞뒤의 산문은 두 항목), function call은 스트리밍된 `fc_` id, `call_id`, 파싱된 인자를 갖고, `output_text`는 message 구간들을 이어 붙인 것입니다. 스트리머가 조각 처리 중 panic하면 이를 격리하고 나머지 인자는 최종 파싱에서 전달됩니다. 비스트리밍 응답은 변하지 않습니다. Hermes, Granite, Mistral, Llama 3에서 객체 `arguments`는 재직렬화한 것이 아니라 모델이 출력한 원본 JSON 텍스트입니다. 최종 파싱이 확인하지 못한 스트리밍 호출은 응답 id와 함께 `warn` 수준으로 기록되고 `incomplete`로 닫히며, 이후 파싱된 호출은 별도 항목으로 내보내집니다. 다른 형식은 이전처럼 생성 이후 호출마다 `output_item.added` 하나, 인자 delta 하나, `.done`, `output_item.done`을 내보냅니다.


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
- 결합된 이력(저장된 입력, 저장된 출력, 또는 새 입력) 어디에든 있는 `compaction` 항목은 이력이 조립된 뒤 확장됩니다. 이력에 capsule이 들어 있던 응답은 자신의 instructions, 그 capsule, 그 이후의 모든 것을 입력으로 저장하므로 이후의 `previous_response_id` 연결도 요약을 계속 replay합니다. 자동으로 compaction된 응답은 자신의 instructions를 입력으로, capsule을 첫 번째 출력 항목으로 저장하며, 이는 같은 방식으로 replay됩니다.


## 대화 compaction

`POST /v1/responses/compact`(별칭 `/responses/compact`)는 오래된 턴을 모델이 작성한 요약으로 대체하여 긴 agent 세션이 다시 컨텍스트 윈도우에 들어가도록 합니다. 비스트리밍이며 아무것도 저장하지 않습니다.

요청 본문: `model`, `input`(문자열 또는 항목 배열), 그리고 선택적으로 `instructions`, `tools`, `max_output_tokens`(요약 예산, 기본값 1024, 1에서 16384), `keep_tokens`(그대로 유지할 턴의 예산, 기본값은 슬롯별 컨텍스트 윈도우의 1/16, 최대 8192). 샘플링 필드는 `/v1/responses`에서와 같이 허용됩니다. `stream: true`, `previous_response_id`, `conversation`은 400을 반환합니다. 전체 이력을 `input`에 보내십시오.

```jsonc
{
  "id": "resp_...",
  "object": "response.compaction",
  "created_at": 1760000000,
  "model": "...",
  "output": [
    {"type":"message", "role":"system", "content":"..."},
    {"type":"message", "role":"user", "content":"latest question"},
    {"type":"compaction", "id":"cmp_...", "encrypted_content":"eyJ2IjoxLC..."}
  ],
  "usage": {"input_tokens": 1830, "output_tokens": 212, "total_tokens": 2042, ...}
}
```

`output`은 system 및 developer 메시지, 유지된 턴, 그리고 정확히 하나의 `compaction` 항목을 담습니다. 버전 1은 유지된 턴도 capsule 안에 저장하므로, 반환되는 본문이 원래 미디어 payload보다 커질 수 있습니다. 서버는 replay 요청 크기 제한에 맞지 않는 출력을 거부하므로 새 항목을 위한 여유를 남겨 두십시오. `usage`는 요약 생성 한 번을 집계합니다. 세션을 이어 가려면 `[...output, <new items>]`을 `/v1/responses`에 보내십시오. 요청 수준의 `instructions`는 compaction pass에 적용되므로, 필요하면 연속 요청에 다시 보내거나 지속적인 instructions를 system/developer 입력 메시지에 넣으십시오. 유지된 턴이 `output`에도, capsule 안에도 나타나므로 그 본문은 이를 두 번 담습니다. 클라이언트는 system 및 developer 메시지와 `compaction` 항목, 그리고 새 항목만 보내면 더 작은 본문 크기로 같은 replay를 얻을 수 있습니다.

**선택.** system 및 developer 메시지는 항상 자신의 역할로 유지됩니다. 가장 최근 user 메시지는 항상 유지됩니다. `keep_tokens`에 들어가는 가장 최근의 온전한 턴들이 유지되며, 가장 최근 턴은 그것만으로 예산을 초과하더라도 유지됩니다. 예산은 렌더링된 유지 컨텍스트에서 측정되므로 system 및 developer 메시지, `instructions`, 템플릿 오버헤드도 포함합니다. 이것만으로 `keep_tokens`를 초과하면 이전 턴이 하나도 들어가지 않아 가장 최근 턴(그리고 최근 user 메시지)만 유지됩니다. 이력은 user 메시지 직전 또는 마지막으로 열려 있던 function call을 닫는 출력 바로 뒤에서만 분할되므로, `function_call`이 자신의 `function_call_output`과 분리되는 일은 없습니다. 출력 없이 호출로 끝나는 이력은 그 tool 그룹 전체를 유지합니다. 분할 이전의 모든 것이 요약됩니다.

**요약.** 요약될 접두부(system 및 developer 메시지와 `instructions` 포함)는 모델의 채팅 템플릿을 통해 렌더링되고 고정된 handoff 지시가 user 턴으로 덧붙으며, `/v1/responses`와 같은 worker 경로에서 temperature 0, thinking 끈 상태, 도구 없이 생성이 한 번 실행됩니다. 토큰 수는 `/v1/responses/input_tokens` 경로를 사용합니다. 성공적으로 가져온 이미지 URL은 compaction마다 캐시되어 모든 집계와 요약 프롬프트에 재사용됩니다. 미디어 지원 여부와 이미지 개수 검사는 prefetch 전에 실행되며, 읽기에 실패한 경우에는 일반 렌더러가 오류를 내도록 맡겨집니다. 미디어가 있는 본문은 여전히 worker에서 집계되므로, 각 집계는 `/v1/responses/input_tokens`처럼 worker 슬롯을 하나 사용합니다.

**실패.** 다음 경우 어느 것도 컨텍스트를 대체하지 않습니다.

| 조건 | 상태 | 오류 |
|-----------|--------|-------|
| 컨텍스트 윈도우 하나보다 긴 입력(다단계 요약은 구현되어 있지 않음) | 400 | `type: exceed_context_size_error`, `code: context_length_exceeded` |
| 요약 프롬프트와 `max_output_tokens`의 합이 윈도우를 초과 | 400 | 동일 |
| 요약된 컨텍스트가 입력보다 짧지 않음 | 400 | 동일 |
| compaction된 출력이 replay 시 요청 본문 제한을 초과 | 400 | `invalid_request_error`. 유지되는 미디어 또는 `keep_tokens`를 줄이십시오 |
| 최근 턴 이전에 안전한 분할 지점이 없음 | 400 | `invalid_request_error` |
| 요약이 비어 있거나 `max_output_tokens`에서 멈춤 | 502 | `server_error` |

오류 타입은 서버의 기존 컨텍스트 초과 오류와 일치하고, 코드는 OpenAI의 것이므로 어느 쪽 클라이언트든 이를 인식합니다.

**Capsule.** `encrypted_content`는 `{"v":1,"summary":"...","retained":[<input items>]}`의 base64url(패딩 없음)입니다. OpenAI 필드 이름과 달리 설계상 암호화되어 있지 않습니다. 클라이언트가 요약된 대화를 이미 가지고 있으므로 기밀성이 더해 주는 것이 없고, 암호화나 HMAC 키는 새 키 저장소 없이는 서버 재시작 후에 유지되지 않기 때문입니다. 따라서 capsule은 클라이언트 데이터로 취급됩니다. 요청 본문 제한으로 한정되고, 유지된 항목은 일반 `input` 항목과 같은 파서와 검증을 거치며, 다른 capsule을 중첩하는 capsule은 거부됩니다. 디코딩되지 않거나 `v`가 1이 아닌 capsule은 400 `invalid_request_error`를 반환합니다.

**Replay.** `input`(또는 연결된 이력)에 `compaction` 항목이 들어 있으면, 가장 최신 항목 이전의 모든 것이 버려지되 system 및 developer 메시지는 예외이며(그대로 다시 보낸 중복은 한 번만 유지), capsule의 요약(assistant 메시지로)과 유지된 항목이 그 자리를 차지하고, 이어서 그 이후의 항목이 옵니다. 이는 `/v1/responses`, `/v1/responses/input_tokens`, `/v1/responses/compact` 자신에 적용되므로, 이미 compaction된 세션을 다시 compaction하면 이전 요약이 새 요약에 합쳐집니다.

알려진 제한:

- 분할이 가장 최근에 완료된 tool 그룹을 요약 속으로 넘길 수 있습니다. upstream mlx-vlm은 그 그룹을 항상 그대로 유지합니다.
- 요약은 system 및 developer 메시지 바로 뒤에 assistant 턴으로 replay되므로, system 이후 첫 턴이 user 턴이어야 하는 채팅 템플릿(Gemma, 오래된 Mistral)은 replay된 프롬프트를 거부할 수 있습니다. upstream mlx-vlm도 같은 순서를 사용합니다.

### 자동 compaction

`/v1/responses` 요청(`/responses`도 해당)은 다음으로 옵트인합니다.

```json
{"context_management": [{"type": "compaction", "compact_threshold": 24000}]}
```

생성 전에 서버는 `/v1/responses/input_tokens` 경로로 렌더링된 입력의 토큰 수를 세어냅니다. 입력은 연결된 `previous_response_id` 또는 `conversation` 항목과 `input`을 합친 전체 이력이며, 이미 존재하는 capsule은 확장된 상태입니다. 토큰 수가 `compact_threshold`에 도달하거나 컨텍스트 윈도우에 출력을 위한 여유가 남지 않으면, 이력은 `POST /v1/responses/compact`와 정확히 같은 방식(같은 선택, 기본 `keep_tokens`, 실패 처리)으로 compaction되고, 생성은 compaction된 컨텍스트에서 실행됩니다. 요약 예산은 컨텍스트 윈도우의 1/16이며 128에서 1024 토큰 사이입니다. 사용하지 않은 `max_output_tokens`는 임계값에 산입되지 않습니다. compaction은 생성 도중에는 실행되지 않습니다.

이후 응답 `output`은 `compaction` 항목으로 시작하고 생성된 항목이 뒤따릅니다. `usage`는 요약 pass가 아니라 최종 생성만 집계합니다. 저장된 응답은 system 및 developer 메시지를 입력으로, capsule을 첫 번째 출력 항목으로 유지하므로, `previous_response_id` 이어가기는 요약과 유지된 턴을 replay하고 그 이어가기가 임계값에 도달할 때에만 다시 compaction합니다. `conversation` 기록은 새 입력과 capsule을 포함한 출력을 추가하며 같은 방식으로 replay됩니다. 이력을 다시 보내는 클라이언트는 출력을 그대로 추가하거나, 가장 최신 `compaction` 항목과 그 이후의 모든 것만 유지할 수 있습니다.

`context_management`를 포함하는 모든 요청에서(compaction 여부와 관계없이), 생성은 출력 한도(`max_output_tokens` 또는 서버 기본값)와 프롬프트 이후 남은 컨텍스트 중 작은 쪽으로 제한됩니다. 그 상한이나 `max_output_tokens`에서 멈춘 응답은 `status: "incomplete"`, `incomplete_details: {"reason": "max_output_tokens"}`, `completed_at` 없음, 그리고 중단된 message 항목에 `status: "incomplete"` 표시를 반환합니다(스트리밍은 `response.incomplete`로 끝남). 제한된 상한보다 높은 제한적 reasoning 예산은 그 한도보다 토큰 하나 아래로 낮춰지므로, thinking을 강제로 끝내더라도 보이는 출력을 위한 여유가 남습니다. 여유 슬롯이 없는 서버는 어떤 집계도 하기 전에 503으로 응답합니다. `context_management`가 없는 요청은 변하지 않습니다. 추가 집계도, 상한도 없고 `length` 중단도 여전히 `completed`로 보고됩니다.

compaction된 스트리밍 요청은 `response.in_progress` 이후, compaction 항목의 `response.output_item.added`(`output_index` 0. 생성된 항목은 인덱스 1부터 이어지며 `response.completed`도 같은 항목을 나열함) 이전에 두 개의 MLX 확장 이벤트를 보냅니다.

```text
event: mlx.compaction.started
data: {"type":"mlx.compaction.started","sequence_number":2,"response_id":"resp_...","input_tokens":5786}

event: mlx.compaction.completed
data: {"type":"mlx.compaction.completed","sequence_number":3,"response_id":"resp_...","input_tokens_before":5786,"input_tokens_after":854}
```

이 숫자는 compaction 전후의 렌더링된 입력을 설명합니다. compaction은 스트림이 열리기 전에 끝나므로 두 이벤트는 그 뒤에 함께 도착하며, 실시간 진행 상황이 아니라 숫자만 전달합니다. compaction이 없으면 이벤트도 없습니다.

**실패.** compaction 실패는 어떤 출력이 생성되기 전에 `POST /v1/responses/compact`와 같은 HTTP 오류(위 표 참조)를 반환합니다. 스트림이 아직 열리지 않았으므로 스트리밍 요청도 마찬가지입니다. 예를 들어 이력이 윈도우 하나를 넘거나 요약이 이를 줄이지 못하면 400 `exceed_context_size_error`(코드 `context_length_exceeded`)입니다. 넘칠 컨텍스트에서는 아무것도 생성하지 않습니다. 이력에 안전한 분할 지점이 없으면(턴 하나이거나 끝나지 않은 tool 그룹 하나) upstream mlx-vlm처럼, 입력이 출력을 위한 여유를 여전히 남기면 compaction 없이 요청을 진행하고, 그렇지 않으면 `context_length_exceeded`로 실패합니다. 잘못된 `context_management`(배열이 아님, 항목이 둘 이상, `compaction`이 아닌 `type`, 또는 1에서 100000000 범위의 정수가 아닌 `compact_threshold`)는 400 `invalid_request_error`를 반환합니다.

구현되지 않은 것: `/v1/chat/completions`와 `/v1/messages`의 자동 compaction, Codex `compaction_trigger` 입력 항목, 그리고 윈도우 하나보다 긴 이력의 다단계 요약.

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

모델이 호출과 함께 내보낸 `call_id`는 `function_call` 항목(및 그 스트리밍 이벤트)으로 그대로 전달됩니다. 서버는 모델이 id를 내보내지 않은 경우에만 `call_` id를 발급합니다.

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
- 내장 도구 또는 MCP 커넥터 실행 없음;
- 디스크에 영속화되는 응답 저장소 없음;
- 업로드 저장소를 통한 `input_file` 수집 또는 `file_id` 해석 없음.
