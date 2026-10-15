# 서버 기능 및 배포 가이드

이 문서는 `mlxcel-server`와 `mlxcel serve` 진입점에 대한 운영자용 안내서입니다.
루트 README의 기능 요약을 확장하되,
[`llama-server-compat.md`](llama-server-compat.md)에 있는 필드 단위의 호환성 기록은
중복하지 않습니다.

두 진입점은 동일한 HTTP 서버를 시작하며 서버 옵션도 서로 일치하도록 유지됩니다.
`mlxcel serve`는 추가로 `--estimate-memory`와 `--force`를 제공하여,
시작 전에 한 번만 수행하는 메모리 사전 점검(preflight)을 지원합니다.

## 모델 및 스토어 인자

`--models-dir`의 의미는 이 플래그를 오프라인 명령이 소유하는지, 서버 진입점이 소유하는지에 따라 달라집니다.

| 대상 | 옵션 | 의미 |
| --- | --- | --- |
| `mlxcel generate`, `run`, `inspect`, `download`, `list`, `rm` | `--models-dir PATH` | mlxcel이 관리하는 모델 스토어를 재정의합니다. |
| `mlxcel-server`, `mlxcel serve` | `--model-store-root PATH` | 저장소 해석과 다운로드에 사용되는, mlxcel이 관리하는 모델 스토어를 재정의합니다. |
| `mlxcel-server`, `mlxcel serve` | `--models-dir PATH` | 멀티 모델 라우터 모드를 시작하고 체크포인트 하위 디렉터리를 직접 탐색합니다. |

`MLXCEL_MODELS_DIR`은 모델 스토어에 대한 서버 측 환경 변수 재정의로 그대로 유지됩니다.
서버 명령줄에서는 `--models-dir`을 `-m/--model`과 함께 사용할 수 없습니다.
기본이 아닌 스토어에서 단일 모델을 시작하려면 `--model-store-root`를 사용하십시오.

단일 모델 예시:

```bash
mlxcel-server -m Qwen3.5-0.8B-4bit --port 8080

mlxcel-server -m mlx-community/Qwen3.5-0.8B-4bit \
  --model-store-root /var/lib/mlxcel/models --port 8080
```

저장소 id, `MLXCEL_DEFAULT_ORG` 아래의 이름만 있는 모델, 로컬 체크포인트 경로를 사용할 수 있습니다.
`--revision`은 HuggingFace 리비전을 해석하지만, mlxcel 스토어는 리비전별로 네임스페이스가 나뉘지 않습니다.
동일한 저장소의 여러 리비전을 유지해야 하는 배포 환경에서는 별도의 `--model-store-root`를 사용하십시오.

## HTTP 인터페이스

로드된 모델이 해당 모달리티를 처리할 수 없더라도 라우트는 마운트됩니다.
이 경우 핸들러는 체크포인트에 따라 라우트 탐색 결과가 달라지게 하는 대신, 구조화된 capability 오류를 반환합니다.

| 인터페이스 | 주요 라우트 | 참고 |
| --- | --- | --- |
| OpenAI | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/models` | Responses는 id로 조회(retrieve), 삭제(delete), 취소(cancel)도 지원합니다. |
| Anthropic | `/v1/messages`, `/v1/messages/count_tokens` | 로드된 채팅 템플릿과 공유 토큰 계산 경로를 사용합니다. |
| Retrieval | `/v1/rerank`, `/v1/reranking` | Cohere/Jina 호환 리랭킹 인터페이스입니다. |
| Audio | `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations` | Kokoro TTS, 오디오를 지원하는 채팅 모델, Whisper는 [`audio-api.md`](audio-api.md)에 설명되어 있습니다. |
| Native llama-server | `/completion`, `/completions`, `/embedding`, `/embeddings`, `/tokenize`, `/detokenize`, `/infill`, `/apply-template` | 네이티브 응답 스키마는 OpenAI `/v1/...` 스키마와 의도적으로 구분됩니다. |
| 프롬프트 검사 | `/chat/completions/input_tokens`, `/responses/input_tokens` 및 `/v1` 별칭 | 생성과 동일한 경로로 렌더링하고 토큰화합니다. |
| 운영 | `/health`, `/props`, `/slots`, `/metrics`, `/lora-adapters` | 일부 변경 및 조회 세부 기능은 해당 플래그로 활성화됩니다. |
| mlxcel 확장 | `/v1/cache/stats`, `/v1/cache/reset`, `/v1/internal/mtp-policy` | 프롬프트 캐시 및 적응형 speculative decoding 상태입니다. |

`/v1`이 아닌 대부분의 편의용 별칭은 계속 사용할 수 있습니다.
두 가지 별칭은 마이그레이션 시 중요한 예외입니다. `/completions`와 `/embeddings`는 이제 네이티브 `llama-server` 스키마를 반환합니다.
OpenAI 객체를 기대하는 클라이언트는 `/v1/completions`와 `/v1/embeddings`를 사용해야 합니다.

`compat/llama-server/b10621/` 아래에 체크인된 b10621 매니페스트가 공식 호환성 목록입니다.
고정된(pinned) 모든 옵션, 라우트, 네이티브 completion 필드를 지원(supported), 별칭(aliased), 해당 없음(not applicable),
설계상 다름(different by design) 중 하나로 분류하며, 보류된 항목은 없습니다.
정확한 스키마, 차이점, 마이그레이션 참고 사항은
[`llama-server-compat.md`](llama-server-compat.md)를 참조하십시오.

## 컨텍스트, 배치, 캐싱

`--ctx-size`는 전체 컨텍스트 예산입니다. 기본값 `0`은 모델로부터 예산을 도출합니다.
`--parallel -1`은 자동 설정이며 현재 통합 컨텍스트 예산 방식으로 네 개의 슬롯으로 결정됩니다.
모든 슬롯이 전체 윈도우를 사용할 수 있고, 스케줄러가 모든 슬롯에 걸친 공유 활성 토큰 총합을 강제합니다.
명시적인 양수 `--parallel`은 `--kv-unified`가 함께 설정되지 않는 한 슬롯별로 분할된 윈도우를 유지합니다.
`--no-kv-unified`는 자동 슬롯 수에서도 분할 윈도우를 강제합니다.
캐시 레이아웃상 배치할 수 없는 패밀리는 로드 후 유효 decode 행 하나로 제한되지만, 요청별 컨텍스트 윈도우는 줄어들지 않습니다.
`/props`, `/slots`, `/v1/models`가 보고하는 값은 로드 후의 전체 윈도우 값입니다.

```bash
mlxcel-server -m Qwen3.5-0.8B-4bit \
  --ctx-size 65536 --parallel 4 --port 8080
```

연속 배칭(continuous batching)과 프롬프트 프리픽스 캐싱은 기본적으로 활성화되어 있습니다.
네이티브 `cache_prompt` 요청 필드를 사용하면 요청별로 조회와 기증(donation)을 모두 끌 수 있습니다.
캐시 히트 시 prefill이 두 부분으로 나뉘어 수행되어 반올림 결과가 달라질 수 있으므로,
greedy 출력이 캐시 상태에 의존하면 안 되는 경우에는 `cache_prompt: false`를 보내십시오
([`llama-server-compat.md`](llama-server-compat.md#reproducible-greedy-output-send-cache_prompt-false) 참조).
`/v1/cache/stats`는 캐시 상태를 보고하며, `POST /v1/cache/reset`은 캐시를 비웁니다.
컨텍스트 시프팅(context shifting)은 b10621과 일치하도록 기본적으로 비활성화되어 있습니다.
애플리케이션이 너무 긴 대화의 일부를 의도적으로 유지하려면 `--context-shift`를 활성화하고 `--keep`을 설정하십시오.

스케줄러와 분리형(disaggregated) 서빙에 대한 자세한 내용은 [`CONTINUOUS_BATCHING.md`](CONTINUOUS_BATCHING.md)를,
KV cache 모드는 [`turbo-kv-cache.md`](turbo-kv-cache.md)를 참조하십시오.

## 멀티 모델 라우터 모드

`-m` 없이 라우터 모드를 시작합니다.

```bash
mlxcel-server --models-dir /srv/models \
  --model-store-root /var/lib/mlxcel/models \
  --models-max 4 --port 8080
```

레지스트리는 다음 세 가지 소스를 우선순위가 높아지는 순서로 조정합니다.

1. 모델 스토어 아래의 캐시 항목.
2. `--models-dir` 아래의 체크포인트 하위 디렉터리.
3. 선택적 `--models-preset` INI 파일의 섹션.

요청은 JSON `model` 필드 또는 GET 라우트의 `model` 쿼리 매개변수로 항목을 선택합니다.
`--models-max`는 LRU 방식의 축출(eviction)로 상주 모델 집합의 크기를 제한합니다.
자동 로드(autoload)는 기본적으로 켜져 있으며, `--no-models-autoload`로 전역적으로 끄거나
관리 요청의 `autoload`로 제어할 수 있습니다.

라우터 관리는 인벤토리 조회와 재로드에 `GET /models` 또는 `/v1/models`를,
백그라운드 다운로드에 `POST /models`를, 캐시가 소유한 모델 삭제에 `DELETE /models?model=...`를,
상태 이벤트에 `GET /models/sse`를 사용합니다.
모델별 `/props`, `/slots` 및 유사한 GET 라우트는 `?model=...`을 받습니다.
모든 관리 라우트는 일반 API 키 미들웨어를 거칩니다.

## LoRA 어댑터

기본 LoRA 경로는 어댑터를 융합하지 않은 상태로 유지하므로, 재시작 없이 스케일을 변경할 수 있습니다.

```bash
mlxcel-server -m models/base \
  --lora adapters/code,adapters/style \
  --lora-scaled adapters/domain:0.5

curl http://localhost:8080/lora-adapters

curl http://localhost:8080/lora-adapters \
  -H 'Content-Type: application/json' \
  -d '[{"id":0,"scale":0.8},{"id":2,"scale":0.25}]'
```

각 요청은 승인(admission) 시점에 어댑터 스케일의 스냅샷을 가져오므로, 실시간 업데이트가 진행 중인 생성에 영향을 주지 않습니다.
네이티브 completion 요청은 자체 `lora` 선택을 제공할 수도 있습니다.
`--lora-init-without-apply`는 모든 어댑터를 스케일 0으로 시작합니다.

`--lora-fuse`는 어댑터를 기본 가중치에 통합하여 decode 오버헤드를 없애지만, 이 경우 스케일은 프로세스가 살아 있는 동안 고정됩니다.
융합 모드에서는 실시간 업데이트와 요청별 선택이 거부됩니다.
분산 tensor 및 pipeline 로더는 현재 융합 동작을 전제로 합니다.

어댑터는 융합 경로이든 비융합 경로이든 로드되는 체크포인트에 대응되어야 합니다.
`adapters.safetensors`의 모든 텐서는 모델이 실제로 보유한 기본 가중치에 대한
`<layer>.lora_a` / `<layer>.lora_b` 쌍의 한쪽이어야 합니다.
다른 아키텍처용으로 학습된 어댑터, 이름이 바뀐 projection, 또는 불필요한 텐서가 있으면,
기본 가중치로 응답하면서 어댑터가 로드되었다고 보고하는 서버를 시작하는 대신,
문제가 되는 모든 이름을 나열하며 로드가 실패합니다.
DoRA 어댑터는 아예 거부됩니다. 크기(magnitude) 벡터 없이 저랭크 쌍만 적용하면
기본 모델과도, 파인튜닝 모델과도 일치하지 않는 가중치가 만들어지기 때문입니다.
pipeline 병렬 실행에서 어댑터의 레이어를 하나도 소유하지 않은 스테이지는 아무것도 적용하지 않으며, 이는 여전히 유효한 로드입니다.

## 샘플링, 구조화된 출력, 추론

서버는 top-k/top-p/min-p, typical-p, XTC, DRY, repetition/frequency/presence 페널티,
Mirostat, 동적 temperature, adaptive-p, logit bias, 확률 보고를 포함하여
llama-server에서 흔히 사용되는 샘플링 단계를 제공합니다.
시작 시 기본값은 지원되는 요청 스키마에서 재정의할 수 있습니다.

mlxcel은 또한 `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, 네이티브 `/completion`에서
불리언 `p_less` 요청 필드(기본값 `false`)를 받습니다.
이 필드는 하이퍼파라미터가 필요 없는 p-less 절단을 활성화합니다.
페널티와 top-n-sigma 이후, temperature로 스케일된 softmax에서의 확률이
충돌 확률 `sum(p^2)`보다 낮은 토큰이 마스킹되므로,
뾰족한 분포는 argmax 쪽으로 수렴하고 평평한 분포는 어휘의 대부분을 유지합니다.
가장 확률이 높은 토큰은 항상 남고, greedy 요청은 필터를 건너뛰며, 서버 전체 기본값은 없습니다.
CLI에서는 `mlxcel generate --p-less`와 `mlxcel chat --p-less`가 이에 해당합니다.

`--grammar`, `--grammar-file`, `--json-schema`, `--json-schema-file`은 출력을 제약합니다.
네이티브 completion 및 infill 요청은 추가로 GBNF lazy-trigger 필드를 받습니다.
문법은 샘플링 중에 파싱되고 강제되며, 잘못된 제약은 힌트로 받아들여지지 않고 거부됩니다.

추론(reasoning)을 지원하는 체크포인트는 `--reasoning-format`, 선택적 별칭, assistant prefill,
토큰 예산, 네이티브 chat-control 엔드포인트를 통한 런타임 배치를 지원합니다.
모델이 생성한 tool call은 파싱하여 클라이언트에 반환할 수 있지만, mlxcel은 서버 측에서 도구를 실행하지 않습니다.

### 도구 호출과 `tool_choice`

`/v1/chat/completions`의 `tool_choice`와, 이를 변환한 Responses 및 Anthropic Messages의 `tool_choice`는
단순히 되돌려 주는 것이 아니라 실제로 강제됩니다.

- 생략 또는 `"auto"`: 선언된 모든 도구가 프롬프트에 렌더링됩니다.
- `"none"`: 어떤 도구도 렌더링되지 않습니다.
- `"required"`: 모든 도구가 렌더링되고 프롬프트에 지시문이 추가됩니다. `tools`가 없거나 비어 있으면 요청은 400으로 거부됩니다.
- `{"type": "function", "function": {"name": "f"}}`: `f`만 렌더링되며 모델은 이를 호출하도록 지시받습니다. `f`가 선언되지 않았거나, `type`이 `function`이 아니거나, 이름이 비어 있으면 요청은 400으로 거부됩니다.

지시문은 첫 번째 system 메시지에 추가되며, 없으면 마지막 user 메시지에 추가되고,
그것도 없으면 선두의 새 system 메시지로 삽입됩니다.
요청 객체 자체는 수정되지 않으며, 프롬프트 캐시는 실제로 렌더링된 프롬프트를 키로 사용합니다.

도구 호출의 와이어 형식이 고정된 래퍼 안의 JSON 객체인 템플릿에서는,
도구 스키마로 만든 구조화 출력 문법을 통해 호출이 추가로 강제됩니다.
따라서 `required`는 항상 선언된 함수에 대한 호출을 만들고,
이름을 지정한 형식은 항상 해당 함수를 만들며, `finish_reason: "tool_calls"`가 반환됩니다.
래퍼는 로드된 채팅 템플릿에서 읽습니다: Hermes / Qwen(`<tool_call>`),
Mistral Nemo(`[TOOL_CALLS]`), Llama 3(`<|python_tag|>` 또는 `"parameters":`).
JSON 와이어 형식이 없는 포맷(ATEM, Gemma 4, Qwen3-Coder 및 GLM 같은 XML 방언, Kimi K2, pythonic)은
지시문과 좁혀진 도구 목록만 받으며, 강제된 선택이 호출 없이 끝나면 포맷 이름과 함께 `warn` 수준으로 기록됩니다.

문법 엔진이 표현할 수 없는 도구 스키마는 요청을 실패시키는 대신 동일한 지시문 전용 경로로 폴백합니다.
컴파일 실패는 `warn` 수준으로 기록되고 생성은 제약 없이 실행됩니다.
클라이언트가 명시적으로 요청한 `response_format` 스키마만이 컴파일 실패를 400으로 바꿉니다.

강제된 `tool_choice`는 `response_format` 스키마와 함께 사용할 수 없습니다(400).
Anthropic의 `tool_choice` 값 `{"type": "any"}`와 `{"type": "tool", "name": "f"}`는
각각 `required`와 이름을 지정한 형식에 매핑되므로, `/v1/messages`에도 동일한 규칙이 적용됩니다.

## 관측성, 영속성, 재개 가능한 스트림

- `GET /health`와 `/v1/health`는 준비 상태(readiness)를 제공합니다.
- `GET /props`는 해석된 모델, 모달리티, 캐시, 컨텍스트, speculative, 엔드포인트 설정을 보고합니다.
- `GET /slots`는 실시간 요청 슬롯 상태를 노출합니다. `--slot-save-path DIR`은 `/slots/:id_slot`에서 저장, 복원, 삭제 동작을 활성화합니다.
- `--metrics`는 `GET /metrics`에서 Prometheus 출력을 활성화하며, b10621 `llamacpp:` 패밀리와 mlxcel 전용 패밀리를 포함합니다.

`X-Conversation-Id`가 있는 스트리밍 요청은 재개 가능한 전송을 위해 보존됩니다.
`GET /v1/stream?conv_id=...&from=N`은 이를 재생하고 이어서 따라가며,
`POST /v1/streams/lookup`은 알려진 id를 보고하고,
`DELETE /v1/stream?conv_id=...`는 이를 취소하고 제거합니다.
이 헤더가 없으면 클라이언트 연결 해제 시 기존과 같이 생성이 취소됩니다.

`--sleep-idle-seconds N`은 유휴 시간이 지나면 모델 워커와 프롬프트 캐시를 해제합니다.
슬립 중에도 health 및 검사 라우트는 계속 사용할 수 있으며,
다음 추론 요청이 모델을 다시 로드하고 자체 prefill을 수행합니다.
음수 값은 슬립을 비활성화하며, 이것이 기본값입니다.

## 실시간 설정

`--settings`는 `/v1/settings`와 `/settings`에 대한 인증된 `GET` 및 `PATCH`를 활성화합니다.
이 엔드포인트는 기본적으로 존재하지 않습니다.
API 키 없이 loopback이 아닌 TCP 리스너에 노출하려 하면 시작이 거부되며,
loopback 및 Unix 소켓 리스너는 인증 없이 실행할 수 있습니다.

```bash
mlxcel-server -m Qwen3.5-0.8B-4bit \
  --host 0.0.0.0 --api-key change-me --settings

curl http://localhost:8080/v1/settings \
  -H 'Authorization: Bearer change-me'

curl -X PATCH http://localhost:8080/v1/settings \
  -H 'Authorization: Bearer change-me' \
  -H 'Content-Type: application/json' \
  -d '{"default_temperature":0.2,"reasoning_budget":1024}'
```

GET은 타입이 지정된 스키마, 현재 값, 안정적인 fingerprint를 반환합니다.
일반 객체 PATCH는 병합을 수행하며, 명시적 형식은 `{"op":"merge|replace","values":{...}}`입니다.
서로 독립적인 유효한 설정은 하나의 원자적 교체(atomic swap)를 통해 함께 게시되고,
읽기 전용이거나 잘못된 이름은 `rejected`로 반환됩니다.
변경 가능한 그룹에는 요청 타임아웃, 생성 기본값, DRY, 언어 bias, 추론 예산,
chat-template kwargs, 루프 감지, diffusion 기본값이 포함됩니다.

## 전송 및 접근 제어

API 키는 반복 가능한 `--api-key` 값, 쉼표로 구분된 값, 또는 `--api-key-file`로 설정합니다.
키가 활성화되면 `/`, `/health`, `/v1/health`만 공개되며,
모델, 메트릭, 설정, 라우터, 생성 라우트는 유효한 bearer 키가 필요합니다.

기본 CORS 모드는 b10621을 따르며 허용적입니다.
명시적인 목록에는 `--allowed-origins`를 사용하고,
기존 llama-server 명령을 마이그레이션할 때는 `--cors-*` 호환 플래그를 사용하십시오.
TLS는 `--ssl-cert-file`과 `--ssl-key-file`을 모두 지정하면 활성화됩니다.
`.sock`으로 끝나는 `--host`는 Unix 도메인 소켓 리스너를 시작하며,
`--api-prefix`는 모든 라우트를 하나의 경로 아래에 중첩시킵니다.

`AIP_MODE=PREDICTION`이면 서버는 Vertex AI 커스텀 컨테이너의
`AIP_HTTP_PORT`, `AIP_HEALTH_ROUTE`, `AIP_PREDICT_ROUTE` 변수도 따릅니다.
전체 환경 변수 목록은 [`environment-variables.md`](environment-variables.md)를 참조하십시오.

## 의도적인 비목표

mlxcel은 llama-server의 내장 Web UI를 제공하지 않으며,
도구, MCP 서버, 에이전트 루프, UI CORS 프록시를 실행하지 않습니다.
이러한 호환성 플래그를 활성화하면, 해당 기능이 활성 상태인 것처럼 조용히 가장하는 대신 시작 시 실패합니다.
`/tools`와 `/cors-proxy`는 클라이언트가 기대하는 비활성 기능 응답 형태를 유지합니다.
실행은 클라이언트가 담당하므로, 클라이언트가 선언한 도구와 파싱된 `tool_calls`는 계속 지원됩니다.

여기에서 반복하지 않은 의도적인 차이점을 포함하여 모든 b10621 옵션과 네이티브 필드에 대해서는
[`llama-server-compat.md`](llama-server-compat.md)를 참조하십시오.
