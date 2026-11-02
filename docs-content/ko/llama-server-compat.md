# llama-server 호환성 경계 (b10621)

`mlxcel-server`는 많은 `llama-server` 플래그, `LLAMA_ARG_*` / `LLAMA_API_KEY` 환경 변수, 라우트, 네이티브 요청 필드를 받아들이지만, "받아들인다"는 것이 "동작이 동일하다"는 뜻은 아닙니다. 하나의 고정된 upstream 레퍼런스에 대해 검증된 호환성 경계는 이 저장소에서 기계가 읽을 수 있는 manifest로 관리됩니다.

```
compat/llama-server/b10621/
```

manifest가 기준(authority)입니다. 이 페이지는 manifest를 읽는 방법, 적용(enforce)되는 방식, 그리고 재생성하는 방법을 설명합니다. Epic #1431은 경계를 넓히는 작업을 추적하며, 이슈 #1443에서 manifest가 도입되었습니다.

## 고정된 레퍼런스

| 항목 | 값 |
|---|---|
| 릴리스 | [`b10621`](https://github.com/ggml-org/llama.cpp/releases/tag/b10621) (2026-08-25 게시) |
| 커밋 | [`c1d0e7a004015f23bc0233470b747b596f29b264`](https://github.com/ggml-org/llama.cpp/tree/c1d0e7a004015f23bc0233470b747b596f29b264) |
| 아카이브 | `llama-b10621-bin-macos-arm64.tar.gz` |
| SHA-256 | `429c8270608600188035e5e92f7d78dffb7900904fe7dd7e6a84f48068cd13cf` |

배포된 바이너리는 323개의 고유한 long-option 표기를 가진 249개의 도움말 항목, 134개의 `LLAMA_*` 환경 변수(그리고 `HF_TOKEN`과 `MTMD_BACKEND_DEVICE`), 등록된 53개의 HTTP 메서드/경로 쌍, 74개의 네이티브 `/completion` 요청 필드를 노출합니다. 이 개수는 `pin.json`에 고정되어 있으며 변경이 있을 때마다 CI에서 다시 검증됩니다.

## manifest 읽기

`pin.json`은 고정된 레퍼런스, 인벤토리 개수, shard 맵(`shards[name].owners`, 해당 shard의 항목을 소유할 수 있는 구현 이슈 번호의 집합이며, [Sharding](#sharding) 참고)을 기록합니다. 그 밖의 모든 `*.json` 파일은 세 종류의 항목을 담은 영역(area) shard입니다. 세 종류는 `option`(도움말 항목마다 하나이며, 받아들이는 모든 표기, 환경 변수 바인딩, 기본값을 포함), `route`(메서드/경로), `native_request_field`(별칭을 포함한 네이티브 `/completion` 스키마 필드)입니다. `pin.json`과 모든 shard에는 `schema_version: 4`가 들어 있습니다.

모든 항목은 epic #1431에서 정의한 정책에 따라 정확히 하나의 호환성 정책 상태를 가집니다.

| 상태 | 의미 |
|---|---|
| `supported` | 표기, 값의 범위, 기본값, 우선순위, 관찰 가능한 동작이 일치하며, 항목의 `divergence` 목록이 비어 있습니다. |
| `aliased` | 다른 mlxcel 표기, 라우트 또는 요청 필드가 테스트된 변환을 통해 동등한 동작을 제공합니다. claim은 mlxcel 식별자를 명시하며, 이는 b10621의 식별자와 달라야 합니다. mlxcel이 b10621의 이름 자체에 응답한다면 해당 항목은 `aliased`가 아니라 `supported`입니다. |
| `not_applicable` | MLX/CUDA에 대응하는 기능이 없습니다. mlxcel은 해당 옵션을 실행 가능한 진단 메시지와 함께 거부하거나 의미상 아무 영향이 없는 형태만 받아들이며, 테스트와 문서가 있습니다. |
| `deferred` | 아직 사실이 아니며, 연결된 구현 이슈가 이를 소유합니다. 받아들이지만 무시되는 플래그는 `supported`가 아니라 항상 `deferred`입니다. |
| `by_design` | 구현되어 제공되지만, 기록되고 테스트된 방식으로 b10621과 영구적으로 다릅니다. 동작이 다르므로 `supported`가 아니고, 동등한 mlxcel 식별자가 없으므로 `aliased`가 아니며, 개념이 적용되고 제공되므로 `not_applicable`이 아니고, 아무도 이를 닫지 않을 것이므로 `deferred`가 아닙니다. 영구성은 항목별로 `rationale` 객체에서 논증되고 항목의 `test`로 고정됩니다. |

`aliased`, `not_applicable`, `deferred` 항목에는 모두 연결된 이슈가 있으며, 해당되는 경우 `notes`와 테스트 id도 있습니다. mlxcel이 파싱하지만 동작하지 않는 플래그는 `mlxcel` claim 블록에 수용 여부가 고정된 `deferred`이며, 결코 `supported`가 아닙니다. `by_design` 항목은 의도적으로 **열린 이슈가 필요 없습니다**. 추가 작업에 대해 닫혀 있다는 것이 이 상태의 본질적인 의미이므로, 해당 항목은 `--check-issues-open` 게이트에 포함되지 않습니다. 해당 동작을 확립한 일련의 과정에 대한 출처로 `issue` 번호를 유지할 수 있으나, 그 번호는 여전히 shard의 owner 집합에 속해야 합니다.

### `divergence`

모든 항목은 `divergence` 목록을 가집니다. 이 목록은 짧은 문자열들로, 각 문자열은 해당 항목에 대해 mlxcel이 b10621과 외부에서 관찰 가능하게 다른 한 가지 방식을 명시합니다. **비어 있지 않은 `divergence`는 `supported`를 금지하며**, validator는 이를 하드 에러로 처리하면서 네 가지 정직한 대안(`aliased`, `not_applicable`, `deferred`, `by_design`)을 명시하고 소유 이슈를 요구합니다.

이 필드가 존재하는 이유는 산문이 게이트가 될 수 없었기 때문입니다. `notes`에는 `--timeout`, `--models-dir`, `--cache-type-k/v`, `POST /completions`에서의 의미 충돌, `--dry-penalty-last-n`의 반전된 DRY 비활성화 sentinel, `--repeat-penalty` / `--frequency-penalty` / `--presence-penalty`의 penalty 윈도 드리프트와 같은 divergence가 적혀 있었습니다. 상태가 이미 정직하다면 그 방식도 통하지만, 반대의 경우는 막지 못합니다. 22개 항목이 관찰 가능하게 다르면서도 한때 `supported`로 표기되었고, 그중 두 개는 "BEHAVIOR DRIFT"로 시작하는 `notes` 필드 아래에 있었으며, CI 게이트가 그 잘못된 claim을 고정해 버렸습니다. epic #1431의 남은 모든 하위 이슈는 더 많은 항목을 `supported`로 전환할 것이므로, 이 규칙은 기계가 읽을 수 있어야 합니다. `notes`는 맥락과 추론을 유지하고, `divergence`는 구체적으로 관찰 가능한 차이와 그 소유 이슈를 담습니다.

두 필드는 모두 `state`, `issue`, `test`, `rationale`, `mlxcel`과 함께 항목 수준의 필드입니다. 항목은 해당 종류의 fact 키와 이 일곱 개의 정책 키를 그 순서대로 정확히 가지며, 그 외에는 아무것도 가지지 않습니다. `divergance`처럼 철자가 틀린 키(또는 `mlxcel` 블록 안에 잘못 넣은 `divergence`)는 아무것도 기록하지 못한 채 조용히 넘어가는 대신 게이트에서 실패합니다.

### `rationale`

모든 항목은 `rationale` 키를 가집니다. `by_design` 항목을 제외하면 어디서나 `null`이며, `by_design` 항목에서는 기록된 divergence가 영구적이라는 것에 대한 기계 검증 가능한 논증입니다(schema 3에서 `divergence`가 설정한 것과 같은 패턴으로, `notes`의 산문은 게이트가 될 수 없습니다). 다른 상태에 null이 아닌 `rationale`이 있으면 하드 에러입니다. 이 객체의 키 집합은 닫혀 있습니다.

- `kind`: 정확히 `"architectural"` 또는 `"policy"`입니다. `architectural`은 mlxcel이 구조적으로 b10621의 동작을 만들어 낼 수 없다는 뜻입니다(예: 임베딩 패밀리가 자체 forward pass 안에서 풀링하기 때문에 풀링되지 않은 hidden state가 남아 있지 않음). `policy`는 mlxcel이 만들어 낼 수 있지만 의도적으로 그러지 않는다는 뜻입니다.
- `reason`: divergence가 단순히 재진술되는 것이 아니라 왜 영구적인지를 설명하는 비어 있지 않은 문자열입니다.
- `revisit_if`: `kind`가 `policy`일 때에만 비어 있지 않아야 하며, 결정을 다시 검토하려면 무엇이 바뀌어야 하는지를 서술합니다. `kind`가 `architectural`일 때는 `null`이어야 합니다. 이를 통해 구조적 불가능성과 정책적 선택이 기계적으로 구별됩니다.

`by_design` 항목은 추가로 비어 있지 않은 `divergence`(없으면 `supported`입니다), 비어 있지 않은 `notes`, 그리고 영구적 동작 자체를 고정하며 해석 가능한 `test` 포인터를 가져야 합니다. 이 요건은 `deferred`의 단일 이슈 번호보다 의도적으로 더 충족하기 어렵게 되어 있습니다. 다섯 번째 상태는 노력을 멈추는 방법이 되기도 하므로, "영구적"이라는 것은 체크박스를 통과시키려고 주장하는 것이 아니라 항목별로 테스트를 뒷받침하여 논증해야 합니다. 이 모든 조건을 충족하지 못하는 항목은 `deferred`로 남습니다. 알아 둘 만한 구조적 한계가 하나 있습니다. manifest의 단위는 옵션 전체이므로, 값별 divergence(예: 받아들여지는 값들 중 하나만 거부되는 경우)는 값별 상태가 아니라 항목의 `divergence` 문자열에 기록됩니다.

### `mlxcel` claim 블록

항목에는 현재 바이너리가 이미 받아들이는 것을 설명하는 `mlxcel` claim 블록이 있을 수도 있습니다. 필드는 `accepted_spellings`, `accepted_on_one_binary_only`, `env`, `env_binding`(`clap` 또는 `env_test`가 있는 테스트된 `runtime` 브리지), `defaults`, `hidden`, `route`, `field`입니다. 이는 닫힌 키 집합이며, validator는 블록 안의 다른 모든 키를 거부하므로, 오타가 난 키는 아무것도 검사하지 않는 claim을 조용히 기록하는 대신 게이트에서 실패합니다. claim은 상태와 무관하게 CI에서 검증되므로, 부분적 지원이 조용히 퇴행할 수 없습니다. `deferred` 항목도 claim을 유지합니다. mlxcel은 `POST /rerank`를 실제로 마운트하고 `--metrics`를 실제로 받아들이며, CI는 둘 모두 그 상태를 유지하도록 합니다. 해당 항목이 `deferred`인 이유는 받아들여지는 표면 뒤의 동작이 여전히 다르기 때문입니다.

## 마이그레이션: b10621에 도달하기 위해 바뀐 것

경계를 넓히면 mlxcel이 이미 가지고 있던 동작이 바뀝니다. 아래 각 항목은 기존 배포 환경에서 알아챌 수 있는 변경입니다.

### `--timeout`은 HTTP 소켓 타임아웃입니다 (#1432)

`--timeout` / `LLAMA_ARG_TIMEOUT`은 b10621과 마찬가지로 기본값 3600초의 HTTP 소켓 읽기/쓰기 타임아웃입니다. 이전에는 600초의 decode watchdog이었습니다. 이 watchdog은 동작과 기본값을 유지한 채 mlxcel 고유의 `--decode-timeout` / `MLXCEL_DECODE_TIMEOUT`으로 이동했으며, 이것 없이 `--timeout`을 설정하면 두 가지를 모두 명시하는 마이그레이션 경고가 기록됩니다.

기본 CORS 응답도 b10621을 따릅니다. 자격 증명이 활성화된 `--cors-origins *`는 리터럴 `*`를 내보내는 대신 요청의 `Origin`을 그대로 반환하고, 메서드 및 헤더 목록은 `OPTIONS` preflight에서만 전송되며, `Access-Control-Expose-Headers`는 더 이상 전송되지 않고, 모든 경로에 대한 `OPTIONS`는 인증과 라우팅보다 먼저 응답됩니다. 기본 SSE ping 간격은 15초에서 30초로 바뀌었으며, 이는 b10621의 `--sse-ping-interval` 기본값입니다. Unix domain socket은 `.sock`으로 끝나는 `--host`로 선택되며, 일반 호스트와 함께 쓴 `--port 0`은 이제 임시(ephemeral) TCP 포트에 바인딩됩니다. 예전의 `--port 0` 소켓 표기도 deprecation 경고와 함께 여전히 동작합니다.

### `/completions`와 `/embeddings`는 OpenAI 별칭이 아니라 네이티브 라우트입니다 (#1441)

b10621은 `/completion`과 `/completions`를 하나의 핸들러로, `/v1/completions`를 다른 핸들러로 보내며, `/embedding`과 `/embeddings`를 `/v1/embeddings`와 비교해 같은 방식으로 처리합니다. mlxcel은 이 모든 경로에서 OpenAI 형태로 응답했기 때문에, 네이티브 스키마를 읽는 `llama-server` 클라이언트는 파싱할 수 없는 객체를 받았습니다.

| 경로 | 이전 | 현재 |
|---|---|---|
| `POST /completion` | 네이티브 형태, 부분적 | 네이티브 형태, b10621 키 집합 |
| `POST /completions` | **OpenAI** 형태 | **네이티브** 형태 |
| `POST /v1/completions` | OpenAI 형태 | OpenAI 형태 (변경 없음) |
| `POST /embedding` | 마운트되지 않음 | **네이티브** 형태 |
| `POST /embeddings` | **OpenAI** 형태 | **네이티브** 형태 |
| `POST /v1/embeddings` | OpenAI 형태 | OpenAI 형태 (변경 없음) |

네이티브 completion 객체는 `index`, `content`, `tokens`, `id_slot`, `stop`, `model`, `tokens_predicted`, `tokens_evaluated`, `generation_settings`, `prompt`, `has_new_line`, `truncated`, `stop_type`, `stopping_word`, `tokens_cached`, `timings`(`cache_n`이 맨 앞)를 포함합니다. b10621의 네이티브 스트림에는 `[DONE]` sentinel이 없으므로, 스트리밍 형태는 `[DONE]` 대신 같은 객체로 끝납니다. 네이티브 임베딩 응답은 `{index, embedding}`의 순수 JSON 배열이며, `embedding`은 배열의 배열입니다.

**`/completions` 또는 `/embeddings`를 가리키며 OpenAI 형태를 파싱하던 클라이언트는 `/v1/completions` 또는 `/v1/embeddings`로 옮겨야 합니다.** 이 두 경로는 변경되지 않았으며 OpenAI 호환을 유지합니다.

하나의 네이티브 요청 필드는 적용되지 않고 거부됩니다. `n_cmpl`(및 그 별칭 `n`)이 1보다 큰 경우이며, mlxcel은 요청당 하나의 completion만 제공하고 결과 객체의 JSON 배열을 반환하는 경로가 없기 때문입니다. 슬롯이 하나일 때 이 거부는 upstream 자체의 거부와 바이트 단위로 동등합니다. upstream의 스키마 제한은 `1 <= value <= n_parallel`이며 clamp하는 대신 거부합니다. 고정된 바이너리에 `-np 1`을 주면 `Field 'n_cmpl': Value must be between 1 <= value <= 1, but got 2`라고 응답합니다. 값 `1`은 받아들여지므로, 전체 스키마를 기본값으로 보내는 클라이언트가 거절당하지는 않습니다.

나머지 네 개는 #1477에서 그 목록을 벗어나 이제 적용됩니다. 각각 스키마 설명이 아니라 고정된 바이너리의 측정값을 기준으로 합니다.

- `return_tokens`는 최상위 `tokens` 배열을 생성된 원시 id로 채우고, 그렇지 않으면 `[]`로 둡니다. id는 `content`를 다시 토큰화한 것이 아니라 샘플링된 것입니다. `stop: ["Paris"]`일 때 바이너리는 `content: " "`와 `tokens: [12095]`로 응답하므로, stop 시퀀스가 텍스트에서 제외한 토큰도 여전히 집계되고 보고됩니다. 스트리밍은 별개이며 무조건적입니다. 필드를 보냈는지와 무관하게 모든 토큰별 프레임이 `tokens`에 해당 토큰의 id를 담고, 마지막 스트리밍 프레임은 `[]`를 담습니다.
- `return_progress`는 스트리밍 응답을 content가 없는 `prompt_progress: {total, cache, processed, time_ms}` 프레임으로 시작하며, 슬롯이 시작될 때 하나, prefill 청크마다 하나씩 보냅니다. `processed`는 캐시가 공급한 토큰도 포함해서 세므로, 캐시된 프롬프트를 반복하면 0이 아니라 `processed == cache`에서 시작합니다. 이 필드는 `stream` 없이는 효과가 없으며, 이는 upstream의 `stream && return_progress` 게이트와 같습니다.
- `n_indent`는 생성된 줄의 들여쓰기가 주어진 공백 문자 수보다 작아지면 completion을 중단하며, `stop_type: "limit"`로 보고됩니다. upstream의 커서는 디코딩된 토큰마다 한 줄씩 전진하므로, 문제의 줄의 텍스트가 나타난 뒤 한두 토큰 후에 중단이 이루어지며, 응답 텍스트는 그 줄의 선행 공백 다음 첫 문자에서 잘립니다. 스트리밍된 프레임에는 이미 추가 텍스트가 포함되어 있으며, 이는 b10621과 mlxcel에서 똑같습니다. 값의 범위는 `0..=2147483647`이고 범위를 벗어나면 clamp하지 않고 거부합니다.
- `t_max_predict_ms`는 한 요청의 예측 단계를 제한하며, 첫 토큰부터 측정하고 내보낸 텍스트에 개행이 포함된 조각에서만 검사하고, 마찬가지로 `stop_type: "limit"`로 보고됩니다. 이 값은 요청별이며 서버 전체의 `--decode-timeout` / `MLXCEL_DECODE_TIMEOUT` watchdog에는 영향을 주지 않습니다. 값의 범위는 `-1..=9223372036854775807`이며, `-1`과 `0`은 모두 이를 비활성화합니다.

같은 변경에서 `generation_settings`는 b10621의 49개 `task_params` 키 중 39개에서 47개로 늘었습니다. `samplers`, `timings_per_token`, `logit_bias`, `lora`, `chat_format`, `generation_prompt`, `reasoning_format`, `reasoning_in_content`가 이제 보고되며, 각각 mlxcel이 실제로 해석한 값에서 가져옵니다. 여전히 없는 두 개는 `backend_sampling`과 `speculative.types`입니다. 전자의 항목은 mlxcel의 sampler가 곧 backend 그래프이며 전환할 CPU 체인이 없다고 기록하고, 후자는 mlxcel에는 없는 upstream의 draft-model 타입을 가리킵니다. 둘 다 임의로 만든 값으로 보고하지 않고 생략하며, 이는 `GET /props`가 따르는 정책과 같고, mlxcel이 각각에 대한 대응 기능을 갖추면서 그 여덟 개 키가 보고 집합으로 옮겨진 것과 같은 정책입니다.

두 가지 스트리밍 프레임 형태가 함께 바뀌었습니다. 이제 모든 partial 프레임은 upstream의 sentinel인 `id_slot: -1`을 담습니다. upstream의 `send_partial_response`는 슬롯 id를 기록하지 않고 마지막 프레임만 슬롯을 명시합니다. 또한 partial 프레임의 `tokens_predicted`는 전송된 프레임 수가 아니라 슬롯이 생성한 토큰 수이며, stop 매처나 증분 detokenizer가 한 조각을 보류했을 때마다 두 값이 달라집니다.

`verbose`는 #1477에서 노력 때문이 아니라 증거에 따라 그 목록에서 빠졌습니다. b10621은 `__verbose` 디버그 블록을 OAI 호환 응답 빌더(`server-task.cpp`)에서만 작성하며, 네이티브 completion 객체는 바로 `to_json_non_oaicompat()`이므로 `verbose: true`는 upstream에서도 아무것도 바꾸지 않습니다. 고정된 바이너리로 측정한 결과, 이 필드를 설정했을 때의 최상위 키 집합은 설정하지 않았을 때와 동일합니다. mlxcel은 이제 이를 받아들이고 무시하며, 이는 upstream이 하는 일입니다. 이를 거부하는 것이 divergence였습니다.

`tokens_cached`도 #1477에서 의미가 바뀌었으며, 이를 캐시 히트 수치로 읽던 클라이언트에는 **호환성을 깨는 변경**입니다. b10621은 요청 이후 슬롯의 캐시 점유량을 보고하며, 이는 `tokens_evaluated + tokens_predicted - 1`(saturating)이지 prefix 캐시가 해당 요청에 공급한 양이 아닙니다. 고정된 바이너리에 대한 여섯 번의 측정이 모두 일치하며, 여기에는 upstream이 여전히 `tokens_predicted: 1`로 응답하는 `n_predict: 0` 프롬프트 전용 사례와, 히트로 수치가 바뀌지 않는 완전 캐시 히트 요청이 포함됩니다. 캐시가 공급한 개수는 `timings.cache_n`이며, 이는 변경되지 않았고 `timings.prompt_n`이 이로부터 파생됩니다. 히트 크기를 알고 싶은 클라이언트는 `cache_n`을 읽어야 합니다.

네 개의 네이티브 필드가 같은 라우트에서 무시됨에서 적용됨으로 바뀌었으며, 각각 스키마 설명이 아니라 고정된 바이너리를 기준으로 확인했습니다(스키마 설명은 그중 두 개에 대해 틀립니다).

- `response_fields`는 완성된 응답을 나열된 경로로 투영(project)합니다. `/`를 포함한 경로는 객체를 따라 탐색한 뒤 **경로 문자열 전체**를 키로 저장하므로, `generation_settings/n_predict`는 이름이 말 그대로 `generation_settings/n_predict`인 루트 키로 반환됩니다. 스키마 설명은 값이 루트로 "unnested"된다고 하지만, 바이너리는 그렇게 하지 않습니다. 존재하지 않는 경로, 객체가 아닌 값으로 들어가는 경로, 빈 경로는 모두 오류 없이 생략되며, 문자열 배열이 아닌 값은 요청을 실패시키지 않고 전체 객체를 반환합니다. 투영은 비스트리밍 본문과 마지막 스트리밍 프레임에 적용되며, 토큰별 프레임에는 적용되지 않습니다.
- `stream_options`와 그 하위 필드 `stream_options.include_usage`는 선언되고 타입 검사됩니다. 네이티브 라우트에서는 마지막 프레임이 항상 카운트와 타이밍 블록을 담으므로 이 값은 양쪽 모두에서 효과가 없습니다. boolean이 아닌 `include_usage`는 upstream과 같은 문구로 거부되고, 객체가 아닌 `stream_options`는 허용되어 무시됩니다.
- `timings_per_token` 프레임은 이제 실제 prefill 수치를 담습니다. `cache_n`, `prompt_n`, `prompt_ms`, `tokens_evaluated`는 마지막 프레임 전까지 모든 프레임에서 0이었지만, 이제는 첫 프레임부터 마지막 프레임이 보고하는 것과 같은 값을 보고합니다.
- `n_predict`는 serde의 값 범위가 아니라 upstream의 전체 값 범위를 받아들입니다. `-1`은 "컨텍스트가 허용하는 만큼"을 뜻하며 422로 거부되는 대신 유효 컨텍스트 윈도로 clamp되고, `0`은 프롬프트를 평가하며(설명은 토큰이 없다고 하지만 바이너리가 실제로 하는 대로 두 구현 모두 정확히 한 토큰을 내보냅니다), `[-1, 2147483647]` 밖의 값은 upstream의 문구로 거부됩니다.

`timings` 내부의 두 가지 파생 값도 같은 시점에 바이너리에 맞게 수정되었으므로, 이를 그래프로 그리는 클라이언트는 이전과 다른 숫자를 보게 됩니다. `prompt_n`은 이제 전체 프롬프트가 아니라 `tokens_evaluated - cache_n`이고, `predicted_per_token_ms` / `predicted_per_second`는 `predicted_ms`가 첫 토큰부터 측정되므로 `predicted_n`이 아니라 `predicted_n - 1`로 나눕니다. draft 요청에서는 첫 토큰 시점이 target prefill의 끝이므로, speculative decoding에서도 같은 분리가 성립합니다(#1592). 자세한 내용은 [`docs/speculative-acceptance.md`](speculative-acceptance.md#per-request-acceptance-on-the-response-itself)를 참고하십시오.

### `stop`은 무시되는 대신 출력을 잘라냅니다 (#1466)

`stop`은 출하된 MLX 서빙 경로에서, 모든 completion 라우트와 스트리밍/비스트리밍 형태 모두에서 적용됩니다. 생성은 첫 번째 일치에서 끝나고, 일치한 문자열은 내보낸 텍스트에서 제외되며, 네이티브 응답은 `stop_type: "word"`와 `stopping_word`에 일치 항목을 보고합니다. 이전에는 파싱되어 스케줄러로 전달되었지만 이후 한 번도 읽히지 않았으므로, 문자열에서 멈추라고 요청한 요청이 출력에 그 문자열을 포함한 채 `n_predict`까지 실행되었습니다.

일치 검사는 생성되는 대로 디코딩된 텍스트에 대해 수행되므로 토큰 경계에 걸친 stop 문자열도 포착되며, 꼬리 부분이 stop 문자열이 될 수 있는 디코딩된 조각은 다음 조각이 그것을 확정할 때까지 스트림에서 보류됩니다. 따라서 스트리밍된 청크를 이어 붙인 결과는 항상 비스트리밍 `content`와 정확히 같습니다. 클라이언트는 이후 토큰이 일치로 바꿔 버리는 텍스트를 관찰할 수 없습니다. 동률이면 텍스트에서 가장 앞선 일치가 이기고, 위치가 같으면 요청의 `stop` 목록에서 첫 번째 항목이 이깁니다.

일치는 upstream과 마찬가지로 reasoning을 제거한 뷰가 아니라 원시 생성 스트림에 대해 수행됩니다. 따라서 reasoning 모델에서 `<think>` 블록 안에 나타나는 stop 문자열은 그 지점에서 요청을 끝냅니다. `/v1/messages`는 이를 Anthropic의 `stop_reason: "stop_sequence"`로, 일치한 문자열을 `stop_sequence`에 담아 보고합니다.

### 토큰화, 템플릿, infill 유틸리티 (#1442)

생성하는 대신 프롬프트를 검사하는 b10621의 여섯 라우트가 이제 제공되며, `/tokenize`와 `/detokenize`는 스키마의 나머지 부분을 지원하게 되었습니다.

`/tokenize`는 b10621의 네 가지 스위치를 모두 지원합니다. `content`는 혼합 형태, 즉 문자열 또는 요소가 문자열이거나 이미 토큰화된 id인 배열을 받으며, 없을 수도 있습니다. 없을 경우 오류가 아니라 빈 토큰 목록으로 응답합니다. `add_special`의 기본값은 `false`입니다. `parse_special`의 기본값은 `true`이며, `false`로 설정하면 텍스트에 쓰인 special-token 표기가 해당 토큰으로 인식되지 않습니다. `with_pieces`는 응답을 평평한 id 배열에서 토큰마다 하나의 `{id, piece}` 객체로 바꾸며, 토큰의 바이트가 유효한 UTF-8이면 `piece`는 JSON 문자열이고 그렇지 않으면 바이트 값의 배열입니다. 이 두 번째 형태는 예외적인 경우가 아닙니다. byte-level BPE 토큰은 흔히 멀티바이트 문자의 일부를 담으며, 클라이언트는 그 배열로 텍스트를 다시 조립합니다. 모든 piece의 바이트를 이어 붙이면 입력이 정확히 복원됩니다.

`/detokenize`는 upstream의 `tokens_to_str`처럼 special token을 건너뛰지 않고 렌더링하며, `tokens`가 없거나 비어 있을 때 실패하는 대신 `{"content": ""}`로 응답합니다. 문자를 이룰 수 없는 바이트는 U+FFFD로 반환되며, upstream의 JSON writer가 replacing 에러 핸들러로 실행되므로 upstream도 이렇게 내보냅니다.

`/apply-template`는 chat-completions 본문을 로드된 chat template으로 렌더링하여 생성 없이 `{"prompt": "..."}`로 응답합니다. `/chat/completions/input_tokens`, `/v1/chat/completions/input_tokens`, `/responses/input_tokens`, `/v1/responses/input_tokens`는 `/v1/messages/count_tokens`가 이미 사용하던 키인 `{"input_tokens": N}`으로 응답합니다. 다섯 개 모두 생성 경로가 실행하는 것과 같은 렌더링과 같은 인코딩을 실행하므로, 그 숫자는 어긋날 수 있는 별도의 추정치가 아니라 해당 요청이 실제로 prefill될 토큰 수입니다. 이미지, 비디오 클립, 오디오가 들어 있는 본문은 model worker에서 집계합니다. 각 미디어 파트는 그곳에서만 soft token으로 변환되므로, 생성이 실행하는 것과 같은 soft-token 확장을 거친 뒤 prefill 직전에 멈추며, 그 수치는 같은 본문이 생성될 때 보고하는 `usage.prompt_tokens`(또는 `usage.input_tokens`)와 같습니다. 이런 집계는 생성처럼 worker 슬롯을 하나 사용하므로 바쁜 서버는 이를 거부할 수 있습니다. 텍스트 전용 본문은 여전히 요청 스레드에서 인코딩되며 worker를 기다리지 않습니다. Responses 쌍은 `/v1/responses`가 사용하는 것과 같은 translator를 거치므로, `previous_response_id`와 `conversation` 복원도 함께 집계됩니다. 로드된 체크포인트가 소비할 수 없는 미디어 파트는 렌더링이 이를 해석하기 전에 생성 라우트가 거부하는 방식과 동일하게 여기서도 거부되므로, 다섯 개 중 어느 것도 생성 경로가 거절했을 프롬프트를 집계하지 않습니다. 렌더링에는 프롬프트 캐시가 켜져 있을 때 서버가 content만 있는 assistant 턴에 다시 주입하는 reasoning이 생성 라우트와 동일하게 포함되므로, 후속 턴은 실제로 prefill되는 프롬프트를 집계합니다. 생성할 수 없는 서버(`--embeddings` 또는 `--reranking`으로 시작했거나 chat 모델이 로드되지 않은 경우)에서는 각 집계 라우트가 자신의 생성 라우트와 동일한 501과 본문을 반환합니다. 스트리밍 `/v1/messages`는 `message_start.usage.input_tokens`에 같은 값을 보고합니다. 텍스트 전용 본문은 로컬에서 집계해 이벤트를 즉시 내보내고, 미디어가 있는 본문은 worker가 prefill 시점에 확장된 프롬프트 길이를 보고할 때까지 `message_start`를 보류합니다. 그 전까지 스트림에는 keepalive 주석만 실리므로, `--sse-ping-interval -1`을 쓰면 미디어 요청이 큐에서 대기하고 prefill하는 동안 클라이언트는 바이트를 전혀 받지 못합니다. `/apply-template`도 같은 경로로 렌더링하므로 그 prompt에도 다시 주입된 reasoning이 표시됩니다.

`/infill`은 어휘가 fill-in-the-middle 마커를 선언하는 모델에서 제공되며, 그렇지 않으면 누락된 마커를 명시하는 upstream의 문구로 거부됩니다. 마커 표기는 `llama-vocab.cpp`가 검색하는 목록이므로 Qwen, StarCoder, Granite, DeepSeek-Coder, CodeLlama 체크포인트는 별도 설정 없이 해석됩니다. `--spm-infill`은 middle 마커는 마지막에 둔 채 suffix 블록을 prefix 블록 앞으로 옮기며, 이는 upstream이 하는 교환입니다. 한 가지 순서로 학습된 모델은 다른 순서의 프롬프트에도 유창하지만 틀린 답을 내므로, 이 플래그는 취향이 아니라 정확성 스위치입니다.

`/infill`에는 의도적이고 영구적인 두 가지 차이가 남아 있으며, manifest 항목은 지원을 주장하는 대신 이를 `by_design`(kind `policy`)으로 기록합니다.

- `input_prefix`, `input_suffix`, `input_extra[].text`, `input_extra[].filename`, `prompt`의 텍스트에 모델 자체의 FIM 마커 표기가 포함되어 있으면 해당 필드와 마커를 명시하는 진단과 함께 **거부됩니다**. b10621은 이 필드들을 special-token 파싱을 끈 채 토큰화하므로 마커는 그곳에서 리터럴 텍스트로 남습니다. mlxcel의 생성 진입점은 파싱을 켠 채 다시 토큰화되는 프롬프트 문자열을 받으며, 문자열로는 "이 문자들은 그 토큰이 아니다"를 표현할 수 없으므로, 텍스트를 그대로 통과시키면 클라이언트가 작성하지 않은 파일이 FIM 프롬프트를 재구성하여 유창하지만 틀린 completion을 반환하게 될 수 있습니다. 명시적으로 실패하는 것이 더 작은 divergence입니다.
- FIM 컨텍스트는 prefill 배치에 맞게 잘리지 않습니다. b10621은 마지막 `3*(n_batch/4)`개의 prefix 토큰과 처음 `(n_batch/4) - (2 + prompt)`개의 suffix 토큰만 유지하지만, mlxcel은 prefix와 suffix 전체를 제공하며, 이는 너무 긴 요청의 프롬프트 텍스트를 버리지 않고 clamp하는 다른 모든 곳의 정책과 일치합니다.

### 임베딩 및 reranking 모드, pooling, 정규화 (#1452)

b10621의 `--embedding` / `--embeddings`, `--rerank` / `--reranking`, `--pooling`, `--embd-normalize`는 두 서버 바이너리 모두에서 받아들여지며, `LLAMA_ARG_EMBEDDINGS`, `LLAMA_ARG_RERANKING`, `LLAMA_ARG_POOLING`이 바인딩됩니다. 두 모드 플래그는 upstream의 플래그가 하는 두 가지 일을 하고, upstream에는 필요 없는 한 가지를 더 합니다.

첫째, **제한**합니다. 생성 라우트는 플래그를 명시하여 `501`로 응답하므로, 클라이언트는 `llama-server --embeddings`에서 보게 될 것과 같은 "이 서버는 생성하지 않는다"를 봅니다. 둘째, **선택**합니다. mlxcel 서버는 하나의 가중치 집합을 재사용하는 대신 전용 임베딩 또는 reranking 워커를 실행하므로, `--embeddings`는 `-m` 또는 `--embedding-model`에서 임베딩 체크포인트를 확인할 수 있어야 하고, `--reranking`은 `-m` 또는 `--reranker-model`에서 reranker를 확인할 수 있어야 합니다. 어떤 것도 제공할 수 없는 모드를 요청하는 명령줄은 유일한 라우트가 영원히 501로 응답하는 서버를 띄우는 대신 플래그를 명시하며 시작 시 실패합니다. 이 시작 시 거부는 upstream에는 없는 유일한 동작입니다. upstream의 `--embeddings`는 `-m`이 무엇이든 그것으로 임베딩하지만, mlxcel에는 causal chat 체크포인트에서 임베딩을 만들어 내는 경로가 없기 때문입니다.

`--pooling`은 `mean`, `cls`, `last`를 mlxcel 자체 커널에 매핑하며, 이 플래그는 체크포인트의 `1_Pooling/config.json`과 `MLXCEL_EMBEDDING_POOLING` 변수보다 우선합니다. 나머지 두 값은 pooling 커널이 전혀 아닙니다. `rank`는 b10621이 모델을 reranking 경로에 올리는 방식이므로 `--reranking`의 동의어로 받아들여지고, `none`은 토큰마다 하나의 벡터를 요구하는데 모든 임베딩 패밀리가 엔진이 출력을 보기 전에 자체 forward pass 안에서 풀링하므로 mlxcel이 만들 수 없습니다. `none`은 그 이유와 함께 시작 시 거부됩니다. mlxcel 고유의 `max` pooling은 b10621 표기가 없으며 `MLXCEL_EMBEDDING_POOLING`과 체크포인트 설정을 통해서만 사용할 수 있습니다.

`--embd-normalize`는 upstream의 산술로 b10621의 전체 값 범위를 구현합니다. `-1`은 없음, `0`은 부호 있는 int16 범위로의 최대 절댓값 재스케일, `1`은 taxicab, `2`는 euclidean, 2보다 큰 모든 값은 해당 p-norm입니다. 영벡터는 NaN이 아니라 0으로 정규화되며, 이는 upstream의 `norm = sum > 0 ? 1/sum : 0` 규칙이고 임베딩 라우트가 유한하지 않은 응답을 500으로 거부하기 때문에 여기서 중요합니다. `2`는 모든 mlxcel 임베딩 응답이 이미 사용하던 L2 커널에 위임하므로, 기본 경로는 이전과 정확히 같은 숫자를 생성합니다. 이 값은 upstream이 읽는 것처럼 `/embedding`, `/embeddings`, `/v1/embeddings`에서 요청별 `embd_normalize`로도 읽을 수 있으며, `mlxcel embed`도 같은 `--pooling`과 `--embd-normalize`를 받으므로 오프라인과 서버 표면이 동일하게 해석됩니다.

기본값 하나는 다릅니다. 플래그가 설정되지 않으면 mlxcel은 `config.json`의 체크포인트 자체 `normalize` 플래그를 따르며, 이를 설정하지 않은 모든 체크포인트에서는 euclidean이고 `false`로 설정한 체크포인트에서는 없음입니다. b10621은 GGUF에 그런 메타데이터가 없으므로 항상 `2`가 기본값입니다.

네 개의 #1452 manifest 항목은 이러한 divergence를 `by_design`으로 기록합니다. 시작 시 거부와 `--pooling` 값 매핑은 `architectural`이고(causal chat 체크포인트에 대한 임베딩 경로가 없고, `none`에 해당하는 풀링되지 않은 hidden state가 없으며, `rank`가 가리킬 pooling 커널이 없음), `--embd-normalize` 기본값은 `policy`입니다(상수 `2`를 기본값으로 하는 것은 가능하지만 체크포인트 작성자가 선언한 `normalize`를 덮어쓰게 됨). 단위의 한계에 유의하십시오. `--pooling`은 하나의 항목이므로, manifest는 `mean`, `cls`, `last`는 제공되고 `none`만 거부된다는 값별 구분을 표현할 수 없으며, 그 구분은 항목의 `divergence` 문자열이 담습니다.

#### Rerank 엔벌로프

네 개의 rerank 라우트는 이제 Cohere의 형태뿐 아니라 b10621의 형태로도 응답합니다.

| 요청이 문서 목록을 표기하는 방식 | 응답 |
|---|---|
| `documents` (Jina / Cohere) | `{"model", "object": "list", "usage", "results": [{"index", "relevance_score"}]}`, `return_documents`일 때 `document`가 함께 반환됨 |
| `texts` (b10621 / TEI) | `{"index", "score"}`의 순수 배열, `return_text`일 때 `text`가 함께 반환됨 |

이전에는 Jina 엔벌로프에 `object`가 없었고 TEI 형태는 전혀 제공되지 않았으며, 둘 다 네 라우트 모두에서 기록된 divergence였습니다. 두 목록 표기를 모두 담은 본문은 Jina 엔벌로프를 유지하고, 세 가지 형태 오류(`"query" must be provided`, `"query" must be a string`, `"documents" must be a non-empty string array`)는 upstream의 문구를 사용합니다. 같은 이유로 `/embedding` 계열은 `"input" or "content" must be provided`로 응답합니다.

#### 제한된 모드에서의 readiness

`--embeddings` 또는 `--reranking`으로 시작한 서버에는 "로드"될 chat 워커가 없으므로, `/health`는 프로세스가 살아 있는 동안 `503 {"status": "loading model"}`로 응답했으며, 이는 어떤 컨테이너 probe 뒤에서도 해당 모드를 사용할 수 없게 만듭니다. 이 두 모드에서는 이제 `/health`가 대신 모드가 선택한 워커의 상태를 보고합니다. 이 변경은 의도적으로 플래그에만 한정됩니다. `-m`이 임베딩 체크포인트이지만 플래그 없이 시작된 서버는 여전히 chat provider에 대해 보고하므로, 현재 존재하는 어떤 것도 동작이 바뀌지 않습니다. queue-full 및 loading 본문 드리프트와 함께 그 더 넓은 경우는 `GET /health`의 manifest 항목에 기록되어 있으며 #1440에 속합니다.

#### `/props`와 `/v1/models`의 확인된 capability

`/props`에는 전달된 값이 아니라 서버가 확인한 내용을 보고하는 `capabilities` 블록이 추가됩니다. 생성이 켜져 있는지, 어떤 모드 플래그가 이를 제한했는지, 그리고 로드된 각 보조 모델에 대해 그 id와 실제로 적용 중인 pooling 및 `embd_normalize`를 보고합니다. 따라서 체크포인트가 덮어쓴 `--pooling` 값도 보입니다. `/v1/models`는 각 항목에 `capabilities` 배열(`completion`, `embedding`, `rerank`)을 붙이며, 이는 클라이언트가 어떤 id를 어디로 보내야 하는지 알아내는 데 필요한 것입니다. 이 필드는 비어 있으면 생략되므로 일반 생성 배포는 OpenAI 객체 형태를 정확히 유지합니다.

### Speculative decoding 표면 (#1433)

b10621의 모든 speculative 옵션은 두 서버 바이너리에서 파싱됩니다. `--spec-draft-model`(`-md`, `--model-draft`, mlxcel의 `--draft-model` 표기 포함)은 MTP / DFlash draft 체크포인트를 선택하고, `--spec-draft-n-max`(제거된 `--draft` / `--draft-n` / `--draft-max` 표기가 별칭으로 유지됨)는 draft 토큰 상한을 설정합니다. 이 상한의 정식 변수는 `LLAMA_ARG_SPEC_DRAFT_N_MAX`이며, 제거된 `LLAMA_ARG_DRAFT_MAX`는 fallback으로 인정됩니다. `--spec-type`은 mlxcel이 실행할 수 있는 것을 변환합니다. `none`은 b10621과 정확히 같이 speculation을 비활성화하고, `draft-mtp` / `draft-dflash`는 `--draft-kind`에 매핑되고, `draft-simple`(b10621의 일반 draft-model speculation)은 `--draft-kind full-model`에 매핑되며(#2757), 그 밖의 모든 서브시스템(n-gram 모드 포함)은 진단과 함께 시작 시 실패합니다. mlxcel 고유의 drafter 없는 n-gram speculation은 `--draft-kind prompt-lookup`(#2255, [`docs/CONTINUOUS_BATCHING.md`](CONTINUOUS_BATCHING.md#prompt-lookup-under-continuous-batching) 참고)이며, `--spec-type none`은 이것도 끕니다. 나머지 `--spec-draft-*` 계열은 GGML 그룹처럼 inert-or-reject로 분류된 숨겨진 호환성 표면입니다. 무해한 값(upstream 기본값, full-offload `--spec-draft-ngl` 표기, `f16` draft 캐시 타입, n-gram 선택자를 고를 수 없는 동안의 모든 n-gram 튜닝 값)은 받아들여지고, mlxcel이 재현할 수 없는 동작을 바꾸는 모든 값은 모델 로드 전에 시작 시 실패합니다. 해석된 설정은 `/props`의 `speculative` 아래(draft 모델의 basename만, kind override, `n_max`)에 보고됩니다. 네이티브 `/completion`의 `speculative.*` 점 표기 필드는 받아들여지고 효과가 없으며, 해당 스키마 블록이 컴파일에서 제외된 b10621과 일치합니다. 응답 측면에서, draft된 요청의 `timings` 객체는 upstream이 표기하고 게이트하는 방식 그대로(verify 라운드가 실행되었을 때만 존재) b10621 자체의 선택적 `draft_n` / `draft_n_accepted` 쌍을 담으며, upstream에는 대응이 없는 두 개의 mlxcel 키도 더합니다. 하나는 이 쌍만으로는 라운드당 평균 허용 길이를 복원할 수 없게 되므로 필요한 `draft_rounds`이고, 다른 하나는 mlxcel의 어느 drafter가 요청을 처리했는지 알려 주는 `draft_kind`입니다. `/v1/chat/completions`는 이 블록 전체를 비스트리밍 응답과 `finish_reason` 청크의 최상위 `timings` 객체로 담으며, 이는 b10621이 자체 OpenAI chat 응답에 넣는 객체입니다. 거기에 남은 차이는 존재 여부입니다. upstream은 모든 chat completion에서 이를 내보내지만 mlxcel은 draft된 요청에서만 내보내며, 두 chat 라우트 항목 모두 이를 기록합니다(#1314). [`docs/speculative-acceptance.md`](speculative-acceptance.md)를 참고하십시오.

### Context shift는 기본적으로 꺼져 있으며, retention은 운영자가 설정할 수 있습니다 (#1472)

`--context-shift` / `--no-context-shift`(`LLAMA_ARG_CONTEXT_SHIFT`), `--keep`, 네이티브 `n_keep` / `n_discard` 요청 필드는 이제 두 서버 바이너리에서 b10621의 컨텍스트 retention 의미를 가집니다. **기본값이 바뀌었습니다**. #1472 이전에는 `--ctx-size` 또는 `--max-kv-size`가 KV 윈도를 제한할 때마다 mlxcel이 가장 오래된 토큰을 조용히 앞에서부터 잘라내고(고정된 4개 토큰의 attention sink 유지) 클라이언트에게 알리지 않았습니다. 이제는 b10621의 기본적으로 비활성화된 context shift와 일치합니다.

- 슬롯별 컨텍스트에 들어가지 않는 프롬프트는 shift가 활성화되어 있든 아니든 upstream의 문구(`request (N tokens) exceeds the available context size (M tokens), try increasing it`)로 admission 시점에 거부됩니다. shift는 decode 중의 성장만 관장합니다.
- shift가 비활성화된 상태(기본값)에서 한계에 도달한 생성은 `truncated: true`와 `stop_type: "limit"`(OpenAI `finish_reason: "length"`)로 정상적으로 중단되며, 이는 b10621의 한계에서 중단하는 형태입니다.
- `--context-shift`가 있으면 스케줄러는 유지되는 prefix 뒤의 토큰을 버려 공간을 확보합니다. `--keep`(요청별로는 `n_keep`이며, `-1` = 초기 프롬프트 전체, 토크나이저가 앞에 붙이는 BOS에 대해 토큰 하나 추가, `bound - 4`로 clamp)은 모든 shift에서 살아남고, 요청별 `n_discard`는 버리는 깊이를 설정합니다(`0`이 기본값이며 유지되지 않는 윈도의 절반을 버립니다). prefill 청크가 한계를 넘어서면 상한이 지켜지도록 버리는 양이 넘어선 만큼으로 늘어납니다.

**예전의 rolling-window 동작에 가까운 것으로 복원하려면 `--context-shift --keep 4`를 전달하십시오.** 이전 동작은 단계마다 초과분만 정확히 버렸고 새 기본 discard는 윈도의 절반이므로 유지되는 텍스트는 다르지만, attention sink가 고정된 채 생성이 다시 무제한이 됩니다. 이전 trim에서 이어지는 두 가지 주의점이 있습니다. shift는 Turbo 양자화 KV 레이어에서는 기록된 no-op이며(시작 시 경고), VLM 요청은 upstream이 멀티모달을 예외로 두는 것처럼 이 모든 장치에서 제외됩니다.

KV cache가 모델 내부에 있는 모델(`ModelOwned` 시퀀스 상태 레이아웃: Gemma 3/4, Qwen 3.5, ExaOne 4, GPT-OSS, Ministral 3, Step 3.5, Mellum, Laguna, Nemotron-NAS, LongCat 그리고 [`docs/supported-models.md`](supported-models.md#kv-cache-modes-on-model-owned-families)에 나열된 다른 패밀리)에는 shift가 trim할 pool cache가 없습니다. 슬롯별 KV 상한(`--max-kv-size` 또는 `--ctx-size` 윈도)이 적용되면 서버는 이런 모델을 context shift를 끈 채로 서빙하고, 모델의 스케줄러가 처음 시작될 때 모델당 한 번 경고를 남깁니다(`--sleep-idle-seconds`에서 깨어난 모델은 이를 반복하지 않으며, 멀티 모델 서버에서는 영향을 받는 모델마다 각자 경고를 남깁니다). 따라서 상한은 컨텍스트 윈도까지 조용히 커지는 대신 admission 거부와 `truncated: true` 중단으로 지켜집니다(#2766, #2783).

같은 이슈에서 배칭 및 슬롯 개수 항목도 닫혔고, #1815는 upstream의 자동 컨텍스트 동작을 마무리했습니다. `--parallel`은 b10621의 전체 값 범위를 받아들이며 기본값은 `-1`(auto)이고, upstream 자체의 auto와 정확히 같이 4개 슬롯으로 해석되며, auto는 기본적으로 `kv_unified`를 활성화합니다. 통합 컨텍스트 예산 방식에서는 모든 슬롯이 `--ctx-size` 윈도 전체를 보면서, 스케줄러는 활성 시퀀스 전체에 걸쳐 하나의 공유 live-token 예산을 적용합니다. 명시적인 `--parallel N` 또는 명시적인 `--max-batch-size`는 `--kv-unified`도 명시되지 않는 한 슬롯별 분리된 윈도를 유지하며, `--no-kv-unified`는 분리된 윈도를 강제합니다. `--batch-size`는 `--prefill-chunk-size`에 `aliased`로 분류됩니다(같은 논리적 prefill 배치이며, ADR 0007 이후 upstream과 같은 2048 기본값). `--ubatch-size`는 `not_applicable`입니다(통합 메모리에는 물리적 마이크로 배치가 없으며, 값은 받아들여지고 시작 시 고지되며 `/props`에 보고됨). `--swa-full`은 `not_applicable`이며 설정되면 거부됩니다. sliding-window 패밀리는 체크포인트의 `sliding_window`로부터 자체 ring 캐시를 구성하고, 이 플래그가 upstream에서 확보해 주는 상태 연산은 ring 크기가 아니라 스케줄러가 소유한 캐시에 게이트되어 있기 때문입니다. 단일 대시 표기인 `-np`, `-kvu`, `-no-kvu`, `-ub`는 argv pre-pass를 통해 재작성됩니다.

### prompt cache는 모든 텍스트 라우트를 포괄합니다 (#1473)

이슈 #1473 이전에는 `--cache-prompt`와 요청별 `cache_prompt` 필드가 chat 형태의 라우트만 도달하는 캐시를 켜고 껐습니다. `/v1/chat/completions`, `/v1/messages`, `/v1/responses`는 prefix를 조회하고 되돌려 기증(donate)했지만, `/v1/completions`와 네이티브 `/completion`은 prompt-cache 요청 컨텍스트를 전혀 만들지 않았으므로 플래그가 무엇이든 매 요청마다 공유 prefix를 다시 prefill했습니다. 이제 두 raw-prompt 라우트 모두 같은 seam을 통해 컨텍스트를 만들며, 키는 모델 id와 고정된 raw-prompt 템플릿 시그니처이므로 raw 프롬프트가 chat 렌더링된 prefix를 가져다 쓸 수 없습니다(두 raw 라우트는 서로 하나의 버킷을 공유하는데, 같은 토큰화이고 템플릿이 없다는 점이 같으므로 올바릅니다). `cache_prompt`는 b10621 클라이언트가 보내는 네이티브 `/completion` 스키마에 선언되어 있으며, `false`이면 조회만이 아니라 조회와 기증 되돌리기를 모두 건너뛰도록 컨텍스트를 보류합니다. `/v1/completions`에는 그런 필드가 없으므로 서버 전체 스위치가 적용되며, 이제 chat 라우트와 똑같이 비스트리밍 응답과 스트리밍 usage 청크 모두에서 `usage.prompt_tokens_details.cached_tokens`를 보고합니다.

410 토큰 프롬프트로 `qwen3-0.6b-4bit`에서 측정한 결과: `/completion`은 cold에서 `timings.cache_n` 0 / `prompt_n` 410, 반복에서 400 / 10을 보고하고, `cache_prompt: false`는 바이트 단위로 동일한 텍스트와 함께 0 / 410으로 돌아가며, 네 번째 요청은 다시 히트하므로 opt-out은 저장소를 방해하지 않습니다. `tests/prompt_cache_compat_e2e.rs`는 chat 라우트가 이미 가졌던 것과 같은 cold / hit / opt-out / hit-again 계약으로 두 새 라우트를 검증합니다.

`--cache-ram`은 `--prompt-cache-capacity-bytes`에 `aliased`입니다(upstream의 `-1`과 `0` sentinel을 갖는 MiB이며, 요청된 값은 정확히 적용되고 설정되지 않은 기본값만 upstream의 8192에 대해 2048 MiB로 다릅니다). `--cont-batching`은 `supported`입니다. `-cb`와 `-nocb`는 argv pre-pass를 통해 재작성되며, `--no-cont-batching`은 이제 decode 폭만 고정하는 대신 prefill/decode 인터리빙을 완전히 중단하는데, 스케줄러의 mixed-step 게이트가 `max_batch_size > 1`과 AND되기 때문입니다. 이는 `params_base.cont_batching || batch.size() == 0`일 때만 대기 중인 프롬프트를 배치에 추가하는 upstream 자체의 게이트와 일치합니다.

### Control vector는 무시되지 않고 거부됩니다 (#1449)

b10621의 `--control-vector`와 `--control-vector-scaled`는 activation-steering 벡터를 로드하여 모델의 residual stream에 더합니다. mlxcel에는 control-vector 적용 경로가 없으며, 적용하지 않은 채 벡터를 받아들이면 배포가 무엇을 서빙한다고 믿는지를 조용히 바꿔 버리므로, 두 플래그는 (숨겨진 상태로) 파싱되지만 설정된 모든 벡터는 가장 가까운 mlxcel 기능으로 logit 수준 조향(`--lang-bias`, 요청별 `logit_bias`)을 명시하는 진단과 함께 모델 로드 전에 시작 시 실패합니다. 스케일 0도 여전히 설정된 벡터 집합이므로 부분 적용되지 않고 전체가 거부됩니다. `--control-vector-layer-range START END` 단독은 벡터가 없으면 아무것도 설정하지 않으므로, upstream과 정확히 같이 효과 없이 받아들여집니다. 벡터가 없는 구성만 서빙될 수 있으므로, control-vector 구성에 걸친 프롬프트/KV 캐시 재사용은 구조상 불가능합니다.

### 재개 가능한 스트림과 chat-completion 제어 (#1444)

b10621의 WebUI는 생성 도중 탭을 닫았다가 답변을 잃지 않고 다시 열 수 있습니다. `X-Conversation-Id` 헤더를 담은 스트리밍 요청은 SSE 바이트의 서버 측 복사본을 유지하며, 세 개의 라우트가 그 복사본을 관리합니다. mlxcel은 upstream과 같은 상수(4 MiB 버퍼와 front-drop, 완료 후 300초 TTL, 60초 GC)로 같은 수명 주기를 모든 스트리밍 표면에서 구현합니다. 해당 표면은 `/chat/completions`와 `/v1/chat/completions`, `/v1/completions`, 네이티브 `/completion`과 `/completions`, `/v1/responses`, `/v1/messages`입니다.

- `GET /v1/stream?conv_id=<id>&from=N`은 버퍼링된 바이트를 바이트 오프셋 `N`부터 재생한 뒤, 생성이 끝날 때까지 라이브 출력을 따라갑니다. 재생되는 바이트는 와이어 상의 SSE 프레임 그대로이므로, 클라이언트가 연결을 끊기 전에 받은 것은 재생의 prefix와 바이트 단위로 일치하며 그 오프셋부터 재개하면 정확히 나머지를 얻습니다. 버려진 버퍼 prefix 아래로 떨어진 오프셋은 upstream의 400 "Stream offset lost, please restart"로 응답합니다.
- `{"conversation_ids": [...]}`를 사용하는 `POST /v1/streams/lookup`은 호출자가 이미 알고 있는 id(정확히 일치하거나 `<id>::<model>` 변형)에 대해 `{conversation_id, is_done, total_bytes, started_at, completed_at}`를 보고하며, 서버는 질의받지 않은 세션을 절대 나열하지 않습니다.
- `DELETE /v1/stream?conv_id=<id>`는 명시적인 Stop입니다. 멱등적인 204이며, 연결 끊김이 사용하는 것과 같은 스케줄러 토큰으로 생성을 취소하고 버퍼를 제거합니다.

세션이 연결되어 있는 동안 클라이언트 연결 끊김은 의도적으로 생성을 취소하지 **않습니다**. 생성은 출력이 버퍼에 커밋된 채 자연스러운 중단 지점까지 실행되며, 이것이 재개를 무손실로 만듭니다. 헤더가 없으면 아무것도 바뀌지 않습니다. 연결 끊김은 여전히 시퀀스를 중단시키고 재생할 세션도 존재하지 않습니다.

실시간 제어도 같은 식별자를 사용합니다. `reasoning_control: true`를 설정한 chat 또는 completions 요청은 시퀀스의 thinking 트래커에 런타임 force-end 플래그를 무장(arm)시키며, `{"id": "<chatcmpl-...>", "action": "reasoning_end"}`를 사용하는 `POST /v1/chat/completions/control`이 이를 설정합니다. 다음에 샘플링되는 reasoning 토큰은 소진된 `--reasoning-budget`이 닫는 것과 똑같이 thinking 블록의 닫힘으로 대체되며, 이미 스트리밍된 이벤트는 영향을 받지 않습니다. 응답 계약은 upstream의 것입니다. id가 없거나 알 수 없는 action이면 400, 알 수 없는 id이거나 무장되지 않은 completion이면 200 `{"success": false, "message": ...}`, 플래그가 설정되면 200 `{"success": true}`입니다.

한 가지 의도적인 divergence가 이 항목들을 `supported`에서 제외하며, manifest에 `by_design`(kind `policy`)으로 기록되어 있습니다(구성된 모든 키가 upstream처럼 단일 신뢰 도메인을 이룬다고 결정할 때에만 재검토 가능). `--api-key`가 두 개 이상 구성되어 있으면 세션과 제어 항목은 그것을 만든 키로 범위가 한정됩니다. 다른 키의 `GET`은 알 수 없는 id와 같은 404로 응답하고, 그 lookup은 세션을 생략하며, 그 `DELETE`는 204 no-op이고, 그 제어 요청은 "no active completion"으로 응답하므로, 어떤 엔드포인트도 키 사이에서 존재 여부를 알아내는 oracle이 되지 않습니다. b10621은 구성된 모든 키를 동등하게 취급하지만, #1444는 한 키가 다른 키의 스트림을 조회하거나 제어할 수 없도록 요구하며, manifest는 그 차이를 숨기는 대신 기록합니다. 키가 0개 또는 1개이면 동작은 upstream과 동일합니다. mlxcel은 추가로 보관되는 완료 세션을 256개로 제한하여(가장 오래된 것부터 제거) 서로 다른 conversation id가 계속 바뀌어도 재생 버퍼가 제한을 유지하도록 합니다.

OpenAI Responses의 retrieve/cancel 표면(`GET /v1/responses/:id`, `POST /v1/responses/:id/cancel`)은 변경되지 않았으며 별도의 메커니즘으로 남아 있습니다. 이것과 스트림 수명 주기 사이의 변환은 주장하지 않습니다.
### Health, props, slots, metrics, 슬롯 영속화 (#1440)

`/health`와 `/v1/health`는 이제 b10621과 정확히 같이 동작합니다. 준비 완료 응답은 `200 {"status": "ok"}`뿐이고, 준비되지 않은 응답은 upstream의 `503 {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}` 엔벌로프이며, 부하는 더 이상 응답을 바꾸지 않습니다. **예전의 `503 {"status": "no slot available"}` 포화 신호를 위해 `/health`를 감시하던 배포는 `GET /slots?fail_on_no_slot=1`로 옮겨야 합니다.** 이는 유휴 슬롯이 없을 때 b10621의 `503 no slot available` 엔벌로프로 응답합니다. 예전의 풍부한 health 본문(배치 게이지, 관측성 카운터)을 파싱하던 probe는 이제 그 데이터를 `GET /slots`와 `GET /metrics`에서 찾을 수 있습니다.

`GET /props`는 upstream과 같이 무조건 마운트되며 b10621의 키 집합을 보고합니다. `default_generation_settings`는 이제 평평한 맵이 아니라 upstream의 `{params, n_ctx}` 형태입니다(**`default_generation_settings.temperature`를 읽던 클라이언트는 `default_generation_settings.params.temperature`를 읽어야 합니다**). 페이로드에는 모델 정체성(`model_alias`, `model_ftype`, `model_path`), `modalities`, `media_marker`, 세 개의 엔드포인트 토글, `ui` / `ui_settings`, `chat_template`과 `chat_template_caps`, `bos_token` / `eos_token`, `build_info`, `is_sleeping`, `cors_proxy_enabled`가 포함됩니다. `--ctx-size 0`이 모델 윈도를 도출할 때 `n_ctx`는 `/slots`와 `/v1/models`가 보고하는 것과 같은 양수의 유효 슬롯별 값이며, unified 모드에서는 나눈 몫이 아니라 모든 슬롯에 대해 구성된 윈도 전체입니다. mlxcel 고유의 해석된 설정 키는 최상위 확장으로 유지됩니다(`kv_unified`, `kv_cache_mode`, `kv_bits`, `speculative`, `capabilities`, 그리고 `geometry` 아래의 컨텍스트/배치 기하). `--props`는 이제 `POST /props`만 게이트하며, 이는 아무것도 바꾸지 않고 `{"success": true}`를 확인 응답하는데, upstream의 핸들러와 정확히 같습니다.

`GET /slots`는 실제 요청별 슬롯 레지스트리에서 b10621의 슬롯 객체를 보고합니다. 생성 요청은 스케줄러 prefill admission 시점부터(prefill 이벤트가 없는 백엔드에서는 첫 디코딩된 토큰부터) 끝날 때까지 `--parallel` 중 가장 낮은 빈 슬롯을 차지합니다. `id`, `is_processing`, `id_task`, 프롬프트 토큰 분할(`n_prompt_tokens`, `n_prompt_tokens_processed`, `n_prompt_tokens_cache`), 해석된 `params`, `next_token` 진행 블록은 chat completions, completions, Responses, Anthropic messages, 네이티브 completion 라우트에서 라이브 값입니다. prefill 동안 `n_prompt_tokens_processed`는 처리된 토큰에서 캐시된 토큰을 뺀 값으로 증가하며, decode 중과 종료 후에 `n_prompt_tokens`는 프롬프트 더하기 수락된 디코딩 토큰입니다. `prompt`와 `generated`는 `LLAMA_SERVER_SLOTS_DEBUG`가 설정되지 않으면 가려지며, 이는 b10621 자체의 디버그 게이트입니다. 비활성화된 엔드포인트(`--no-slots`)는 이전의 404 대신 upstream의 501 진단으로 응답하며, `--metrics` 없는 `/metrics`와 `--slot-save-path` 없는 슬롯 action도 마찬가지입니다. 슬롯은 라우트가 요청을 받아들일 때가 아니라 요청의 첫 진행 신호에서 확보되므로, 여전히 스케줄러 큐에서 대기 중인 요청은 슬롯을 차지하지 않는다는 규칙이 유지됩니다. 의도적 차이로 기록된 채 남은 것은 `params` 키의 부분집합이며, 이는 `/props`를 따릅니다. mlxcel이 동작시키지 않는 키는 만들어 내는 대신 생략됩니다.

`--slot-save-path DIR`은 upstream의 응답 스키마, 파일명 검증(`fs_validate_filename`을 규칙 단위로 이식), 원자적 쓰기, 순회 및 심볼릭 링크 탈출을 거부하는 정규 경로 한정과 함께 `POST /slots/:id_slot?action=save|restore|erase`를 활성화합니다. restore는 저장한 모델 및 토크나이저 정체성에 묶입니다. 영속화되는 것은 슬롯의 KV 캐시가 아니라 토큰 스트림이며, restore는 토큰을 복원하고 다음 요청이 다시 prefill하거나(또는 서버 자체의 prompt cache에서 채택) 하는데, 이것이 항목에 기록된 divergence입니다.

`GET /metrics`는 b10621의 `llamacpp:` 카운터 및 게이지 패밀리로 이름 하나하나 그대로 시작하고, `Process-Start-Time-Unix` 헤더를 담으며, 처리량 게이지를 두 번의 scrape 사이 윈도에 대해 평균내므로 llama-server용으로 작성된 Prometheus scrape 설정이 그대로 동작합니다. `mlxcel_` 패밀리는 같은 본문에서 그 뒤를 잇습니다. `/props`, `/models`, `/health`와 함께 유휴 sleep 중에도 응답합니다.

### Vertex AI (GCP) custom-container 호환성 (#1456)

b10621은 Google Cloud Vertex AI custom container를 순전히 `AIP_*` 환경 변수만으로 서빙하며, mlxcel도 이제 두 서버 바이너리에서 같은 방식으로 동작합니다. `AIP_MODE=PREDICTION`일 때: `AIP_HTTP_PORT`(기본값 8080)는 기록되는 경고와 함께 `--port`를 덮어쓰고, `AIP_HEALTH_ROUTE`(앞의 슬래시가 보장됨)는 health 핸들러의 GET 별칭이 되며, `AIP_PREDICT_ROUTE`(기본값 `/predict`)는 prediction fan-out을 마운트합니다. 등록된 라우트와 충돌하는 predict 경로는 모델 로드 전에 시작 시 실패합니다. `AIP_MODE`가 설정되지 않으면 아무것도 등록되지 않으며 변수는 효과가 없습니다.

`POST /predict`는 최대 128개 항목의 `{"instances": [...]}`를 받아들입니다. 각 인스턴스는 `@requestFormat`에 대상 라우트를 명시합니다. 등록된 라우트의 camelCase 별칭(`chatCompletions`, `embeddings`, `applyTemplate`, ...)이거나 등록된 경로를 그대로 쓸 수 있으며, 이 필드는 제거되고 `stream` 필드는 경고와 함께 강제로 꺼지며, 나머지는 predict 요청 자체의 헤더와 함께 구성된 라우터를 통해 프로세스 내에서 디스패치되므로 API 키 인증, 요청 검증, 큐 admission이 직접 호출과 정확히 같이 적용됩니다. 결과는 요청 순서대로 `{"predictions": [...]}`로 반환되며, 인스턴스별 실패는 해당 슬롯의 오류 객체로 나타납니다. 별칭 테이블은 라우터 자체가 마운트하는 것과 같은 라우트 인벤토리에서 도출되므로, 새로 추가된 라우트는 자동으로 별칭이 생깁니다.

두 가지 내부 사항은 응답을 바꾸지 않으면서 다르며, divergence가 아니라 manifest notes에 기록됩니다. 인스턴스 실행은 동시 디스패치 8개로 제한되며(upstream은 모든 인스턴스를 한꺼번에 시작하고, 어느 쪽이든 순서는 보존됨), `completions`처럼 별칭이 충돌하는 경우(`/completions` 네이티브 대 `/v1/completions` OpenAI)는 등록 순서에 따라 결정적으로 `/v1` 라우트로 해석되는 반면, upstream은 정렬되지 않은 맵을 순회하므로 어느 쪽이 이기는지 명시되어 있지 않습니다. predict 라우트도 health 별칭도, b10621과 마찬가지로 공개 엔드포인트가 아닙니다. API 키가 구성되어 있으면 Vertex AI는 키를 제시해야 합니다.

`AIP_*` 변수는 [`environment-variables.md`](environment-variables.md)에 문서화되어 있습니다.

### Web UI, tools, MCP, CORS proxy, agent 모드 (#1435, #1838)

이 그룹의 모든 것은 b10621의 내장 브라우저 UI를 위해 존재합니다. `--ui` / `--webui`는 `/`에서 SvelteKit 번들을 제공하고, `--path`는 이를 디렉터리로 대체하며, `--tools`와 `--tools-runtime`은 `/tools`를 통해 서버가 실행하는 도구를 해당 UI에 노출하고(upstream 자체의 개발자 문서는 이 엔드포인트를 UI 내부용으로 표시함), `--mcp-servers-config` / `--mcp-servers-json`은 stdio MCP 자식 프로세스로 같은 엔드포인트에 공급하고, `--ui-mcp-proxy`는 브라우저가 원격 MCP 서버에 도달할 수 있도록 범용 `/cors-proxy` URL 프록시를 열며, `--agent`는 프록시와 모든 도구를 한꺼번에 켭니다. mlxcel은 `--ui` / `--webui`를 번들된 WebUI의 별칭으로 받아들이고 셸을 `${api-prefix}/webui/`에서 제공하며 `/`를 health 엔드포인트로 유지합니다. 여전히 서버 측 도구를 실행하지 않고, MCP 자식 프로세스를 생성하지 않으며, 임의의 정적 디렉터리를 제공하지 않고, 범용 CORS 프록시를 노출하지 않습니다.

구체적으로, 두 서버 바이너리 모두에서: `--ui` / `--webui`는 번들된 WebUI를 활성화하고, `--no-ui` / `--no-webui`는 비활성화하며, `LLAMA_ARG_UI`는 b10621의 `parse_bool_value` 규칙과 `LLAMA_ARG_NO_UI` 별칭으로 해석됩니다. CLI 플래그가 우선하며 반복된 활성화/비활성화 형태는 last-wins 파싱을 사용합니다. 지원되지 않는 인접 형태(`--ui-config`, `--ui-config-file`, `--path`, `--tools`, `--tools-runtime`, `--mcp-servers-config`, `--mcp-servers-json`, `--ui-mcp-proxy`, `--webui-mcp-proxy`, `--agent`, 단일 대시 `-ag`)는 여전히 모델 로드 전에 지원되는 대안을 명시하는 한 줄 진단과 함께 시작 시 실패합니다. 지원되지 않는 표면에 대한 부정 형태(`--no-agent`, `--no-ui-mcp-proxy`, `--no-webui-mcp-proxy`)는 계속 효과가 없습니다. `--webui`와 `--no-webui`는 mlxcel 고유의 제품 플래그이며, 이 그룹에서 `--help`의 `WebUI Options` 아래에 나열되는 유일한 표기입니다(#1900). `--ui`, `--no-ui`, 그리고 거부되거나 효과가 없는 모든 형태는 shard의 `mlxcel.hidden` 클레임이 기록하듯 계속 숨겨집니다. WebUI 자체는 [docs/webui/README.md](webui/README.md)에 문서화되어 있습니다.

`GET`/`POST /tools`와 `GET`/`POST /cors-proxy`는 b10621 자체의 비활성화된 기능 stub으로 마운트됩니다. `{"error":{"message":"this feature is disabled","type":"feature_disabled"}}`를 담은 403이며, 바이트 형태가 upstream의 `res_403`과 같습니다. 이 기능을 꺼 둔 llama-server 배포의 클라이언트는 여기서도 동일한 응답을 봅니다. 이를 켜 두었던 배포는 기록된 divergence입니다. CORS 프록시는 구현되지 않고 거부되므로, 그 SSRF 표면(loopback, link-local, metadata-service, DNS-rebinding 도달 가능성)은 구조상 여기에 존재하지 않습니다. chat-completion 요청에서 클라이언트가 선언한 `tools`는 무관하며 계속 동작합니다. mlxcel은 도구 호출에 대한 모델 출력을 파싱하지만 서버 측에서는 아무것도 실행하지 않습니다. OpenAI Responses의 내장 도구 계약으로 표현되는 서버 측 MCP 도구 루프는 b10621 호환성 밖의 제품 기능으로서 #1457에서 별도로 추적됩니다.
### `--models-dir`은 router 모드이며, 저장소 루트는 `--model-store-root`로 이동했습니다 (#1438)

`--sleep-idle-seconds`는 #1440 이후 제공되며, manifest의 마지막 `deferred` 슬롯을 가져간 항목입니다. upstream의 sleep은 유휴 윈도 이후 컨텍스트와 모델을 파괴하고, `/props`, `/models`, `/metrics`의 스냅샷을 만들며, 요청이 서버를 깨우고 `load_model`이 끝날 때까지 다른 모든 라우트를 차단합니다. `-1`(기본값)과 모든 음수 값은 이를 비활성화하고, `0`은 이를 무장시키며, 이는 upstream 자체의 `< 0` 게이트입니다.

mlxcel은 모델 provider를 교체하는 대신 서빙 **워커**를 재웁니다. 스케줄러의 idle arm은 이미 요청 채널에서 차단되므로, 구성된 윈도는 이를 만료 시 루프에서 반환하는 제한된 대기로 바꿉니다. 그러면 워커는 스케줄러를 drop하여 그것이 소유한 `LoadedModel`과 함께 가중치를 해제하고, prompt-prefix 캐시를 비우며(분리된 KV 블록이 sleep을 넘어 살아남아 그 목적의 절반을 무력화하는 별도의 핸들), 같은 채널에서 대기하다가 다음 요청에서 다시 로드하고 그 요청을 재구성된 스케줄러에 넘깁니다. 따라서 깨어나는 것은 그 요청 자체의 prefill이며, upstream의 차단형 wake와 정확히 같고, idle arm에 도달했다는 것은 이미 decode 배치도 대기 중인 prefill도 없다는 뜻이므로 진행 중인 어떤 것도 잃을 수 없습니다. `/health`는 upstream의 것과 마찬가지로 sleep 중에도 200으로 계속 응답합니다. 실제 체크포인트에서 `--sleep-idle-seconds 5`로 측정한 결과: `is_sleeping`은 윈도 이후 true가 되고 completion 요청 후 0.39초 만에 다시 false가 됩니다.

하나의 차이는 `by_design`으로 기록됩니다. **upstream의 네 개보다 많은 라우트가 sleep 중에도 도달 가능하며, 서버를 깨우지 않고 응답합니다.** b10621은 자체 핸들러가 방금 파괴한 컨텍스트를 읽기 때문에 정확히 그 이유로 `/props`, `/models`, `/metrics`를 캐시합니다. mlxcel의 핸들러는 `AppState`(설정, 토크나이저, chat template, metrics 핸들)를 읽으며, 그중 어느 것도 sleep이 해제하지 않으므로 스냅샷 없이 라이브로 응답합니다. 이는 `/tokenize`, `/detokenize`, `/apply-template` 및 다른 프롬프트 검사 라우트에도 마찬가지이며, upstream은 이들에 응답하려면 서버를 깨워야 하지만 mlxcel은 그럴 필요가 없습니다. 이들이 깨울 이유가 없습니다. 깨움은 모델을 복원하기 위해 존재하는데, 이들 중 어느 것도 모델을 읽지 않기 때문입니다.

슬롯 save와 restore는 KV 이미지가 아니라 슬롯의 토큰 스트림과 모델 및 토크나이저 정체성을 영속화하며, 처리 중인 슬롯을 대상으로 한 action은 upstream이 슬롯이 비워질 때까지 미루는 것과 달리 `503`으로 응답합니다. 둘 다 manifest 항목에 근거와 함께 `by_design`으로 기록되어 있습니다. KV 이미지는 파일을 이를 기록한 캐시 모드, 양자화 방식, paged 블록 크기에 묶어 버리지만, 토큰 스트림은 그중 어느 것으로든 복원되고 prefill로 같은 KV를 다시 구성하기 때문입니다.

`--models-dir` / `LLAMA_ARG_MODELS_DIR`은 이제 `mlxcel-server`와 `mlxcel serve`에서 b10621의 router 의미를 가집니다. 모델 인자 없이 시작하면 router 표면을 제공하며, b10621이 자식 프로세스를 생성하는 곳에서 mlxcel은 프로세스 내 모델 풀을 사용합니다. 풀은 세 가지 소스를 b10621의 우선순위(cache < models-dir < preset)로 조정합니다. 첫째는 모델 캐시(`--model-store-root` / `MLXCEL_MODELS_DIR` / `MLXCEL_CACHE_DIR`이 가리키는 mlxcel 저장소)이며, 그 `<owner>/<name>` 스냅샷은 제거 가능한 `source: "cache"` 항목으로 나열됩니다. 둘째는 `--models-dir` 디렉터리로, `config.json`을 가진 각 직계 하위 디렉터리가 디렉터리 이름으로 명명된 하나의 모델입니다(디렉터리 밖으로 해석되는 심볼릭 링크는 건너뜁니다). 셋째는 `--models-preset` INI 섹션입니다. 요청의 `model` 이름은 발견된 레지스트리(및 preset 별칭)에 대해서만 해석되며, 관리 라우트는 다른 모든 것과 같은 API 키 뒤에 있습니다. `GET /models`(및 `/v1/models`)는 `status`, `aliases`, `tags`, `architecture`, `source`, `can_remove`를 포함한 b10621 router 인벤토리를 보고하고, `?reload=1`은 다시 스캔하며, `POST /models/load` / `POST /models/unload`는 upstream의 정확한 거부로 응답하고, `--models-max`(기본값 4)는 LRU 제거로 로드된 집합을 제한하며, `--models-autoload`와 요청별 `?autoload=` 스위치는 온디맨드 로드를 제어합니다. 요청은 POST에서는 JSON 본문의 `model`, GET에서는 `?model=`로 라우팅되므로, `GET /slots?model=X` 또는 `GET /props?model=X`는 이름이 지정된 모델 자체의 엔드포인트에 도달하며, 이는 upstream의 프록시 계약과 정확히 같습니다.

`POST /models`는 새 모델을 캐시에 다운로드합니다. 이름이 검증되고(비어 있지 않음, 올바른 형식의 repo id, 그다음 hub에 대한 동기 메타데이터 probe로, upstream의 검증 후 다운로드 순서이며, owner가 없는 이름은 먼저 `mlx-community/<name>`으로 확장되고 이는 `-m`이 적용하는 확장과 같음), 라우트는 `{"success": true}`로 응답하고, 백그라운드 태스크가 MLX 스냅샷을 가져오는 동안 항목은 URL별 진행률과 함께 `status: "downloading"`으로 나열됩니다. `DELETE /models?model=X`는 캐시에서 온 모델을 디스크에서 제거하며(진행 중인 다운로드를 취소하거나 실행 중인 인스턴스를 먼저 언로드함), 그 외 모든 것에 대해서는 upstream의 정확한 거부로 응답합니다. 삭제는 저장소 루트에 대해 containment 검사를 받으므로, HTTP로 전달된 이름이 캐시 밖의 것을 삭제할 수는 없습니다. `GET /models/sse`는 `{"model", "event"[, "data"]}` 어휘 전체를 스트리밍합니다. `models_reload`, `model_status`, `status_change`, `download_progress`, `download_finished`, `download_failed`, `model_remove`입니다.

`--models-preset`은 b10621의 INI preset을 로드합니다. `[name]` 섹션은 모델을 정의하거나(`model =`은 체크포인트 디렉터리를, `hf-repo =`는 캐시 스냅샷을 지정) 발견된 모델 위에 설정을 덧씌우고, `[*]`는 모든 모델 아래에 계단식으로 적용되는 전역 preset이며, 키는 대시 없는 long-option 표기 또는 `LLAMA_ARG_*` 환경 변수 표기를 받고, 각 섹션은 CLI와 같은 파이프라인으로 다시 해석되는 router 시작 설정의 모델별 복제본 위에 덧씌워집니다(명시적으로 주어진 router CLI 플래그가 이기며, 이는 upstream 자체의 덧씌우기 순서입니다). preset 전용 `load-on-startup`과 `dedup-cache-models` 옵션은 적용되고 `stop-timeout`은 효과 없이 파싱됩니다. 변환되는 집합(`model`, `hf-repo`, `alias`, `tags`, `ctx-size`, `parallel`, `temp`, `top-k`, `top-p`, `min-p`, `seed`, `n-predict`) 밖의 키는 조용히 무시되는 대신 키와 섹션을 명시하는 진단과 함께 시작 시 실패하며, `--models-preset`을 모델 인자와 함께 주는 것도 preset이 router 모드 밖에서는 아무것도 조종하지 않으므로 마찬가지로 시작 시 실패합니다.

**예전의 mlxcel 의미는 이동했습니다.** `--models-dir`은 이전에 `-m <owner/name>` 해석과 자동 다운로드가 사용하는 로컬 모델 저장소 루트를 가리켰으나, 이제 두 서버 바이너리 모두에서 `--model-store-root`가 그 역할을 합니다(`MLXCEL_MODELS_DIR`은 여전히 동작하며, `mlxcel download` / `list` / `rm` / `generate` 하위 명령은 llama-server 표면이었던 적이 없으므로 `--models-dir` 표기를 유지합니다). `--models-dir`과 모델 인자를 함께 쓰는 서버 명령줄은 어느 한쪽 의미를 조용히 고르는 대신 대체 플래그를 명시하는 진단과 함께 시작 시 실패합니다. `--tags`는 `/v1/models` 모델 객체가 보고하는 참고용 태그를 설정합니다.

단일 모델 `GET /models` / `GET /v1/models`도 b10621의 형태로 이동했습니다. OpenAI `data` 항목은 `aliases`(전체 `--alias` 목록), `tags`, `owned_by: "llamacpp"`(이전에는 `"user"`), 그리고 `config.json`과 safetensors 헤더에서 도출한 체크포인트 사실의 `meta` 블록(`n_params`는 양자화된 `U32` 페이로드를 선언된 비트 폭으로 풀어서 계산)을 담으며, 그 옆에 b10621의 Ollama 호환 `models` 블록이 있고 그 `details.format`은 정직하게 `"safetensors"`로 표시됩니다.

### Multi-adapter LoRA, unfused로 서빙 (#1439)

`--lora`는 쉼표로 구분된 adapter 경로를 받고(각각 scale 1.0이며, mlxcel 고유의 `--adapter` 표기는 별칭으로 유지), `--lora-scaled`는 쉼표로 구분된 `FNAME:SCALE` 쌍을 받습니다. adapter는 **unfused**로 서빙됩니다. 각 adapter는 linear 레이어가 forward마다 더하는 저랭크 항 `scale * (x @ A) @ B`를 기여하며, 사용자 scale은 adapter 자체의 `alpha / r`에 곱해지고, 이는 upstream이 적용하는 곱과 정확히 같습니다. 기본 가중치는 결코 수정되지 않으므로 scale은 라이브입니다. 서빙 레이어가 쓰고 레이어가 읽는 공유 핸들 뒤에 있습니다. 숫자가 아니거나, NaN이거나, 무한대인 scale, 없는 adapter 디렉터리, 텐서 shape가 기본 projection에 맞지 않는 adapter, 모델이 공유 linear 생성자 밖에서 만드는 adapter 레이어(적용되지 않을 것이므로), tied-embeddings 체크포인트의 `lm_head` 쌍, Llama decoder(`llama`, `tie_word_embeddings: true`인 `qwen2`)와 Qwen3이 아닌 패밀리의 tied 체크포인트에 있는 `embed_tokens` 쌍(이런 adapter는 먼저 `mlx_lm.fuse`로 병합하십시오. 지원되는 패밀리에서 mlx-lm의 `LoRAEmbedding` 쌍은 adapter의 단일 scale 뒤에서 임베딩 조회와 tied head에 모두 unfused로 적용됩니다), mixture-of-experts expert adapter(expert별 3차원 `lora_a` / `lora_b`, 미지원), 그리고 multi-adapter 또는 scaled 로딩을 tensor/pipeline parallelism과 결합하는 것은 모두 시작 시 실패합니다. `GET /lora-adapters`는 모든 adapter를 b10621의 항목 형태(`id`, `path`, `scale`, `task_name`, `prompt_prefix`)로, 라이브 서버 기본 scale과 함께 보고하므로 교체가 거기서 보입니다.

`POST /lora-adapters`는 upstream의 규칙(나열된 id는 scale을 설정하고, 나열되지 않은 id는 0.0으로 떨어지며, 알 수 없는 id는 무시됨)으로 해석되는 새 scale 벡터를 런타임에 적용합니다. 요청별 네이티브 `lora` 필드도 같은 방식으로 해석되며 해당 요청에만 적용됩니다. 둘 다 upstream의 타이밍을 따릅니다. 요청은 admission 시점에 자신의 scale을 스냅샷하고, 스케줄러는 배치마다 한 번 실행 중인 그룹의 스냅샷으로 공유 핸들을 쓰며, b10621 자체의 `can_batch_with` 규칙은 배치를 하나의 스냅샷으로 유지하므로, 교체가 이미 진행 중인 생성을 바꾸는 일은 없고 서로 다른 adapter 집합을 지정한 두 동시 요청이 서로에게 새어 들어갈 수 없습니다. `--lora-init-without-apply`는 시작 지점에서의 같은 메커니즘입니다. adapter는 scale 0.0으로 로드 및 검증되고, 이후 POST로 재시작 없이 올릴 수 있습니다. scale 0.0의 항은 어떤 연산도 추가하지 않으므로, 적용되지 않은 adapter를 보유한 서버는 adapter 없이 시작한 서버와 바이트 단위로 동일하게 응답합니다.

**`--lora-fuse`(mlxcel 고유)는 예전 동작으로 되돌립니다.** adapter가 로드 시 기본 가중치에 구워 넣어지며, 이는 decode에서 비용이 들지 않지만 scale이 프로세스 동안 고정되므로, `POST /lora-adapters`와 효과가 있는 요청별 `lora`는 플래그를 명시하는 진단과 함께 거부됩니다. 이것은 b10621 표기가 아니며 b10621 명령줄에서는 도달할 수 없습니다. tensor-parallel 및 pipeline 로더는 이를 함의하는데, 런타임 항이 아직 따라가지 못하는 방식으로 기본 가중치를 샤딩하기 때문입니다. `--no-batch`는 기본 가중치를 조용히 서빙하는 대신 adapter를 아예 거부하는데, 레거시 순차 워커가 두 경로보다 먼저 만들어졌기 때문입니다.

### Mirostat, dynamic temperature, adaptive-p, min_keep, logit bias, 확률 리포트, seed fold (#1485)

b10621의 나머지 샘플링 표면은 Rust sampler 단계로 구현되었으며, C++이나 브리지 변경은 필요하지 않았습니다. Mirostat(`--mirostat` 1/2, tau/eta, 네이티브 필드)은 활성화된 동안 전체 체인을 대체하며, 이는 b10621의 `common_sampler_init`이 그때 sampler 목록을 건너뛰는 것과 정확히 같습니다. penalty, DRY, 모든 truncation 필터가 우회되고, 시퀀스별 surprise 목표 `mu`가 단계에 걸쳐 이어집니다. Dynamic temperature(`--dynatemp-range` / `--dynatemp-exp`), `min_keep` 하한, adaptive-p는 fused 경로에 슬롯이 없는 파라미터를 사용하면서 고정된 b10621 필터 순서를 재현하는 Rust 확장 체인에서 실행되며, 이들 중 아무것도 사용하지 않는 모든 구성은 fused 경로에서 바이트 단위로 동일하게 유지됩니다. Adaptive-p는 sampler 목록이 `adaptive_p` 단계를 명시할 때만 활성화되며(upstream 자체의 규칙), 따라서 `--samplers` / `--sampler-seq` / `samplers` 필드는 이제 고정된 기본 순서에 선택적으로 `adaptive_p` / `a`를 확장한 것을 받아들이고, 다른 모든 순서는 여전히 거부됩니다. `-l` / `--logit-bias`와 네이티브 `logit_bias` 필드는 upstream의 형태(쌍 또는 객체, 문자열 키는 토큰화됨, `false`는 금지)로 공유 토큰 bias 맵에 반영되며, `n_probs`(별칭 `logprobs`) / `post_sampling_probs`는 네이티브 라우트에서 원시 logit 기반 pre-sampling 또는 선형 확률의 post-chain으로 b10621의 `completion_probabilities` 리포트를 생성합니다.

기존 배포가 알아챌 수 있는 세 가지 동작 변경이 있습니다. 첫째, seed 도메인이 플래그와 모든 라우트의 `seed` 필드에서 b10621의 uint32 공간으로 접힙니다(fold). `-1`(및 `4294967295`)은 랜덤 sentinel로 유지되지만, 이전에 랜덤이던 `--seed -2`는 이제 결정적 seed `4294967294`이며, -1 미만 필드 값에 대한 #1485 이전의 422는 사라졌습니다. 둘째, `--dry-sequence-breaker`는 b10621의 의미를 가집니다. 플래그가 없고 DRY가 활성화되어 있으면 이전에 mlxcel은 breaker 없이 DRY를 실행했지만, 이제는 upstream의 기본 breaker 집합(`\n`, `:`, `"`, `*`)이 적용되고, `none`은 명시적인 빈 집합이며, breaker가 정확히 하나의 토큰으로 인코딩될 필요는 더 이상 없고(멀티 토큰 breaker에서 시작이 더는 실패하지 않음), breaker 토큰 데이터는 디코딩된 텍스트가 breaker를 담고 있는 토큰을 어휘에서 스캔하여 도출되기 때문이며 이는 upstream의 `get_overlapping_token_sequences`입니다. 네이티브 `dry_sequence_breakers` 필드는 이제 비어 있지 않은 문자열 배열을 받으며(upstream의 값 범위와 정확히 같은 오류 문구), OpenAI 형태의 라우트는 mlxcel 고유의 토큰 id 배열을 유지합니다. 셋째, `/props`와 `generation_settings`는 새로 해석된 설정을 보고하며, `dry_sequence_breakers`는 해석된 토큰 id가 아니라 breaker 문자열로 보고됩니다.

`--temp`, `--top-k`, `--top-p`는 `by_design`으로 기록됩니다. 플래그가 없으면 서버 기본값은 여전히 체크포인트의 `generation_config.json`에서 해석되며(mlx-lm 동등성, 의도된 mlxcel 동작), 플래그가 주어지면 동작은 b10621과 정확히 일치합니다.

### GBNF 문법과 lazy trigger (#1485)

`--grammar`, `--grammar-file`, `--json-schema`, `--json-schema-file`이 제공되며, `POST /completion`과 `POST /infill`의 네이티브 `json_schema` 필드(별칭 `grammar`), `grammar_lazy`, `grammar_triggers`, `preserved_tokens`도 마찬가지입니다. 문법은 실제로 생성을 제약합니다. 모든 단계의 logit은 출력을 파싱 가능하게 유지하는 토큰으로 마스킹되며, 문법이 그렇지 않으면 요청은 사전에 거부됩니다.

GBNF는 b10621 자체의 문법으로 파싱되며, 구문 요약만으로는 틀리기 쉬운 부분도 포함합니다. `_`는 유효한 식별자 문자가 아니지만 `-`와 선행 숫자는 유효하고, `#`은 줄 끝까지 이어지며 공백으로 취급되고, 빈 alternative는 유효하며 의미가 있고, `x{0}`은 항목을 지우며, `x{0,5000}`처럼 지나치게 큰 최댓값은 조용히 무제한으로 넓혀지지만 지나치게 큰 최솟값은 오류이고, 반복 예산은 중첩에 걸쳐 곱셈으로 적용되며, 반복 연산자는 연쇄될 수 있고, `.`은 개행을 포함한 모든 코드 포인트와 일치하며, 좌재귀는 도달 가능한 규칙만이 아니라 모든 규칙에 걸쳐 거부됩니다. 토큰 terminal도 동작합니다. `<[1000]>`, `<think>`와 그 `!` 부정은 텍스트로 잘못 파싱되는 대신 토큰 수준 요소가 됩니다.

세 가지가 의도적으로 b10621과 다릅니다. GBNF 소스는 256 KiB로 제한되며, 더 큰 문법은 제한을 명시하는 진단과 함께 거부됩니다. upstream에는 소스 크기 제한이 없으므로, 반복 없는 단순한 규칙을 많이 모아 만든 큰 문법은 거기서는 받아들여지고 여기서는 거부됩니다. upstream 자체의 보호 장치는 곱셈 방식의 반복 예산이며, 이는 문법이 얼마나 *확장*될 수 있는지를 제한하므로 이미 mlxcel의 4 MiB lowered-grammar 상한을 제한하지만, 소스 바이트에 대해서는 아무 말도 하지 않습니다. 나머지 둘입니다. `json_schema` 값은 먼저 GBNF로 변환되는 대신 llguidance의 JSON-Schema 프런트엔드를 통해 컴파일되므로, upstream이 변환된 문법 텍스트를 보고하는 스키마 요청에서 `generation_settings.grammar`는 비어 있으며, 출력은 어느 쪽이든 스키마에 부합합니다. 그리고 lazy-grammar trigger 패턴은 `std::regex`가 아니라 `fancy-regex`로 컴파일되며, 컴파일할 수 없는 패턴은 sampler 초기화 시 예외를 던지는 대신 400으로 거부됩니다. trigger 일치가 토큰 안에서 시작될 때는 matcher가 토큰 전체 단위로 진행하므로, 일치 오프셋부터의 버퍼링된 바이트는 재생(replay) 전에 다시 토큰화됩니다.

우선순위는 자체 필드 설명이 아니라 b10621의 코드를 따릅니다. 스키마 분기는 `json_schema`가 존재하고 **그리고** `grammar`가 없을 때만 선택되므로, `{"grammar": "", "json_schema": {...}}`는 요청 문법 없이 실행되며 `--grammar` 기본값이 있으면 그대로 둡니다. 문자열이 아닌 `grammar`는 upstream의 `json_value`가 빈 기본값으로 fallback하는 것과 같은 방식으로 무시됩니다. 시작 실패도 upstream의 구분을 따릅니다. 잘못된 형식의 `--json-schema`는 upstream이 인자 핸들러 안에서 변환하기 때문에 서버가 listen하기 전에 중단되지만, 잘못된 형식의 `--grammar`는 upstream이 요청이 sampler를 구성할 때까지 GBNF를 파싱하지 않기 때문에 시작 오류를 기록한 뒤 제약된 요청마다 거부합니다.

## Sharding

manifest는 epic #1431의 동시 구현 체인이 서로 겹치지 않는 파일을 편집하도록 영역별로 shard됩니다. 소유권은 산문이 아니라 기계가 읽을 수 있습니다. `pin.json`의 `shards` 맵은 shard마다 그 안의 항목을 소유할 수 있는 구현 이슈 번호의 집합을 기록하며(예를 들어 `shards["authentication"].owners == [1437]`), `scripts/ci/check_llama_compat_manifest.py`는 `issue`가 자신의 shard owner 집합의 구성원이 아닌 항목을 실패시킵니다. 이것이 두 개의 동시 체인이 같은 파일을 편집하는 것을 막는 방법입니다. 리뷰어뿐 아니라 파일 자체가 두 번째 체인의 항목을 거부합니다.

| Shard | 체인 | 소유 이슈 |
|---|---|---|
| `transport-tls-cors.json` | A | #1432 |
| `authentication.json` | A | #1437 |
| `routes.json` | A | #1438, #1440, #1441, #1442, #1452, #1466, #1477 |
| `embeddings-and-rerank.json` | A | #1452 |
| `observability-and-slots.json` | A | #1440 |
| `router-models.json` | A | #1438 |
| `lora-adapters.json` | A | #1439 |
| `model-source.json` | B | #1434, #1438 |
| `ggml-runtime.json` | B | #1445 |
| `chat-templates.json` | B | #1447, #1470 |
| `multimodal-and-audio.json` | B | #1451, #1446 |
| `ui-tools-mcp-gcp.json` | B | #1435, #1456 |
| `streams-and-realtime.json` | B | #1444 |
| `runtime-and-context.json` | C | #1449, #1450, #1453, #1472, #1473 |
| `sampling-and-grammar.json` | C | #1436, #1377, #1466, #1485 |
| `speculative.json` | C | #1433 |
| `logging-and-presets.json` | C | #1448 |

라우트와 네이티브 요청 필드는 그것을 소유한 이슈의 shard에 있으므로(예를 들어 audio-transcription 라우트는 `multimodal-and-audio.json`에, 네이티브 샘플링 필드는 `sampling-and-grammar.json`에 있음), 체인이 다른 체인의 파일을 편집할 필요가 전혀 없습니다.

## 적용(Enforcement)

세 개의 게이트가 manifest와 바이너리를 함께 묶어 두며, 세 개 모두 CI와 `make verify`에서 실행됩니다.

1. `scripts/ci/check_llama_compat_manifest.py`(`llama-compat manifest` CI 잡, `make verify-llama-compat`): 개수, 상태, 이슈 링크, shard 소유권, 항목 키 allowlist, `mlxcel` claim 키 allowlist, `divergence` 규칙, 테스트 id, 정규 직렬화에 대한 오프라인 구조 검증입니다. 모든 항목의 `issue`는 자신의 shard의 `pin.json` owner 집합에 속해야 하고, `mlxcel` 블록은 인정되는 아홉 개의 키만 사용할 수 있으며, `supported` 항목은 divergence를 기록할 수 없고, `by_design` 항목은 비어 있지 않은 `divergence`, `notes`, 해석되는 `test` 포인터, 올바른 형식의 `rationale`을 가져야 합니다(다른 모든 상태에서는 `rationale`이 null). CI 잡은 `--check-issues-open`을 추가하므로 닫힌 이슈를 가리키는 `deferred` 항목은 실패하며, `by_design` 항목은 이 검사에 절대 포함되지 않아서, 남은 divergence가 영구적으로 기록되면 이슈를 닫을 수 있습니다. `scripts/ci/check_llama_compat_manifest_test.sh`는 이 validator 자체의 부정(negative) 커버리지입니다. manifest의 임시 복사본(`--manifest-dir`로 전달)을 변형하고 각 규칙이 실제로 이를 거부하는지 단언하므로, 규칙이 항상 통과만 하는 것으로 퇴화할 수 없습니다.
2. `tests/llama_compat_manifest.rs`: 숨겨진 `--dump-flag-surface` 기계 인터페이스를 통해, 숨겨진 호환성 인자를 포함하여 `mlxcel serve`와 `mlxcel-server` 둘 다의 실제 clap 표면에 대해 모든 option claim을 검증하며, 반대 방향도 검증합니다. claim이 없는 option 항목은 어느 바이너리에서도 받아들여지지 않아야 하므로, 항목을 전환하지 않고 b10621 플래그를 추가하면 실패합니다. 또한 sentinel 자체가 `--help`에 절대 렌더링되지 않는지도 단언합니다. 아카이브 게이트가 걸린 전체 인벤토리 적합성 테스트도 포함합니다. `MLXCEL_LLAMA_B10621_DIR`을 압축 해제된 공식 아카이브 디렉터리로 설정하면 실제 `llama-server --help`에서 option 인벤토리를 다시 도출하여 정확히 비교합니다(CI는 이를 건너뛰며, 아카이브를 결코 다운로드하지 않음).
3. `src/server/llama_compat_tests.rs`: 라우트 claim을 실제 라우터에 대해, 네이티브 필드 claim을 `NativeCompletionRequest`에 대해 양방향으로 검증하고, `make verify`가 거부할 manifest를 `cargo test` 실행이 통과시킬 수 없도록 `divergence` 규칙을 다시 서술합니다. manifest 항목을 전환하지 않고 b10621 라우트를 마운트하거나 b10621 필드를 받아들이면 실패하며, 이것이 조용한 드리프트를 검토 가능한 diff로 바꿉니다. `aliased` claim도 양방향으로 검사됩니다. mlxcel 식별자는 해석되어야 하고 b10621 식별자는 해석되지 않아야 하므로, 별칭이 완전한 지원으로 잘못 표기될 수 없습니다.

## 모델 소스

`mlxcel-server`와 `mlxcel serve`는 b10621의 모델 소스 어휘를 받아들이지만, mlxcel은 MLX 런타임이어서 GGUF 파일이 아니라 SafeTensors 체크포인트 디렉터리를 로드합니다. 값이 GGUF 아티팩트만 가리킬 수 있는 모든 플래그는 가중치를 하나도 읽기 전에, 값, 이유, 구체적인 대체안을 명시하는 진단과 함께 시작 시 거부됩니다. 아래 분류는 `model-source.json` shard입니다.

| 플래그 | 환경 변수 | 상태 | 동작 |
|---|---|---|---|
| `-m` / `--model` | `LLAMA_ARG_MODEL` | `aliased` | MLX 체크포인트 디렉터리, HuggingFace `owner/name` repo id, 또는 `MLXCEL_DEFAULT_ORG`에 대해 확장되는 이름만 있는 값을 받습니다. `.gguf` 경로, `-NNNNN-of-NNNNN.gguf` 분할 shard, 모든 URL은 거부됩니다. |
| `--hf-repo` | `LLAMA_ARG_HF_REPO` | `aliased` | `-m`에 전달된 같은 값과 정확히 같이 해석됩니다. 둘 다 주어지면 llama-server와 일치하게 `-m`보다 우선하며, 대체된 값은 로그에 기록됩니다. `:<quant>` 접미사는 거부됩니다. |
| `--hf-token` | `HF_TOKEN` | `aliased` | 스냅샷 다운로드를 인증합니다. 환경 변수보다 우선하며, `--help`에 렌더링되거나 로그에 기록되거나 디스크에 쓰이지 않습니다. |
| `--offline` | `LLAMA_ARG_OFFLINE` | `supported` | b10621처럼 프로세스 전체에서 캐시 사용을 강제하고 모든 다운로드를 금지합니다. `-m`, `--hf-repo`, `--embedding-model`, `--reranker-model`, `mlxcel download`, moondream starmie 토크나이저 가져오기, 요청 경로의 미디어 URL이 모두 거부됩니다. |
| `-a` / `--alias` | `LLAMA_ARG_ALIAS` | `aliased` | b10621과 마찬가지로 쉼표로 구분된 목록이며, b10621은 별칭을 `std::set`에 보관하고 첫 번째 요소를 서빙합니다. mlxcel도 동일하게 정렬하므로 `--alias zebra,apple`은 양쪽 모두 `apple`을 서빙합니다. 나머지는 기록됩니다(`/v1/models`는 아직 이를 보고하지 않으며, #1438에서 추적). |
| `--warmup` / `--no-warmup` | — | `supported` | `--no-warmup`은 시작 warmup 단계를 실제로 건너뜁니다. |
| `--hf-file` | `LLAMA_ARG_HF_FILE` | `not_applicable` | 거부됨: 저장소 안의 GGUF 파일 하나를 선택하지만, MLX는 스냅샷 전체를 로드합니다. |
| `--model-url` | `LLAMA_ARG_MODEL_URL` | `not_applicable` | 거부됨: mlxcel은 URL이 아니라 저장소 식별자로 해석합니다. HuggingFace URL은 사용할 `--hf-repo` 값으로 변환되어 안내됩니다. |
| `--docker-repo` | `LLAMA_ARG_DOCKER_REPO` | `not_applicable` | 거부됨: Docker Hub 모델 저장소는 GGUF를 배포합니다. |

항상 거부되는 세 개의 플래그는 `--help`에서 숨겨져 있으므로, 운영자에게 노출되는 표면은 GGUF 백엔드를 암시하지 않습니다. 그래도 파싱은 되므로 llama-server 명령줄은 clap의 "unexpected argument" 오류 대신 진단에 도달합니다.

llama.cpp는 이 옵션들을 대시 하나와 여러 글자로 씁니다(`-hf`, `-hfr`, `-hff`, `-hft`, `-mu`, `-dr`). clap은 대시 하나를 한 글자 short 옵션들의 묶음으로 읽으므로 `-hf`는 `-h -f`로 파싱되어 종료 상태 0으로 `--help`를 렌더링합니다. upstream이 받아들이는 명령줄이 실행도 되지 않고 오류도 보고하지 않는 것입니다. argv pre-pass(`src/cli/llama_short_flags.rs`)가 clap이 보기 전에 정확히 그 토큰들을 long 표기로 다시 씁니다. 이 pre-pass는 어떤 옵션이 다음 인자를 소비하는지 빌드된 clap 명령에서 확인하고 `--` 종결자에서 멈추므로, 우연히 그 토큰 중 하나와 같은 철자의 *값*은 결코 다시 쓰이지 않습니다.

형식 게이트는 서버가 아니라 공유 `-m` 리졸버에 있으므로, `mlxcel generate`, `mlxcel chat`, `mlxcel serve`, `mlxcel-server` 모두 GGUF 참조를 동일하게 거부합니다. 이슈 #1434가 이 shard를 소유합니다.

## GGML 런타임, 배치, 메모리 옵션

`ggml-runtime.json`의 모든 것은 **GGML** 백엔드를 설명합니다. 스레드 풀이 어떤 CPU 코어에서 실행되는지, 몇 개의 레이어를 VRAM에 복사할지, 모델을 GPU에 걸쳐 어떻게 분할할지, GGUF 파일을 `mmap`할지 `mlock`할지, 어떤 RPC 서버에 작업을 맡길지, 어떤 GGML 양자화기가 KV 캐시를 저장할지입니다. mlxcel은 모든 텐서 연산을 하나의 Metal 또는 CUDA 장치에서 MLX로 실행하며 GGML 백엔드가 없으므로, 거의 아무것도 대응되는 것이 없습니다.

각 옵션은 숨겨진 호환성 인자로 받아들여지며 그 **값**은 어떤 가중치도 읽기 전에 시작 시 분류됩니다.

- b10621에서의 의미를 mlxcel이 이미 만족하거나 아무것도 요구하지 않는 값은 효과가 없으며 조용히 받아들여집니다. `--split-mode none`, `<= 0`인 모든 값의 `--threads`(upstream은 이를 하드웨어 동시성으로 대체), `--cpu-strict 0`, `--poll 50`, `--prio 0`, `--flash-attn on|auto`, `--gpu-layers all`, `--main-gpu 0`, `--load-mode auto|mmap`, `--mmap`, `--no-direct-io`, `--repack`과 `--no-repack`, `--kv-offload`, `--op-offload`, `--no-perf`, `--n-cpu-moe 0`, `--fit off`, `--backend-sampling`, 그리고 upstream 핸들러가 deprecation 경고일 뿐 아무것도 하지 않는 `--defrag-thold`의 모든 값입니다.
- 그 밖의 것은 옵션, 값, 플랫폼 제약, 그리고 있는 경우 mlxcel 대안을 명시하는 진단과 함께 시작을 중단합니다. `--split-mode row`와 `--tensor-split`은 `--tp-size` / `--pp-size`와 [`distributed.md`](distributed.md)를, `--rpc`는 `--node-role` / `--cluster-peers`를, `--mlock`과 `--load-mode mlock`은 `MLXCEL_WIRED_LIMIT`를, `--device`는 `MLXCEL_DEVICE`를, `--perf`는 `--metrics`를, `--fit on`은 `--estimate-memory`를, `--no-kv-offload`는 `--cache-type-k` / `--cache-type-v`를, `--check-tensors` / `--list-devices`는 `mlxcel` CLI의 `inspect` 하위 명령을 안내합니다.

`--load-mode`, `--mlock`, `--mmap`, `--direct-io`는 모두 upstream에서 하나의 필드에 쓰며 명령줄 순서대로 적용되고, b10621은 뒤의 셋을 첫 번째로 대체하도록 deprecated 처리하면서 이들을 결합하면 last-wins라고 경고합니다. mlxcel은 `--load-mode`가 있으면 이를 기준으로 삼고 deprecated된 셋은 전혀 분류하지 않으므로, `--no-mmap --load-mode mmap`은 upstream과 정확히 같이 여기서 시작됩니다. `--load-mode` *뒤에* 쓴 deprecated 플래그는 upstream에서는 이기지만 여기서는 무시되며, 그 잔여 차이는 네 개 항목 모두에 기록되어 있습니다.

llama.cpp는 이 중 대부분을 대시 하나와 여러 글자로 쓰는데(`-ngl`, `-fa`, `-sm`, `-ctk`, ...), clap은 이를 한 글자 short 옵션들의 묶음으로 읽습니다. `src/cli/llama_short_flags.rs`의 argv pre-pass가 clap이 보기 전에 정확히 그 토큰들을 다시 쓰므로, `-ngl 99`는 알 수 없는 인자로 실패하는 대신 `--gpu-layers`에 도달합니다. 이 pre-pass는 어떤 옵션이 다음 항목을 소비하는지 빌드된 clap 명령에서 확인하므로 우연히 그 토큰과 같은 철자의 값은 결코 다시 쓰이지 않으며, 그 테이블은 mlxcel의 short 옵션을 가릴 수 없습니다.

`--gpu-layers`는 분류에 체크포인트가 필요한 유일한 옵션입니다. 체크포인트의 `num_hidden_layers`만이 전체 offload(mlxcel은 항상 모든 레이어를 가속기에서 실행하므로 효과 없음)와 부분 offload를 구분합니다. 따라서 모델 참조가 해석된 후에 검사되며, 다른 모든 옵션은 그 전에 검사되므로 `--numa distribute`는 수 기가바이트 다운로드 후가 아니라 즉시 보고됩니다.

이 모든 것은 `--help`에서 숨겨져 있습니다. 이들은 mlxcel 기능이 아니라 호환성 표면이며, 이를 렌더링하면 존재하지 않는 GGML 백엔드를 암시하게 됩니다. mlxcel 고유의 Metal, Accelerate, CUDA, neural-accelerator, TurboQuant 옵션은 영향을 받지 않으며 계속 표시됩니다.

### KV cache 타입

`--cache-type-k`와 `--cache-type-v`는 이 shard에서 거부하는 대신 별칭으로 처리하는 유일한 항목이며, 정확히 한 가지 값만 별칭으로 처리합니다. b10621은 `f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1`을 받아들이며, `f16`은 양쪽에서 같은 것, 즉 양자화되지 않은 half-precision KV 저장을 가리키므로 mlxcel의 `KVCacheMode::Fp16`에 매핑됩니다.

여섯 개의 양자화된 이름은 **같은 것의 다른 이름이 아니라 서로 다른 양자화기**입니다. `q8_0`은 블록별 scale을 가진 블록 단위이고, mlxcel의 `int8`은 토큰별 absmax이며, Turbo 모드는 Walsh-Hadamard 회전을 사용한 PolarQuant입니다. 이슈 #1445는 수치적 또는 저장 동등성이 입증되지 않았고 이름의 유사성은 증거가 아니므로 이들 중 어느 것도 TurboQuant 모드에 매핑하지 않기로 의도적으로 결정했습니다. `f32`와 `bf16`은 다른 이유로 거부되므로 별도의 문장을 둡니다. 이들은 양자화되지 않은 GGML float 타입이며, mlxcel의 KV 캐시는 f16 또는 자체 양자화 모드 중 하나를 저장하고 선택할 수 있는 f32나 bf16 저장이 없습니다. 각 거부는 값, 어휘가 왜 이어지지 않는지, 그리고 존재하는 mlxcel 모드를 명시합니다. 동등성을 입증하고자 한다면 [`benchmarks.md`](benchmarks.md)의 teacher-forced logit-trace 절차와 결정된 위치에서의 불일치 리포트를 따라야 합니다.

mlxcel 고유의 `int8`, `fp16+turbo4`, `fp16+turbo3`, `turbo4`, `turbo4-delegated`는 같은 플래그에서 계속 사용할 수 있습니다. [`turbo-kv-cache.md`](turbo-kv-cache.md)를 참고하십시오.

### 환경 변수 바인딩

값을 받는 옵션은 `LLAMA_ARG_*` 변수를 clap을 통해 바인딩하며, clap은 문자열을 변경하지 않고 그대로 전달합니다. 값이 없는 플래그와 `--x` / `--no-x` 쌍은 그렇지 않습니다. b10621은 값이 없는 옵션을 환경 변수에서 `on`, `enabled`, `true`, `1`일 때만(`common_arg_utils::is_truthy`, 대소문자 구분 비교) 발동시키며, bool 쌍은 `parse_bool_value`와 false를 의미하는 `LLAMA_ARG_NO_*` 별칭을 통해 읽습니다. clap 자체의 boolish 파서는 더 넓은 어휘를 받아들이고 그 밖에서는 *오류를 내므로*, 이 아홉 개는 대신 런타임에 b10621의 규칙으로 해석됩니다. 따라서 `LLAMA_ARG_CPU_MOE=0`은 `--cpu-moe`를 활성화하지 않으며, `LLAMA_ARG_PERF=sometimes`는 upstream이 예외를 던지는 것과 정확히 같이 시작을 중단합니다.

## Chat template, reasoning, 출력 파싱

`chat-templates.json`은 이름이 일치한다고 해서 동작이 일치할 가능성이 가장 낮았던 shard입니다. mlxcel은 이미 Jinja 엔진, `reasoning_content`, thinking budget, `reasoning_effort` kwarg를 가지고 있었지만, `--reasoning-format`, `--reasoning`, `--skip-chat-parsing`, `--prefill-assistant`는 전혀 파싱되지 않았으므로, 이 중 하나를 요구하는 llama-server 명령줄은 아예 실패했고 배포는 thoughts를 `reasoning_content`에서 옮길 수 없었습니다.

### Thoughts가 가는 곳

`--reasoning-format`(`LLAMA_ARG_THINK`)이 배치를 선택하며, mlxcel은 두 chat 경로 모두에서 네 가지 값을 모두 지원합니다.

| 값 | `message.content` | `message.reasoning_content` |
|---|---|---|
| `none` | 태그를 포함한 thoughts가 파싱되지 않은 채 그대로 | 없음 |
| `deepseek` | 답변만 | thoughts |
| `deepseek-legacy` | `<think>` 태그를 포함한 답변 | thoughts |
| `auto` (기본값) | 여기서는 `deepseek`와 같음 | thoughts |

mlxcel의 #1447 이전 동작은 무조건 `deepseek`였습니다. 같은 배치가 같은 라우트를 제공하는 disaggregated router front에도 적용됩니다.

스트리밍 형태도 #1470 이후에는 구분자를 포함하므로, `none`과 `deepseek-legacy`에서 `delta.content`를 이어 붙인 결과가 비스트리밍 `message.content`와 바이트 단위로 같습니다. 이전에는 불가능했습니다. 스트리밍 필터는 thinking 구분자를 일치시키면서 소비하고, 블록을 열린 상태로 준비시키는 generation prompt는 모델이 열기 마커를 전혀 내보내지 않는다는 뜻이기 때문입니�
