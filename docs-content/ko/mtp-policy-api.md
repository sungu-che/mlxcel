# 적응형 MTP 정책 API (`/v1/internal/mtp-policy`)

`mlxcel-server`는 현재 서비스 중인 조합(pairing)에 대해 싱글톤(B=1) MTP speculative burst를 실행할 가치가 있는지를 머신별로 판단합니다. 처음 몇 개의 적격 요청을 프로파일링하여 활성화 또는 거부 판정으로 확정하고, 재시작 시 다시 프로파일링하지 않도록 그 판정을 영속화합니다. 판단 방식은 [환경 변수](environment-variables.md)의 `MLXCEL_MTP_ADAPTIVE`를, 스케줄러 내 위치는 [Continuous batching](CONTINUOUS_BATCHING.md)을 참조하십시오.

이 판정은 "이 조합에서 MTP가 여기서 도움이 되는가"에 대한 유일한 머신별 답이며, 호스트 애플리케이션이 사용자에게 보여 주고 싶어 하는 값입니다. 이 엔드포인트가 이를 읽는 공식적인 방법입니다.

구현 소스 맵:

| 모듈 | 역할 |
|--------|----------------|
| `src/server/batch/mtp_policy.rs` | 정책 상태 머신, 영속화된 hint, 게시되는 `MtpPolicySnapshot`. |
| `src/server/batch/observability.rs` | 워커가 마지막으로 게시한 스냅샷을 보관합니다. |
| `src/server/routes/mtp_policy.rs` | 라우트 핸들러와 와이어 본문. |

## 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|-------------|
| GET | `/v1/internal/mtp-policy` | 실행 중인 조합에 대한 적응형 MTP 정책 상태를 보고합니다. |

`/v1`이 없는 별칭은 마운트되지 않습니다.

모델을 서비스하는 노드에서는 CLI 플래그로 제어되는 `/props`, `/slots`, `/metrics`와 달리 이 엔드포인트가 항상 마운트됩니다. 유일한 예외는 disaggregated 라우터 프런트엔드(`--node-role router`)입니다. 라우터 앱은 `/health`, `/router/stats`, `/v1/chat/completions`, `/v1/completions`만 마운트하고 나머지는 모두 404를 반환하며, 이는 이미 `/v1/models`, `/v1/cache/stats`, `/metrics`에 대해 그렇게 하는 것과 같습니다. 마운트된 노드에서는 정책이 전혀 실행되지 않는 경우를 포함해 모든 상태에서 올바른 형식의 본문으로 응답하므로, 소비자는 무조건 폴링할 수 있으며 "보고할 것이 없음"과 "이 서버는 이 경로를 서비스하지 않음"을 구분할 필요가 없습니다. 다른 모든 라우트와 동일한 API 키 미들웨어 뒤에 있으며, 페이로드는 영속화된 hint만큼 거칩니다. 즉 판정, 수락률, 샘플 수, 조합 식별 정보뿐입니다. 프롬프트 데이터도, 토큰 id도, 요청을 식별하는 어떤 것도 없습니다.

## 응답 본문

```json
{
  "schema_version": 1,
  "state": "settled",
  "reason": null,
  "decline_detail": null,
  "verdict": "enable",
  "mtp_enabled": true,
  "target": "Gemma4-12B",
  "drafter": "Gemma4-12B-MTP",
  "hardware": "M5-16c",
  "block_size": 4,
  "acceptance_rate": 0.62,
  "samples": 4,
  "samples_required": 4,
  "samples_remaining": null
}
```

| 필드 | 타입 | 의미 |
|-------|------|---------|
| `schema_version` | integer | 이 본문의 와이어 스키마 버전. [버전 관리 및 호환성](#versioning-and-compatibility)을 참조하십시오. |
| `state` | string | `"settled"`, `"profiling"`, `"forced"`, `"unavailable"`, 또는 `"exactness_declined"`. |
| `reason` | string 또는 null | 정책을 사용할 수 없는 이유. `state`가 `"unavailable"`일 때만 null이 아닙니다. |
| `decline_detail` | string 또는 null | 정확성(exactness) 프로브의 한 줄 이유. `state`가 `"exactness_declined"`일 때만 null이 아니며, 부팅 시 WARN 로그에 남는 문장과 같습니다. |
| `verdict` | string 또는 null | 확정되면 `"enable"` 또는 `"decline"`, 그렇지 않으면 `null`. 영속화된 hint의 `verdict` 필드와 같은 값입니다. |
| `mtp_enabled` | boolean 또는 null | *정책*이 현재 B=1 MTP burst를 막고 있지 않은지 여부. 판정이 아니라 실시간 게이트입니다. 프로파일링은 정책 수준에서 MTP를 강제로 켜므로 프로파일링 중에는 `true`입니다. burst가 실제로 실행 중이라는 뜻은 아닙니다. 이후의 런타임 게이트(타깃 지원, 정확성, capability 검사)가 모든 burst 시도를 여전히 거부할 수 있으며, 이 경우 `mtp_enabled: true`임에도 프로파일링은 샘플을 전혀 누적하지 못합니다. |
| `target` | string 또는 null | 서비스 중인 모델 디렉터리의 basename. |
| `drafter` | string 또는 null | draft 모델 디렉터리의 basename. |
| `hardware` | string 또는 null | 영속화된 판정의 키가 되는 대략적인 하드웨어 클래스 레이블. Apple 호스트는 세대와 코어 수 형식을 유지합니다(예: `"M5-16c"`). Apple이 아닌 호스트는 GPU 백엔드를 접두사로 붙이고, 백엔드가 제공하는 경우 디바이스 아키텍처도 붙입니다(예: `"cuda-Unknown-0c"` 또는 `"rocm-gfx1151-Unknown-0c"`). 따라서 CUDA 판정은 ROCm에서 재사용되지 않습니다(issue #1887). |
| `gpu_vendor` | string | 서비스 호스트의 GPU 벤더: `"Apple"`, `"Nvidia"`, `"Amd"`, `"Unknown"`. |
| `gpu_device` | string 또는 null | 백엔드가 보고하는 디바이스 이름(예: `"AMD Radeon 8060S Graphics"`). 백엔드가 제공하지 않으면 null. |
| `gpu_architecture` | string 또는 null | 실행 중인 백엔드 고유 체계의 아키텍처 문자열: ROCm에서는 `gfx1151`, CUDA에서는 `sm_89`, Metal에서는 Apple GPU 패밀리 문자열. 벤더 간에 비교할 수 없으므로 `gpu_vendor`와 함께 읽으십시오. |
| `block_size` | integer 또는 null | 조합의 키가 되는 draft 블록 크기(K). |
| `acceptance_rate` | number 또는 null | 대략적으로 측정된 수락률(제안된 draft 토큰 대비 수락된 토큰). 프로파일링 중에는 진행 중인 값, 확정되면 최종 값, 아무것도 측정되지 않았으면 `null`. 확정되면 영속화된 hint와 정확히 일치하도록 소수점 둘째 자리로 반올림되며, 프로파일링 중에는 반올림되지 않은 원래의 진행 중 몫입니다. |
| `samples` | integer | 지금까지 누적된, 또는 확정된 판정의 근거가 된 적격 샘플 수. forced 또는 unavailable이면 `0`. |
| `samples_required` | integer | 프로파일링 윈도우가 확정되기까지 필요한 적격 샘플 수. |
| `samples_remaining` | integer 또는 null | 아직 필요한 적격 샘플 수. 다른 상태에서는 대기 중인 것이 없으므로 `state`가 `"profiling"`일 때만 null이 아닙니다. |

`target`, `drafter`, `hardware`, `block_size` 네 필드가 조합 키입니다. 판정을 보여 주는 소비자는 이들이 표시하려는 조합과 일치하는지, 특히 `block_size`를 확인해야 합니다. 하나의 K에서 프로파일링된 판정은 다른 K로 이어지지 않으며, `--draft-block-size` / `MLXCEL_DRAFT_BLOCK_SIZE`를 변경하면 새 프로파일링 윈도우가 시작됩니다. `--num-draft-tokens`는 별개의 오프라인 전용 `generate` 설정이며 정책 키에 영향을 주지 않습니다.

### 상태

| `state` | 의미 |
|---------|---------|
| `settled` | 이 프로세스에서 측정되었거나 영속화된 hint에서 복원된 판정이 적용 중입니다. `verdict`가 어느 쪽인지 알려 줍니다. |
| `profiling` | 아직 샘플을 누적하는 중입니다. 판정은 아직 없으며, `samples_remaining`이 필요한 단일 요청 생성이 몇 번 더 남았는지 알려 줍니다. 그동안 MTP는 정책 수준에서 강제로 켜지므로 `mtp_enabled`는 `true`입니다. 제한: 정책 아래의 요청별 런타임 게이트(타깃 지원, 시퀀스별 capability 검사)가 모든 burst 시도를 거부하면 샘플이 기록되지 않으며 이 상태가 `samples: 0`으로 지속됩니다. 이러한 고갈의 구조적 사례 하나인 정확성 프로브 거부는 issue #1298 이후 `exactness_declined`로 보고됩니다. |
| `exactness_declined` | 런타임 정확성 프로브가 해당 조합을 거부했습니다. 이 하드웨어에서는 사용 가능한 어떤 커널 선택으로도 multi-token verify 블록이 단일 토큰 체인과 바이트 단위로 동일하지 않으므로, B=1 burst가 디스패치되지 않고 요청은 기존 decode로 처리됩니다. `decline_detail`에 프로브의 이유가 담기고, `mtp_enabled`는 `false`이며, 판정은 영원히 도착하지 않으므로 `profiling`이 아닌 것입니다. `MLXCEL_MTP_ALLOW_INEXACT=1`은 그래도 실행하도록 선택하지만, temperature 0 바이트 동일성 계약을 포기하게 됩니다. |
| `forced` | `MLXCEL_ENABLE_MTP_B1`이 결정을 고정했습니다. 프로파일링도 측정도 이루어지지 않았으므로 `verdict`, `acceptance_rate`, `samples`에는 측정값이 없습니다. `mtp_enabled`는 여전히 고정 결과가 무엇으로 해석되었는지를 보고합니다. |
| `unavailable` | 실행 중인 적응형 정책이 없습니다. `reason`이 이유를 알려 줍니다. |

`forced`는 의도적으로 `settled`에 합치지 않았습니다. 운영자의 고정은 측정된 판정이 아니며, 이를 "이 머신에서 MTP가 가치 있다고 측정되었음"으로 렌더링하는 소비자는 아무도 측정하지 않은 것을 보고하는 셈입니다. 고정은 고정으로 렌더링하십시오.

### Unavailable 이유

| `reason` | 의미 |
|----------|---------|
| `no_mtp_dispatch` | 서버에 MTP speculative 디스패치가 없습니다. drafter가 제공되지 않았거나 drafter가 다른 종류의 speculative로 해석되었습니다. B=1 MTP burst가 실행되지 않으므로 `mtp_enabled`는 `false`입니다. |
| `adaptive_disabled` | `MLXCEL_MTP_ADAPTIVE`가 꺼져 있어 측정도 영속화도 이루어지지 않습니다. 대신 B=1 burst를 결정하는 것은 `MLXCEL_ENABLE_MTP_B1`이 함께 설정된 경우(운영자 고정), 그렇지 않으면 하드웨어별 정적 게이트입니다. `mtp_enabled`는 둘 중 결정한 쪽을 보고합니다. |
| `worker_not_ready` | 아직 어떤 배치 워커도 정책 상태를 게시하지 않았습니다. 모델이 아직 로딩 중이며 일시적인 상태입니다. 모든 워커 변형은 시작 시 게시하며, MTP 경로가 아예 없는 레거시 단일 시퀀스(`--no-batch`) 워커와 OpenXLA 워커도 `worker_not_ready`를 정상 상태로 남겨 두는 대신 즉시 `no_mtp_dispatch`를 게시합니다. `mtp_enabled`는 `null`입니다. |

이를 구분하는 것이 이 엔드포인트의 요점입니다. 다른 프로세스에서 hint 파일을 읽어서는 구분할 수 없었습니다. 빈 디렉터리는 "아직 프로파일링 중", "MTP가 구성되지 않음", "캐시 루트가 다른 곳으로 해석됨"을 모두 한꺼번에 의미했고, 소비자는 어느 경우든 빈 화면으로 퇴화했습니다.

## 버전 관리 및 호환성

본문에는 `1`에서 시작하는 정수 `schema_version`이 포함됩니다.

하나의 `schema_version` 내에서 mlxcel은 다음을 보장합니다.

- 기존 필드는 이름, 타입, 의미를 유지합니다.
- `state`, `reason`, `verdict` 레이블 집합은 늘어나기만 합니다. 새로운 state나 새로운 unavailable 이유는 버전 증가 없이 나타날 수 있습니다.
- 새 필드가 추가될 수 있습니다.

따라서 소비자는 알 수 없는 필드를 무시해야 하며, 인식할 수 없는 `state`, `reason`, `verdict` 레이블은 오류가 아니라 "렌더링할 수 있는 판정 없음"으로 취급해야 합니다. 알 수 없는 레이블에서 곧바로 실패하는 소비자는 레이블을 추가하는 릴리스에서 깨질 것입니다.

`schema_version`은 이러한 보장을 깨는 모든 변경에 대해 증가합니다. 필드 제거 또는 이름 변경, 필드 타입 변경, 의미 변경, 레이블 집합 축소가 해당합니다. 증가는 호환성을 깨는 변경이며 changelog 항목이 작성됩니다. 받은 `schema_version`을 인식하지 못하는 소비자는 필드를 추측하지 말고 본문을 읽을 수 없는 것으로 취급하고 아무것도 표시하지 않아야 합니다.

`schema_version`은 영속화된 hint의 `version`(`HINT_VERSION`)과 독립적입니다. hint 버전은 디스크 형식과 그 뒤의 판정 의미론을 추적하며, 저장된 판정을 폐기하고 다시 프로파일링해야 할 때마다 증가합니다. 두 숫자는 서로 무관하며 차이가 벌어질 것입니다.

## 영속화된 hint 파일과의 관계

확정된 판정은 여전히 `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy/<key-hash>.json`에 기록되며, 이 동작은 변하지 않았습니다. 이 파일들은 판정이 재시작을 견디게 하는 수단이지 인터페이스가 아닙니다. `HINT_VERSION`, 하위 디렉터리 이름, hint 본문은 비공개이며 패치 릴리스에서 예고 없이 바뀔 수 있습니다.

이 엔드포인트는 hint 파일이 담고 있는 모든 것(hint에는 이미 조합의 `hardware` 레이블이 포함됨)과 파일이 표현할 수 없는 상태인 profiling, forced, unavailable을 노출합니다. 파일 대신 이것을 읽으십시오.

## 예시: MTP가 실행 중인가, 그리고 이유는 무엇인가

```bash
curl -s http://127.0.0.1:8080/v1/internal/mtp-policy | jq
```

프로파일링 중:

```json
{
  "schema_version": 1,
  "state": "profiling",
  "reason": null,
  "decline_detail": null,
  "verdict": null,
  "mtp_enabled": true,
  "target": "Gemma4-12B",
  "drafter": "Gemma4-12B-MTP",
  "hardware": "M5-16c",
  "block_size": 4,
  "acceptance_rate": 0.58,
  "samples": 2,
  "samples_required": 4,
  "samples_remaining": 2
}
```

이는 "측정 중: 단일 요청 생성 4건 중 2건 완료"로 렌더링됩니다.

drafter가 없는 서버:

```json
{
  "schema_version": 1,
  "state": "unavailable",
  "reason": "no_mtp_dispatch",
  "decline_detail": null,
  "verdict": null,
  "mtp_enabled": false,
  "target": null,
  "drafter": null,
  "hardware": null,
  "block_size": null,
  "acceptance_rate": null,
  "samples": 0,
  "samples_required": 4,
  "samples_remaining": null
}
```

정책이 보고할 수 있는 조합은 싱글톤 MTP 패밀리입니다. assistant drafter를 쓰는 Gemma 4, `qwen3_5_mtp` 헤드를 쓰는 Qwen 3.5 / 3.6 / 3.8, 체크포인트에 포함된 `mtp.safetensors`를 쓰는 Inkling, 그리고 `mlxcel split-mtp`가 추출하는 `glm4_moe_lite_mtp` 블록을 쓰는 GLM-4.7-Flash(`glm4_moe_lite`)입니다(issue #1326). 이들 모두 정책이 샘플을 보기 전에 동일한 정확성 프로브를 통과하므로, 어느 것에서든 `exactness_declined`가 나타날 수 있습니다.

두 가지 커널 선택 모두에서 정확성 프로브에 실패하는 조합이 있는 세대 15+ 호스트(M3 Ultra와 M5 Max에서 Gemma 4 31B + bf16 assistant 조합으로 측정됨, issue #1279):

```json
{
  "schema_version": 1,
  "state": "exactness_declined",
  "reason": null,
  "decline_detail": "verify block position 0 differs from the single-token chain in 245722 of 524288 logit bytes. Disabling qmv_wide did not make it exact either.",
  "verdict": null,
  "mtp_enabled": false,
  "target": "gemma-4-31b-it-4bit",
  "drafter": "gemma-4-31b-it-assistant-bf16",
  "hardware": "M5-40c",
  "block_size": 4,
  "acceptance_rate": null,
  "samples": 0,
  "samples_required": 4,
  "samples_remaining": null
}
```

이는 "MTP가 구성되었으나 거부됨; 기존 decode 실행 중"으로 렌더링되며, 이유는 원문 그대로 확인할 수 있습니다.

동일한 상태는 `/health` 관측성 스냅샷의 `observability.mtp_policy` 아래에도 유사하지만 버전이 없는 형태로 포함됩니다(state 필드를 `status`로 표기하고 `schema_version`과 `samples_remaining`을 생략함). 이는 이미 해당 엔드포인트를 폴링하는 운영자를 위한 것입니다. `/health`는 안정성 보장이 없는 운영자용 인터페이스이며, 위의 계약을 갖는 것은 `/v1/internal/mtp-policy`입니다.
