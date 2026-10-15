# Speculative decoding 수용 규칙

Speculative decoding(추측 디코딩)은 가벼운 draft 모델로 토큰을 제안하고 비용이 큰 target 모델로 이를 검증합니다. *수용 규칙(acceptance rule)*은 어떤 제안이 살아남을지를 결정합니다. 이는 두 가지를 동시에 결정합니다. target forward 한 번당 한 라운드가 내보내는 토큰 수(처리량)와, 내보낸 스트림이 따르는 분포(정확성)입니다. 이 문서는 각 코드 경로가 어떤 규칙을 실행하는지, 각 규칙이 무엇을 보장하는지, 그리고 로그를 통해 주어진 실행이 어떤 규칙을 사용했는지 알아내는 방법을 설명합니다.

## 규칙

`p`는 검증 위치에서 target 모델의 유효한(effective) 다음 토큰 분포(토큰 bias, 페널티, 온도를 적용하지 않은 분포에 대한 top-k / top-p / min-p 필터링, 재정규화된 필터링 행에 대한 XTC, 그리고 최종 온도 스케일링을 거친 후)를 뜻하고, `q`는 drafter가 제안을 실제로 추출한 분포를 뜻합니다.

### Sampler-match (`temperature > 0`에서의 기본값)

드래프트된 토큰이 `p`에서 새로 추출한 값과 같을 때에 한해 수용하며, 두 분기 모두에서 그 추출값을 내보냅니다. **무손실(lossless)**: 내보낸 모든 토큰이 target에서 새로 추출한 조건부 샘플이므로, 스트림은 정확히 target 단독 샘플과 같습니다. 수용 확률은 `sum_x p(x) q(x)`입니다.

이는 `SpeculativeGenerator`가 항상 해온 방식입니다. 이슈 #902는 검증 루프가 target의 *argmax*와 비교한다고 설명하지만, 이 생성기에는 부정확한 설명이며 그 차이가 핵심입니다. 코드는 `sample_token_optimized(pos_logits, target_sampling, history)`를 호출하는데, 이는 `temperature == 0`일 때만 `argmax`입니다.

### Greedy argmax (`temperature == 0`)

샘플러가 우연히 `argmax`인 경우의 sampler-match입니다. 같은 논거로 무손실이며, 정확히 greedy target 디코딩입니다.

### Argmax-against-argmax (MTP, DFlash, DSpark, Muse Glimmer assistant)

온도와 관계없이 드래프트된 토큰이 `argmax(target_logits)`와 같을 때 수용하고, 불일치하면 argmax를 내보냅니다. `temperature > 0`에서는 **무손실이 아닙니다**. 서빙되는 스트림은 `p`의 샘플이 아니라 target의 greedy 스트림입니다. 수용 확률은 `q(argmax p)`입니다.

Muse Glimmer assistant drafter(#1343)는 이 규칙 아래 동일한 DFlash 라운드 루프에서 실행되며, Qwen 3.5 DFlash drafter와 디스패치 차이가 없습니다. 제안은 요청의 샘플러에 따라 마스크된 슬롯마다 샘플링되고 target의 argmax에 대해 수용됩니다. `temperature == 0`에서는 아래의 block-versus-chain 정확성 프로브(probe)를 전제로, target의 rotating 캐시가 가진 2048 토큰 슬라이딩 윈도우를 넘어서도 정확히 greedy target 디코딩입니다.

이 드래프터의 verify 폭은 DFlash 루프에서 유일한 적응형 폭이므로 프로브가 한 번 이상 실행됩니다. 라운드 루프는 drafter가 선언한 깊이인 4개 행으로 워밍업하고, 측정 윈도우가 그만한 가치가 있음을 보일 때만 요청된 상한까지 넓히며, 게이트는 버스트가 시작되기 전에 두 폭을 모두 통과시킵니다. 상한만 프로브하면 워밍업 폭이 측정되지 않은 채 허용되는데, 이는 형식적인 문제가 아닙니다. forward 폭이 MLX가 디스패치하는 양자화 matmul 커널을 선택하며, `docs/benchmarks.md`는 동일한 비교가 폭 8에서 20.6퍼센트, 폭 32에서 0.0퍼센트의 불일치를 보인다고 기록합니다.

LFM2 DSpark drafter는 DFlash 라운드 루프 위에서 실행되므로 이 규칙을 따르며, 디스패치에 한 가지 차이가 있습니다. greedy 전용이라는 점입니다. drafter의 Markov 헤드는 greedy target에 대해 학습된 체인을 제안하며, 서버는 샘플러가 argmax가 아닌 요청(`top_k != 1`인 `temperature > 0`)을 편향된 greedy 스트림으로 서빙하는 대신 경고 한 번과 함께 클래식 디코드로 보냅니다. `temperature == 0`에서는 아래의 block-versus-chain 정확성 프로브를 전제로 정확히 greedy target 디코딩입니다.

### Modified rejection sampling (옵트인)

Leviathan et al. 2023, "Fast Inference from Transformers via Speculative Decoding"(Algorithm 1)과 Chen et al. 2023, "Accelerating Large Language Model Decoding with Speculative Sampling"에서 가져왔습니다.

1. `u ~ U[0, 1)`을 추출합니다. `u * q(t) <= p(t)`이면 드래프트된 토큰 `t`를 수용합니다.
2. 첫 번째 거부에서 `normalize(relu(p - q))`에서 추출한 토큰 하나를 내보내고 체인을 끝냅니다.
3. 드래프트된 모든 토큰이 수용되었다면 `p`에서 추출한 보너스 토큰을 내보냅니다.

무손실입니다. 점별로(pointwise) `min(q, p) + relu(p - q) == p`이기 때문입니다(잔차(residual)의 정규화 상수가 거부 확률을 정확히 상쇄함). 수용 확률은 `sum_x min(p(x), q(x))`입니다.

## 수용 상한

`q`로 분포된 제안과 `p`로 분포된 방출(emission)에 대해, 둘이 일치할 수 있는 최대 확률은 `sum_x min(p(x), q(x))`이며, 이는 최대 커플링(maximal-coupling) 한계이고 `1 - TV(p, q)`와 동등합니다. **모든** 분포 보존 규칙은 여기까지로 제한됩니다.

| 규칙 | 무손실 | 수용 확률 |
|------|----------|------------------------|
| Sampler-match | 예 | `sum_x p(x) q(x)` |
| Modified rejection sampling | 예 | `sum_x min(p(x), q(x))` (상한) |
| Argmax-against-argmax | **아니요** | `q(argmax p)` (상한에 의해 제한되지 않음) |

`[0, 1]`에서 `min(a, b) >= a * b`이므로 rejection sampling은 항상 sampler-match 이상으로 자주 수용합니다. 그러나 `q(argmax p)`에는 그런 제약이 없습니다. `p(a) = 0.3`, `q(a) = 0.95`이면 argmax 규칙은 0.95로 수용하지만 올바른 어떤 규칙이든 상한은 약 0.30입니다.

**결과를 분명히 말하면:** 어떤 구현도 분포 보존 규칙이 argmax 수용에 비해 평균 수용 길이를 엄밀하게 개선하도록 만들 수 없습니다. 이는 고쳐야 할 결함이 아니라 비편향성의 대가입니다. 이슈 #902의 요구 결과("온도 0.7에서 평균 수용 드래프트 길이가 argmax 수용에 비해 엄밀하게 개선")는 일반적으로 달성할 수 없으며, 개선이 실제로 존재하고 증명 가능한 sampler-match에 대한 것으로 읽어야 합니다.

## 현재 각 경로가 실행하는 규칙

| 경로 | `temperature == 0` | `temperature > 0` 기본값 | `temperature > 0` 옵트인 |
|------|--------------------|---------------------------|--------------------------|
| `SpeculativeGenerator`(클래식, 더 이상 사용되지 않으며 v0.8.0에서 제거됨, #2176; `mlxcel generate --draft-model`) | greedy argmax (무손실) | **sampler-match** (무손실) | modified rejection sampling (무손실, 수용 최적) |
| 엔진의 토큰 전용 루프에서의 `PromptLookupDrafter`(`mlxcel generate --prompt-lookup`) | greedy argmax (무손실) | **sampler-match** (무손실이며 여기서는 수용 최적) | 필요 없음 |
| 서버 스케줄러에서의 `PromptLookupDrafter`(`--draft-kind prompt-lookup`, 동일한 `Engine::verify_round`) | greedy argmax (무손실) | **sampler-match** (무손실이며 여기서는 수용 최적) | 필요 없음 |
| Gemma 4 MTP 라운드 루프 | argmax (여기서는 무손실) | argmax-against-argmax (**편향됨**) | 연결되지 않음 |
| DFlash 라운드 루프(Qwen 3.5 DFlash drafter) | argmax (여기서는 무손실) | argmax-against-argmax (**편향됨**) | 연결되지 않음 |
| DFlash 라운드 루프(LFM2 DSpark drafter) | argmax (여기서는 무손실, 프로브 게이트) | 클래식 디코드로 전환 | 연결되지 않음 |
| DFlash 라운드 루프(Muse Glimmer assistant drafter) | argmax (여기서는 무손실, 프로브 게이트) | argmax-against-argmax (**편향됨**) | 연결되지 않음 |

MTP와 DFlash verify 경로는 샘플러와 관계없이 `argmax_per_position` / `argmax_logits_to_array`로 target 토큰을 선택하며, 이들의 drafter는 추출에 사용한 분포 없이 토큰 id만 반환합니다(`Drafter::draft_block`은 `Vec<i32>`를 반환). 이들을 무손실로 만들려면 drafter 인터페이스와 `MtpTarget::verify_forward`의 출력을 모두 확장해야 합니다. 이 경로들은 이 기능이 수용률만이 아니라 정확성을 얻게 해 주는 곳이며, argmax 대비 수용 트레이드오프가 공짜 이득이 아니라 실제 결정이 되는 곳입니다.

Prompt lookup은 결정적으로 제안하므로 `q`는 드래프트된 토큰 `d`에서 one-hot입니다. 그러면 두 수용 확률은 `p(d)`로 일치하며(`sum_x p(x) q(x) = sum_x min(p(x), q(x)) = p(d)`), 거부 시 두 규칙 모두 `d`가 아니라는 조건을 건 `p`에서 추출한 값을 내보냅니다. 따라서 이 경로에서는 sampler-match가 이미 수용 최적 규칙이며 옵트인할 것이 없습니다.

CUDA 서버의 prompt-lookup 거버너는 추가로, 수용된 제안 수에 1을 더한 값의 EMA가 4 토큰 아래로 내려가고, 최소 두 번의 드래프트 라운드 이후이며 가장 최근 라운드도 하한 미만일 때 수확이 낮은 행을 일시 중지합니다. 하한을 충족한 최근 라운드는 일시 중지 없이 초기 EMA 부채를 상환할 수 있습니다. 수용된 토큰 EMA가 하한 이상이었던 후에 발생한 고립된 수용 0 라운드는 한 라운드의 유예를 받으며, 부분적인 저수확 수용은 그렇지 않습니다. 유예는 연속된 첫 번째 저수확 라운드를 넘어 연장되지 않으며, 일시 중지에는 여전히 내보낸 토큰 EMA가 하한 미만이어야 합니다. 이 측정된 필드 경계 휴리스틱은 수용 규칙 자체를 바꾸지 않습니다. 이는 제안 비용에 대한 결정이지 sampler-match 수용 규칙의 변경이 아닙니다. 일시 중지된 행도 여전히 내보낸 토큰을 관찰하고 섀도 룩업을 실행하며, 일치하는 섀도 토큰 일곱 개가 verify를 복원합니다. 단순 요청(plain ask), 철회된 제안, 게이트로 제외된 행은 verify 증거를 추가하지 않습니다. CLI에는 손익분기 하한이 없으며 두 토큰 섀도 확인을 유지합니다. `adaptive = false`는 하한을 무시하고, graded 정책은 변경되지 않습니다.

## RNG 의존성

Modified rejection sampling은 이전 규칙들이 사용하지 않던 난수를 추출합니다.

* 수용 테스트를 위해 검증된 드래프트 위치마다 `U[0, 1)` 하나.
* 잔차를 위해 거부마다 categorical 추출 하나.

둘 다 융합된(fused) 샘플러가 소비하는 것과 같은 스트림인 MLX의 기본 키 시퀀스를 거치므로, `mlxcel`의 `--seed`(MLX `random::seed`)는 여전히 실행을 정확히 재현합니다. 그러나 스트림이 이전과 다르게 소비되므로 **같은 시드에서도 내보낸 토큰이 변경 이전 실행과 다릅니다**. 이는 `MLXCEL_SAMPLING_GUMBEL`이 분포를 바꾸지 않으면서 스트림을 바꾸는 것과 같은 의미입니다. 이 변경이 반영되기 전에 기록된 토큰 스트림을 재현하려면 `MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT=0`을 설정하십시오.

Greedy(`temperature == 0` 또는 `top_k == 1`)는 새로운 것을 추출하지 않습니다. 규칙 선택기가 어떤 분포 텐서도 만들기 전에 argmax 규칙을 반환하므로 temperature 0 출력은 바이트 단위로 동일합니다.

## 옵트인이며 기본값이 아닌 이유

`MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT=1`(`true`, `yes`, `on`도 가능)은 modified rejection sampling을 활성화합니다. **기본적으로 꺼져 있으며**, `SpeculativeGenerator::with_stochastic_acceptance(bool)`로 프로그램적으로 환경 변수를 재정의할 수 있습니다.

기본값이 꺼짐인 이유는, 현재 이 규칙이 닿는 유일한 경로에서 대체 대상 규칙이 이미 무손실이었기 때문입니다. 따라서 얻는 것은 수용률뿐이며, 가능한 이득은 정확히 `sum_x min(p, q) / sum_x p(x) q(x)`인데, drafter가 확신할 때마다 이 값은 1로 수렴합니다. `q(t*) ~ 1`이면 `min(p, q)`와 `p * q`가 일치하기 때문입니다. Llama-3.1-8B / Llama-3.2-1B 쌍에서 온도 0.7로 측정하면 약 1.02입니다. 검증된 위치마다 어휘 전체에 대한 패스 두 번과 추가 호스트 동기화 한 번은 이론적 수용률 2퍼센트의 가치가 없습니다.

이는 `MLXCEL_FUSED_QK_NORM`(#326) 및 #905 융합과 같은 패턴을 따릅니다. 구현, 증명, 테스트는 반영되어 사용할 수 있지만, 측정이 정당화할 때까지 기본값은 움직이지 않습니다.

## 실행에서 규칙 읽어내기

**`mlxcel` CLI는 기본적으로 `tracing` subscriber를 설치하지 않으므로**, 아래의 info 수준 계측은 `SpeculativeGenerator`를 실행하는 유일한 바이너리에서는 보이지 않습니다. 따라서 수용 요약은 모든 speculative 실행 후에 **stdout**으로 무조건 출력됩니다.

```
[Speculative acceptance] rule=stochastic rounds=153 proposed=612 positions_tested=195 accepted=46 per_position_acceptance=0.2359 acceptance_rate=0.0752 mean_accepted_len=0.3007 (stochastic (modified rejection sampling))
```

`rule=`은 안정적인 식별자이며 `stochastic`, `greedy-argmax`, `sampler-match`, `sampler-match-no-proposal-distribution` 중 하나입니다. A/B의 한 팔이 조용히 폴백이 아님을 증명해 주는 것이 이 값입니다.

### 두 수용률은 서로 바꿔 쓸 수 없습니다

| 필드 | 정의 | 용도 |
|-------|------------|-----|
| `per_position_acceptance` | `accepted / positions_tested` | **이론이 예측하는 값.** 팔 사이 및 `sum_x min(p, q)`와 비교합니다. |
| `acceptance_rate` | `accepted / proposed` | 각 드래프트 블록 중 얼마나 살아남았는지. 블록 길이에 따라 변합니다. |
| `mean_accepted_len` | `accepted / rounds` | target forward당 얻는 토큰 수. 처리량을 좌우합니다. |

`proposed`는 모든 드래프트 블록의 모든 위치를 세지만, 체인은 첫 거부에서 멈추고 그 뒤 위치는 테스트하지 않습니다. 따라서 `acceptance_rate`는 체인이 잘린 만큼 정확히 낮아지며, 체인 길이 분포가 다른 두 팔 사이에서 이를 비교하는 것은 의미가 없습니다. 이 둘을 혼동하면 올바른 구현이 2배 회귀처럼 보이기에 충분합니다.

### 선택적 tracing

`RUST_LOG`를 설정하면 이제 오프라인 CLI 명령에 subscriber가 설치되어(stderr에 기록하므로 파이프된 stdout은 깨끗하게 유지됨), 한 번만 출력되는 `speculative acceptance rule active` 줄, 호출별 `speculative decode finished` 줄, drafter 자동 감지 줄이 나타납니다. `RUST_LOG`가 설정되지 않으면 기본 출력은 변하지 않습니다. `mlxcel serve`는 자체 subscriber를 설치하므로 영향을 받지 않습니다.

### 응답 자체에 담기는 요청별 수용 정보

`mlxcel-server`는 **한 요청**에 대해 speculation이 한 일을 그 요청 자신의 응답에 보고하므로, 클라이언트는 서버 로그에 접근하지 않고도 "drafter가 도움이 되었는가"에 답할 수 있습니다(#1314). 이 블록은 drafter가 verify 라운드를 최소 한 번 실행한 경우에만 나타납니다. 클래식 디코드 배포와 prefill 안에서 끝난 speculative 요청은 이전과 같은 본문으로 응답합니다.

네이티브 `/completion` 및 `/completions` 라우트에서 카운터는 기존 `timings` 객체에 실려 갑니다.

```json
"timings": {"cache_n": 0, "prompt_n": 12, "predicted_n": 64, "...": "...",
            "draft_n": 72, "draft_n_accepted": 55, "draft_rounds": 9,
            "draft_kind": "dflash"}
```

`draft_n`과 `draft_n_accepted`는 llama-server 고유의 선택적 쌍이며 업스트림이 표기하는 방식 그대로, 업스트림이 덧붙이는 방식 그대로 덧붙이므로 llama-server timings를 이미 읽는 클라이언트는 mlxcel의 것도 그대로 읽습니다. `draft_rounds`와 `draft_kind`는 mlxcel의 추가 항목입니다. 업스트림에는 라운드 수도 drafter 종류라는 개념도 없으며, 라운드 수가 없으면 이 쌍만으로는 라운드당 평균 수용 길이를 복구할 수 없습니다.

`/v1/chat/completions`는 전체 블록을 최상위 `timings` 객체로 싣습니다. 비스트리밍 응답과 `finish_reason`을 담은 스트리밍 청크에서 그렇습니다. b10621은 자체 OpenAI chat 응답에 `timings` 객체를 넣으며 네이티브 라우트가 사용하는 것과 동일한 구조체로 만들어지므로, mlxcel의 것도 `draft_*` 절반만이 아니라 같은 객체입니다. 키를 탐색한 뒤 그곳에서 `predicted_per_second`를 읽는 클라이언트는 기대하는 키를 찾게 됩니다. 남아 있는 유일한 차이는 존재 여부입니다. 업스트림은 모든 chat completion에서 이 블록을 내보내지만 mlxcel은 드래프트된 요청에서만 내보내며, 이는 `compat/llama-server/b10621/routes.json`에서 두 chat 라우트 모두에 대해 기록되어 있습니다.

`finish_reason` 청크만 이를 싣습니다. 카운터는 실행의 총계이며, 스트림 중간 프레임이 진행 중인 `draft_n`을 보고하면 아직 총계가 아닌 값을 보고하는 셈이 되기 때문입니다. 같은 이유로 네이티브 라우트의 토큰별 `timings_per_token` 프레임은 아홉 개의 기본 키를 싣고 네 개는 싣지 않습니다.

기본 키는 드래프트된 요청에서도 클래식 요청과 같은 의미를 가집니다(#1592). `prompt_ms`는 큐 대기와 target prefill을 포함하고, `predicted_ms`는 prefill이 끝난 순간부터의 모든 verify 라운드를 포함하므로, `predicted_per_second`는 드래프트된 디코드 속도이며 같은 프롬프트에 대한 클래식 수치와 비교할 수 있습니다. 버스트는 스트리밍 전에 끝까지 실행되므로, 완료된 토큰이 재생되는 시점이 아니라 prefill이 끝나는 시점에 첫 토큰 시각을 찍습니다. 수정 전에는 그 재생 시점이 기준이었고, `prompt_ms`가 버스트 전체를 흡수하며 `predicted_ms`는 1밀리초 미만의 재생 시간을 보고했습니다(96 토큰 요청에서 초당 95000 토큰). 시작 후 첫 번째 드래프트 요청은 drafter의 디스크 로드 비용도 부담하는데, 이는 프로세스당 한 번의 비용이며 모든 `timings` 필드에서 제외되고 `Drafter loaded` 로그 줄에 남습니다. [`docs/llama-server-compat.md`](llama-server-compat.md#completions-and-embeddings-are-native-routes-not-openai-aliases-1441)의 `timings` 참고 사항을 확인하십시오.

라운드당 평균 수용 길이는 `(draft_n_accepted + draft_rounds) / draft_rounds`입니다. 모든 라운드가 수용된 드래프트와 보너스 토큰 하나를 내보내기 때문입니다. 이는 보고하지 않고 클라이언트에 맡겨져 있으므로, 다른 집계를 원하는 클라이언트가 이 값에 묶이지 않습니다. `draft_kind`는 `--draft-kind` 이름을 담습니다. 현재 실제로는 `dflash`, `mtp`, `prompt-lookup`입니다. `internal-mtp`는 클래식 디스패치로 해석되며 이를 버스트 게이트가 거부하므로, 이 종류로 speculative하게 서빙되는 요청은 없고 어떤 블록도 보고하지 않습니다.

이 블록은 요청의 전체 라운드 루프를 소유하는 세 경로를 다룹니다. DFlash B=1 버스트, MTP B=1 버스트, 그리고 tick-cooperative MTP 슬라이스(카운터는 한 슬라이스의 것이 아니라 모든 슬라이스에 걸친 세션 총계)이며, 여기에 prompt lookup의 tick별 verify 라운드가 더해지는데 이 카운터는 요청에 대한 drafter의 총계입니다(forward가 뒤따르지 않아 철회된 제안은 세지 않음). 기본값이 꺼져 있는 B>1 배치 버스트는 라운드 루프가 행별 수용 카운터 없이 행별 토큰만 반환하므로 아무것도 보고하지 않습니다.

분리형(disaggregated) 라우터 프런트도 아무것도 보고하지 않습니다. 이는 완료된 `GenerationResult`가 아니라 prefill/decode 핸드오프 프로토콜에서 스트림을 조립하는데, 그 프로토콜은 생성된 토큰 수만 싣고 수용 카운터는 싣지 않으므로, 그곳에 블록을 노출하는 것은 응답 형태의 변경이 아니라 핸드오프 프로토콜의 변경입니다.

## 처리량 수치를 신뢰하기 전에 이득 검증하기

`MLXCEL_SPECULATIVE_ACCEPT_DIAG=1`은 두 번째 줄을 추가합니다.

```
[Speculative acceptance diagnostic] closed_form_sum_min=0.7302 closed_form_sum_prod=0.5651 (the measured per-position acceptance sits at sum_prod under the default sampler-match rule and at sum_min under the opt-in acceptance-optimal rule; sum_min >= sum_prod always)
```

`closed_form_sum_min`은 실행이 실제로 테스트한 verify 위치에 대해 평균한 `sum_x min(p(x), q(x))`이며, modified rejection sampling이 수용하는 확률입니다. `closed_form_sum_prod`는 `sum_x p(x) q(x)`이며 sampler-match 규칙이 수용하는 확률입니다. `a, b`가 `[0, 1]`에 있을 때 `min(a, b) >= a * b`이므로, `sum_min >= sum_prod`는 모든 분포 쌍에 대해 항상 성립합니다. 따라서 이 진단은 "수용률이 이상해 보인다"를 산술적 진술로 바꿔 줍니다.

* 측정된 `per_position_acceptance`는 실제로 실행된 규칙의 닫힌 형태(closed form)에 있어야 합니다. 기본 sampler-match 규칙에서는 `closed_form_sum_prod`, 옵트인 수용 최적 규칙에서는 `closed_form_sum_min`입니다. 어느 쪽인지 알려면 위 줄의 `rule=`을 읽으십시오. 둘 중 어느 쪽에도 있지 않다면 수용 테스트가 잘못된 것입니다.
* `closed_form_sum_min`은 `closed_form_sum_prod` 이상이어야 합니다. 그렇지 않다면 `p` 또는 `q`가 잘못된 것입니다.

테스트된 위치마다 어휘 전체에 대한 리덕션 두 번과 호스트 readback 한 번이 들기 때문에 기본적으로 꺼져 있습니다.

### 이득의 크기는 코드가 아니라 drafter의 엔트로피에 달려 있습니다

drafter가 거의 결정적일 때마다 `sum_x min(p, q)`는 `sum_x p(x) q(x)`로 수렴합니다. `q(t*) ~ 1`이면 `min(p(t*), q(t*)) = p(t*)`이고 `p(t*) q(t*) ~ p(t*)`이므로 두 규칙은 같은 비율로 수용하고 이 변경은 아무것도 얻지 못합니다. 이득은 `q`가 퍼져 있을 때 나타납니다. Llama-3.1-8B-Instruct-4bit와 Llama-3.2-1B-Instruct drafter로, 200 토큰, `--num-draft-tokens 4`, `--no-chat-template`, `--seed 1234`, 각각 하나의 궤적으로 측정했습니다.

| 온도 | `closed_form_sum_min` | `closed_form_sum_prod` | 비율 | 측정된 `per_position_acceptance` |
|---|---|---|---|---|
| 0.7 | 0.7422 | 0.5823 | 1.27x | 0.5870 |
| 1.0 | 0.7983 | 0.6071 | 1.31x | 0.6000 |
| 1.5 | 0.7808 | 0.0002 | 매우 큼 | 0.0000 |

이 실행들은 기본 sampler-match 규칙을 사용하므로 측정값은 `sum_min`이 아니라 `sum_prod`에 있습니다. `sum_min`은 옵트인 규칙이 도달할 상한이며, *비율* 열은 둘 사이의 처리량 여유(headroom)입니다. 0.7과 1.0에서 여유는 약 1.3x입니다. 1.5에서는 두 분포가 충분히 퍼져 있어 독립적인 target 추출이 drafter의 토큰에 사실상 절대 떨어지지 않으므로, 커플링 상한이 여전히 0.78인데도 sampler-match는 전혀 수용하지 못합니다. 그 행이 옵트인 규칙을 옹호하는 논거의 모양입니다.

어떤 처리량 수치보다도 먼저 `closed_form_sum_min / closed_form_sum_prod`를 읽으십시오. 이는 처리량 측정이 보여 줄 수 있는 상한입니다.

이 수치들은 draft 캐시 되감기(rewind) 수정(#994) 이후 다시 측정한 것입니다. 이 페이지의 이전 수치는 거부 라운드마다 draft KV cache가 두 항목씩 너무 많이 잘리던 시기에 측정되었으며, 이로 인해 drafter가 굶주리고 두 닫힌 형태가 측정된 비율과 함께 끌려 내려갔습니다.

## 변경 측정하기

클래식 `SpeculativeGenerator`는 이 규칙이 바뀐 경로이며 오프라인 CLI에서만 접근할 수 있습니다(서버는 speculative 작업을 MTP 및 DFlash 라운드 루프를 통해 라우팅). `speculative_bench`가 아니라 draft 모델을 사용하는 `mlxcel generate`에 대해 A/B를 실행하십시오. `speculative_bench`의 쌍은 모두 MTP 또는 DFlash이므로 변경되지 않은 폴백을 자기 자신과 비교하게 됩니다.

```bash
# Baseline arm: the default sampler-match rule.
MLXCEL_SPECULATIVE_ACCEPT_DIAG=1 \
  ./target/release/mlxcel generate \
    -m models/<target> --draft-model models/<draft> --num-draft-tokens 4 \
    --temp 0.7 --seed 1234 -n 256 -p "<prompt>"

# Opt-in arm: modified rejection sampling.
MLXCEL_SPECULATIVE_STOCHASTIC_ACCEPT=1 MLXCEL_SPECULATIVE_ACCEPT_DIAG=1 \
  ./target/release/mlxcel generate \
    -m models/<target> --draft-model models/<draft> --num-draft-tokens 4 \
    --temp 0.7 --seed 1234 -n 256 -p "<prompt>"
```

`rule=` 필드가 두 팔에서 다른지(`sampler-match` 대 `stochastic`) 확인한 다음, `per_position_acceptance`와 `mean_accepted_len`을 비교하십시오. `--temp 0.0`에서 반복하면, 두 팔은 동일한 출력과 동일한 `rule=greedy-argmax`를 생성해야 합니다.

**drafter 종류 줄에 대한 참고.** 명시적인 `--draft-kind`가 없으면 CLI는 drafter의 `config.json`에서 *자동 감지한* drafter 종류를 출력하는데, 이는 자주 `dflash`이며, 그런 다음 이를 무시하고 어쨌든 클래식 `SpeculativeGenerator`를 실행합니다. 이제 두 번째 줄이 실제로 어떤 경로가 실행되는지 알려 줍니다.

```
Resolved drafter kind: dflash (block_size = 16, default)
Drafter kind was auto-detected only; running the classic SpeculativeGenerator path (pass --draft-kind explicitly to select a kind-specific round loop).
```

`--draft-kind`를 명시적으로 전달해도 `mlxcel generate`에서 DFlash 라운드 루프를 얻는 것은 **아니며**, 오프라인 경로가 이를 구성하지 않는다는 오류를 반환합니다.

**실제 DFlash 또는 DSpark drafter는 로드 전에 거부됩니다.** 위에서 자동 감지된 `dflash`는 일반적인 작은 전체 모델에 적용된 `DEFAULT_DRAFTER_KIND` 폴백이며, 이는 클래식 `SpeculativeGenerator`가 원하는 바로 그것이므로 그 조합은 계속 동작합니다. 실제 DFlash drafter 체크포인트는 다른 객체입니다. `embed_tokens`도 `lm_head`도 포함하지 않는데, 바인딩할 때 target에서 둘 다 빌려 오기 때문에 클래식 경로는 이를 `LoadedModel`로 전혀 로드할 수 없습니다. CLI는 `--draft-kind`의 유무와 관계없이 이를 즉시 거부하고, DFlash 라운드 루프를 실제로 구동하는 경로로 `mlxcel-server`를 지목합니다(#1168). LFM2 DSpark drafter(#1339)와 Muse Glimmer assistant(#1343)도 같은 형태이며 같은 거부를 받습니다. 이 검사가 존재하기 전에는 drafter의 일반적인 `"model_type": "qwen3"` 때문에 Qwen 3 로더로 보내져 `Weight not found: model.embed_tokens.weight`로 실패했습니다.

판별 기준은 해석된 `DrafterKind`가 아니라 구조적인 것입니다(drafter의 `config.json`에 있는 `dflash_config` 블록 및/또는 `architectures: ["DFlashDraftModel"]` 또는 `["Lfm2DSparkDraftModel"]`, 또는 Muse assistant 자체의 `model_type: muse_glimmer_assistant` / `["MuseGlimmerAssistantModel"]`). 해석된 종류에 의존하면 모든 일반 클래식 drafter가 함께 거부되는데, 이들은 모두 `dflash`로 자동 해석되기 때문입니다. 동일한 구조적 검사가 `get_model_type` 내부에서도 실행되므로 `mlxcel generate -m <dflash-dir>`도 "not a standalone model"로 거부됩니다.

## 회귀 방지 장치

`src/lib/mlxcel-core/src/speculative/`에 있습니다.

* `distribution_tests.rs::speculative_stream_matches_the_target_distribution` — 12000 토큰에 대해(`MLXCEL_SPEC_DIST_SAMPLES`로 재정의 가능) 내보낸 스트림이 target의 정확한 categorical과 일치하는지에 대한 카이제곱 적합도 검정이며, 자유도 5, 25.7448 초과 시 기각(alpha = 1e-4).
* `distribution_tests.rs::chi_square_rejects_known_wrong_acceptance_rules` — 검정력(power) 보정. 같은 통계량이 잔차 대신 `p`에서 재샘플링하는 규칙과 argmax-against-argmax 규칙을 기각해야 합니다.
* `distribution_tests.rs::filtered_target_support_is_never_violated` — 허용 오차 0: `top_k = 2` target의 지지집합(support) 밖의 토큰이 하나라도 내보내지면 실패합니다.
* `distribution_tests.rs::temperature_zero_stream_is_byte_identical_to_greedy_target_only` — greedy 출력에 대한 허용 오차 0.
* `distribution_tests.rs::kv_cache_length_is_exact_in_both_termination_regimes`: 메인 캐시 rewind 산술을 전부 거부 및 전부 수용 체제에서 고정합니다.
* `distribution_tests.rs::draft_cache_tracks_the_emitted_sequence_in_every_acceptance_regime`: draft 캐시 rewind 산술. 전부 거부, 전부 수용, 혼합 체제 전반에서 `draft offset + tokens owed == main offset`을 단언하며, 이는 drafter가 target과 정확히 같은 접두부를 조건으로 한다는 진술입니다.
* `distribution_tests.rs::a_fully_accepted_round_owes_its_last_proposal_and_the_next_round_pays_it`: 어떤 trim도 공급할 수 없는 하나의 항목. 완전히 수용된 라운드의 마지막 제안은 draft 모델을 거쳐 forward되지 않은 채 내보내지므로, `pending_draft_context`에 실려 다음 라운드의 맨 앞에서 재생됩니다.
* `stochastic_accept_tests.rs` — `min(1, p/q)`에 대한 수용률, `normalize(relu(p - q))`에 대한 잔차, 그리고 `sum min(p, q)` 최적값.
