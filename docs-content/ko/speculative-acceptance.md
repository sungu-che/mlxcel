# Speculative decoding 수용 규칙

Speculative decoding(추측 디코딩)은 가벼운 draft 모델로 토큰을 제안하고 비용이 큰 target 모델로 이를 검증합니다. *수용 규칙(acceptance rule)*은 어떤 제안이 살아남을지를 결정합니다. 이는 두 가지를 동시에 결정합니다. target forward 한 번당 한 라운드가 내보내는 토큰 수(처리량)와, 내보낸 스트림이 따르는 분포(정확성)입니다. 이 문서는 각 코드 경로가 어떤 규칙을 실행하는지, 각 규칙이 무엇을 보장하는지, 그리고 요청이 speculation의 동작 결과를 어떻게 보고하는지를 설명합니다.

## 규칙

`p`는 검증 위치에서 target 모델의 유효한(effective) 다음 토큰 분포(토큰 bias, 페널티, 온도를 적용하지 않은 분포에 대한 top-k / top-p / min-p 필터링, 재정규화된 필터링 행에 대한 XTC, 그리고 최종 온도 스케일링을 거친 후)를 뜻하고, `q`는 drafter가 제안을 실제로 추출한 분포를 뜻합니다.

### Sampler-match (`temperature > 0`에서의 기본값)

드래프트된 토큰이 `p`에서 새로 추출한 값과 같을 때에 한해 수용하며, 두 분기 모두에서 그 추출값을 내보냅니다. **무손실(lossless)**: 내보낸 모든 토큰이 target에서 새로 추출한 조건부 샘플이므로, 스트림은 정확히 target 단독 샘플과 같습니다. 수용 확률은 `sum_x p(x) q(x)`입니다.



### Greedy argmax (`temperature == 0`)

샘플러가 우연히 `argmax`인 경우의 sampler-match입니다. 같은 논거로 무손실이며, 정확히 greedy target 디코딩입니다.

### Argmax-against-argmax (MTP, DFlash, DSpark, Muse Glimmer assistant)

온도와 관계없이 드래프트된 토큰이 `argmax(target_logits)`와 같을 때 수용하고, 불일치하면 argmax를 내보냅니다. `temperature > 0`에서는 **무손실이 아닙니다**. 서빙되는 스트림은 `p`의 샘플이 아니라 target의 greedy 스트림입니다. 수용 확률은 `q(argmax p)`입니다.

Muse Glimmer assistant drafter(#1343)는 이 규칙 아래 동일한 DFlash 라운드 루프에서 실행되며, Qwen 3.5 DFlash drafter와 디스패치 차이가 없습니다. 제안은 요청의 샘플러에 따라 마스크된 슬롯마다 샘플링되고 target의 argmax에 대해 수용됩니다. `temperature == 0`에서는 아래의 block-versus-chain 정확성 프로브(probe)를 전제로, target의 rotating 캐시가 가진 2048 토큰 슬라이딩 윈도우를 넘어서도 정확히 greedy target 디코딩입니다.

이 drafter의 verify 폭은 DFlash 루프에서 유일한 적응형 폭이므로 프로브가 한 번 이상 실행됩니다. 라운드 루프는 drafter가 선언한 깊이인 4개 행으로 워밍업하고, 측정 윈도우가 그만한 가치가 있음을 보일 때만 요청된 상한까지 넓히며, 게이트는 버스트가 시작되기 전에 두 폭을 모두 통과시킵니다. 상한만 프로브하면 워밍업 폭이 측정되지 않은 채 허용되는데, 이는 형식적인 문제가 아닙니다. forward 폭이 MLX가 디스패치하는 양자화 matmul 커널을 선택하며, `docs/benchmarks.md`는 동일한 비교가 폭 8에서 20.6퍼센트, 폭 32에서 0.0퍼센트의 불일치를 보인다고 기록합니다.

LFM2 DSpark drafter는 DFlash 라운드 루프 위에서 실행되므로 이 규칙을 따르며, 디스패치에 한 가지 차이가 있습니다. greedy 전용이라는 점입니다. drafter의 Markov 헤드는 greedy target에 대해 학습된 체인을 제안하며, 서버는 샘플러가 argmax가 아닌 요청(`top_k != 1`인 `temperature > 0`)을 편향된 greedy 스트림으로 서빙하는 대신 경고 한 번과 함께 클래식 디코드로 보냅니다. `temperature == 0`에서는 아래의 block-versus-chain 정확성 프로브를 전제로 정확히 greedy target 디코딩입니다.



## 수용 상한

`q`로 분포된 제안과 `p`로 분포된 방출(emission)에 대해, 둘이 일치할 수 있는 최대 확률은 `sum_x min(p(x), q(x))`이며, 이는 최대 커플링(maximal-coupling) 한계이고 `1 - TV(p, q)`와 동등합니다. **모든** 분포 보존 규칙은 여기까지로 제한됩니다.

| 규칙 | 무손실 | 수용 확률 |
|------|----------|------------------------|
| Sampler-match | 예 | `sum_x p(x) q(x)` |
| Argmax-against-argmax | **아니요** | `q(argmax p)` (상한에 의해 제한되지 않음) |

`q(argmax p)`에는 그런 제약이 없습니다. `p(a) = 0.3`, `q(a) = 0.95`이면 argmax 규칙은 0.95로 수용하지만 올바른 어떤 규칙이든 상한은 약 0.30입니다. 따라서 일반적으로 어떤 분포 보존 규칙도 argmax 수용에 비해 평균 수용 길이를 엄밀하게 개선할 수 없습니다.

Modified rejection sampling(Leviathan et al. 2023, Chen et al. 2023)은 이 상한에 도달합니다. 옵트인 구현이 클래식 draft 모델 루프에 포함되어 있었으나 그 루프와 함께 v0.8.0에서 제거되었습니다(#2754). 그 루프를 대체한 full-model drafter(#2757)는 엔진의 토큰 전용 루프가 쓰는 sampler-match 규칙을 실행하며, 이는 무손실이지만 수용 최적 규칙을 복원하지는 않습니다. 다음 절을 참고하십시오.

## 현재 각 경로가 실행하는 규칙

| 경로 | `temperature == 0` | `temperature > 0` |
|------|--------------------|-------------------|
| 엔진의 토큰 전용 루프에서의 `PromptLookupDrafter`(`mlxcel generate --prompt-lookup`) | greedy argmax (무손실) | **sampler-match** (무손실이며 여기서는 수용 최적) |
| 서버 스케줄러에서의 `PromptLookupDrafter`(`--draft-kind prompt-lookup`, 동일한 `Engine::verify_round`) | greedy argmax (무손실) | **sampler-match** (무손실이며 여기서는 수용 최적) |
| 엔진의 토큰 전용 루프에서의 `FullModelDrafter`(`mlxcel generate --draft-model <일반 모델>`, `mlxcel-server --draft-model <일반 모델>`, #2757) | greedy argmax. 프로브 게이트 없음. plain decode와 바이트 단위로 동일하다고 보장되지 않음(prompt lookup과 같은 부류이며 같은 스트림. [오프라인 라우팅](#오프라인-라우팅) 참고) | **sampler-match** (무손실): drafter는 자신의 argmax를 제안하고, target의 추출이 결정 |
| Gemma 4 MTP 라운드 루프 | argmax (여기서는 무손실) | argmax-against-argmax (**편향됨**) |
| DFlash 라운드 루프(Qwen 3.5 DFlash drafter) | argmax (여기서는 무손실) | argmax-against-argmax (**편향됨**) |
| DFlash 라운드 루프(LFM2 DSpark drafter) | argmax (여기서는 무손실, 프로브 게이트) | 클래식 디코드로 전환 |
| DFlash 라운드 루프(Muse Glimmer assistant drafter) | argmax (여기서는 무손실, 프로브 게이트) | argmax-against-argmax (**편향됨**) |

이 페이지에서 "클래식 디코드"는 speculative하지 않은 일반 디코딩을 뜻합니다. MTP와 DFlash verify 경로는 샘플러와 관계없이 `argmax_per_position` / `argmax_logits_to_array`로 target 토큰을 선택하며, 이들의 drafter는 추출에 사용한 분포 없이 토큰 id만 반환합니다(`Drafter::draft_block`은 `Vec<i32>`를 반환). 이들을 무손실로 만들려면 drafter 인터페이스와 `MtpTarget::verify_forward`의 출력을 모두 확장해야 합니다.

Prompt lookup은 결정적으로 제안하므로 `q`는 드래프트된 토큰 `d`에서 one-hot입니다. 그러면 두 수용 확률은 `p(d)`로 일치하며(`sum_x p(x) q(x) = sum_x min(p(x), q(x)) = p(d)`), 거부 시 두 규칙 모두 `d`가 아니라는 조건을 건 `p`에서 추출한 값을 내보냅니다. 따라서 이 경로에서는 sampler-match가 이미 수용 최적 규칙입니다.

CUDA 서버의 prompt-lookup 거버너는 추가로, 수용된 제안 수에 1을 더한 값의 EMA가 4 토큰 아래로 내려가고, 최소 두 번의 드래프트 라운드 이후이며 가장 최근 라운드도 하한 미만일 때 수확이 낮은 행을 일시 중지합니다. 하한을 충족한 최근 라운드는 일시 중지 없이 초기 EMA 부채를 상환할 수 있습니다. 수용된 토큰 EMA가 하한 이상이었던 후에 발생한 고립된 수용 0 라운드는 한 라운드의 유예를 받으며, 부분적인 저수확 수용은 그렇지 않습니다. 유예는 연속된 첫 번째 저수확 라운드를 넘어 연장되지 않으며, 일시 중지에는 여전히 내보낸 토큰 EMA가 하한 미만이어야 합니다. 이 측정된 필드 경계 휴리스틱은 수용 규칙 자체를 바꾸지 않습니다. 이는 제안 비용에 대한 결정이지 sampler-match 수용 규칙의 변경이 아닙니다. 일시 중지된 행도 여전히 내보낸 토큰을 관찰하고 섀도 룩업을 실행하며, 일치하는 섀도 토큰 일곱 개가 verify를 복원합니다. 단순 요청(plain ask), 철회된 제안, 게이트로 제외된 행은 verify 증거를 추가하지 않습니다. CLI에는 손익분기 하한이 없으며 두 토큰 섀도 확인을 유지합니다. `adaptive = false`는 하한을 무시하고, graded 정책은 변경되지 않습니다.



## 오프라인 라우팅

`mlxcel generate --draft-model <dir>`은 세 가지 라운드 루프 중 하나를 실행합니다. MTP(명시적 `--draft-kind mtp` 또는 MTP `model_type`), DFlash(오프라인 DFlash arm이 있는 target(현재는 Laguna)에서의 DFlash drafter 디렉터리, 명시적이거나 자동 감지됨), 또는 full-model drafter(#2757)입니다. full-model drafter는 DFlash 계열 마커도 MTP `model_type`도 없는 디렉터리에서 자동 감지되거나 `--draft-kind full-model`로 선택되는 일반적인 더 작은 체크포인트입니다. full-model drafter는 엔진의 토큰 전용 루프, 즉 `--prompt-lookup`이 실행하는 것과 같은 루프이며, draft 모델 자체의 KV cache를 target의 것 옆에서 되감습니다. target의 tokenizer를 공유하는 drafter가 필요하고, target보다 큰 draft 어휘는 로드 전에 거부됩니다. 그 밖의 경우는 drafter를 로드하기 전에 `--draft-kind mtp`, `--draft-kind dflash`, `--draft-kind full-model`, `--prompt-lookup`을 명시하는 오류로 실패합니다. 캐시 trim으로 되감을 수 없는 상태를 가진 target 또는 drafter(순환, 하이브리드, 모델 소유 패밀리)는 prompt lookup에 적용되는 `Engine::verify_rounds_unsupported`의 같은 규칙으로 거부되며(#2495), `mlxcel-server`에서도 마찬가지입니다. CLI는 실행 후 stderr에 drafter의 수용 및 draft-forward 횟수를 출력합니다.

greedy 정확성은 토큰 전용 루프의 것이며, MTP와 DFlash 경로가 block-versus-chain 프로브를 통해 도달하는 수준에는 미치지 못합니다. temperature 0에서 스트림은 모든 위치에서 target 자신의 argmax이지만 plain decode와 바이트 단위로 동일하다고 보장되지 않으며, 그 차이를 기준으로 게이트하지도 않습니다. #2757의 M1 Ultra 기록에서 실제 쌍 실행 36회 중 32회가 plain decode와 일치했습니다. 나머지 네 번은 target 단독에서 비롯됩니다. draft 모델도 draft 캐시도 없는 `--prompt-lookup`이 같은 프롬프트에서 같은 다른 스트림을 내보내며, target을 자신의 plain 스트림에 대해 teacher-forced로 재생하면 그 KV 히스토리가 멀티 토큰 forward로 구축될 때 같은 뒤집힘이 재현됩니다. #2757의 pull request가 그 증거를 담고 있습니다. drafter의 제안만으로는 스트림을 움직일 수 없습니다. target이 모든 위치를 자신의 캐시로부터 결정하기 때문이며, 정렬이 어긋난 draft 캐시는 수용률만 잃습니다. 샘플링된 스트림은 카이제곱 및 Markov 가드를 통과합니다.

오프라인 arm이 없는 target의 DFlash 또는 DSpark drafter는 같은 거부와 함께, 더 많은 target에 대해 DFlash 라운드 루프를 구동하는 `mlxcel-server`로의 안내를 받습니다(#1168). 이 체크포인트들은 `embed_tokens`도 `lm_head`도 제공하지 않으며, 바인딩할 때 둘 다 target에서 빌려 오기 때문입니다. LFM2 DSpark drafter(#1339)와 Muse Glimmer assistant(#1343)도 같은 형태입니다.

판별 기준은 해석된 `DrafterKind`가 아니라 구조적인 것입니다(drafter의 `config.json`에 있는 `dflash_config` 블록 및/또는 `architectures: ["DFlashDraftModel"]` 또는 `["Lfm2DSparkDraftModel"]`, 또는 Muse assistant 자체의 `model_type: muse_glimmer_assistant` / `["MuseGlimmerAssistantModel"]`). 일반 전체 모델도 `dflash`로 자동 해석되므로 해석된 종류로는 둘을 구별할 수 없습니다. 동일한 구조적 검사가 `get_model_type` 내부에서도 실행되므로 `mlxcel generate -m <dflash-dir>`도 "not a standalone model"로 거부됩니다.

## 실행 읽어내기


`RUST_LOG`를 설정하면 오프라인 CLI 명령에 subscriber가 설치되어(stderr에 기록하므로 파이프된 stdout은 깨끗하게 유지됨) drafter 자동 감지 줄이 나타납니다. `RUST_LOG`가 설정되지 않으면 기본 출력은 변하지 않습니다. `mlxcel serve`는 자체 subscriber를 설치하므로 영향을 받지 않습니다. `mlxcel generate --prompt-lookup`은 실행 후 stderr에 drafter 통계를 출력합니다.



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

라운드당 평균 수용 길이는 `(draft_n_accepted + draft_rounds) / draft_rounds`입니다. 모든 라운드가 수용된 드래프트와 보너스 토큰 하나를 내보내기 때문입니다. 이는 보고하지 않고 클라이언트에 맡겨져 있으므로, 다른 집계를 원하는 클라이언트가 이 값에 묶이지 않습니다. `draft_kind`는 `--draft-kind` 이름을 담습니다. `dflash`, `mtp`, `prompt-lookup` 중 하나입니다.

이 블록은 요청의 전체 라운드 루프를 소유하는 세 경로를 다룹니다. DFlash B=1 버스트, MTP B=1 버스트, 그리고 tick-cooperative MTP 슬라이스(카운터는 한 슬라이스의 것이 아니라 모든 슬라이스에 걸친 세션 총계)이며, 여기에 prompt lookup의 tick별 verify 라운드가 더해지는데 이 카운터는 요청에 대한 drafter의 총계입니다(forward가 뒤따르지 않아 철회된 제안은 세지 않음). 기본값이 꺼져 있는 B>1 배치 버스트는 라운드 루프가 행별 수용 카운터 없이 행별 토큰만 반환하므로 아무것도 보고하지 않습니다.

분리형(disaggregated) 라우터 프런트도 아무것도 보고하지 않습니다. 이는 완료된 `GenerationResult`가 아니라 prefill/decode 핸드오프 프로토콜에서 스트림을 조립하는데, 그 프로토콜은 생성된 토큰 수만 싣고 수용 카운터는 싣지 않으므로, 그곳에 블록을 노출하는 것은 응답 형태의 변경이 아니라 핸드오프 프로토콜의 변경입니다.




































## 회귀 방지 장치

`src/lib/mlxcel-core/src/engine/speculative_distribution_tests.rs`에 있으며, 모두 엔진의 토큰 전용 verify 루프에 대한 것입니다.

* `speculative_stream_matches_the_target_distribution`(및 `_at_block_size_one`): 12000 토큰에 대해(`MLXCEL_SPEC_DIST_SAMPLES`로 재정의 가능) 내보낸 스트림이 target의 정확한 categorical과 일치하는지에 대한 카이제곱 적합도 검정이며, 자유도 5, 25.7448 초과 시 기각(alpha = 1e-4). target이 컨텍스트에 독립적이므로 이 두 테스트는 샘플러 경로가 스트림에 들어가는 것을 지키지만 잘못된 수용/거부 결정은 볼 수 없습니다.
* `markov_stream_keeps_the_target_conditional`: 분포가 이전 토큰에 의존하는 target에서 `(previous, next)`에 대한 같은 통계량, 자유도 30. 이것이 수용/거부 결정에 대한 확률적 가드입니다.
* `chi_square_rejects_known_wrong_acceptance_rules`: 호스트 추출에 대한 검정력(power) 보정. 같은 통계량이 제안을 검사 없이 그대로 내보내는 규칙과 argmax-against-argmax 규칙을 기각해야 합니다.
* `filtered_target_support_is_never_violated`: 허용 오차 0. `top_k = 2` target의 지지집합(support) 밖의 토큰이 하나라도 내보내지면 실패합니다.
* `temperature_zero_stream_is_byte_identical_to_greedy_target_only`: greedy 출력에 대한 허용 오차 0.
* `acceptance_accounting_is_exact_in_both_deterministic_regimes`: 전부 거부 및 전부 수용 체제에서의 `SpeculativeRounds` 카운터.

`engine/verify_round_rollback_tests.rs`(`kv_cache_length_is_exact_in_both_acceptance_regimes`, `kv_cache_stays_consistent_across_mixed_accept_reject_rounds`)는 모든 라운드 후에 모든 레이어에서 상태가 프롬프트에 보류 중인 토큰을 제외한 모든 내보낸 토큰을 더한 것을 담고 있음을 고정합니다. `src/lib/mlxcel-core/src/cache/kv_trim_tests.rs`는 `KVCache::trim`을, `src/commands/generate_tests.rs`는 오프라인 라우팅 매트릭스를 고정합니다.

