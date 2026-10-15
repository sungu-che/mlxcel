# f32 승격 감사

`lablup/mlxcel#1711`은 `mistral4.rs`가 호스트 `&[f32]`로 Llama-4 attention 스케일을 만들었기 때문에 곱셈이 `q`를 f32로 승격시켰고, 그 dtype이 residual stream을 타고 이후 모든 레이어로 전파되어 각 matmul이 자신의 가중치를 승격하도록 강제한다는 것을 발견했습니다. 해당 체크포인트는 캐스트 이전에는 mlx-vlm의 37%, 이후에는 100%로 측정되었으며, 이는 M5 Max에서 5.19배입니다.

해당 이슈의 설명은 f32 상수가 만들어져 activation과 결합되는 다른 모든 위치에 대한 정적 감사를 요청했습니다. 이 파일은 그 결과를 기록하며, 유용한 부분은 부정적인 쪽입니다. 후보 위치 50곳 중 결함은 하나였습니다. `scripts/audit_f32_promotion.py`는 동일한 두 가지 검색을 실행하며, 그 하나의 결함이 수정된 지금은 broad 모드에서 49곳을 보고합니다.

## 검색 대상

f32 배열이 dtype을 지정하지 않은 채 `from_slice_f32`, `full_f32`, `zeros_f32`, `ones_f32`, `arange_f32`로 만들어지고, 이후 다른 피연산자와 함께 `multiply`, `add`, `subtract`, `divide`, `maximum`, `minimum`에 도달합니다. MLX는 더 넓은 피연산자 쪽으로 승격하므로, 그 다른 피연산자가 activation이면 결과는 f32로 해당 줄을 벗어납니다.

세 가지 개선이 중요하며, 각각은 먼저 틀려 본 뒤에 알게 되었습니다. 앞서 캡처한 dtype을 받는 생성자(`full_f32(&[1], alpha, input_dtype)`)는 출처가 아니며, 이를 놓치면 첫 목록에 50곳이 아니라 80곳이 올라갑니다. 의도적으로 f32로 계산하고 마지막에 캐스트하는 블록 역시 결함이 아니며, 캐스트는 보통 몇 줄 뒤의 파생 변수에 적용되므로, 생성된 이름의 캐스트만 찾는 검색은 블록 전체를 보고합니다. 그리고 생성 직후 캐스트로 다시 바인딩된 생성자는 고쳐진 것이지 고장난 것이 아닙니다. 이 확인이 없으면 좁은 검색은 방금 수리하게 만든 바로 그 위치를 보고했습니다.

## 50곳 중 49곳이 결함이 아닌 이유

**승격이 residual stream에 도달하기 전에 캐스트로 제거됩니다.** `topk_scores`에 f32 `routed_scaling_factor`를 곱하는 다섯 개의 MoE router(`glm4_moe`, `solar_open`, `deepseek`, `glm4_moe_lite`, `dots1`)는 모두 점수를 `switch_layers::moe_weighted_sum`에 전달하며, 이 함수는 곱셈 전에 `array_dtype(expert_out)`로, 합산 후에는 요청된 출력 dtype으로 캐스트합니다. f32 연산은 실제로 일어나지만 `[n_tokens, k]` 텐서에 국한됩니다. `ernie4_5_moe_vl`도 같은 함수에 도달합니다.

**블록이 의도적으로 f32로 계산하고 마지막에 캐스트합니다.** `llama4.rs`는 attention 스케일을 만들기 전에 `q_dtype`을 캡처하고 `q`에 곱하기 전에 캐스트하며, 이것이 `#1711`이 llama4를 이 패턴의 원형이며 변경이 필요하지 않다고 말한 의미입니다. `gemma3n.rs`의 sparse activation은 `astype(..., array_dtype(&gate))`로 끝납니다. `deepseek_v4_hyper`의 Sinkhorn 게이트는 `astype(&collapsed, x_dtype)`로 끝납니다. `bailing_moe_linear`의 청크 decay는 같은 표현식 안에서 캐스트합니다.

**해당 양이 설계상 f32입니다.** `deepseek_v4_indexer`는 입력이 f32라고 문서화하고 int32 인덱스를 반환합니다. `glm4v`와 `hunyuan_vl`의 Rope 각도 테이블은 의도적으로 위치를 먼저 f32로 캐스트합니다. `kimi_linear`는 `FLOAT16`을 명시적으로 전달하며, 검색이 이를 표시한 것은 리터럴이 `array_dtype`이 아니기 때문일 뿐입니다.

**해당 위치는 token마다가 아니라 이미지마다 또는 요청마다 한 번 실행됩니다.** 50곳 중 열네 곳은 `src/vision/encoders/`에, 여덟 곳은 `src/vision/` 아래의 다른 곳에, 여섯 곳은 `src/audio/`에, 한 곳은 로드 시점의 `lora/loader.rs`에 있습니다. 그곳의 승격은 decode가 아니라 prefill에 비용을 발생시키며, 어느 것도 밖으로 새어 나가는 것으로 확인되지 않았습니다.

## 실제 결함이었던 하나

`solar_open.rs`의 `gptq_to_mlx_tensors`입니다. 4단계는 `quantized_matmul`과 `gather_qmm`이 해당 dtype을 기대하기 때문이라는 주석과 함께 scale을 f16으로 캐스트합니다. 그런 다음 5단계에서 bias를 계산하는데 두 분기가 서로 달랐습니다. `qzeros` 분기는 `scales_dtype`을 `compute_mlx_biases_from_qzeros`로 전달했지만, 대칭 분기는 호스트 `f32`로 zero point를 만들고 MLX가 승격하도록 두어 f16 scale 옆에 f32 bias를 반환했습니다. `qzeros`가 없는 GPTQ 체크포인트는 모든 양자화 matmul에 불일치하는 쌍을 로드했습니다.

이 함수는 문제를 일으킨 줄의 두 줄 위에서 자신의 의도를 밝히고 있었으며, 이는 기억해 둘 만한 형태입니다. 이것을 찾아내는 검색은 일반 감사보다 좁습니다. 의도된 dtype을 명시하는 함수 안에 있으면서 해당 호출에서 그것을 사용하지 않는 위치입니다.

`gptq_biases_match_the_scales_dtype_in_both_branches`가 이를 고정하며, 쌍을 이루는 것이 불변식이므로 f16을 두 번 명시하는 대신 두 dtype이 같다는 것을 단언합니다. 캐스트를 되돌리면 테스트는 `left: 10` 대 `right: 9`로 실패합니다.

## 다시 실행하기

```bash
python3 scripts/audit_f32_promotion.py --mode narrow   # 19 sites
python3 scripts/audit_f32_promotion.py --mode broad    # 49 sites
```

이 감사는 테스트가 아니라 검색이며 CI에 연결되어 있지 않습니다. 여기서 새로운 f32 생성자를 금지하는 것은 없습니다. 요점은 broad 형태가 거짓 양성률이 높고, 단 하나의 결함은 narrow 형태에서 나왔다는 것입니다. 스크립트는 수정과 같은 방식으로 확인되었습니다. 수리된 코드에 대해서는 아무것도 보고하지 않고, 캐스트를 제거하는 즉시 `solar_open.rs:196`을 다시 지목합니다.
