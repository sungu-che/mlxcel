# f32 attention widen 감사

`lablup/mlxcel#1710`은 `phi-2-4bit`가 f16에서 32개 레이어 중 29번째 레이어에서 NaN이 되고 이후 모든 token이 `!`로 decode된 뒤, 업스트림의 f32 attention 가드를 `src/models/phi.rs`와 `src/models/stablelm.rs`로 포팅했습니다. 이 결함이 살아남을 수 있었던 것은 `lablup/mlxcel#1709`의 `gelu_approx`가 우연히 residual stream을 widen하여 점수를 범위 안에 유지하고 있었기 때문입니다. 해당 이슈는 업스트림이 이런 방식으로 가드하는 다른 패밀리들도 mlxcel 포트에 그 가드를 가지고 있는지 물었고, 답 없이 닫혔습니다. 이 파일이 그 답입니다.

유용한 부분은 부정적인 쪽입니다. 두 리터럴 검색이 반환하는 업스트림 파일 여덟 개 중 세 개는 이미 일치하고, 두 개는 실패 모드가 다른 별개의 가드이며, 하나는 이 가드가 아예 아니고, 하나는 mlxcel 포트가 없으며, 하나는 변경이 필요했습니다. 이 두 리터럴은 두 번째 가드 클래스를 포괄하지 못하므로 두 번째 검색이 필요합니다. 이를 통해 mlxcel 패밀리 네 개가 더 추가되며 그중 세 개도 일치합니다. widen이 정당하게 필요하지 않은 패밀리가 있다는 것도 하나의 결과입니다.

## 검색 대상

`ml-explore/mlx-lm` 및 `Blaizzy/mlx-vlm`으로 범위를 한정하여, 리터럴 `queries.astype(mx.float32)`와 `keys.astype(mx.float32)`에 대해 GitHub 코드 검색 또는 로컬 클론에서의 `git grep`을 사용합니다.

```bash
git -C mlx-lm  grep -nF -e "queries.astype(mx.float32)" -e "keys.astype(mx.float32)" origin/main -- '*.py'
git -C mlx-vlm grep -nF -e "queries.astype(mx.float32)" -e "keys.astype(mx.float32)" origin/main -- '*.py'
```

**2026-09-14 재실행**, 둘 다 2026-09-12자인 `ml-explore/mlx-lm` `dcbcf78` 및 `Blaizzy/mlx-vlm` `45d6e125` 기준입니다. 업스트림은 계속 변하므로 이후의 독자는 표를 믿지 말고 다시 실행해야 합니다.

리터럴은 출발점일 뿐 판정이 아닙니다. 세 가지 서로 다른 것이 이에 일치합니다.

**오버플로 가드**, 즉 #1710이 다룬 것입니다. f16의 최대 유한값은 65504이고, `q @ k^T` 곱이 이를 넘을 수 있으며, softmax가 그 결과인 inf를 NaN으로 바꿉니다. MLX가 점수 matmul을 더 넓은 피연산자에 맞추어 승격하므로, queries만 widen해도 충분합니다.

**블록 선택 가드**, 이는 오버플로 가드와 실패 모드를 공유하지 않습니다. 점수가 KV 블록에 대한 이산적인 top-k 선택을 좌우하며, 반올림이 경계선 근처의 블록을 뒤바꿉니다. 이를 잘못 처리해도 NaN이나 눈에 띄는 고장은 없고 다른 블록이 선택될 뿐입니다. `minimax_m3_vl`은 구조상 이를 보여주고 `qwen4_exp`는 주석으로 이를 밝힙니다.

**둘 다 아님**, 리터럴이 스케일 곱셈이나 평균 reduction에 나타나고 바로 다시 캐스트되는 경우입니다.

## 판정

| 업스트림 위치 | 업스트림이 widen하는 것 | 클래스 | mlxcel | 판정 |
|---|---|---|---|---|
| [mlx-lm `phi.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/phi.py), [mlx-vlm `phi/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phi/language.py) | queries | 오버플로 | `src/models/phi.rs:199` | 일치, 변경 없음 |
| [mlx-lm `stablelm.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/stablelm.py), [mlx-vlm `stablelm/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/stablelm/language.py) | queries와 keys | 오버플로 | `src/models/stablelm.rs:197-198` | 일치, 변경 없음 |
| [mlx-lm `phixtral.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/phixtral.py), [mlx-vlm `phixtral/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/phixtral/language.py) | queries | 오버플로 | `src/models/phixtral.rs:507` | keys와 values를 과도하게 widen함; **일치하도록 축소** |
| [mlx-vlm `paligemma/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/paligemma/language.py) | queries와 keys, `model_type != "gemma"`일 때만 | 오버플로, 단 아래 참조 | 없음; 텍스트 attention은 `src/models/gemma.rs:181`과 `src/models/gemma2.rs:194` | widen 불필요, 측정됨 |
| [mlx-vlm `minimax_m3_vl/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/minimax_m3_vl/language.py) (사본 네 개) | indexer queries와 keys | 블록 선택 | `src/models/minimax_m3_indexer.rs:219`, widen 안 됨 | 미검증, 체크포인트 없음 |
| [mlx-vlm `qwen4_exp/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/qwen4_exp/language.py) | pooled keys와 query | 블록 선택 | 포트 없음 | 할 일 없음 |
| [mlx-vlm `nemotron_labs_diffusion/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/nemotron_labs_diffusion/language.py) | queries와 keys | 오버플로 | 포트 없음 | 할 일 없음 |
| [mlx-vlm `muse_glimmer/language.py`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/muse_glimmer/language.py) | 이 가드가 아님 | 둘 다 아님 | 해당 없음 | 할 일 없음 |

아래에서 언급하는 모든 체크포인트는 f16으로 배포되므로, 각각에 대해 65504 상한이 실제 쟁점입니다. 어느 것도 bf16이 아니며, bf16은 float32의 지수 범위를 가지므로 widen은 범위가 아니라 정밀도에 관한 논거가 됩니다.

## 변경된 항목: phixtral

`src/models/phixtral.rs`는 queries, keys, **그리고** values를 widen했습니다. 두 업스트림 모두 queries만 widen하고 keys와 values는 자신의 dtype으로 전달합니다. 이는 `3576d734`가 `phi.rs`와 `stablelm.rs`에서 제거한 것과 동일한 과도한 widen이며, 이 파일만 남겨져 있었습니다.

축소는 queries 캐스트는 유지하고 나머지 둘을 제거합니다. `phixtral-4x2_8-4bit`로 M1 Ultra에서 측정한 결과:

- **생성 결과가 바이트 단위로 동일합니다.** `MLXCEL_PRINT_TOKEN_IDS=1 ... -p "Paris is" -n 32 --temp 0`은 두 바이너리에서 동일한 32개 id를 생성합니다.
- **Decode 처리량은 변하지 않습니다.** 11-token 프롬프트에서 `-n 128`: 이전 중앙값 95.06 tok/s, 이후 94.94, 1.6% 편차에서 비율 0.9987. 1061-token 프롬프트에서 `-n 256`, 즉 두 배가 된 V 읽기가 가장 큰 비용을 치를 것으로 예상되는 경우: 이전 57.22, 이후 57.19, 1.1% 편차에서 비율 0.9995.

두 번째 결과는 #1829가 작성될 때의 기대, 즉 decode step마다 KV cache의 f32 사본 두 개를 제거하면 처리량이 회복될 것이라는 기대와 모순됩니다. 이 체크포인트에서는 어느 컨텍스트 길이에서도 그렇지 않습니다. 이 축소는 속도 수치가 아니라 업스트림과의 동등성, 그리고 출력이 증명 가능하게 동일한 작업에 비용을 치르지 않는다는 점에 근거합니다. 규모를 비교하면, #1710이 *추가한* 가드는 `phi-2-4bit`를 164.10 tok/s에서 128.76 tok/s로 떨어뜨렸으며, 이 가드는 그보다 두 자릿수 작습니다.

## paligemma와 Gemma 2에 아무것도 필요하지 않은 이유

업스트림의 paligemma attention에는 두 개의 분기가 있습니다. `text_config.model_type == "gemma"`일 때는 widen 없이 `scaled_dot_product_attention`을 호출하며, 이는 PaliGemma 1이고 `src/models/gemma.rs`와 일치합니다. PaliGemma 2(`model_type == "gemma2"`)가 택하는 `else` 분기는 완전한 수동 eager attention입니다. queries를 스케일하고, f32 점수를 계산하고, `attn_logit_softcapping` tanh를 적용하고, f32로 softmax한 뒤, 다시 캐스트합니다. 그 주석은 동기를 밝히고 있으며, 오버플로가 아닙니다.

> Match HF eager semantics: compute logits in fp32, softmax in fp32, then cast attention weights back to query dtype before value matmul.

따라서 mlxcel에게 남는 질문은 영향을 받는 체크포인트에서 점수가 f16 범위를 벗어날 수 있는지뿐입니다. 이는 실제 decode의 모든 레이어에서 `q @ k^T`를 f32로 계산하여 절댓값의 최댓값을 취하는 방식으로 직접 측정했습니다. 일부러 f32를 썼습니다. f16으로 측정하면 포화되어 알고자 한 답을 가려 버리기 때문입니다.

| 체크포인트 | 레이어 샘플 수 | max &#124;q @ k^T&#124; | 65504 아래 여유 |
|---|---|---|---|
| `paligemma2-3b-ft-docci-448-6bit`, 이미지 포함 | 624 | 659.9 | 99x |
| `gemma-2-2b-it-4bit` | 650 | 270.1 | 242x |
| `gemma-2-9b-8bit` | 1050 | 900.8 | 73x |

각 실행은 24개 token을 생성했습니다. 이 점이 중요합니다. 첫 시도는 PaliGemma 2를 `--image` 없이 실행했는데, 이 경우 즉시 EOS를 내보내고 아무것도 생성하지 않은 생성의 prefill forward 세 번에서 78개의 샘플을 수집했습니다. 퇴화된 실행에서 얻은 측정은 측정이 아닙니다.

비교하자면 #1710은 16-token 프롬프트에서 stablelm의 최악 곱을 5534.5로 측정했으며, 이는 11.8x의 여유이고 안전하다고 판단했습니다. Gemma 2의 73x는 여기서 가장 빠듯하지만 그보다 여섯 배 안전합니다. 또한 widen은 `src/models/gemma2.rs`를 공유하는 모든 일반 Gemma 2 텍스트 체크포인트에도 적용되므로, 손이 닿지 않는 범위를 위해 패밀리 전체가 비용을 치르게 됩니다.

Gemma 2의 한 가지 속성은 판정을 바꾸지는 않지만 실패 모드를 바꾸므로 기록해 둘 가치가 있습니다. `attn_logit_softcapping`은 50.0이고 `tanh(inf)`는 1.0이므로, 실제로 오버플로한 점수는 NaN이 되는 대신 상한에서 포화됩니다. 출력은 오버플로한 항목들 사이의 순서를 잃게 되며, 이는 phi의 경우보다 조용한 실패입니다.

## 검증되지 않은 것

**MiniMax-M3.** `src/models/minimax_m3_indexer.rs:219`는 업스트림 사본 네 개가 모두 두 피연산자를 widen하는 곳에서, 입력 dtype으로 `matmul(q, k_t)`를 사용해 token 점수를 계산합니다. 이는 블록 선택 클래스이므로 잘못된 판정이 조용히 지나갑니다. NaN은 없고 sparse attention에 들어가는 KV 블록 집합만 달라집니다. 이 경로는 `should_apply_sparse(kv_len)`가 성립할 때, 즉 `kv_len > 2 * topk_blocks * block_size`일 때만 도달하므로 실행해 보려면 긴 컨텍스트가 필요합니다. 이 머신에는 M3 체크포인트가 없고, `docs/supported-models.md`는 이 패밀리가 머신의 용량을 초과한다고 기록합니다. 통과가 아니라 미검증으로 기록합니다. `ok`로 보고된 건너뛰기 때문에 `lablup/mlxcel#1718`의 NaN 가드가 두 머신 어디에서도 아무것도 다루지 못하게 되었기 때문입니다(`tests/common/mod.rs`).

## 검색 리터럴은 블록 선택 클래스를 포괄하지 못합니다

두 리터럴은 오버플로 가드의 변수 이름을 따서 지어졌습니다. 블록 선택 위치들은 우연히만 일치하며, 그중 하나는 아예 일치하지 않습니다.

`minimax_m3_vl`은 `idx_queries`와 `idx_keys`가 해당 부분 문자열을 포함하기 때문에 일치합니다. `qwen4_exp`는 `pooled_keys`에서는 일치하지만 `query`에서는 일치하지 않으므로 두 위치 중 하나를 통해서만 드러납니다. 같은 클래스인 DeepSeek-V4의 HiSA indexer는 피연산자 이름이 `q`와 `pooled`이므로 둘 다 일치하지 않습니다. 이것은 mlxcel에 있으며, #1829의 범위 노트가 이를 가리켰기 때문에 여기서 발견되었을 뿐입니다.

따라서 이 클래스에는 변수 이름 두 개가 아니라 sparse 블록 선택을 구현하는 파일들을 대상으로 한 별도의 검색이 필요합니다.

```bash
git -C mlx-vlm grep -nF "astype(mx.float32)" origin/main -- '*/hisa_kernel.py' '*_vl/language.py'
grep -rn "FLOAT32" src/models/*indexer*.rs src/models/glm_moe_dsa.rs
```

동일한 ref에서 실행하면 mlxcel 패밀리 네 개가 더 나오며, 그중 세 개가 업스트림과 일치합니다.

| 패밀리 | 업스트림 | mlxcel | 판정 |
|---|---|---|---|
| DeepSeek-V4 HiSA indexer | `q`와 `pooled`를 widen, [`hisa_kernel.py:42-43`](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/models/deepseek_v4/hisa_kernel.py) | `src/models/deepseek_v4_indexer.rs:222-223`이 둘 다 widen | 일치, 변경 없음 |
| DeepSeek-V3.2 indexer | widen 없음; 유일한 f32 캐스트는 MoE router sigmoid, [`deepseek_v32.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v32.py) | `src/models/deepseek_v32_indexer.rs`, widen 없음 | 일치, 변경 없음 |
| GLM-MoE-DSA | f32 캐스트가 전혀 없음, [`glm_moe_dsa.py`](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/glm_moe_dsa.py) | `src/models/glm_moe_dsa.rs`, widen 없음 | 일치, 변경 없음 |
| MiniMax-M3 indexer | widen함 | `src/models/minimax_m3_indexer.rs:219`, widen 없음 | 유일한 격차; 위를 참조 |

DeepSeek-V4가 유용한 항목입니다. 그 indexer는 `minimax_m3_vl`과 같은 이유로 widen하며, mlxcel도 이미 그렇게 하고, 포트는 자체 doc 주석에 dtype을 명시합니다(`q`는 `[B, H, L, D]` f32). 두 리터럴 검색만 신뢰한 독자는 이 패밀리가 가드되지 않았다고 결론 내렸을 것입니다.

## 검색에 다시 일치할 두 항목

이후 재실행에서 다시 문제 삼지 않도록 기록해 둡니다.

`muse_glimmer/language.py`는 `queries.astype(mx.float32)`에 일치하지만 이 가드가 아닙니다. f32로 `qk_scale_factor`를 곱한 뒤 attention 전에 바로 `queries.dtype`으로 다시 캐스트합니다.

`qwen4_exp/language.py`(Qwen3.8-Flash-Next, 이후 GLM-5 Next)에는 의미가 서로 다른 두 개의 일치가 있습니다. 931행에서는 pooled keys의 평균을 f32로 취하고 바로 다시 캐스트하는데, 이는 reduction 정밀도입니다. 975행에서는 블록 선택 점수를 f32로 계산하며, 이는 `minimax_m3_vl`과 같은 클래스입니다. 이 파일은 업스트림에 2026-08-26에 추가되었고 #1829의 표에는 없습니다. mlxcel은 이 패밀리를 포팅하지 않으므로 할 일은 없지만, 누락이 바로 그곳의 표가 일곱 개 파일이고 여기의 표가 여덟 개 파일인 이유입니다.
