# MLA 행렬 흡수(matrix-absorbed) decode

Multi-head latent attention(MLA)은 K와 V를 token당 하나의 latent 벡터로 압축하는 DeepSeek 계열의 attention입니다. 이 페이지는 latent를 먼저 up-projection하지 않고 그 latent에 대해 직접 attention을 수행하는 mlxcel의 행렬 흡수 decode 경로를 다룹니다.

상태: opt-in이며 기본값은 꺼짐입니다. MLA 지오메트리를 가진 합성 텐서에서 검증되었습니다. 실제 DeepSeek 체크포인트에서는 아직 검증되지 않았습니다.

## 항등식

MLA는 폭이 `kv_lora_rank`(출시된 모든 체크포인트에서 512)인 latent `c` 하나와 폭이 `qk_rope_head_dim`(64)인 공유 rope 스트림 `k_pe`를 저장합니다. `kv_b_proj`는 `c`를 head별 key와 value로 up-projection합니다. 그 두 행 블록을 `W_UK`와 `W_UV`라고 하면,

```
score:  q_nope . (W_UK c)  ==  (W_UK^T q_nope) . c
output: attn @ (W_UV c)    ==  (attn @ c) @ W_UV
```

`W_UK`를 query projection에 접어 넣고 `W_UV`를 attention 뒤로 옮기면 산술은 바꾸지 않으면서 decode 그래프에서 up-projection이 사라집니다. 그러면 attention은 하나의 latent KV head를 공유하는 `num_heads`개의 query head로 `c`에 대해 실행되며, 점수는 `kv_lora_rank`에 대한 내적과 `qk_rope_head_dim`에 대한 내적, 두 내적의 합이 됩니다.

## 얻는 것

KV cache가 head 수에 따라 증가하지 않게 됩니다.

| 지오메트리 | 압축 해제 시 바이트/token/레이어 (f16) | latent 바이트/token/레이어 (f16) | 비율 |
|---|---|---|---|
| DeepSeek-V2-Lite (head 16개) | 10240 | 1152 | 8.9x |
| DeepSeek-V3 (head 128개) | 81920 | 1152 | 71.1x |

이 값은 벤치마크 하니스가 출력하는 것과 동일한 함수인 `mlxcel_core::mla::decompressed_bytes_per_token`과 `latent_bytes_per_token`에서 가져온 것이므로, 표가 코드와 어긋날 수 없습니다.

비용은 고정된 가중치 메모리입니다. 흡수된 contraction은 dense batched matmul이고 MLX에는 3차원 head별 양자화 피연산자를 유지하는 batched quantized matmul이 없으므로, fold 과정에서 로드 시점에 `kv_b_proj`를 dequantize하여 dense로 보유합니다. DeepSeek-V2-Lite의 경우 27개 레이어에 걸쳐 f16으로 약 113 MiB이며, 4-bit 원본은 약 28 MiB입니다. 이 교환은 캐시 절감이 이를 넘어서는 컨텍스트 길이를 지난 뒤에야 이득이 됩니다. 하니스는 두 수치를 모두 출력하여 손익분기점이 보이도록 합니다.

## 플래그

| 변수 | 효과 |
|---|---|
| `MLXCEL_MLA_ABSORBED=1` | `(ckv, kpe)`를 캐시하고 흡수 decode를 실행합니다(Stage 1). 기본값 off. |
| `MLXCEL_MLA_SPLIT_KV=1` | 추가로 latent 범위를 청크로 나누고 부분 상태를 병합합니다(Stage 2). 위 항목을 함의하며, 그것 없이는 무시됩니다. |

두 변수 모두 대소문자 구분 없이 `1`, `true`, `on`, `yes`를 받습니다. 설정되지 않은 변수를 포함해 그 외는 모두 off이므로, 오타는 아무도 요청하지 않은 것을 조용히 활성화하는 대신 압축 해제 경로로 degrade됩니다.

`MLXCEL_MLA_ABSORBED`가 설정되지 않으면 fold가 만들어지지 않고, 어떤 가중치도 dequantize되지 않으며, 패밀리는 기존의 압축 해제 경로를 변경 없이 실행합니다.

## 적용 대상

| 패밀리 | 이 작업 이전 | 현재 |
|---|---|---|
| `deepseek_v2` | 압축 해제된 캐시 | 플래그 하에서 흡수됨 |
| `deepseek_v3` | 이미 흡수됨(`sanitize_weights`의 `embed_q` / `unembed_out` 분해) | 변경 없음 |
| `deepseek_v32` | 이미 흡수됨, 동일한 분해 | 변경 없음 |
| `minicpm3` | 압축 해제된 캐시 | 변경 없음, 여전히 압축 해제 |
| `dots1` | 압축 해제된 캐시 | 변경 없음, 여전히 압축 해제 |

V3와 V3.2는 이 작업 이전부터 흡수 변환의 자체 사본을 가지고 있었습니다. 공유 `mlxcel_core::mla` 모듈은 이들이 수렴할 수 있는 곳이며, MiniCPM3와 dots1을 다음에 fold할 수 있는 곳이기도 합니다.

## 캐시 레이아웃과 계속 동작하는 것

latent 캐시는 FP16 모드의 `KVCache`이며, 두 슬롯은 비대칭 텐서를 보유합니다.

| 슬롯 | 보유하는 것 | Shape |
|---|---|---|
| `keys` | `ckv` | `[B, 1, L, kv_lora_rank]` |
| `values` | `kpe` | `[B, 1, L, qk_rope_head_dim]` |

FP16 업데이트 경로는 이미 각 쪽의 head 차원을 자신의 shape에서 읽으므로 새로운 캐시 타입이 필요하지 않습니다. 이 점이 중요합니다. 진정으로 새로운 타입이면 패밀리를 SSM 하이브리드가 사용하는 `ModelOwnedSequenceState` 탈출구로 밀어 넣어야 하며, 이는 batching, paged decode 백엔드, 일반 prompt-cache 기증 경로의 비용을 치르게 합니다. 대신 `offset`, `live_start`, `trim`, `trim_front`, `nbytes`, `can_trim_prompt_cache`는 모두 latent 행에서도 기존 의미를 유지합니다.

슬롯 할당은 `src/models/deepseek_v3.rs`가 항상 해 오던 것과 일치하므로, 공유 타입으로 fold된 V2와 V3의 기존 코드는 동일한 버퍼를 설명합니다.

메시지와 함께 거절되어 압축 해제 경로로 폴백하는 경우:

- FP16이 아닌 모든 KV cache 모드(`int8`, `fp16+turbo4`, `turbo4`, `fp16+turbo3`, `turbo4-delegated`). 이들의 token별 quantization은 head별 K/V 행에 맞춰 보정된 것이며, 이를 읽는 모든 query head에 의해 복원 오차가 증폭되는 512 폭 latent에는 맞지 않습니다.
- Paged backing. pool은 양쪽에 하나의 head 차원을 사용하는 `[num_blocks, page_size, Hkv, head_dim]`을 할당하는데, 비대칭 `(512, 64)` 분할은 이를 채울 수 없습니다.

거절은 캐시별로 결정되고 그 답은 캐시의 수명 동안 고정되므로, 첫 step에서 거절하는 실행은 모든 step에서 거절하며 캐시가 두 레이아웃을 섞는 일은 없습니다.

## Prefill

Prefill은 압축 해제 경로에 남아 있으며, 여전히 로드되어 있는 `kv_b_proj`를 통해 살아 있는 latent 윈도우를 up-projection합니다. 흡수는 decode 변환입니다. 전체 윈도우에 대한 한 번의 up-projection을 모든 query에 적용되는 fold와 맞바꾸는 것이므로, 캐시된 윈도우가 query 수에 비해 클 때만 이득입니다.

이를 위해 원래의 `kv_b_proj`를 유지한다는 것은 흡수된 prefill이 압축 해제 경로가 사용하는 것과 동일한 양자화 가중치에 대한 동일한 산술이라는 뜻입니다. 대가는 캐시가 더 이상 압축 해제된 형태로 보유하지 않기 때문에, chunked prefill이 이전 청크에서 이미 처리한 행을 다시 up-projection한다는 점입니다.

Speculative 및 MTP multi-token verify step도 이 동일한 경로를 사용하므로, 폴백하지 않고 latent 캐시에 대해 계속 동작합니다.

## Stage 2: split-KV

흡수 decode는 head마다 query 행 하나가 캐시된 전체 윈도우에 대해 수행되므로 병렬성이 `batch * num_heads`로 제한되고 컨텍스트가 늘어도 늘지 않습니다. Stage 2는 latent 범위를 청크로 나누고 그 부분 softmax 상태를 나중에 결합하며, 이는 이슈 #898이 paged decode에 대해 한 관찰과 동일합니다.

병합은 이슈 #898의 `paged_attention_merge_states`이며, 커널, FFI 시그니처, C++ launcher를 변경하지 않고 재사용합니다. 그 계약은 의도적으로 paging과 무관합니다. `v_in [N, H, D]`는 자체 분모로 이미 나눈 부분 결과, `lse_in [N, H]`는 **log2** 단위, `o_indptr [M + 1]`은 그룹화입니다. MLA는 `H = num_heads`, `D = kv_lora_rank`, `M = batch`, `N = batch * chunks`로 여기에 맞습니다.

log2 요구 사항은 조용히 실패하는 유일한 조항입니다. 자연로그 LSE도 병합은 되며 그럴듯하지만 잘못된 가중 평균을 반환하므로, 변환은 이름 있는 상수(`mla::split_kv::LOG2_E`)로 두었고 `merge_rejects_natural_log_lse_units`는 자연로그 변형이 레퍼런스와 일치하지 *않음*을 단언하는 음성 대조군입니다.

부분 결과 생산자는 현재 MLX 연산으로 구성됩니다. 올바르며 fused 부분 커널을 검증할 때의 레퍼런스이지만, Stage 1이 하나의 큰 matmul을 실행하는 곳에서 `C`개의 작은 matmul을 실행하므로 Stage 1보다 느릴 것으로 예상됩니다. fused 부분 커널이 존재하기 전에는 Stage 2 처리량 수치를 인용하지 마십시오.

## 벤치마크

```bash
export DEVELOPER_DIR=/Applications/Xcode-26.6.0.app/Contents/Developer
cargo run --release --features metal,accelerate --example mla_absorbed_decode_bench -- \
    --contexts 4096,16384,32768 --batches 1,4 --steps 64 --warmup 16
```

세 가지 방식에 걸쳐 decode step당 MLA attention 블록 하나를 측정하고 KV 메모리 표를 출력합니다. 실제 체크포인트가 아닌 합성 텐서이므로, 수치는 종단 간 모델 처리량이 아니라 변경된 레이어의 처리량입니다.

모든 행은 `mlxcel_core::mla::stats`의 `paths=` 필드로 끝나며, 이는 측정 구간 앞뒤로 취하고 리셋한 값입니다. 하니스는 카운터가 해당 방식의 레이블과 일치하지 않는 행에 `WARNING`을 출력합니다. **`paths=` 필드가 자신의 방식을 가리키지 않는 행은 보고하지 마십시오.** 이슈 #899는 한 번도 활성화되지 않은 fused decode 경로를 배포했고 그 전/후 벤치마크는 폴백을 자기 자신과 비교했습니다. 그 null 결과는 깨끗해 보였고 거의 받아들여질 뻔했습니다.

실제 체크포인트의 경우, `MLXCEL_MLA_ABSORBED=1 mlxcel generate -m models/<deepseek> ...`는 로드 시 몇 개 레이어가 fold되었는지와 전/후 바이트/token을 알리는 stderr 한 줄을 출력합니다. 해당 줄이 `0/27 layers`라고 읽히는 실행은 폴백을 실행하고 있는 것입니다. 이 줄은 stdout이 아닌 stderr로 가므로 생성된 텍스트와 섞이지 않으며 `RUST_LOG`에 의존하지 않습니다.

## 관련 항목

- 구현은 `src/lib/mlxcel-core/src/mla/`.
- 병합 커널과 그 계약은 `src/lib/mlxcel-core/src/paged_v2/`.
- 이 경로가 거절하는 KV cache 모드는 [`docs/turbo-kv-cache.md`](turbo-kv-cache.md).
