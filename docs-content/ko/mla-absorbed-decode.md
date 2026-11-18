# MLA 행렬 흡수(matrix-absorbed) decode

Multi-head latent attention(MLA)은 K와 V를 token당 하나의 latent 벡터로 압축하는 DeepSeek 계열의 attention입니다. 이 페이지는 latent를 먼저 up-projection하지 않고 그 latent에 대해 직접 attention을 수행하는 mlxcel의 행렬 흡수 decode 경로를 다룹니다.

상태: DeepSeek V2에 대해 opt-in이며 기본값은 꺼짐입니다. deepseek-v2-lite-chat-4bit-mlx에서 기본 압축 해제 레이아웃과 비교 측정되었습니다(이슈 #3102). 왜 opt-in으로 유지되는지는 [DeepSeek V2의 기본 레이아웃](#deepseek-v2의-기본-레이아웃)을 참조하십시오.

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

비용은 고정된 가중치 메모리이며, 그 양은 체크포인트에 따라 다릅니다. 양자화된 `kv_b_proj`는 역양자화(dequantizing) 없이 fold됩니다. 패킹된 행들은 rank 축을 따라 헤드별로 재그룹화되므로, `absorb_queries`와 `unabsorb_output`은 체크포인트 자체의 평면에서 배치된 `quantized_matmul`로 실행됩니다(DeepSeek-V2-Lite에서 30.4 MiB, 4-bit 원본 28 MiB 대비)(#3121). 양자화되지 않은 레이어, NVFP4 글로벌 스케일, 런타임 LoRA 항 또는 모드 불일치는 dense f16 fold(V2-Lite에서 약 113 MiB)를 유지합니다. 로드 시점의 로그 라인이 `folded 27/27 layers (27 quantized, 30.4 MiB)`와 같이 어느 쪽인지 표시합니다.

## DeepSeek V2의 기본 레이아웃

DeepSeek V2는 fused 흡수 decode 커널이 실행되는 모든 곳에서 MLA latent를 기본적으로 캐시하며(#3131), 그 외의 환경이나 `MLXCEL_MLA_ABSORBED=0`에서는 헤드별 K와 V를 압축 해제하여 캐시합니다. 이 기본값은 두 차례의 측정을 거쳐 결정되었습니다. 이슈 #3102는 M5 Max(2026-11-15)에서 deepseek-v2-lite-chat-4bit-mlx를 대상으로 두 레이아웃을 측정했으며, 디코드 처리량을 몇 퍼센트 이내로 유지하고 teacher-forced 트레이스가 실질적인 불일치를 보이지 않는 경우에만 latent 레이아웃을 기본값으로 삼는 규칙을 적용했습니다:

| 측정 항목 | 압축 해제 (당시 기본값) | 흡수 | |
|---|---|---|---|
| decode, pp512/tg128 (중앙값 tok/s) | 200.6 | 180.7 | -9.9% |
| decode, 2048 토큰 프롬프트, tg128 | 168.1 | 161.9 | -3.7% |
| decode, 16384 토큰 프롬프트, tg128 (평균) | 70.9 | 111.2 | +57% |
| 피크, 28,705 토큰 프롬프트, 2048 토큰 조각 | 20.20 GB | 13.52 GB | -6.68 GB |
| 피크, 동일 프롬프트, 단일 패스 | 48.55 GB | 41.02 GB | -7.53 GB |
| logit 트레이스, 불일치하는 확실한 위치 | 레퍼런스 | 3,653개 중 0개 | prefill 폭 2048 및 decode 폭 1 |

트레이스 조건은 충족되었습니다. 모든 불일치는 레퍼런스의 상위 2개가 0.19 logit 이내인 지점에서만 발생했으므로 두 레이아웃은 반올림 클래스에서만 다릅니다. 그러나 처리량 조건은 충족되지 않았습니다. 흡수 decode는 512 토큰 컨텍스트에서 스텝당 약 0.55 ms의 고정 비용을 수반하며, 더 작은 캐시 읽기가 이를 보상하는 것은 대략 3K 토큰을 지난 시점부터였습니다. 짧은 채팅 길이 요청에서 필요 없는 메모리를 아끼기 위해 10% 더 느려지는 기본값은 당시 기각되었고, 압축 해제 레이아웃이 기본값으로 유지되었습니다.

이 결정을 바꾸기 위해 필요한 것은 해당 고정 비용을 없애는 것이었습니다. 이슈 #3121이 그 대부분을 제거하면서 결정이 다시 열렸습니다:

- 이제 fold가 양자화 상태로 실행됩니다.
- 마스크되지 않은 단일 쿼리 흡수 attention은 최대 8192개 키 윈도우까지 fused Metal 커널(`MLXCEL_MLA_FUSED_DECODE`, 기본값 on)에서 실행됩니다.

동일한 체크포인트에서 재측정된 결과(2026-11-16):

| 측정 항목 | 압축 해제 (당시 기본값) | 흡수 | |
|---|---|---|---|
| decode, pp512/tg128 (중앙값 tok/s) | 205.5 | 201.4 | -2.0% |
| decode, 2048 토큰 프롬프트, tg128 | 170.1 | 180.7 | +6.2% |
| decode, 16384 토큰 프롬프트, tg128 | 73.5 | 114.4 | +56% |
| logit 트레이스, 불일치하는 확실한 위치 | 레퍼런스 | 393개 중 0개 | 1024, 4096, 8192 뒤의 decode 폭 1 |

두 조건이 모두 충족되었으며, 이슈 #3131은 측정이 이루어진 Metal에서 흡수 decode를 기본값으로 지정했습니다. 전환된 바이너리에서 재측정된 결과(2026-11-16, 공유 호스트이므로 절대 속도는 낮으며, arm들이 인터리브됨):

| 측정 항목 | `MLXCEL_MLA_ABSORBED=0` (압축 해제) | 기본값 (흡수) | |
|---|---|---|---|
| decode, pp512/tg128 (중앙값 tok/s) | 165.7 | 170.8 | +3.1% |
| decode, 2048 토큰 프롬프트, tg128 | 150.3 | 157.0 | +4.5% |
| decode, 16384 토큰 프롬프트, tg128 | 68.7 | 100.9 | +47% |

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
| `deepseek_v2` | 압축 해제된 캐시 | Metal에서는 기본적으로 흡수됨(#3131), 그 외에서는 기본적으로 압축 해제됨; `MLXCEL_MLA_ABSORBED`로 강제 가능 |
| `deepseek_v3` | 이미 흡수됨(`sanitize_weights`의 `embed_q` / `unembed_out` 분해) | 변경 없음 |
| `deepseek_v32` | 이미 흡수됨, 동일한 분해 | 변경 없음 |
| `minicpm3` | 압축 해제된 캐시 | 변경 없음, 여전히 압축 해제 |
| `dots1` | 압축 해제된 캐시 | 변경 없음, 여전히 압축 해제 |

V3와 V3.2는 이 작업 이전부터 흡수 변환의 자체 사본을 가지고 있었습니다. 공유 `mlxcel_core::mla` 모듈은 이들이 수렴할 수 있는 곳이며, MiniCPM3와 dots1을 다음에 fold할 수 있는 곳이기도 합니다.

## 캐시 레이아웃과 계속 동작하는 것

latent 캐시는 FP16 모드의 `KVCache`이며, 두 슬롯은 비대칭 텐서를 보유합니다.

| 슬롯 | 보유하는 것 | 형태 |
|---|---|---|
| `keys` | `ckv` | `[B, 1, L, kv_lora_rank]` |
| `values` | `kpe` | `[B, 1, L, qk_rope_head_dim]` |

FP16 업데이트 경로는 이미 각 쪽의 head 차원을 자신의 shape에서 읽으므로 새로운 캐시 타입이 필요하지 않습니다. 이 점이 중요합니다. 진정으로 새로운 타입이면 패밀리를 SSM 하이브리드가 사용하는 `ModelOwnedSequenceState` 탈출구로 밀어 넣어야 하며, 이는 batching, paged decode 백엔드, 일반 prompt-cache 기증 경로의 비용을 치르게 합니다. 대신 `offset`, `live_start`, `trim`, `trim_front`, `nbytes`, `can_trim_prompt_cache`는 모두 latent 행에서도 기존 의미를 유지합니다.

슬롯 할당은 `src/models/deepseek_v3.rs`가 항상 해 오던 것과 일치하므로, 공유 타입으로 fold된 V2와 V3의 기존 코드는 동일한 버퍼를 설명합니다.

메시지와 함께 거절되어 압축 해제 경로로 폴백하는 경우:

- FP16이 아닌 모든 KV cache 모드(`int8`, `fp16+turbo4`, `turbo4`, `fp16+turbo3`, `turbo4-delegated`). 이들의 token별 quantization은 head별 K/V 행에 맞춰 보정된 것이며, 이를 읽는 모든 query head에 의해 복원 오차가 증폭되는 512 폭 latent에는 맞지 않습니다.
- Paged backing. pool은 양쪽에 하나의 head 차원을 사용하는 `[num_blocks, page_size, Hkv, head_dim]`을 할당하는데, 비대칭 `(512, 64)` 분할은 이를 채울 수 없습니다.

거절은 캐시별로 결정되고 그 답은 캐시의 수명 동안 고정되므로, 첫 step에서 거절하는 실행은 모든 step에서 거절하며 캐시가 두 레이아웃을 섞는 일은 없습니다.

## 다중 토큰 스텝

흡수는 단일 토큰 디코드에만 국한되지 않습니다. 헤드별로 `r = kv_lora_rank`, `d = qk_nope_head_dim + v_head_dim`, `L`개의 쿼리 및 이번 스텝 업데이트 후의 `T`개의 캐시된 latent 위치에 대해, 흡수 경로의 비용은 `L*r*d + 2*L*T*r`이고 구체화(materializing) 비용은 `T*r*d + L*T*d`입니다. `L < r*d / (r*d/T + 2*r - d)`인 동안 흡수가 우세합니다. `mlxcel_core::mla::use_absorbed_mla`는 호출마다 이 한계를 한 번 평가하며 모든 MLA 패밀리는 쿼리 개수 대신 이 결과에 따라 분기합니다.

단일 쿼리는 항상 흡수됩니다. `T`가 커질수록 이 한계는 `r*d / (2*r - d)`(DeepSeek-V3의 경우 170 쿼리, GLM-4.7-Flash의 경우 398)에 수렴하므로, 긴 캐시 뒤의 speculative verify 블록이나 chunked-prefill 연속은 흡수로 실행됩니다. 콜드 캐시(`T == L`)에서는 배포된 모든 구성에 대해 한계가 실제 `L` 미만으로 떨어지므로, 새로운 prefill은 여전히 구체화 경로를 거치며 기준선과 수치적으로 동일하게 유지됩니다.

두 경로 모두 동일한 latent 캐시를 기록하므로, 아무것도 다시 패킹하지 않고 스텝 간에 두 경로를 전환할 수 있습니다. 다중 토큰 흡수 스텝에는 구체화 경로가 `causal_attention`에서 얻는 causal mask가 필요합니다. `deepseek_v3`과 `deepseek_v32`는 흡수 또는 구체화에 관계없이 모든 다중 토큰 스텝을 `mla::rope_folded_attention`을 통해 실행하며, 이는 rope 절반을 SDPA 헤드 차원에 이어 붙이고 causal 모드 또는 호출자의 마스크를 적용합니다(#3071). 공유 `absorbed_attention`도 다중 토큰 스텝에 대해 동일한 folded 호출을 사용하며, `deepseek_v2`의 `forward_absorbed`는 호출자가 마스크를 전달하지 않은 경우 causal mask를 구성하고 호출자가 제공한 패딩 마스크가 우선합니다. 구체화 측은 여전히 원래의 `kv_b_proj`를 로드된 상태로 유지하며, 캐시가 더 이상 압축 해제된 형태로 보관하지 않으므로 이전 청크가 이미 처리한 행을 다시 up-projection합니다.

## 2단계: split-KV

흡수 decode는 head마다 query 행 하나가 캐시된 전체 윈도우에 대해 수행되므로 병렬성이 `batch * num_heads`로 제한되고 컨텍스트가 늘어도 늘지 않습니다. Stage 2는 latent 범위를 청크로 나누고 그 부분 softmax 상태를 나중에 결합하며, 이는 이슈 #898이 paged decode에 대해 한 관찰과 동일합니다.

병합은 이슈 #898의 `paged_attention_merge_states`이며, 커널, FFI 시그니처, C++ launcher를 변경하지 않고 재사용합니다. 그 계약은 의도적으로 paging과 무관합니다. `v_in [N, H, D]`는 자체 분모로 이미 나눈 부분 결과, `lse_in [N, H]`는 **log2** 단위, `o_indptr [M + 1]`은 그룹화입니다. MLA는 `H = num_heads`, `D = kv_lora_rank`, `M = batch`, `N = batch * chunks`로 여기에 맞습니다.

log2 요구 사항은 조용히 실패하는 유일한 조항입니다. 자연로그 LSE도 병합은 되며 그럴듯하지만 잘못된 가중 평균을 반환하므로, 변환은 이름 있는 상수(`mla::split_kv::LOG2_E`)로 두었고 `merge_rejects_natural_log_lse_units`는 자연로그 변형이 레퍼런스와 일치하지 *않음*을 단언하는 음성 대조군입니다.

부분 결과 생산자는 현재 MLX 연산으로 구성됩니다. 올바르며 fused 부분 커널을 검증할 때의 레퍼런스이지만, Stage 1이 하나의 큰 matmul을 실행하는 곳에서 `C`개의 작은 matmul을 실행하므로 Stage 1보다 느릴 것으로 예상됩니다. fused 부분 커널이 존재하기 전에는 Stage 2 처리량 수치를 인용하지 마십시오.

## 벤치마크

```bash
export DEVELOPER_DIR=/Applications/Xcode-26.6.0.app/Contents/Developer
cargo run --release --features metal,accelerate --example mla_absorbed_decode_bench --     --contexts 4096,16384,32768 --batches 1,4 --steps 64 --warmup 16
```

세 가지 방식에 걸쳐 decode step당 MLA attention 블록 하나를 측정하고 KV 메모리 표를 출력합니다. 실제 체크포인트가 아닌 합성 텐서이므로, 수치는 종단 간 모델 처리량이 아니라 변경된 레이어의 처리량입니다.

모든 행은 `mlxcel_core::mla::stats`의 `paths=` 필드로 끝나며, 이는 측정 구간 앞뒤로 취하고 리셋한 값입니다. 하니스는 카운터가 해당 방식의 레이블과 일치하지 않는 행에 `WARNING`을 출력합니다. **`paths=` 필드가 자신의 방식을 가리키지 않는 행은 보고하지 마십시오.** 이슈 #899는 한 번도 활성화되지 않은 fused decode 경로를 배포했고 그 전/후 벤치마크는 폴백을 자기 자신과 비교했습니다. 그 null 결과는 깨끗해 보였고 거의 받아들여질 뻔했습니다.

실제 체크포인트의 경우, `MLXCEL_MLA_ABSORBED=1 mlxcel generate -m models/<deepseek> ...`(명시적 활성화 표기; 기본값은 tracing을 통해 info 수준, `RUST_LOG=info`로 동일한 줄을 보고함)는 로드 시 몇 개 레이어가 fold되었는지와 전/후 바이트/token을 알리는 stderr 한 줄을 출력합니다. 해당 줄이 `0/27 layers`라고 읽히는 실행은 폴백을 실행하고 있는 것이며, 기본값 하에서도 거절(decline)은 stderr에 출력됩니다. 이 줄은 stdout이 아닌 stderr로 가므로 생성된 텍스트와 섞이지 않으며 `RUST_LOG`에 의존하지 않습니다.

## 관련 항목

- 구현은 `src/lib/mlxcel-core/src/mla/`.
- 병합 커널과 그 계약은 `src/lib/mlxcel-core/src/paged_v2/`.
- 이 경로가 거절하는 KV cache 모드는 [`docs/turbo-kv-cache.md`](turbo-kv-cache.md).
