# TurboQuant KV cache

TurboQuant 모드는 K 및/또는 V cache 텐서를 quantization하여 KV-cache 메모리를 줄입니다. 품질과 속도가 모델 계열, cache 모드, 하드웨어, 서버 경로에 따라 달라진다는 의미에서 이 구현은 실험적입니다. 대상 모델과 작업 부하를 직접 측정해 보지 않았다면 기본 FP16 cache를 사용하십시오.

TurboQuant 알고리즘(PolarQuant rotation, Lloyd-Max codebook, layer-aware V 보호, sparse-V dequant)은 [turboquant_plus](https://github.com/TheTom/turboquant_plus)의 Rust 포트이며, Copyright 2026 Tom Turney, Apache License 2.0으로 라이선스됩니다. Apache-2.0 Section 4(d)에 따라 계승되는 저작자 표시는 최상위 [NOTICE](../NOTICE) 파일을 참조하십시오.

구현 진입점:

- CLI 플래그: `src/cli/turbo_args.rs`
- cache 모드: `src/lib/mlxcel-core/src/cache.rs`
- TurboQuant 헬퍼: `src/lib/mlxcel-core/src/cache/turbo/`
- 품질/동작 테스트: `tests/turbo_kv_e2e.rs` 및
  `src/lib/mlxcel-core/src/cache/*turbo*_tests.rs`

## 사용 가능한 모드

| 사용자용 모드 | 실제 cache 모드 | 비고 |
|------------------|----------------------|-------|
| `fp16` | `KVCacheMode::Fp16` | 기본값이자 baseline. |
| `int8` | `KVCacheMode::Int8` | 토큰별 INT8 absmax quantization. |
| `fp16+turbo4` / `turbo4-asym` | `KVCacheMode::Turbo4Asym` | FP16 K, 4-bit TurboQuant V. 가장 안전한 TurboQuant 시작점. |
| `turbo4` / `turbo4-sym` | `KVCacheMode::Turbo4` | 4-bit TurboQuant K 및 V. K 쪽 quantization은 품질에 민감하므로 검증/allowlist에 등록된 계열에서만 사용하십시오. |
| `turbo4-delegated` | `KVCacheMode::Turbo4Delegated` | V 쪽의 hot/cold 분할과 packed cold 저장소. 대상 하드웨어에서 속도를 검증하십시오. |
| `fp16+turbo3` / `turbo3-asym` / `turbo3` | `KVCacheMode::Turbo3Asym` | FP16 K, 3-bit TurboQuant V. Turbo4 V보다 공격적입니다. |

대칭 Turbo3은 노출되지 않습니다.

## 측정된 속도와 메모리 절충

Quantized KV cache는 메모리 사용량 최적화이지 decode 속도 최적화가 아닙니다. #369 이후 M1 Ultra sweep(`benchmarks/turbo_kv/`, 2026-06-19)에서 모든 quantized 모드는 fp16보다 느리게 decode합니다. dequant 작업(Turbo 모드는 Walsh-Hadamard 역회전과 codebook 조회, int8은 scale-and-add)이 토큰마다 실행되며, 테스트한 context 길이에서는 절약하는 메모리 대역폭보다 더 많은 비용이 듭니다. dense 모델에서는 FFN이 decode를 지배하고 압축이 줄이는 V 읽기가 작업량에서 작은 비중을 차지하므로 격차가 더 큽니다.

decode는 해당 context에서 같은 모델의 fp16 처리량에 대한 비율로, prefill은 8K에서 보고합니다. 아래는 대표적인 체크포인트 두 개이며, 네 모델 전체 sweep(qwen2.5-7b와 142B dots.llm1 MoE 추가)은 `benchmarks/turbo_kv/`에 있습니다.

| 모드 | dense decode 4K / 16K | dense prefill 8K | MoE decode 4K / 16K | MoE prefill 8K | KV 압축 |
|------|-----------------------|------------------|---------------------|----------------|----------------|
| `fp16` | 1.00 / 1.00 | 1.00 | 1.00 / 1.00 | 1.00 | 1x (baseline) |
| `int8` | 0.61 / 0.41 | 1.29 | 0.48 / 0.50 | 1.14 | ~2x |
| `turbo4-delegated` | 0.63 / 0.45 | 1.27 | 0.43 / 0.34 | 1.13 | ~4x V |
| `turbo4-asym` | 0.38 / 0.37 | 0.92 | 0.24 / 0.26 | 0.80 | ~3.8x V |
| `turbo4` (sym) | 0.22 / 0.20 | 0.73 | 0.17 / 0.17 | 0.62 | ~4x K+V |
| `turbo3-asym` | 0.04 / 0.02 | 0.91 | 0.05 / 0.03 | 0.77 | ~5x V |

Dense = `qwen3-8b-4bit`, MoE = `qwen3-30b-a3b-4bit`, M1 Ultra.

수치가 말해 주는 것:

- **`int8`과 `turbo4-delegated`가 빠른 선택입니다.** int8(2x)은 dense와 MoE 모두에서 견고하고, turbo4-delegated는 최근 V를 fp16으로 유지하고 cold tail만 4-bit로 두어 비슷한 decode 속도에서 약 4x V 압축에 도달합니다. 둘 다 prefill도 빠르게 합니다(KV를 덜 쓰기 때문, 1.1-1.3x).
- **`turbo4-asym`은 메모리와 정확성을 위한 선택이지 속도를 위한 선택이 아닙니다.** K는 fp16으로 유지되고 #369 dequant-SDPA 경로는 fp16 reference와 parity가 정확하지만, decode는 약 0.2-0.4x입니다. K를 건드리지 않은 채 약 4x V 압축이 필요하고 decode 비용을 감수할 수 있을 때 선택하십시오. 같은 fp16-K + 4-bit-V 절충에서 속도를 원한다면 대신 `turbo4-delegated`(~0.7x)를 사용하십시오. #370은 격차를 줄이려고 V dequant를 attention 커널에 융합하는 것을 시도했으나 이득이 아니라 3-7x 퇴보로 측정되었습니다. ADR 0002의 2026-06-21 추가 기록을 참조하십시오.
- **대칭 `turbo4`는 압축을 극대화하지만 가장 느립니다.** allowlist에 등록된 계열에서만 사용하십시오(아래 참조).
- **`turbo3-asym`은 사실상 사용할 수 없습니다**(0.02-0.07x, 32K에서 약 0.4 tok/s). 메모리가 극단적으로 부족한 경우를 위해서만 존재하며 권장 모드가 아닙니다.

이 속도 저하는 upstream TurboQuant+ 분석과 일치합니다. 대표 수치인 32K에서의 `+22.8%` decode는 fp16이 아니라 turbo3 자체의 full-dequant 경로와 비교하여 측정되었으며, 그 이득에도 turbo3은 int8의 0.93x에 머뭅니다. 압축은 4.6x의 메모리 절충이며, decode 비용은 회전되고 codebook으로 quantization된 cache를 매 step 다시 dequantize하는 데 내재한 것입니다. +22.8%를 만들어 내는 upstream의 sparse-V 건너뛰기는 mlxcel에는 이어지지 않는데, mlxcel의 Turbo decode가 fused flash-attention이 아니라 분리된 dequant와 네이티브 SDPA이기 때문입니다. [ADR 0002](adr/0002-turbo-kv-split-dequant-vs-fused.md)가 측정된 A/B를 기록합니다. 그 2026-06-21 추가 기록은 #370을 종결합니다. asym을 mlxcel의 fused 커널로 라우팅하면 decode가 3-7x 퇴보했는데, 직접 작성한 fused 커널이 구체화된 V에 대한 네이티브 SDPA보다 느리기 때문이며, 따라서 분리된 dequant와 네이티브 SDPA가 여전히 빠른 구성입니다.

### CUDA (GB10)에서

같은 절충이 CUDA에서도 성립하며, 수치는 `llama-3.1-8b-4bit`로 GB10(sm_121)에서 측정했습니다(2026-07-10, `benchmarks/cuda_gb10_issue635_kvquant_2026-07-10.csv`. 전체 매트릭스는 `benchmark_results/cuda-kv-quant-modes-gb10-2026-07-10.md`). 여섯 모드 모두 Metal 전용 abort 없이 실행됩니다. 모든 커스텀 Turbo/Sparse-V 커널은 macOS에서만 활성화되고 CUDA에서는 일반 MLX 그래프 경로로 fallback합니다.

fp16 대비 Int8 decode는 2K에서 0.74x, 8K에서 0.53x, 32K에서 0.31x이므로 모든 길이에서 느리고 context가 길수록 격차가 벌어집니다. 이 경로가 매 step INT8 윈도우 전체를 fp16으로 dequantize한 다음 표준 fp16 SDPA를 실행하기 때문입니다(읽기 대역폭을 줄이는 fused INT8-KV 커널이 없음). Int8이 주는 것은 메모리입니다. 32K에서 MLX 최대치가 10.37 GB에서 8.18 GB로 떨어져 fp16 KV cache를 절반으로 줄인 것과 일치합니다. CUDA에 대한 권장 사항: 속도를 위해 fp16(기본값)을 유지하고, fp16 KV cache가 긴 context에 맞지 않거나 통합 메모리에 더 많은 동시 시퀀스를 담아야 할 때만 decode 속도 비용을 감수하고 `int8`을 선택하십시오. Turbo 모드는 V-quantization 품질 손실을 더하면서 현재 CUDA에서 속도 이득이 없으므로 CUDA에서 실험적으로 남습니다. 더 작은 메모리 사용량을 decode 이득으로 바꾸려면 fused INT8-KV / paged-attention 커널(#634)이 선행되어야 합니다.

## CLI 및 서버 플래그

같은 TurboQuant 플래그 그룹이 `mlxcel generate`, `mlxcel serve`, `mlxcel-server`에 평탄화되어 포함됩니다.

```bash
# Legacy shorthand.
mlxcel generate -m models/<checkpoint> \
    --kv-cache-mode fp16+turbo4 \
    -p "Hello" -n 100

# llama-server-style split flags.
mlxcel-server -m models/<checkpoint> \
    --cache-type-k fp16 \
    --cache-type-v turbo4 \
    --port 8080
```

지원되는 분할 플래그 조합:

| `--cache-type-k` | `--cache-type-v` | 결과 |
|------------------|------------------|--------|
| `fp16` 또는 `f16` | `fp16` 또는 `f16` | `fp16` |
| `int8` | `int8` | `int8` |
| `fp16` | `turbo4` 또는 `turbo4-asym` | `fp16+turbo4` |
| `turbo4` | `turbo4` | `turbo4` |
| `fp16` | `turbo4-delegated` | `turbo4-delegated` |
| `fp16` | `turbo3` 또는 `turbo3-asym` | `fp16+turbo3` |

분할 플래그와 `--kv-cache-mode`를 모두 지정하면 분할 플래그가 우선하며 경고가 로그로 남습니다.

llama.cpp의 `f16` 표기는 정확히 FP16 저장소의 별칭입니다. `q8_0`, `q4_0`과 같은 GGML quantizer 이름은 mlxcel에 수학적으로 동등한 cache 형식이 없으므로 계속 오류입니다.

환경 변수:

| 변수 | 용도 |
|----------|---------|
| `LLAMA_ARG_CACHE_TYPE_K` | `--cache-type-k`의 환경 변수 fallback. |
| `LLAMA_ARG_CACHE_TYPE_V` | `--cache-type-v`의 환경 변수 fallback. |
| `MLXCEL_KV_BOUNDARY_V_LAYERS` | CLI 플래그가 설정되지 않았을 때의 Boundary-V layer 수. |
| `MLXCEL_SPARSE_V_THRESHOLD` | Sparse-V 임계값. `0`은 sparse-V 동작을 비활성화합니다. |
| `MLXCEL_SPARSE_V_KERNEL` | falsy 값으로 설정하면 사용되는 곳에서 커스텀 sparse/dequant Metal 커널을 비활성화합니다. |
| `MLXCEL_TURBO4_ASYM_DEQUANT_SDPA` | falsy 값은 `Turbo4Asym`의 기본 dequant-first SDPA 경로를 비활성화하고 sparse-V 근사로 fallback합니다. |
| `MLXCEL_TURBO4_DELEGATED_DEQUANT_SDPA` | falsy 값은 delegated 모드의 기본 dequant-first SDPA 경로를 비활성화합니다. |
| `MLXCEL_TURBO4_DELEGATED_FUSED` | truthy 값은 비교용으로 이전의 fused delegated-kernel 경로를 선택합니다. |
| `MLXCEL_TURBO4_DELEGATED_FP16_FAST_PATH` | truthy 값은 delegated 모드 속도 실험을 위해 FP16 V working set을 유지합니다. |
| `MLXCEL_TURBO4_DELEGATED_FP16_SIDECARS` | FP16 fast path의 sidecar 정책. |

대부분의 사용자는 실험적 환경 변수가 필요하지 않습니다. 구현 변형을 벤치마크하는 경우가 아니라면 공개된 CLI 플래그를 사용하십시오. 더 넓은 범위의 런타임 및 진단 환경 변수 참조는 [Environment variables](environment-variables.md)를 참조하십시오.

## Boundary-V layer 보호

`--turbo-boundary-v N`은 Turbo 모드가 활성일 때 첫 `N`개와 마지막 `N`개 transformer layer의 V cache를 FP16으로 유지합니다. 공격적인 V quantization으로 인한 품질 손실을 줄이기 위한 것입니다.

```bash
mlxcel generate -m models/<checkpoint> \
    --kv-cache-mode fp16+turbo4 \
    --turbo-boundary-v 2 \
    -p "Hello" -n 100
```

`0`은 정책을 비활성화합니다. layer 수의 절반보다 큰 값은 런타임이 제한합니다. 이 플래그는 `fp16` 및 `int8` cache 모드에서는 작동하지 않습니다.

continuous batching에서 attention 경로는 각 layer의 cache 뒤에 있는 저장소를 따릅니다([ADR 0008](adr/0008-kv-attention-dispatch-in-the-cache.md)). batched decode step의 Turbo 행은 단일 시퀀스 decode와 같은 dequant-first 변형을 사용하며, Boundary-V(또는 `--kv-skip-last-layer`)가 FP16으로 유지하는 layer는 FP16이 아닌 paged 레이아웃 아래에서 행별 dense update와 SDPA를 실행하는 dense cache를 보관합니다. 둘 다 연산 순서만 다른 동일한 정확한 attention이므로 #2171 이전의 batched 경로와 비트 단위로 동일하지는 않습니다.

## 대칭 Turbo4 allowlist

K 쪽 quantization은 softmax 품질에 크게 영향을 줄 수 있습니다. 따라서 코드에는 `src/lib/mlxcel-core/src/cache/turbo/allowlist.rs`에 allowlist 헬퍼가 있으며, 이것이 이 목록의 기준(source of truth)입니다. 하드코딩된 allowlist 모델 타입 접두사는 다음과 같습니다.

- `qwen3_5`
- `qwen3_5_moe`
- `qwen3_next`

`turbo4`가 다른 dense 4-bit 체크포인트에서 안전하다고 가정하지 마십시오. 더 안전한 시작점은 K를 FP16으로 유지하는 `fp16+turbo4`입니다.

fallback은 단순히 광고되는 것이 아니라 실제로 수행됩니다. 모든 생성 진입점은 cache가 만들어지기 전에 요청된 모드를 `cache::turbo::resolve_kv_cache_mode_for_model`로 해석하므로, 이 목록 밖의 계열에서 `--kv-cache-mode turbo4`는 `fp16+turbo4`로 서빙되며 stderr와 서버 로그에 그 사실이 표시됩니다. CLI 배너, 서버 시작 로그, `/props` 페이로드(`--props`로 엔드포인트가 활성화된 경우의 `kv_cache_mode`, `kv_bits`)에 보고되는 모드는 요청된 모드가 아니라 실제 적용된 모드입니다. 새 항목을 추가한다면 같은 변경에 품질 gate를 포함하십시오.

## 모델 소유 cache 계열

일부 계열은 `forward`에 전달되는 동질적인 `&mut [KVCache]` 슬라이스를 사용하지 않고, 이질적인 attention, sliding-window, recurrent, convolutional 상태를 모델 래퍼 안에 보관합니다. CLI 생성기와 서버 스케줄러는 layer별 실제 KV 모드 표(Boundary-V와 `--kv-bits` / batch-KV 정책 포함)를 한 번 해석하고 그 표를 모델 소유 attention-cache 생성자에도 주입합니다. CLI 배너와 서버 시작 로그는 실제 모드와 `applied to N of M layers`를 보고하며, `GET /v1/cache/stats`에는 `kv_cache_mode_effective`가 포함됩니다.

Gemma 3, AFMoE, Gemma 4 text/VLM/Unified, Qwen 3.5 text/MoE/VLM, Qwen3-Next, Bailing MoE Linear, LFM2/LFM2-VL이 소유하는 attention KV cache와 Llama 4의 일반 attention arm은 자신의 layer index에 주입된 모드를 사용하고, 표가 없을 때만 FP16으로 fallback합니다. Recurrent, convolutional, gated-delta 상태는 KV 텐서가 아니므로 변하지 않습니다. 전체 모델 상태 snapshot을 donate하는 계열의 경우, FP16이 아닌 attention KV 모드는 현재 snapshot 직렬화에서 의도적으로 fail-closed입니다. 모델은 경고를 로그로 남기고 부분적이거나 잘못 표시된 상태를 저장하는 대신 그 snapshot의 donate를 건너뜁니다. 모드는 여전히 live attention layer에 적용되며 실제 모드로 보고됩니다.

Llama 4의 `ChunkedKVCache` layer는 FP16으로 유지됩니다. 아직 quantized sidecar 저장소가 없으며, FP16이 아닌 표가 그 chunked layer에 적용될 상황이면 래퍼가 한 번 경고합니다. Muse Glimmer는 부분 모드를 조용히 실행하는 대신 FP16이 아닌 모드를 시작 시 여전히 거부합니다.

## WHT head 차원 제약

TurboQuant는 Walsh-Hadamard transform을 사용합니다. 구현은 프로덕션 경로에서 head 차원이 2의 거듭제곱일 것을 기대합니다. 지원되지 않는 head 차원의 모델은 해당 cache 경로에서 TurboQuant를 거부하거나 계열별 fallback을 사용해야 하며, 품질 테스트 없이 조용히 padding하지 마십시오.

## WHT 스케일링, 그리고 quantization 타깃이 원하는 스케일

`mlxcel_core::wht`는 스케일을 MLX에 맡기며, MLX는 `1/sqrt(N)`을 적용합니다. 이로써 변환은 정규직교(orthonormal)이자 involution이 되어 `wht(wht(x))`가 `x`를 반환하고, 회전된 값에 맞춰진 codebook은 회전되지 않은 값과 같은 크기를 봅니다. 현재의 모든 호출자가 정확히 이것을 원하며, 다른 스케일을 하드코딩하는 백엔드 포트는 잘못된 것입니다. 왕복(round trip)이 아니라 forward norm 비율을 측정하십시오. `1/sqrt(N)`이면 1이고 스케일 없는 행렬이면 `sqrt(N)`인데, 왕복은 둘을 구별하지 못합니다. `examples/wht_numeric_probe`가 둘 다 보고합니다.

`mlxcel_core::wht_scaled`는 정규직교 인자를 공짜로 실을 수 없는 경우, 즉 block-scaled 저정밀도 형식에 입력되는 회전을 위해 존재합니다. 인자가 어디에 속하는지는 형식의 스케일이 무엇을 표현할 수 있는지가 결정합니다.

| 타깃 | 스케일 저장 | `1/sqrt(N)`이 들어가는 곳 |
|---|---|---|
| `mxfp8`, `mxfp4` | `E8M0` block scale, 2의 거듭제곱 | `log2(N)`이 짝수일 때만 정확히 흡수됨: head_dim 64(`1/8`)와 256(`1/16`)은 가능, 128(`2^-3.5`)은 불가 |
| `nvfp4` | `FP8` block scale과 `FP32` 텐서별 global scale | global scale이 흡수하되 범위를 대가로 치름 |
| `fp8` `E4M3` | 448에서 포화 | 스케일과 포화 검사는 하나의 결정: 스케일 없는 변환은 크기를 최대 `sqrt(N)`배, head_dim 128에서 11.3x 키움 |

이 표가 막는 실패는 lablup/mlxcel#1769와 lablup/mlxcel#1772의 pin 갱신이 이미 한 번 치른 것입니다. `E8M0` 반올림이 block 최대값을 최대 29퍼센트까지 포화시켰는데 하류의 어떤 것도 이를 알리지 않았습니다.

## MLA latent cache는 FP16만 지원

`glm4_moe_lite`, `deepseek_v3`, `deepseek_v32`(`deepseek_v3.2`로도 표기), `glm_moe_dsa`, `kimi_linear`, `kimi_k3`, `longcat_flash_ngram`은 MLA `(kv_latent, k_pe)` 쌍을 하나의 `KVCache`에 저장합니다. "K" 슬롯은 `kv_lora_rank` 폭의 latent(배포된 체크포인트에서 512)를 담고 "V" 슬롯은 `qk_rope_head_dim` 폭의 RoPE 키 스트림(64)을 담습니다. `deepseek_v32`, `deepseek_v3.2`, `glm_moe_dsa`는 `index_head_dim` 폭의 DSA indexer 키를 같은 "K" 슬롯에 이어 붙이므로 기본 `index_head_dim` 128에서 그 폭은 640입니다. 어느 폭에서도 두 슬롯은 head별 K/V 행이 아니므로, 그 내용에 맞게 보정된 quantized 모드는 없습니다.

- 대칭 `turbo4`는 "V" 폭으로 부호 벡터와 codebook을 만들고 "K" 슬롯에도 그것을 적용하여, 64개 항목의 부호 벡터 끝을 넘어 448개의 float를 읽습니다(indexer를 가진 계열에서는 576). `glm4-flash`에서는 생성된 모든 토큰이 `!`로 나왔습니다.
- 비대칭 모드(`fp16+turbo4`, `fp16+turbo3`, `turbo4-delegated`)는 "V" 슬롯, 여기서는 `k_pe`만 quantization합니다. 따라서 **키**를 압축하고 latent는 정확하게 남기며, 이는 위 모드 표의 "FP16 K, 4-bit V" 계약과 반대입니다. 출력이 유한하게 유지되었기 때문에 이것은 눈에 띄지 않았습니다.

이 계열에 대해 `fp16`이 아닌 모드를 요청하면 계열과 요청된 모드를 밝히는 경고 한 번과 함께 `fp16`으로 해석됩니다. `deepseek_v2`는 이 집합에 속하지 않습니다. cache마다 `MlaLatentCache::supports`에 묻고 모드가 quantized이면 압축 해제된 head별 K/V 레이아웃으로 fallback하므로, 그곳에서 quantized 모드는 일반 head별 행에 안착하여 계속 작동합니다.

latent 레이아웃에서 Turbo 모드가 작동하게 하려면 설정 변경이 아니라 새로운 보정과 새로운 커널(`k_pe`는 FP16으로 둔 채 latent용 512 폭 codebook)이 필요합니다.

## Paged cache와 서버 batching

paged decode 레이아웃은 `PagedKvLayout::uniform_with_mode`를 통해 TurboQuant 모드를 받아들입니다. 서버 dispatch는 `--decode-storage-backend paged`가 선택되면 Turbo 모드를 paged 레이아웃으로 라우팅합니다.

### 통합 paged KV cache

continuous-batching 서버는 요청 간 prompt-prefix cache와 시퀀스별 decode 상태를 하나의 참조 카운트 방식 copy-on-write block pool에 함께 보관합니다(epic #116). paged 백엔드는 batch가 가능한 pool 기반 계열의 기본값이며(`--decode-storage-backend auto`는 batching 시 paged로 해석됨), `dense`는 기존의 시퀀스별 cache를 강제합니다.

프롬프트 prefix를 공유하는 두 요청은 그 prefix의 KV block을 한 번만 저장합니다. 채택은 소비하지 않는 clone-and-pin입니다. 차용자는 일치한 prefix block에 대한 pinned 참조를 복제하므로 저장된 항목은 동시 형제 요청과 이후 더 깊은 일치를 위해 살아남으며, 진행 중인 임의의 수의 요청이 저장된 prefix 하나를 동시에 공유할 수 있습니다. 부분 일치도 채택됩니다. Automatic Prefix Caching(APC, 기본적으로 켜짐)에서는 일치가 16토큰 hash block 단위로 검증되고 32토큰 pool block 경계로 내림되며, 차용자는 갈라지는 접미사만 새 block에 re-prefill합니다(가득 찬 공유 block은 절대 변경되지 않으며, 공유된 부분 tail은 copy-on-write로 fork됨). cache 항목은 실제 pool 바이트로 계상되므로 `--prompt-cache-capacity-bytes`(기본값 2 GiB)가 보유량을 실제로 제한하고 LRU 축출이 실제로 작동합니다.

paged adopt와 donate는 pool 기반 Fp16 계열(qwen3, llama3와 같은 dense-natural 백엔드)에서 지원됩니다. 모델 소유 상태 계열과 recurrent 또는 hybrid SSM 모델은 dense 또는 모델 소유 cache를 유지하며 pool 밖에 있습니다.

모델 소유 계열에는 한 가지 설명이 필요한데, paged 백엔드가 이들에게 적용되며 이를 pool 참여로 읽기 쉽기 때문입니다. Gemma 3과 Llama 4는 `supports_batching()`을 보고하므로 `--decode-storage-backend auto`(기본값)에서 스케줄러가 이들을 paged 백엔드에 할당합니다. AFMoE도 모델 소유 상태를 선언하지만 `supports_batching() == false`를 보고하므로 dense 백엔드에 머물며, 거기서는 할당된 백엔드가 이미 `ModelOwned`로 읽히고 donate도 이미 건너뛰었으므로 어느 쪽이든 영향이 없습니다. 그 할당은 **회계 목적일 뿐**입니다. 이들의 실제 K/V는 모델의 `ModelOwnedSequenceState`에 있고, `make_caches()`는 pool에 아무것도 넘기지 않으며, paged block table은 `sync_paged_state_with_lengths`가 이들의 길이를 반영할 수 있도록 존재할 뿐입니다. 이런 시퀀스는 prompt cache에 donate되지 않고 거기서 채택되지도 않습니다. `CachePool::detach_paged`는 layer별 핸들이 없는 paged 시퀀스를 거부하고, `DetachedPagedCacheSet::clone_eligible`은 최소 하나를 요구하며, 스케줄러는 할당된 백엔드가 아니라 모델 자체의 `sequence_state_layout().backend`에 따라 donate를 미리 거부합니다(할당된 백엔드는 바로 이 계열들에서 `PagedKvCache`로 읽히며, 이것이 issue #1346에서 잘못된 turn-2 답변으로 이어졌습니다). 거부는 조용하지 않고 드러납니다. `GET /v1/cache/stats`는 `reject_model_owned_state`를 보고하고 `/metrics`는 `mlxcel_prompt_cache_reject_total{reason="model_owned_state"}`를 제공하므로, 저장소가 정당하게 비어 있는 계열과 고장 난 cache를 구별할 수 있습니다. dense prompt-cache 채택은 분리된 모든 layer의 KV 모드를 현재 해석된 layer별 표와도 비교하며, 불일치는 오래된 모드의 cache를 설치하는 대신 `reject_kv_mode_mismatch`와 `mlxcel_prompt_cache_reject_total{reason="kv_mode_mismatch"}`로 거부됩니다. 이 계열들의 실제 요청 간 재사용은 다음에 설명하는 모델 상태 snapshot 경로(#1335)를 통해 이루어지며, `supports_snapshot_reuse()`에 동의한 계열은 그 분기에서 먼저 처리되어 계속 donate합니다.

### recurrent 상태를 위한 exact-prefix snapshot

Hybrid-SSM 및 linear-attention 계열은 block 공유에서 계속 제외됩니다. recurrent hidden 상태는 radix/APC 토큰 prefix로부터 재구성할 수 없고 임의의 이전 토큰으로 truncate할 수도 없기 때문입니다. 상태가 recurrent이든 아니든, cache를 스케줄러의 pool에 넘기지 않고 `SequenceId`별로 직접 소유하는 모든 계열도 마찬가지입니다. 이런 계열을 위해 `mlxcel-server`에는 직교하는 exact-prefix snapshot 버킷이 있습니다. `supports_snapshot_reuse()`를 구현하는 모델은 turn이 끝날 때 전체 모델 소유 상태를 복사하고, 다음 요청의 토큰이 같은 세션 키 아래에서 저장된 정확한 prefix로 시작하면 이를 새 시퀀스에 복원할 수 있습니다.

snapshot을 지원하는 계열은 다음과 같습니다.

- Recurrent 및 hybrid SSM: Mamba, Mamba2, Jamba, Nemotron-H, Falcon-H1, Granite 4 H, PLaMo 2, LFM2, Qwen3-Next, Bailing MoE Linear, Inkling.
- Attention-cache 계열: Gemma 3, Gemma 4, AFMoE, Llama 4, Muse Glimmer, Qwen 3.5 / 3.6 text 및 MoE.
- 다섯 개의 snapshot hook을 텍스트 모델로 전달하는 VLM 래퍼: Gemma 4 VL 및 Unified, Qwen 3.5 VL, LFM2-VL, Granite 4 Vision, Muse Glimmer VLM, Inkling VL, 그리고 Gemma 3과 Llama 4 VLM 체크포인트가 사용하는 공유 `vision::VisionLanguageModel`. 래퍼 자체는 시퀀스별 상태를 갖지 않으며 prompt-cache 키는 요청의 멀티모달 digest를 포함하므로, 복원이 다른 이미지로 계산된 KV를 제공할 수 없습니다.

위의 목록을 신뢰하는 대신 `grep -rn 'fn supports_snapshot_reuse' src/models
src/vision`으로 목록을 다시 생성하십시오.

attention-cache 계열은 하나의 직렬화기 `src/models/kv_snapshot.rs`를 공유하며, 이는 세 가지 일반 cache 타입을 다루고 각각에 대해 truncate 복원이 할 수 있는 일을 정합니다. N개 토큰의 저장된 snapshot이 그 N개의 진짜 prefix인 프롬프트를 가진 요청에 서비스할 수 있되, 단축된 cache가 더 짧은 프롬프트의 cold prefill과 구별되지 않는 경우에만 가능하므로 truncation이 중요합니다.

- `KVCache`(full attention)는 항상 truncate할 수 있습니다. 모든 토큰이 자기 슬롯을 가지므로 tail을 버리는 것은 `offset`을 되감는 것입니다.
- `RotatingKVCache`(sliding window)는 ring이 wrap되지 않은 동안, 즉 `buffer_size == 0 && idx == offset && offset <= max_size`일 때만 truncate할 수 있습니다. ring이 wrap되면 논리적 토큰 `t`가 더 이상 슬롯 `t`에 있지 않아 되감기가 잘못된 윈도우를 노출합니다.
- `ChunkedKVCache`(Llama 4 iGQA)는 front가 trim되지 않은 동안(`start_position == 0`)에만 truncate할 수 있습니다. front trim 이후 버퍼는 `[start_position, offset)`을 보관하는데, `target_len` 토큰의 cold prefill은 `[target_len - chunk_size, target_len)`을 보관할 것이므로, truncate된 복원은 cold 실행보다 엄격히 더 적은 토큰에 대해 attend하게 됩니다.

exact-prefix 복원은 세 규칙 모두의 영향을 받지 않습니다. 전체 길이 복원은 wrap된 ring과 trim된 front를 포함해 캡처된 그대로 cache를 재현합니다. 더 짧은 prefix 변형만 거부됩니다.

snapshot은 토큰 벡터를 키로 하므로 캐시된 상태는 정확히 그 토큰들만 담아야 합니다. M5 neural-accelerator prefill 정렬은 chunk를 32토큰 타일로 padding한 다음 `CachePool`의 cache에서 padding을 다시 trim하는데, 모델 소유 시퀀스는 거기에 아무것도 없으므로 padding 위치가 모델 자체의 cache에 남아 `offset`이 실제 토큰 수를 넘어서게 됩니다. 따라서 스케줄러는 모델 소유 계열에 대해 `LanguageModel::trim_state`(시퀀스의 id와 함께)도 호출하여 모델 자체의 시퀀스별 상태를 되감으며, 되감기가 실패하면 요청이 중단됩니다. Gemma 3이 이를 구현하며(sliding-window layer는 물리적 ring 버퍼도 잘라 내므로 다음 decode step 이후에 padding key가 남지 않음) 정렬된 경로를 유지합니다. Gemma 4, AFMoE, Llama 4는 아직 이를 구현하지 않아 `supports_padded_prefill()`에 `false`로 응답합니다. pool 기반 계열은 영향이 없습니다.

Snapshot 재사용은 quantized attention cache에 대해 의도적으로 보수적입니다. 위에 나열된 모든 계열은 FP16 모델 소유 attention KV 상태만 직렬화하며, live layer가 Int8 또는 Turbo 모드로 구성되어 있으면 공유 직렬화기는 `keys`/`values`만 복사해 FP16으로 복원하는 대신 명시적 오류와 함께 donate와 복원을 거부합니다. 복원은 직렬화된 snapshot 모드를 live layer별 표와도 비교하므로, 다른 KV 모드에서 캡처된 snapshot은 cache를 손상시키는 대신 cold prefill로 fallback합니다. `ChunkedKVCache`에는 quantized 변형이 없으므로 Llama 4 layer는 구조상 FP16이며, quantized 모드 요청은 cache 생성 시 한 번 경고됩니다.

snapshot 버킷에는 자체 바이트 상한, 항목 상한, TTL, LRU 카운터, hit/miss 지표가 있습니다. `GET /v1/cache/stats`는 `snapshot_*` 필드를 보고하고, `/metrics`는 `mlxcel_prompt_cache_snapshot_hits_total`, `mlxcel_prompt_cache_snapshot_misses_total`, `mlxcel_prompt_cache_snapshot_tokens_reused_total`, 그리고 레이블이 붙은 snapshot 축출을 노출합니다. 이 경로는 의도적으로 전체 prefix만 지원합니다. APC block 매칭에 참여하지 않고, 세션 간에 SSM 상태를 공유하지 않으며, hybrid SSM 계열에 대한 block 공유 예외를 수정하지 않습니다.

block pool은 `--kv-cache-budget <bytes|auto|none>`(환경 변수 `MLXCEL_KV_CACHE_BUDGET`)으로 제한됩니다. 기본값은 `auto`로, 메모리 추정에서 상한을 유도하므로 batched-decode 기본값(#628)이 전체 context 시퀀스를 동시에 실행하다 OOM abort에 이르지 않습니다. `none`(또는 `0`)은 이전의 제한 없는 pool을 복원합니다. 예산 아래에서 스케줄러는 요청을 거부하기 전에 차가운 캐시된 prefix를 축출하고, 그다음 preemption합니다. `GET /v1/cache/stats`는 paged block 사용량(block 크기, 할당, live, 비어 있음, 예약/사용 중인 바이트, 예산)을 보고합니다.

### 측정된 효과

아래 수치는 `models/qwen3-0.6b-4bit`(28 layer, KV head 8개, head dim 128, 런타임 KV dtype bf16이므로 토큰당 약 112 KiB의 KV)를 사용한 M1 Ultra에서 나온 것입니다.

**공유 prefix당 절약되는 메모리.** `N`개의 동시 요청에 걸친 `P`토큰의 공유 prefix는 `N`개가 아니라 pool 사본 하나를 차지합니다. 토큰당 KV 바이트는 `2 (K and V) * n_kv_heads * head_dim * num_layers * dtype_bytes`입니다. 공유되는 1024토큰 system prompt는 약 112 MiB이며, 동시 요청 8개일 때 pool은 이를 여덟 번이 아니라 한 번 저장하므로 대략 `7 * 112 MiB ≈ 784 MiB`를 절약합니다.

**피한 prefill 토큰.** `P`토큰 prefix를 공유하는 `N`개의 요청 중 첫 번째만 이를 prefill하고, 나머지 `N - 1`개는 캐시된 block을 채택하여 `(N - 1) * P`개의 prefill 토큰 forward pass를 건너뜁니다. `tests/paged_prefix_share_parity.rs`는 채택한 요청이 공유 prefill을 건너뛰면서도 cold 실행과 바이트 단위로 동일하게 decode함을 확인합니다.

**End-to-end 서버 메모리 사용량.** 고정된 HTTP 작업 부하 하에서 `mlxcel serve`의 프로세스 전체 물리 footprint(`/usr/bin/footprint`)입니다(`scripts/bench_memory_footprint.py`, `models/llama-3.2-1b-4bit`, M1 Ultra, 기본값, 시나리오 단계의 최대치).

| 시나리오 | v0.1.4 | 현재 기본값 | 공유 토큰 |
|---|---|---|---|
| 유휴 (가중치만) | 957 MiB | 983 MiB | - |
| 동시 요청 8개, 공유되는 약 3.7k 토큰 system prompt | 1653 MiB | 1627 MiB | 29184 (8개 모두 채택) |
| 같은 prefix, 순차 요청 8개 | 1650 MiB | 1413 MiB | 25536 |
| 대화 하나, 8 turn | 2104 MiB | 1301 MiB | 10976 |
| 서로 다른 프롬프트 32개 (churn) | 1558 MiB | 2276 MiB | 1984 |

v0.1.4는 HTTP 경로에서 prompt cache를 전혀 작동시키지 않았으므로(재사용 0), 그 footprint가 cache 없는 하한입니다. 현재 기본값은 모든 공유 시나리오에서 이와 같거나 더 낫고, 공유 prefill 작업도 건너뜁니다(8-way burst는 요청 단계를 3.4x 더 빠르게 끝냄). churn 수치가 더 높은 이유는 donate된 항목이 이제 향후 재사용을 위해 보존되기 때문입니다. 그 보존은 `--prompt-cache-capacity-bytes`가 관장하는 실제 메모리이며 상한에서 LRU 축출로 해제됩니다.

**Decode 처리량.** paged decode의 parity 증거는 의도적으로 범위가 정해져 있습니다. `tests/paged_scheduler_parity.rs`는 B=1/B=2에서 스케줄러 형태의 qwen3 및 llama3 Fp16 dense-natural-cache 할당을 다루며, TF32를 비활성화한 상태에서 greedy 토큰 동일성과 제한된 logit RMS를 요구하는 CUDA qwen3 arm을 포함합니다. 이 증거는 VLM front end, 모델 소유 cache, Turbo/quantized-KV 모드, 또는 모든 CUDA reduction 기하에 대한 바이트 동일성을 주장하지 않습니다. 실제로 작동하는 batched 경로는 #899의 pooled paged decode이며, 모델 forward가 아니라 cache 자체에서 dispatch됩니다(`KVCache::attend`, ADR 0008). gather-then-SDPA 경로는 정확성 reference로 유지됩니다. batch 4에서 네이티브 커널은 512토큰 프롬프트에서 276 tok/s, 4096토큰 프롬프트에서 84 tok/s로 실행되는 반면 gather reference는 146 및 7.7 tok/s입니다(1.9x 및 10.9x). gather reference는 매 step 보이는 윈도우를 다시 구체화하기 때문에 context에 따라 급격히 저하되며, 이것이 실제 경로가 네이티브 커널을 사용하는 이유입니다. 별도의 fused split-K 커널(Metal, 그리고 #634 이후 `mx.fast.cuda_kernel`을 통한 CUDA도 포함)(`MLXCEL_PAGED_ATTENTION_NATIVE`, `paged_decode_attention_pooled`에 입력)은 위의 block-table 커널과는 다른 코드 경로입니다. #331 이후 이는 더 이상 단순한 on/off 스위치가 아닙니다. 적응형 selector가 Metal에서는 ADR 0001이 이득을 측정한 batch>=4 / ctx<=4096 / 단일 slab 구간 안에서만 이를 dispatch하고, CUDA(#634)에서는 Metal 상한이 적용되지 않으므로 batch나 context와 관계없이 단일 slab layer에서 dispatch합니다. 나머지 모든 곳에서는 gather가 실행되며, 환경 변수는 A/B 테스트를 위해 여전히 어느 arm이든 강제로 고정합니다. chunked slab 저장소는 이 구간을 더 좁힙니다. layer가 slab 하나를 넘어 커지면 커널이 거부하기(gather로 fallback) 때문입니다. #710은 pooled 진입점과 그 selector를 라이브러리 전용 API로 퇴역시켜 `mlxcel serve` decode 경로 밖으로 옮겼으며, 이 경로는 위에서 설명한 block-table 커널에 머뭅니다. 변수 자체는 `src/lib/mlxcel-core/src/layers.rs`의 `resolve_dispatch_decision`과 `resolve_paged_v2_dispatch`에 따라 현재 두 소비자를 유지합니다. 라이브러리 전용 pooled 진입점과 서버의 pool 기반 batched paged decode입니다. 서버 측에서는 issue #899가 fused v2 커널을 프로덕션 decode 경로로 만들고 이 변수의 강제 off 값을 kill switch로 명명했으며, 강제 on 값은 측정된 토큰 하한을 우회하여 처리할 수 있는 모든 형태에 v2를 고정합니다. 하한과 기본값은 [Paged decode v2 variables](environment-variables.md#paged-decode-v2-variables)를, dispatch 정책과 결과별 로그 줄은 [Continuous batching](CONTINUOUS_BATCHING.md#seeing-which-path-ran)을 참조하십시오. 경위: #710이 라이브러리 진입점을 퇴역시킨 것이 "서버 설정이 아니다"라는 해석의 출처이며, #899가 이 변수에 두 번째이자 서버 쪽 소비자를 부여했습니다. ADR 0001의 #710 결정 기록, [ADR 0001](adr/0001-paged-attention-gather-vs-fused-kernel.md)을 참조하십시오.

pool 확장은 layer마다 하나의 큰 텐서를 재할당하는 대신 고정 크기 slab을 추가하므로, pool을 확장해도 기존 KV를 복사하지 않고 할당자 cache에 오래된 버퍼 크기의 사다리가 남지 않습니다. 서로 다른 약 4k 토큰 프롬프트를 가진 동시 요청 8개(qwen3-0.6b, 공유할 것이 없는 스트레스 케이스)는 변경 전의 5013 MiB 대비 3739 MiB에서 최대치를 기록하며, 같은 작업 부하에서 dense 백엔드의 1.13x입니다.

## 권장 시작점

| 작업 부하 | 권장 사항 |
|----------|----------------|
| 일반 서빙 | `fp16`으로 시작하십시오. Quantized KV는 메모리 압박을 위한 것이지 속도를 위한 것이 아닙니다. |
| 더 낮은 KV 메모리, 가장 빠른 quantized 옵션 | `int8`(~2x) 또는 `turbo4-delegated`(~4x V)를 사용하십시오. 둘 다 prefill도 빠르게 합니다. |
| K를 정확하게 유지한 최대 V 압축 | `fp16+turbo4`(turbo4-asym). ~0.2-0.4x decode를 감수하며, 품질을 위해 boundary-V를 활성화하십시오. |
| 대칭 `turbo4` 고려 | allowlist에 등록되었거나 검증된 계열에서만 사용하십시오. 가장 느린 모드입니다. |
| 메모리가 극단적으로 부족한 경우에만 | `fp16+turbo3`은 사실상 사용할 수 없을 정도로 decode가 느립니다. 메모리 절감이 속도 비용만큼 가치 있는지 확인하십시오. |

## Advisor 권장 사항 (`--recommend-quant`)

`mlxcel generate --recommend-quant`는 quantization 조언과 함께 참고용 KV-cache 모드 섹션을 출력합니다. 모델 계열과 context 범위별로 `fp16`, `int8`, `turbo4-delegated`, `fp16+turbo4`, `turbo4` 중 하나를 제안하며(decode가 사실상 사용 불가능한 `fp16+turbo3`은 제안하지 않음), 추측하는 대신 벤치마크의 시작점을 얻을 수 있습니다. 제안은 위에서 측정된 절충을 따릅니다. 빠른 선택으로 int8과 turbo4-delegated, K를 정확하게 유지하는 메모리 선택으로 fp16+turbo4, allowlist에 등록된 계열에만 대칭 turbo4입니다.

제안은 참고용이며 옵트인 방식일 뿐입니다. 기본 추론 경로는 변하지 않습니다. 플래그가 없으면 런타임은 여전히 `fp16`을 사용합니다. 제안된 모드를 사용하려면 `--kv-cache-mode`(또는 `--cache-type-k` / `--cache-type-v`)를 직접 지정해야 합니다. advisor는 `config.json`만 읽으며, 가중치를 로드하지 않고 quantized 가중치 dtype도 바꾸지 않습니다. KV-cache 모드는 K/V cache 텐서만 quantization하고 attention을 위해 FP16으로 다시 dequantize하므로, 권장 사항이 프로젝트가 금지하는 bf16에서 f16으로의 quantized 가중치 승격(`#289` 퇴보)을 다시 일으킬 수 없습니다.

제안을 결정하는 기준:

- **모델 계열**은 KV 아키텍처 분류기(`src/execution/kv_arch.rs`)에서 얻습니다: 표준 attention, sliding-window, MLA(DeepSeek), hybrid attention + recurrent, 순수 SSM. 원본 `model_type`도 대칭 Turbo4 allowlist(`src/lib/mlxcel-core/src/cache/turbo/allowlist.rs`)와 대조합니다.
- **Context 범위**는 short(`<=4K` 토큰, 대화형 / 단일 요청), medium(`4K-32K`), long(`>32K`, 장문 context 서빙)으로 구분됩니다. KV-cache 압박이 지배적인 곳이므로 장문 context와 메모리 제약 서빙이 순수 짧은 decode tok/s보다 우선됩니다.

advisor가 적용하는 보수적 규칙:

| 계열 / context | 제안 | 함께 벤치마크 | 이유 |
|------------------|-----------|----------------|-----|
| 모든 계열, 짧은 context | `fp16` | - | KV 사용량이 작으므로 baseline을 유지합니다. |
| 표준 / sliding-window / hybrid, medium | `int8` | `turbo4-delegated` | int8은 가장 빠른 quantized 모드(~2x)이고, turbo4-delegated는 비슷한 속도에서 ~4x V입니다. |
| 표준 / sliding-window / hybrid, long, allowlist 등록 | `turbo4` | `turbo4-delegated` | 계열이 PPL gate를 통과했으며, turbo4는 최대 압축이고 delegated는 더 빠른 fallback입니다. |
| 표준 / sliding-window / hybrid, long, allowlist 미등록 | `turbo4-delegated` | `fp16+turbo4` | 가장 빠른 ~4x V 압축이며, fp16+turbo4는 K를 정확하게 유지하는 대안입니다. 대칭 `turbo4`는 allowlist 밖에서는 제안하지 않습니다. |
| MLA (DeepSeek), medium 또는 long | `int8` | - | latent 차원이 2의 거듭제곱이 아니므로 Turbo Walsh-Hadamard V 경로가 적용되지 않으며, 토큰별 INT8에는 head-dim 제약이 없습니다. |
| 순수 SSM (Mamba/Mamba2) | `fp16` | - | context에 비례하는 KV cache가 없으므로 Turbo 모드가 거의 아무것도 절약하지 못합니다. |
| 2의 거듭제곱이 아닌 head dim (예: head_dim 80의 Phi-2) | `int8` (medium/long), `fp16` (short) | - | Turbo Walsh-Hadamard transform은 2의 거듭제곱 head 차원이 필요합니다. advisor는 이런 계열에서 Turbo 제안을 `int8` 또는 `fp16`으로 낮추며, MLA 처리와 일치합니다. |

head 차원 검사는 `head_dim` 또는 `head_size`가 있으면 `config.json`에서 직접 읽습니다. 명시적인 필드가 둘 다 없으면 hidden size를 attention head 수로 나누며, 각 명명 규칙이 사용하는 대체 필드 이름을 확인합니다. hidden size는 `hidden_size` / `d_model` / `dim` / `model_dim`, head 수는 `num_attention_heads` / `num_heads` / `n_heads` / `n_head`입니다. 이는 KV 아키텍처 분류기의 필드 이름 범위와 일치하므로 두 경로가 유도된 head 차원에 합의합니다. 필요한 필드가 전혀 없으면 advisor는 Turbo가 적용 가능하다고 보수적으로 가정하고 제안을 변경하지 않습니다.

대칭 `turbo4`는 allowlist에 등록된 계열에만 제안됩니다. allowlist 밖에서는 advisor가 long context에서는 `turbo4-delegated`를, medium에서는 `int8`을 앞세우며, 이는 위에서 측정된 절충과 일치합니다. `fp16+turbo3`은 제안되지 않습니다. 모든 제안은 프로덕션에 채택하기 전에 아래 체크리스트로 계열마다 검증해야 할 가설로 취급하십시오.

## 검증 체크리스트

모델 계열에 TurboQuant 모드를 권장하기 전에:

1. 짧은 smoke generation을 실행하여 모드가 로드되고 decode되는지 확인합니다.
2. 대표적인 프롬프트 집합에서 FP16과 출력 품질을 비교합니다.
3. perplexity/NIAH 또는 작업에 적합한 다른 품질 gate를 실행합니다.
4. prefill과 decode 처리량을 따로 측정합니다.
5. 모드를 continuous batching이나 paged decode와 함께 사용할 예정이라면 정확한 서버 경로를 테스트합니다.
6. 하드웨어, MLX 커밋/버전, 모델 체크포인트, 프롬프트/decode 형태, 모든 cache 플래그를 기록합니다.

`tests/turbo_kv_e2e.rs`의 ignored 테스트가 하드웨어/모델 품질 gate를 두기에 적합한 곳이며, 로컬 모델 체크아웃이 필요하므로 기본 `cargo test` 실행에는 포함되지 않습니다.

## 알려진 제한 사항

- TurboQuant는 모든 텍스트 및 VLM 계열에서 균일하게 검증되지 않았습니다.
- VLM 장문 context 및 다중 이미지 프롬프트는 별도의 검증이 필요합니다.
- 이전 세대 Apple Silicon과 non-Hopper/non-Blackwell CUDA 경로는 개발자 벤치마크 머신과 다른 병목을 가질 수 있습니다.
- 실험적 환경 변수 경로는 A/B 테스트에는 유용하지만 새로운 벤치마크 데이터 없이 사용자 대상 권장 사항에 나타나서는 안 됩니다.
- 위에 나열된 MLA latent 계열에서는 TurboQuant나 INT8 모드를 사용할 수 없으며, 무엇을 요청하든 FP16으로 서빙됩니다.
