# 아키텍처 개요

`mlxcel`은 C++ 브리지를 통해 MLX를 호출하는 Rust 추론 런타임입니다. 공개 진입점은
의도적으로 얇게 유지됩니다. CLI 파싱은 가장자리에서 이루어지고, 모델 로딩, 요청 준비,
스케줄링, MLX 연산은 목적별로 분리된 모듈에 있습니다.

## 최상위 구성

```text
src/
├── main.rs                      # `mlxcel` CLI schema and subcommand routing
├── bin/mlx_server.rs            # standalone `mlxcel-server` binary
├── commands/                    # CLI subcommand handlers
├── execution/                   # runtime/device and sampling helpers
├── model_metadata.rs            # model-kind and loading-policy descriptors
├── backend/                     # ComputeBackend seam: which engine executes forward()
├── loading/                     # model loading routers and family registries
├── loaded_model.rs              # LoadedModel enum and LanguageModel dispatch
├── loaded_model_capabilities.rs # multimodal capability routing
├── models/                      # text model implementations and detection
├── multimodal/                  # shared multimodal prompt/runtime helpers
├── vision/                      # vision encoders, processors, connectors
├── audio/                       # audio encoder support
├── server/                      # HTTP server, request translation, scheduler
├── distributed/                 # TP/PP/DI config, transports, registries
├── tokenizer/                   # tokenizer loading helpers
├── lora/                        # LoRA adapter loading
├── lib/mlxcel-core/             # MLX C++ FFI crate and low-level generation primitives
└── lib/mlxcelverse/             # mlxcelverse: MLX build, per-backend overlays, fused kernels
```

## `mlxcel-core`

`src/lib/mlxcel-core/`는 MLX 직접 브리지와 저수준 런타임 구성 요소를 소유합니다.

- `src/lib/mlxcel-core/src/lib.rs` — `cxx::bridge` 정의와 크레이트 export.
- `src/lib/mlxcel-core/src/cache.rs` 및 `src/lib/mlxcel-core/src/cache/` — FP16/INT8/TurboQuant KV cache 변형,
  paged cache 레이아웃, detach/adopt 헬퍼, 캐시 테스트. `cache/attend.rs`에는
  `KVCache::attend`와 `cache::attend_batched`가 있으며, 이는 캐시 뒤의 저장소에 따라
  dense 또는 paged 커널을 선택하는 단일 attention 진입점입니다
  ([ADR 0008](adr/0008-kv-attention-dispatch-in-the-cache.md)). 또한 모델이 직접 소유하는 패밀리(Gemma 3, Llama 4)가
  attention을 수행할 때 거치는 `KvAttention` 트레이트, `RotatingKVCache::attend`, `ChunkedKVCache::attend`,
  `cache::attend_batched_rows`도 여기에 있습니다 ([ADR 0009](adr/0009-engine-step-api.md)).
- `src/lib/mlxcel-core/src/ops.rs`, `src/lib/mlxcel-core/src/dtype.rs`, `src/lib/mlxcel-core/src/streams.rs` — 일반적인 MLX
  연산과 런타임 개념에 대한 래퍼.
- `src/lib/mlxcel-core/src/sampling.rs` — CLI/서버 경로가 공유하는 페널티와 토큰 샘플링.
  샘플러 체인 순서: 토큰 bias 및 페널티, 그다음 temperature가 적용되지 않은 분포에서 평가되는 top-k / top-p / min-p,
  그다음 재정규화된 필터링 행에 대한 XTC, 마지막으로 최종 추출에만 적용되는 temperature 스케일링 한 번
  (llama-server 체인과 일치).
- `src/lib/mlxcel-core/src/sampling_row_step.rs`: `RowSampler`, 행 단위 샘플링 단계의 단일 구현
  (ADR 0007). `SamplerState` 수명 주기, 토큰 이력 순서 규칙, 구조화 출력 마스크와 추출 후 override 훅,
  그리고 단계의 스테이지 목록에서 도출되는 fused-batch 적격성을 포함합니다.
  엔진 step은 서버와 raw-completion 클라이언트 모두에서 이를 통해 토큰을 추출합니다.
- `src/lib/mlxcel-core/src/engine/`: `Engine`, 배치 네이티브 decode 엔진
  ([ADR 0007](adr/0007-unified-batch-native-engine.md), [ADR 0009](adr/0009-engine-step-api.md)).
  모델과 `CachePool`을 소유하며, `open`, `prefill`(계획 조각 하나), `step`(forward 후 모든 행의 샘플링과 종료 단계,
  모든 행 수에 대해 하나의 진입점), `submit` / `finish_rows`(lookahead의 분리된 형태), `close`를 제공합니다.
  `engine/rows.rs`에는 `StepRow`, `StepRowHooks`, `RowOutcome`이 있으며, 서버는 이를 통해 행별로
  구조화 출력 matcher, thinking 예산, stop matcher, 스트리밍을 제공하고 각 행의 토큰, 종료 원인 또는 행 단위 오류를 돌려받습니다.
- `src/lib/mlxcel-core/src/sampling_token_bias.rs`: `compose_token_bias`, 요청 bias, 언어 bias,
  출력 억제에 대한 단일 우선순위 규칙으로, CLI와 서버가 요청당 한 번 호출합니다.
- `src/lib/mlxcel-core/src/generate.rs` — `LanguageModel` 트레이트, `SamplingConfig`, `GenerationStats`.
  decode 루프는 `engine/`에 있습니다. `engine/direct.rs`는 `DirectEngine`으로, raw-completion 클라이언트인 `mlxcel generate`,
  벤치마크, 추론 세션 실행이 사용합니다(ADR 0007, #2176). `engine/speculative.rs`는 prompt lookup 같은 drafter를 위한
  토큰 전용 speculative 루프입니다.
- `src/lib/mlxcel-core/src/drafter/` 및 `src/lib/mlxcel-core/src/speculative/` — speculative decoding 지원.
  `speculative/stochastic_accept.rs`에는 수락 규칙과 분포 보존 보장이 있습니다.
  각 경로가 어떤 규칙을 실행하는지는 [`speculative-acceptance.md`](speculative-acceptance.md)를 참조하십시오.
- `src/lib/mlxcel-core/src/layers.rs`, `src/lib/mlxcel-core/src/weights.rs`, `src/lib/mlxcel-core/src/utils.rs` — 모델 구성 요소,
  SafeTensors 로딩, 마스크, 헬퍼 연산.
- `src/lib/mlxcel-core/src/autotune/`: shape 버킷 기반 커널 autotuner(issue #906). `TunableOp` 계약,
  교차 배치된 N회 중앙값 프로파일링 하니스와 그 불안정 tactic 방지 장치(반복 횟수는 측정된 launch 비용에 비례하며,
  후보는 샘플 자체의 분산보다 더 크게 기본값을 앞서야 함), `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/autotune` 아래의
  영속 tactic 캐시, 연산별 소비자를 포함합니다. 기본적으로 꺼져 있으며 `MLXCEL_AUTOTUNE`이 설정되지 않으면 완전히 비활성입니다.
  `mlxcel tune`이 이를 오프라인으로 구동합니다.
- `src/lib/mlxcel-core/src/bench_rotation.rs`: `examples/` 아래 마이크로벤치 하니스를 위한,
  last-level cache를 고려한 회전 입력 버퍼 ([benchmarks](benchmarks.md) 참조).

`src/lib/mlxcel-core/build.rs`는 `src/lib/mlxcelverse/`를 통해 고정된 MLX 커밋을 빌드하고 브리지 코드를 컴파일합니다.
업스트림 MLX 위에 무엇이 얹혀 있는지는
[mlxcelverse](#mlxcelverse-the-mlx-side-layer)를 참조하십시오.

## mlxcelverse: MLX 측 계층

mlxcelverse는 mlxcel이 업스트림 MLX 위에 구축하는 모든 것(`src/lib/mlxcelverse/`, 해당
[README](../src/lib/mlxcelverse/README.md)에 설명됨)을 가리키는 이름입니다.
mlxcel은 MLX의 포크를 사용하지 않습니다. `src/lib/mlxcelverse/CMakeLists.txt`는
`ml-explore/mlx`를 고정된 하나의 커밋(`GIT_TAG`, 고정의 유일한 기준)으로 가져오고,
mlxcelverse의 소스 오버레이를 체크아웃 위에 복사한 다음, `cxx` 브리지가 링크하는 정적 `libmlx.a`를 빌드합니다.
모든 백엔드(Metal, CUDA, ROCm)가 동일한 커밋에서 빌드됩니다.

mlxcelverse에는 서로 다른 규칙으로 유지 관리되는 두 종류의 코드가 있습니다.

| 종류 | 위치 | MLX pin 업데이트 시 변경되는 것 |
|------|-------|---------------------------------|
| 백엔드별 소스 오버레이: MLX가 컴파일되기 전에 MLX 소스를 대체하거나 추가하는 전체 파일 | `src/lib/mlxcelverse/overlays/metal/`(항상 복사됨), `overlays/cpu/`(항상 복사되고 모든 빌드에서 컴파일됨: f32로 누적하는 CPU 양자화 matmul, #2248), `overlays/cuda/`(CUDA 백엔드 파일과 MLX 코어 파일, CUDA 빌드에서만 복사됨), `overlays/rocm/`(ROCm 백엔드와 이것이 훅을 거는 MLX 코어 파일, ROCm 빌드에서만 복사됨) | 각 오버레이는 새 업스트림 파일과 3-way 병합되고 줄 단위로 비교됩니다. `CONTRIBUTING.md`의 "Bumping the MLX upstream pin"과 `overlays/rocm/README.md`를 참조하십시오 |
| mlxcel 자체 커널: MLX의 공개 커스텀 커널 API(`fast::metal_kernel`, `fast::cuda_kernel`, ROCm에서는 `fast::hip_kernel`) 위에 구축된 fusion 및 확장 함수 | `src/lib/mlxcelverse/kernels/` 및 `src/lib/mlxcel-core/cpp/mlx_cxx_kernels.cpp`의 커널 본문 | 공개 API 호환성만 해당하며, fused launcher는 그래프 레퍼런스와 대조하여 다시 검증됩니다 |

오버레이 복사는 파일 전체에 대한 `configure_file(... COPYONLY)`이므로,
CMake configure를 다시 실행해도(`build.rs`가 매 리빌드마다 수행) 멱등적입니다.
백엔드별 오버레이는 해당 백엔드에 대해서만 복사되므로, Metal 빌드는 CUDA 또는 ROCm으로 수정된 MLX 코어 파일을
컴파일하지 않으며, `rocm`과 `cuda` 기능은 함께 활성화할 수 없습니다.

ROCm 오버레이는 NripeshN/mlx(MIT)의 `rocm-support` 브랜치에서 벤더링한 것으로,
업스트림 초안 ml-explore/mlx#2300의 head입니다.
`overlays/rocm/UPSTREAM`에는 소스 커밋과 이것이 재조정된 MLX pin이 기록되어 있고,
`overlays/rocm/LOCAL_FIXES.md`에는 mlxcel이 그 위에 추가로 적용한 모든 변경이 나열되어 있습니다.
`docs/mlxcelverse/rocm-overlay.md`는 두 업스트림을 따라가는 절차(pin 업데이트, 포크 동기화, `scripts/mlxcelverse/` 도구)이며,
`docs/mlxcelverse/upstream/`에는 포크 측 수정 패키지가 과거 참고용으로 보관되어 있습니다
(제출되지 않았으며, 수정은 오버레이에 남습니다).

## 로딩 파이프라인

일반적인 텍스트 생성 요청은 다음 경로를 따릅니다.

```text
model path
  → src/models/detection.rs reads config.json and returns ModelType
  → src/model_metadata.rs selects loading policy
  → src/loading/ dispatches to config-backed, non-standard, special, or VLM loader
  → tokenizer is loaded
  → LoadedModel + tokenizer are returned to CLI/server
```

중요한 제어 지점:

- `src/models/detection.rs`는 `config.json::model_type`과 관련 설정 힌트를 `ModelType`으로 매핑합니다.
- `src/model_metadata.rs`는 패밀리가 텍스트인지 VLM인지, 어떻게 로드되는지,
  어댑터를 지원하는지를 기록합니다.
- `src/loading/config_backed.rs`, `src/loading/nonstandard.rs`,
  `src/loading/special.rs`, `src/loading/vlm*.rs`에 로딩 구현이 있습니다.
- `src/loaded_model.rs`와 `src/loaded_model_capabilities.rs`는 하위 CLI/서버 코드가
  모든 구체적인 모델 타입에 대해 매칭하지 않도록 합니다.
- `src/backend/`는 compute-backend seam입니다. CLI와 서버의 로드 지점은 `loading::load_model`을 직접 호출하는 대신
  `select_backend().load_model(...)`을 호출하므로, `LanguageModel::forward`를 실행하는 엔진은
  로드 경계에서 선택됩니다. 기본 기능 하에서는 seam이 컴파일 시점에 MLX 백엔드로 접히며 런타임 디스패치가 추가되지 않습니다.
  선택적 `experimental-backend` 기능은 향후 비 MLX 엔진을 위한 슬롯을 예약해 둡니다(issue #338).

## 요청 경로

### `mlxcel generate`

1. `src/main.rs`가 CLI 인자를 파싱합니다.
2. `src/commands/generate.rs`가 프롬프트/미디어 입력과 샘플링 옵션을 준비합니다.
3. 로딩 파이프라인이 `LoadedModel`을 구성합니다.
4. `mlxcel-core`가 decode 루프를 실행하고 출력을 stdout에 씁니다.

### `mlxcel serve` / `mlxcel-server`

1. `src/main.rs` 또는 `src/bin/mlx_server.rs`가 CLI 플래그와 `LLAMA_ARG_*`
   환경 변수 기반 옵션을 파싱합니다.
2. `src/server/startup.rs`가 시작 설정을 해석하고, 모델을 로드하고,
   Axum 애플리케이션을 구성합니다.
3. `src/server/app.rs`가 `/v1/chat/completions`,
   `/v1/completions`, `/v1/responses`, `/health`, `/v1/models` 같은 라우트를 마운트합니다. OpenAI
   오디오 인터페이스도 마운트됩니다(`/v1` 접두사가 있는 경로와 없는 경로 모두):
   `/v1/audio/speech`(text-to-speech), `/v1/audio/transcriptions`, `/v1/audio/translations`(speech-to-text)입니다.
   음성 모델이 `AppState`의 audio-model 슬롯에 연결되기 전까지는 구조화된
   `501 Not Implemented`를 반환합니다.
4. 라우트 핸들러가 요청을 내부 생성 작업으로 변환합니다. 채팅 요청은 `src/server/chat_template.rs`를 통해 렌더링되지만,
   Jinja 템플릿을 제공하지 않고 코드로 렌더링하는 형식을 가진 체크포인트는 예외입니다. Kimi K3의 XTML 렌더러
   (`src/server/kimi_k3_chat.rs`)는 토큰 id를 직접 생성하며, 이는 사전 토큰화된 요청 경로를 통해 스케줄러에 도달합니다.
5. `src/server/batch/`는 활성화된 경우 배치 decode를 스케줄링합니다. `BatchScheduler`는 매 tick마다 무엇을 실행할지 결정하고
   `mlxcel_core::engine::Engine`을 통해서만 모델을 구동합니다
   (`src/server/batch/scheduler/step_rows.rs`는 엔진 step 행에 대한 스케줄러 측 구현입니다).
   `--no-batch`는 `max_batch_size = 1`인 동일한 스케줄러입니다.
6. 스트리밍 응답은 SSE 프레임으로 전송됩니다.

#### Panic 및 스레딩 방침

릴리스 빌드는 `panic = "unwind"`를 사용하므로(issue #375), 의도된 오디오 워커 `catch_unwind`가 프로덕션에서 동작합니다. 오디오 워커(`src/server/audio_worker.rs`의 `run_guarded`)에서 발생한 합성 또는 전사 panic은 요청 단위 오류로 격리됩니다. capability가 비활성화된 XLA 오디오 전처리 기반도 향후 패밀리 기능 생성기 주위에 동일한 요청 단위 가드를 두어, XLA 오디오가 연결되지 않은 동안에도 호스트 워커를 정상 상태로 유지합니다. 모든 핵심 추론 워커 스레드는 의도적으로 반대 방침을 취합니다. `src/worker_failfast.rs`의 `run_core_thread_or_abort`가 배치 및 레거시 서버 워커(`src/server/model_worker.rs`)와 원격 pipeline 스테이지 서비스 스레드(`src/distributed/pipeline/remote_service.rs`)를 감싸므로, 깨진 불변식을 의미하는 panic은 조용히 unwind되어 서버가 생성할 수 없는 상태로 남는 대신 로그를 남기고 프로세스를 abort하여 감독자(supervisor)가 재시작하게 합니다. 분산 pipeline 스테이지에는 자체 `catch_unwind`가 없으며, 스테이지 오류는 코디네이터에서 `Result` 전파와 스테이지 타임아웃 및 health probing으로 처리되어 죽거나 실패한 스테이지가 요청 단위 오류로 드러납니다. 전역 abort panic 훅은 없습니다. 이는 unwind 이전에 실행되어 오디오 워커의 보호 장치를 무력화하기 때문입니다. MLX C++ FFI 예외는 여전히 Rust panic이 아니라 `std::terminate`가 되며, 실패 가능한 `try_*` 경계로 감싸지지 않은 모든 `cxx` 브리지 호출에서 프로세스를 종료시킵니다(issue #382로 추적 중). 배치 스케줄러의 decode 및 prefill eval 호출은 이렇게 감싸져 있어, 잡힌 throw는 프로세스가 아니라 영향을 받은 요청만 실패시키며, 백엔드가 복구 불가능해 보이면 스케줄러를 정상적으로 종료하는 연속 실패 가드가 이를 뒷받침합니다(issue #822). [ADR 0003](adr/0003-release-panic-unwind-with-core-thread-abort.md)을 참조하십시오.

## 플랫폼별 동작

- macOS/Metal, Linux/CUDA, Linux/ROCm 동작은 주로 `src/lib/mlxcelverse/` 아래의
  고정된 MLX 빌드, 해당 백엔드의 mlxcelverse 오버레이, Cargo에 전달되는 기능 플래그에 의해 결정됩니다.
- Apple Silicon 런타임/디바이스 헬퍼는 `src/lib/mlxcel-core/src/hardware.rs`와
  `src/execution/runtime.rs`에 있습니다.
- 커스텀 fused 커널 launcher는 `src/lib/mlxcelverse/kernels/` 아래에 있으며
  C++ 브리지를 통해 호출됩니다. 각 launcher는 Metal JIT 소스를 가지며, 이식된 경우 CUDA 대응 구현도 가지고,
  `mlx::core::metal::is_available()`에 의해 런타임에 선택됩니다: TurboQuant Sparse-V 및 위임된 SDPA,
  paged-attention decode(v1, 그리고 `src/lib/mlxcel-core/src/paged_v2/`에서 구동되고
  `MLXCEL_PAGED_ATTENTION_V2=1`로 선택되는 v2 cross-CTA split-KV 및 merge 커널), Gumbel-max 샘플링.
- CUDA 커널 동작은 대부분 MLX에서 상속됩니다. `mlxcel`은 빌드 시 `MLX_CUDA_ARCHITECTURES`를 통해
  CUDA 아키텍처 목록을 전달합니다.
- ROCm(실험적, `--features rocm`)은 `src/lib/mlxcelverse/overlays/rocm/`의 벤더링된 ROCm 백엔드를 실행합니다.
  빌드는 HIP `gfx` 타깃을 `MLX_ROCM_ARCHITECTURES`(기본값: `rocminfo`가 보고하는 타깃)를 통해 전달하고,
  백엔드의 커널 아카이브와 ROCm 라이브러리를 링크하며, `$ROCM_PATH/lib`를 rpath로 임베드합니다.
  위의 fused launcher는 Metal을 사용할 수 없을 때마다 CUDA를 선택하므로, ROCm에서는 MLX 그래프 폴백이 있는 경우
  그 폴백을 사용합니다. 백엔드 종류별 라우팅은 lablup/mlxcel#1803입니다.
  상태와 알려진 공백은
  [Linux with AMD ROCm](installation.md#linux-with-amd-rocm-experimental)을 참조하십시오.

## 분산 및 멀티 디바이스

`src/distributed/`에는 tensor parallelism, pipeline parallelism, 분리형(disaggregated) 추론 실험이 사용하는
공유 클러스터 설정, transport, registry, 메트릭, 스케줄러 인프라가 있습니다.
운영자용 요약은 [분산 추론](distributed.md)을 참조하십시오.

## 추가 자료

- [새 모델 추가하기](adding-models.md)
- [오디오 입력 전처리](audio-preprocessing.md)
- [지원 모델](supported-models.md)
- [TurboQuant KV cache](turbo-kv-cache.md)
- [OpenAI Responses API](responses-api.md)
- [분산 추론](distributed.md)
