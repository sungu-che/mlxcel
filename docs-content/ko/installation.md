# 설치

`mlxcel`은 루트 Rust 패키지에서 두 개의 네이티브 실행 파일을 빌드합니다.

- `mlxcel` — 명령줄 생성, 모델 목록 조회, 다운로드를 담당합니다.
- `mlxcel-server` — OpenAI/llama-server 스타일 엔드포인트를 제공하는 HTTP 서버입니다.

이 바이너리는 런타임에 Python이나 Node.js를 필요로 하지 않습니다. 완전한 정적
바이너리는 아니며, 플랫폼별 GPU/런타임 라이브러리는 여전히 필요합니다.

## 지원 플랫폼

| 플랫폼 | 상태 | 일반적인 feature 플래그 | 참고 |
|----------|--------|-----------------------|-------|
| Apple Silicon 기반 macOS | 주요 | `metal,accelerate` | 주된 개발 및 검증 대상입니다. |
| NVIDIA CUDA 기반 Linux | 보조 | `cuda` | 릴리스 빌드는 현재 CUDA 13 시대 시스템을 대상으로 하며, 다른 버전은 MLX/CUDA 호환성에 따라 달라집니다. |
| AMD ROCm 기반 Linux | 실험적 | `rocm` | 소스 빌드만 지원합니다. ROCm 10.0 / HIP 7.15의 RDNA 3.5(`gfx1151`, Strix Halo)에서 검증되었으며 lablup/mlxcel#1801에서 추적합니다. [AMD ROCm 기반 Linux](#amd-rocm-기반-linux-실험적)를 참고하십시오. |
| CPU 전용 Linux | 릴리스 대상 아님 | 없음 | 제한된 구성에서는 컴파일될 수 있지만, 이 프로젝트에서 유용하거나 검증된 추론 대상은 아닙니다. |
| NVIDIA CUDA 기반 Windows | 실험적 | `cuda` | 소스 빌드만 지원하며 단일 노드입니다(tensor 또는 pipeline 병렬화 없음). `aarch64-pc-windows-msvc`(NVIDIA RTX Spark N1X)에서 네이티브로 검증되었으며, 빌드 glue는 `x86_64-pc-windows-msvc`도 포함합니다. [NVIDIA CUDA 기반 Windows](#nvidia-cuda-기반-windows-실험적)를 참고하십시오. |
| AMD ROCm 기반 Windows | 실험적 | `rocm` | 소스 빌드만 지원하며 단일 노드입니다. Radeon 8060S(`gfx1151`)와 ROCm 10.1.0을 사용하여 `x86_64-pc-windows-msvc`에서 네이티브로 검증되었습니다. `PATH`에 ROCm `bin` 디렉터리를 두거나 `ROCM_PATH`를 설정하지 않아도 실행됩니다. [AMD ROCm 기반 Windows](#amd-rocm-기반-windows-실험적)를 참고하십시오. |

## Cargo feature 플래그

두 바이너리(`mlxcel`과 `mlxcel-server`)는 같은 루트 패키지에서 빌드되므로
하나의 feature 집합이 둘 모두에 적용됩니다. `cargo build --features <a,b>`로 전달합니다.
배포용 빌드는 기본값과 플랫폼 백엔드 플래그를 활성화하며, 나머지는 선택적 seam,
진단 harness, 테스트용 scaffolding입니다.

| Feature | 기본값 | 효과 |
|---------|---------|--------|
| `surgery` | **on** | Axis A 가중치 로드 surgery입니다. 로드 시점에 `scale` / `add` / `prune` / `replace` / `interpolate` 가중치 공간 편집을 수행하는 `--surgery <config.yaml>`과 `MLXCEL_SURGERY`를 제공하며, `mlxcel-surgery` crate를 가져옵니다. surgery 설정이 제공되지 않으면 로드 경로는 이 feature가 없는 빌드와 바이트 단위로 동일합니다. |
| `webui` | **on** | 관리용 WebUI를 위한 번들된 오프라인 브라우저 셸입니다(issue #1836). 커밋된 `src/webui/assets/` 번들을 `rust-embed`와 `mime_guess`로 내장하므로, 바이너리는 Node 단계 없이 이를 제공합니다. 자세한 내용은 [WebUI 번들링](webui/bundling.md)을 참고하십시오. `--no-default-features`를 지정하면 assets와 내장 의존성이 빠집니다. |
| `metal` | off | Apple Silicon Metal GPU 백엔드입니다(`mlxcel-core/metal`에 위임). macOS에서 표준입니다. |
| `accelerate` | off | Apple Accelerate CPU BLAS 백엔드입니다(`mlxcel-core/accelerate`에 위임). macOS에서 표준입니다. |
| `cuda` | off | NVIDIA CUDA GPU 백엔드입니다(`mlxcel-core/cuda`에 위임). NVIDIA 호스트에서 필수이며, 일반 빌드는 CPU 전용입니다(아래 함정 안내 참고). |
| `rocm` | off | Linux와 Windows x86_64 MSVC의 AMD GPU 백엔드입니다(`mlxcel-core/rocm`에 위임). `src/lib/mlxcelverse/overlays/rocm/`의 ROCm overlay에서 빌드됩니다. `cuda` 또는 `metal`과 함께 사용할 수 없습니다. 실험적이며 [AMD ROCm 기반 Linux](#amd-rocm-기반-linux-실험적)와 [AMD ROCm 기반 Windows](#amd-rocm-기반-windows-실험적)를 참고하십시오. |
| `experimental-backend` | off | MLX가 아닌 compute 백엔드 seam 슬롯을 예약합니다(issue #338). 커널은 포함하지 않으며 런타임 dispatch도 추가하지 않습니다. 향후 MLX가 아닌 엔진(예: FuriosaAI RNGD)이 `ComputeBackend`를 구현할 플러그인 경계만 컴파일합니다. `select_backend()`는 여전히 MLX로 귀결됩니다. |
| `xla-backend` | off | OpenXLA / StableHLO 백엔드 seam입니다(issue #449, [ADR 0004](adr/0004-compute-backend-session-seam-and-stablehlo-family.md)). `mlxcel-xla`를 가져오고 `Backend::Xla` / `Session::Xla` arm과 `MLXCEL_BACKEND=xla` 선택기를 컴파일하지만, 네이티브 실행 엔진은 포함하지 않습니다. 이 crate는 순수 Rust 스텁과 StableHLO 그래프 emitter로 이루어져 있어 CI에서 그대로 빌드됩니다. |
| `xla-iree` | off | `xla-backend`에 실제 IREE 실행(`mlxcel-xla/iree`)을 더합니다. 사전 빌드된 IREE 런타임에 대해 C shim을 컴파일하고 번들된 prefill / decode_step 그래프를 구동합니다. 빌드 시점에 `IREE_DIST`(또는 아래의 소스 빌드 변수)가 필요하므로, CI나 릴리스 기본값이 아니라 로컬 / 선택적 빌드입니다. |
| `xla-reference-diagnostics` | off | `xla-iree`에 `mlxcel-xla/diagnostics`를 더하여 네이티브 IREE 진단에서 device 중립 reference 캡처를 수행합니다. eager MLX 쪽은 기본 CPU 백엔드에 두고, IREE CUDA 소스나 빌드 트리 없이 IREE `local-task`를 검증합니다. `xla_gemma3_reference_check`와 `xla_molmo_reference_check` 예제를 gate합니다. |
| `xla-diagnostics` | off | Gemma3n 중간 oracle 검증을 위한 테스트 전용 forwarding입니다. 고정된 reference는 프로덕션 CUDA 런타임이므로, `xla-diagnostics-cpu` 위에 `cuda`를 활성화합니다. `xla_llava_reference_check`와 `xla_vision_reference_check` 예제를 gate합니다. `xla-iree`의 일부가 아니며, 프로덕션 번들은 변경되지 않습니다. |
| `xla-diagnostics-cpu` | off | `xla-iree`에 `mlxcel-xla/diagnostics`를 더한 것으로, CUDA 드라이버가 없는 머신에서 범위가 제한된 호스트 reference 검사를 위한 CPU 전용 짝입니다. `xla_qwen25_vl_window_check`, `xla_qwen3_vl_deepstack_check`, `xla_youtu_vl_reference_check`, `xla_gemma3n_audio_reference_check`, `xla_gemma3n_sscp_conv_oracle` 예제를 gate합니다. 검증된 `xla-diagnostics` 프로덕션 대상 gate를 의도적으로 대체하지 않습니다. |
| `xla-diagnostics-rocm` | off | `rocm`에 `xla-diagnostics-cpu`를 더한 것으로, 프로덕션 AMD GPU 중간 oracle 검증을 위한 ROCm 기반 짝입니다. |
| `xla-micro-oracle` | off | `xla-iree`에 `mlxcel-xla/micro-oracle`을 더한 것으로, CPU에서 실행 가능한 범위 제한 MLX/IREE 연산자 oracle harness입니다. `xla_numeric_probe` 예제를 gate합니다. CUDA 전용 프로덕션 probe는 `cuda`를 추가할 수 있으나, 공유 리포트 및 비교 계층은 이를 요구하지 않습니다. |
| `test-utils` | off | 테스트 전용 helper입니다. `distributed_integration`, `pipeline_e2e`, `paged_handoff_parity` 통합 테스트를 빌드하는 데 필요합니다(`cargo test --features test-utils`). 바이너리에는 필요하지 않습니다. |
| `voicechat-mic` | off | Nemotron VoiceChat `/v1/realtime` 엔드포인트용 실시간 마이크 및 스피커 클라이언트입니다(issue #1376). `cpal` 오디오 device crate는 `cargo run --example voicechat_microphone --features voicechat-mic`에서만 컴파일되며, 라이브러리, 서버, CI용으로는 컴파일되지 않습니다. [Nemotron VoiceChat](nemotron-voicechat.md)을 참고하십시오. |

`default = ["surgery", "webui"]`이므로 일반 `cargo build`는 surgery와 번들된 WebUI만
활성화합니다. 실제 빌드에서는 항상 그 위에 플랫폼 백엔드를 추가하며, 예를 들어
Apple Silicon에서는 `--features metal,accelerate`, NVIDIA에서는 `--features cuda`를
사용합니다. `mlxcel-surgery` crate와 내장된 WebUI assets를 완전히 제외하려면(surgery
이전 동작과 비교하는 CI parity 테스트나 제약이 있는 임베디드 대상) `--no-default-features`로
빌드하십시오.

```bash
# Metal + Accelerate, no surgery crate, no WebUI assets.
cargo build --release --no-default-features --features metal,accelerate
```

### OpenXLA / StableHLO 백엔드 (`xla-backend`, `xla-iree`)

XLA 경로는 두 단계로 나뉜 선택적 기능이며 Apple Silicon이나 CUDA 배포 빌드에는
들어가지 않으므로, 해당 바이너리는 이를 전혀 컴파일하지 않고 seam은 MLX로 귀결됩니다.

- `xla-backend`는 seam만 컴파일합니다. `Backend::Xla` / `Session::Xla` arm,
  `MLXCEL_BACKEND=xla` 선택, StableHLO 그래프 emitter입니다. 네이티브 툴체인이
  필요 없으므로 CI에서 그대로 빌드됩니다.
- `xla-iree`는 실행 런타임을 추가합니다. 빌드 스크립트가 사전 빌드된 IREE 배포본에
  대해 C shim을 컴파일하므로, 빌드 시점에 다음 중 하나를 설정해야 합니다.
  - `IREE_DIST`: 압축을 푼 `iree-dist-<ver>-linux-<arch>` 트리(CPU / Vulkan
    dist)입니다. dist 자체의 `bin/iree-compile`이 번들된 그래프를 lowering합니다.
  - `IREE_CUDA_HOME` (+ `IREE_CUDA_COMPILE`): GB10 급 GPU 경로를 위해 소스 빌드한
    CUDA 지원 IREE 런타임과 이에 맞는 cuda 지원 `iree-compile`입니다.
    `scripts/iree/setup-cuda.sh`가 이 트리를 생성합니다.
  - `IREE_ROCM_HOME` (+ `IREE_ROCM_COMPILE`): AMD ROCm GPU 경로(`gfx1151` / `gfx1100` /
    `gfx942`)를 위해 소스 빌드한 HIP/ROCm 지원 IREE 런타임(`IREE_HAL_DRIVER_HIP=ON`)과
    이에 맞는 rocm 지원 `iree-compile`입니다. `scripts/iree/setup-rocm.sh`(또는
    `make iree-rocm`)가 이 트리를 생성하고 이에 맞는 환경을 출력합니다.
  - `IREE_MACOS_HOME` (+ `IREE_MACOS_COMPILE`): Apple Silicon 개발 경로를 위해 소스 빌드한
    macOS 런타임과 Metal 지원 `iree-compile`입니다. `scripts/iree/setup-macos.sh`가
    이 트리를 생성하고 이에 맞는 환경을 출력합니다.

런타임에는 `MLXCEL_BACKEND=xla`로 백엔드를 선택하고, `MLXCEL_XLA_*` 변수(device,
precision, packed quant)로 조정합니다. 전체 목록은
[환경 변수](environment-variables.md#openxla--stablehlo-백엔드-변수)를,
설계는 [ADR 0004](adr/0004-compute-backend-session-seam-and-stablehlo-family.md)를
참고하십시오.

## Apple Silicon 기반 macOS

사전 요구 사항:

- Apple Silicon Mac.
- Rust 2024 edition과 호환되는 Rust 툴체인.
- Xcode Command Line Tools (`xcode-select --install`).
- Metal 툴체인 구성 요소.
- `PATH`에서 사용할 수 있는 CMake.
- `ffmpeg` 5.0 이상. 비디오 입력이 필요한 경우에만 필요합니다(`brew install ffmpeg`).
  이는 빌드 의존성이 아니라 런타임 의존성입니다. 빌드와 모든 텍스트, 이미지,
  오디오 경로는 이것 없이도 동작하며, 없을 때 `--video`는 이름이 붙은 오류를
  보고합니다. [비디오 입력과 ffmpeg](#비디오-입력과-ffmpeg)를 참고하십시오.

```bash
# One-time: install the Metal shader compiler if it is not already present.
xcodebuild -downloadComponent MetalToolchain

git clone https://github.com/lablup/mlxcel.git
cd mlxcel
cargo build --release --features metal,accelerate
```

macOS에서는 일반 `cargo build --release`에서도 MLX가 Metal과 Accelerate를 활성화하며, 위의 명시적 feature는 의도한 백엔드를 문서화하기 위한 것입니다. MTP `qmv_wide` 재시도와 같은 런타임 제어는 일반 빌드를 포함하여 실제로 빌드된 백엔드를 따릅니다. `MLXCEL_BUILD_METAL=OFF`는 Cargo `metal` feature가 선택되어 있더라도 Metal 백엔드와 그 제어를 비활성화합니다.

체크포인트를 로드하지 않고 런타임 제어를 검증하려면 `cargo run -p mlxcel-core --profile test-fast --example metal_runtime_switch_probe`를 실행하십시오. QMV 선택과 decode command-buffer 재정의를 모두 확인한 뒤 이전 값으로 복원합니다. Metal device를 사용할 수 없으면 skip을 보고합니다.

[#1988](https://github.com/lablup/mlxcel/issues/1988)의 영향을 받는 이전 일반 macOS 빌드에서는 기존 서버 명령 앞에 `MLXCEL_QMV_WIDE=0`을 붙여 새 프로세스를 시작하십시오. 이는 exactness gate를 켠 상태로 시작 시점에 더 좁은 Metal projection 커널로 고정합니다. 처리량이 줄어들 수 있으며, Metal에만 적용되고, 로드된 모델의 probe가 여전히 통과해야 합니다.

빌드 결과물:

```text
target/release/mlxcel
target/release/mlxcel-server
```

macOS 릴리스 워크플로는 필요할 때 `mlx.metallib` 아티팩트도 함께 패키징합니다.
바이너리를 수동으로 배포하는 경우에는 실행 파일만으로 항상 충분하다고 가정하지 말고,
릴리스 워크플로에 비추어 런타임 패키지 구성을 확인하십시오.

## CUDA 기반 Linux

사전 요구 사항은 배포판과 CUDA 버전에 따라 다릅니다. 최소한 다음이 필요합니다.

- Rust 2024 edition과 호환되는 Rust 툴체인.
- CMake와 C++20을 지원하는 컴파일러.
- `nvcc`를 포함한 CUDA toolkit.
- 선택한 CUDA toolkit과 호환되는 NVIDIA 드라이버.
- 고정된 MLX 빌드가 요구하는 cuDNN 및 CUDA 런타임 라이브러리.
- BLAS 및 LAPACK 개발 패키지(C 헤더 포함). MLX의 CMake는 `cblas.h`와
  `lapacke.h`를 찾으므로, 런타임 라이브러리뿐 아니라 `lapacke` 헤더도 있어야 합니다.
- `ffmpeg` 5.0 이상. 비디오 입력이 필요한 경우에만 필요합니다
  (`sudo apt-get install -y ffmpeg`). macOS와 마찬가지로 런타임 전용이며,
  [비디오 입력과 ffmpeg](#비디오-입력과-ffmpeg)를 참고하십시오.

Debian/Ubuntu(x86_64 또는 aarch64)에서의 빌드 패키지는 다음과 같습니다.

```bash
sudo apt-get install -y \
    build-essential cmake git \
    libopenblas-dev liblapack-dev liblapacke-dev
# CUDA toolkit (nvcc) and cuDNN come from NVIDIA's apt repository, e.g.
#   cuda-toolkit-13-0  cudnn9-cuda-13
```

`liblapacke-dev`는 `lapacke.h`를 제공하는 패키지입니다. `liblapack-dev`만으로는
이 헤더가 빠지며, MLX CMake configure 단계가 `LAPACK_INCLUDE_DIRS`를 `NOTFOUND`로
설정하여 실패합니다.

빌드 예시:

```bash
git clone https://github.com/lablup/mlxcel.git
cd mlxcel
cargo build --release --features cuda
```

> **CPU 전용 빌드 함정.** Linux에서 일반 `cargo build --release`는 기본 feature
> (`cuda` 없음)를 사용하므로 CPU 전용 바이너리가 생성됩니다. 로드와 생성은 여전히
> 되지만, MLX가 호스트 CPU에서 조용히 실행되어 GPU 처리량의 극히 일부만 나옵니다
> (GB10에서 수백 tok/s여야 할 것이 한 자릿수 tok/s). 그래서 실수를 알아차리기 쉽지 않습니다.
> NVIDIA 호스트에서는 항상 `--features cuda`를 지정하십시오.

CUDA가 `/usr/local/cuda` 아래에 설치되어 있지 않으면 `CUDA_HOME`을 설정하십시오.

```bash
CUDA_HOME=/opt/cuda cargo build --release --features cuda
```

### CUDA 아키텍처 선택

> **Volta(sm_70)에는 CUDA 12.x 툴체인이 필요합니다.** CUDA 13은 Volta 지원을 제거했으므로, 그 `nvcc`는 아무것도 컴파일하기 전에 `nvcc fatal : Unsupported gpu architecture 'compute_70'`으로 `compute_70`을 즉시 거부합니다. 공개된 릴리스 아카이브는 CUDA 13으로 빌드되므로 sm_70 코드를 전혀 포함하지 않으며, 프로젝트의 CUDA CI runner에는 CUDA 13이 설치되어 있으므로 `cuda-sm70-compile` gate는 실패하지 않고 그곳에서 skip됩니다. V100이나 다른 Volta 카드용으로 빌드하려면 CUDA 12.x가 설치된 호스트에서 소스 빌드를 해야 합니다. CUDA 12.9.41에서 이를 검증했으며, 해당 버전은 sm_70을 문제없이 컴파일합니다.


`src/lib/mlxcel-core/build.rs`는 `MLX_CUDA_ARCHITECTURES`를 읽습니다. 설정되지 않은 경우
빌드 스크립트가 `nvidia-smi`로 compute capability를 감지하고, 이를 릴리스 워크플로가
표기하는 방식으로 적습니다. Hopper에는 CUDA의 아키텍처 전용 `a` 접미사가 붙고
(`90`은 `90a`가 됨), Blackwell(`sm_100`, `sm_120`, `sm_121`)은 접미사 없이 유지됩니다.
감지에 실패하면 `90a`로 fallback됩니다. 따라서 자동 감지된 빌드와 공개된 빌드는 포함하는
아키텍처의 범위가 다를 뿐, 그 아키텍처에서 얻는 기계어 코드는 결코 다르지 않습니다.
이것이 핵심입니다. issue
#1943 이전에는 규칙이 SM 90 이상의 모든 대상에 접미사를 붙였으므로, GB10에서의 기본 빌드는
`121a`를 생성한 반면 같은 카드용 릴리스 아카이브에는 `121`이 들어 있었습니다.

명시적으로 설정한 `MLX_CUDA_ARCHITECTURES`는 그대로 사용되므로, 직접 지정할 때도 같은
방식으로 표기하십시오. Hopper는 `90a`, Blackwell은 접미사 없이 지정합니다. 이 섹션의
나머지 부분은 Blackwell이 접미사 없이 지정되는 이유를 설명합니다.

Blackwell에서는 `a` 접미사가 다른 것을 결정합니다. MLX는
`mlx/backend/cuda/quantized/nvfp4_quantize.cuh`의 하드웨어 block-float 변환기,
즉 `cvt.rn.satfinite.e2m1x2.f32`를 발행하는 변환기를, nvcc가 아키텍처 전용 대상을
컴파일할 때에만 컴파일합니다(nvcc는 이를 `__CUDA_ARCH_SPECIFIC__`로 알립니다). 따라서 접미사 없는
`121` 빌드에서는 이들이 컴파일에서 빠지며, NVFP4와 MXFP4 양자화는 빌드 출력에 아무런 표시 없이
스칼라 CUTLASS 변환 시퀀스로 fallback됩니다.

목록에 `121a`를 넣으면 모든 translation unit에 그 비용을 부담시켜 이를 해결할 수 있습니다.
CUTLASS는 같은 매크로에서 `CUTLASS_ARCH_MMA_SM121A_ENABLED`를 도출하므로 decode 커널도
다르게 컴파일되고, 접미사가 일치하는 device에서는 아키텍처 전용 이미지가 드라이버가 무시할 수
있는 대안이 아니라 실제로 로드되는 이미지가 됩니다. 이 프로젝트의 GB10에서는 어떤 decode
커널도 호출하지 않는 변환기 하나를 고치기 위해 아카이브가 12.7 MB 늘어나고 모든 Blackwell
머신에서 `qmv`가 달라지게 됩니다.

따라서 목록은 접미사 없이 유지되고, `src/lib/mlxcelverse/CMakeLists.txt`가 이러한 변환기에 도달할 수 있는 유일한 translation unit인 `fp_quantize.cu`에만 아키텍처 전용 이미지를 추가합니다. 요청하지 않아도 하드웨어 경로를 얻으며, 그것이 다른 어떤 것에도 영향을 주지 않습니다. `121a`로 빌드하는 것은 전진이 아니라 퇴보이며 스스로를 무력화하기도 합니다. 목록이 이미 아키텍처 전용으로 지정한 capability는 주입 과정에서 건너뛰는데, nvcc에 같은 `--generate-code`를 두 번 요청하는 것은 no-op이 아니라 오류이기 때문입니다. `121f`는 더 나쁩니다. 변환기의 dispatcher gate는 만족시키지만 자체 gate는 만족시키지 못해 컴파일 자체가 실패합니다.

Hopper의 `90a`는 다른 경우입니다. 두 릴리스 목록이 모두 배포하는 값이므로 자동 감지된 표기가 이와 일치합니다. 현재 MLX pin에서는 이 접미사가 더 이상 단독으로 어떤 것도 gate하지 않습니다. upstream 커밋 `44540d12`가 `qmm_sm80`, `qmm_sm90`, `gather_gemm`을 런타임 NVRTC 컴파일로 옮기고, 이 섹션의 이전 버전이 인용하던 `MLX_CUDA_SM90A_ENABLED` 정의를 제거했습니다. 또한 `jit_module.cpp`는 이제 실행 중인 device에서 NVRTC `--gpu-architecture`를 도출하고, compute capability 9 이상에서는 스스로 `a`를 덧붙입니다. 고정된 트리를 `90`과 `90a`로 각각 크로스 컴파일하면 결과가 일치합니다. `qmm_sm90.cu`, `qmm.cu`, `qmm_sm80.cu`는 어느 표기에서도 device 함수를 내보내지 않으며, `qmv.cu`와 `fp_qmv.cu`는 cubin이 아키텍처 전용임을 표시하는 `EF_CUDA_ACCELERATORS` 헤더 플래그를 제외하면 동일한 SASS를 내보냅니다. `CUTLASS_ARCH_MMA_SM90A_ENABLED`는 여전히 `__CUDA_ARCH_FEAT_SM90_ALL`을 기준으로 하므로, 이후의 pin에서는 다시 의미를 가질 수 있습니다.

```bash
# Hopper / GH200-style target, spelled the way the release workflow spells it.
MLX_CUDA_ARCHITECTURES=90a cargo build --release --features cuda

# GB10 / DGX Spark-style target used by the release workflow. Plain: the
# hardware NVFP4/MXFP4 converter is added to one translation unit by CMake.
MLX_CUDA_ARCHITECTURES=121 cargo build --release --features cuda

# Multiple targets, if your MLX/CUDA toolchain supports them.
MLX_CUDA_ARCHITECTURES="90a;121" cargo build --release --features cuda
```

완성된 빌드가 무엇을 담고 있는지 확인하려면 환경 변수가 아니라 아카이브를 읽으십시오.
올바른 Blackwell 빌드는 `fp_quantize.cu.o`에 fp4 명령이 있고, 전체 아카이브에 아키텍처
전용 이미지가 정확히 하나 있으며, forward-JIT PTX는 그대로입니다.

```bash
A=$(ls -d target/release/build/mlxcel-core-*/out/build/lib/libmlx.a | head -1)
mkdir -p /tmp/mlxq && (cd /tmp/mlxq && ar x "$A" fp_quantize.cu.o qmv.cu.o)
cuobjdump --dump-sass /tmp/mlxq/fp_quantize.cu.o | grep -c F2FP.SATFINITE.E2M1  # > 0
cuobjdump --list-elf  /tmp/mlxq/qmv.cu.o | grep -c 'sm_[0-9]*a'                 # 0: decode untouched
cuobjdump --list-elf  "$A" | grep -c 'sm_[0-9]*a'                               # 1
cuobjdump --dump-ptx  "$A" | grep -c '.target sm_121$'                          # > 0
```

하나의 체크아웃에는 feature 집합과 아키텍처 목록마다 하나씩, 이러한 빌드 디렉터리가 여러 개
있을 수 있으므로, 읽고 있는 것이 의도한 것인지 확인하십시오.
`grep MLX_CUDA_ARCHITECTURES "$(dirname "$A")/../CMakeCache.txt"`가 해당 빌드가 구성된 목록을 알려 줍니다.

PTX에서 `cvt.rn.satfinite.e2m1x2`를 찾지 마십시오. 주입은 cubin을 내보내고 PTX는 내보내지 않으며, 내보내지는 PTX는 fallback arm을 타는 접미사 없는 `compute_121` pass에서 나오므로, 올바른 빌드에서도 해당 grep은 0을 반환합니다. 두 빌드를 구분하는 것은 위의 SASS 개수입니다.

바이너리가 빌드된 아키텍처 목록이 실행되는 GPU를 포함하지 않으면, 첫 번째 커널 launch에서 불투명한 CUDA 로드 오류로 나중에 실패하는 대신 시작을 거부하고, 자신이 가진 목록과 찾아낸 compute capability를 모두 밝힙니다(issue #1537). 이는 두 경우에 발생합니다. 매트릭스가 `80`에서 시작하는 공개된 x86_64 아카이브를 V100 같은 Ampere 이전 카드에서 실행하는 경우, 그리고 `nvidia-smi`를 사용할 수 없었던 호스트에서 만든 소스 빌드가 `90a`로 fallback되어 자신의 빌드 머신에서 실행되지 못하는 경우입니다. 두 경우 모두 해결책은 `MLX_CUDA_ARCHITECTURES`를 대상 device로 설정하여 위와 같이 다시 빌드하는 것입니다. 실행 중인 capability, 컴파일된 목록, device가 cubin으로 서비스되는지 JIT 컴파일된 PTX로 서비스되는지를 출력하려면 `MLXCEL_TRACE_ARCH`를 설정하십시오([환경 변수](environment-variables.md) 참고). 같은 요약은 시작 시 `Detected N GPU(s)` 줄 옆에도 나타납니다. `MLXCEL_DEVICE=cpu`는 이 거부를 우회하므로, 잘못된 아키텍처로 빌드된 바이너리를 올바른 것을 빌드하는 동안 CPU에서 실행할 수 있습니다.

저장소의 릴리스 워크플로는 self-hosted runner에서 두 개의 Linux CUDA 대상을 각각
하나의 fat 바이너리로 빌드합니다. aarch64는 GH200(`90a`), GB200(`100`), GB10(`121`)을
하나의 빌드(`90a;100;121`)로 포괄하고, x86_64는 Ampere부터 Blackwell까지
(`80;86;89;90a;100;120`)를 포괄합니다. 위에서 설명한 이유로 두 대상 모두 Blackwell은 접미사
없이 지정되며, 하드웨어 변환기는 CMake가 하나의 translation unit에 추가합니다. 각
대상에 대해 `mlxcel` CLI와 `mlxcel-server`는 별도의 아카이브(`mlxcel-...`과
`mlxcel-server-...`, 각각 약 347 MB)로 배포되므로 사용자는 필요한 것만 다운로드하면
됩니다. 배포되는 모든 릴리스에는 공급망 투명성과 취약점 스캔을 위해
`sbom-<version>.cyclonedx.json.gz`라는 이름의 CycloneDX SBOM도 함께 제공됩니다. 그 외의
GPU/OS 조합은 로컬 검증이 필요한 소스 빌드로 취급하십시오.

### ROCm 아키텍처 선택

`src/lib/mlxcel-core/build.rs`는 `MLX_ROCM_ARCHITECTURES`를 읽습니다. 설정되지 않은 경우 빌드 스크립트가 `rocminfo`에 GPU agent의 `gfx` 대상을 물어보며, agent가 보고되지 않으면 이름이 붙은 오류와 함께 실패합니다. 이는 `/dev/kfd`에 접근할 수 없는 컨테이너에서 발생합니다. 이 변수를 설정하면 감지를 완전히 우회하고 그대로 사용되므로, 보이는 GPU가 없는 빌드 호스트에서도 특정 GPU용 바이너리를 만들 수 있습니다.

```bash
# The Radeon 8060S (Ryzen AI MAX+ 395) used for ROCm validation.
MLX_ROCM_ARCHITECTURES=gfx1151 cargo build --release --features rocm

# Multiple targets, semicolon-separated.
MLX_ROCM_ARCHITECTURES="gfx1100;gfx1151" cargo build --release --features rocm
```

모든 항목은 HIP target ID여야 합니다. 즉 `gfx` 뒤에 소문자 16진수 숫자가 오거나(`gfx1151`), generic 대상(`gfx11-generic`)이거나, feature 접미사가 붙은 대상(`gfx90a:xnack+`)이어야 합니다. `native`, 대문자 이름, 옵션 문자열처럼 그 밖의 값은 CMake configure 단계를 `Not a HIP target ID`로 중단시키는데, 각 항목이 `--offload-arch` 컴파일러 인수가 되기 때문입니다.

HIP의 커버리지는 CUDA의 커버리지처럼 순서가 있는 관계가 아닙니다. CUDA cubin은 더 높은 minor 리비전에서 실행되고 PTX는 major 버전을 넘어 앞으로 JIT되므로, 목록이 명시하지 않은 device도 포괄할 수 있습니다. HIP code object는 하나의 `gfx` 대상용으로 빌드되어 그 대상에서만 실행되며 JIT fallback이 없으므로, `gfx1151`과 `gfx1150`은 숫자가 인접해 있어도 서로 무관하고, 목록은 명시한 대상만을 정확히 포괄합니다. 대상 feature 접미사(`gfx90a:xnack+`)는 다른 대상을 명명하는 것이 아니라 한 대상에서 code object 기능을 선택하므로, 목록을 device와 비교할 때는 무시됩니다.

컴파일된 `gfx` 목록에 실행되는 device가 없는 바이너리는, 첫 번째 커널 launch에서 불투명한 HIP 오류로 나중에 실패하는 대신 시작을 거부하고 둘 다 밝힙니다(issue #1805). 실행 중인 대상, 컴파일된 목록, 포함 여부를 출력하려면 `MLXCEL_TRACE_ARCH`를 설정하십시오([환경 변수](environment-variables.md) 참고). 같은 요약은 시작 시 `Detected N GPU(s)` 줄 옆에 device 이름 및 메모리와 함께 나타납니다. `MLXCEL_DEVICE=cpu`는 이 거부를 우회하므로, 잘못된 대상으로 빌드된 바이너리를 올바른 것을 빌드하는 동안 CPU에서 실행할 수 있습니다.

### 사전 빌드된 CUDA 아티팩트: 런타임 요구 사항

MLX의 CUDA 백엔드는 일부 커널을 처음 실행될 때 NVRTC로 런타임에 컴파일하므로
(gather와 기타 인덱싱 커널, 그리고 2026-07 MLX pin 이후로는 양자화 matmul 커널도),
사전 빌드된 바이너리는 런타임 라이브러리뿐 아니라 배포 호스트에서 사용할 수 있는
CUDA 헤더도 필요로 합니다.

- **CCCL(libcu++) 헤더**는 사전 빌드된 Linux CUDA 아카이브(aarch64와 x86_64 모두)
  안에 번들되어 있습니다. 각 아카이브는 `bin/` + `include/cccl/`로 풀리며, 이는 MLX의
  JIT가 실행 파일 기준으로 찾는 구성입니다(`<exe-dir>/../include/cccl`).
  `mlxcel`/`mlxcel-server`는 `bin/` 아래에, `include/cccl/` 디렉터리는 그 옆에 두고,
  구조를 평평하게 만들지 마십시오. 런타임은 번들된 헤더를 실행 파일의 정식 경로
  (`/proc/self/exe`)에서 찾으므로, 상대 경로 `./mlxcel`을 포함한 어떤 실행 방식도
  동작합니다. mlxcel을 임베딩하면서 평평한 바이너리 구조를 유지하는 경우처럼 JIT가
  CCCL 헤더를 명시적으로 가리키게 하려면 `MLXCEL_CCCL_DIR`을 설정하십시오.
- **CUTLASS/CuTe 헤더**도 같은 방식으로 번들됩니다(`bin/` 옆의 `include/cute/`와
  `include/cutlass/`). 2026-07 이후의 MLX pin은 양자화 matmul 커널(`qmm`,
  `gather_gemm`)을 NVRTC로 JIT 컴파일하며, 이 커널들은 `<cute/...>`/`<cutlass/...>`를
  include합니다. JIT는 이를 `<exe-dir>/../include`에서 찾으며, 평평한 embedded 구조처럼
  재정의하려면 `cute/`와 `cutlass/`를 담은 디렉터리를 `MLXCEL_CUTLASS_DIR`로
  지정하십시오. 소스 빌드는 자동으로 빌드 트리로 fallback됩니다. 이 헤더가 없으면
  양자화 모델을 처음 실행할 때 `cannot open source file "cute/numeric/numeric_types.hpp"`로
  실패합니다.
- **CUDA toolkit 헤더**(`cuda_runtime.h` 등)는 호스트에서 가져옵니다. CUDA toolkit을
  설치하고 `/usr/local/cuda`에 없다면 `CUDA_HOME`(또는 `CUDA_PATH`)을 설정하십시오.
  이것이 없으면 첫 번째 NVRTC 컴파일이 `cannot open source file` 오류로 실패합니다.
- **CUDA 공유 라이브러리**도 호스트 toolkit에서 가져옵니다. 바이너리는 `cudart`,
  `cublas`, `cublasLt`, `cufft`, `cusolver`, `nvrtc`, cuDNN을 동적으로 링크합니다.
  cuSOLVER는 MLX pin이 `81ba1c6a`로 옮겨 가면서 이 목록에 추가되었으며(MLX의 CUDA
  백엔드가 이제 Cholesky에 이를 사용하고 시작 시 초기화합니다), 따라서 이를 빠뜨린
  runtime 전용 설치는 실행 시 `libcusolver.so...: cannot open shared object file`로
  실패합니다. 전체 CUDA toolkit 패키지에는 이 모두가 포함되어 있습니다.
- GPU에서 실행하려면 CUDA toolkit과 일치하는 NVIDIA 드라이버가 있어야 합니다.

컴파일된 커널은 디스크에 캐시됩니다(`MLX_PTX_CACHE_DIR`; `mlxcel`과 `mlxcel-server`는 이를 기본적으로 `~/.cache/mlxcel/cuda-ptx/<mlx-commit>`으로 설정하고, 다른 바이너리는 사용자별 `$TMPDIR/mlx-<euid>/<mlx-version>/ptx`로 fallback하며, Windows에서는 `%TEMP%\mlx\<mlx-version>\ptx`를 사용합니다). 따라서 각 커널 변형의 첫 실행만 NVRTC 비용을 부담합니다. 여기에는 mlxcel의 fused decode 경로(fused MoE, SSM) 뒤에 있는 커스텀 커널도 포함됩니다. 업스트림 MLX는 이 커널을 모든 프로세스마다 다시 컴파일했으므로, GB10에서 `mlxcel generate`마다 약 1초씩 더해지곤 했습니다(lablup/mlxcel#2492). 항목은 GPU의 compute capability와 NVRTC 버전을 딴 하위 디렉터리에 놓이므로(예: `sm121-nvrtc13.0`), 여러 GPU나 toolkit이 공유하는 캐시가 한쪽에 다른 쪽용으로 빌드된 코드를 넘겨주는 일이 없습니다.

캐시 디렉터리는 디렉터리 자체에서 `/`까지 경로의 모든 구성 요소가 본인 또는 root 소유이고 group이나 other가 쓸 수 없는 경우에만 사용됩니다(`/tmp`처럼 root 소유의 sticky 디렉터리는 괜찮습니다). 쓰기 권한이 있는 누구든 프로세스가 로드하는 컴파일된 코드를 심어 둘 수 있기 때문입니다(lablup/mlxcel#2539). 그렇지 않으면 mlxcel은 `mlxcel: CUDA kernel disk cache disabled: <directory> <reason>` 한 줄을 출력하고(reason은 `is not owned by the current user`, `is group- or other-writable`, `is a symbolic link`, `is not a directory`, `cannot be resolved` 중 하나입니다), 매 실행마다 모든 커널을 컴파일합니다. 이를 해결하려면 지정된 디렉터리에서 group과 other의 쓰기 권한을 제거하거나(`chmod go-w <directory>`), `MLX_PTX_CACHE_DIR`을 private 디렉터리로 지정하십시오. Ubuntu에서 사용자 private group의 기본값인 `umask` `002`는 group 쓰기 가능 디렉터리를 만들기 때문에, `mlxcel`과 `mlxcel-server`는 기본 캐시 디렉터리를 모드 `0700`으로 만들고, 이전 릴리스가 그렇게 만든 경우 자신이 관리하는 디렉터리에서 group 쓰기 권한을 제거합니다. 대상은 `cuda-ptx`와 `cuda-ptx/<mlx-commit>`이며, `MLXCEL_CACHE_DIR`이 설정되지 않은 경우에는 `~/.cache/mlxcel` 자체도 포함됩니다. 본인이 소유한 디렉터리만 변경됩니다. other 쓰기 권한은 `umask`가 남기는 종류가 아니므로 복구되지 않으며, 명시적으로 지정된 group 쓰기 가능 `MLX_PTX_CACHE_DIR`이나 `MLXCEL_CACHE_DIR`, 또는 group 쓰기 가능 `~/.cache`와 마찬가지로 통지와 함께 계속 거부됩니다.

`MLX_PTX_CACHE_DIR`과 `MLXCEL_CACHE_DIR`은 이 캐시에서 절대 경로여야 합니다. 상대 경로는 현재 디렉터리를 기준으로 해석되는데, 그 디렉터리는 누구든 여러분 대신 고를 수 있기 때문입니다. 이 경우 통지는 `MLX_PTX_CACHE_DIR must be an absolute path, got "<value>"`라고 표시되며 아무것도 생성되지 않습니다. Windows에서는 `\cache`와 `C:cache`도 상대 경로로 취급됩니다. `MLX_PTX_CACHE_DIR`이 설정되어 있으면 `MLXCEL_CACHE_DIR`은 참조되지 않으며, 두 변수 중 어느 쪽이든 값이 비어 있으면 설정되지 않은 것으로 취급됩니다.

Windows에서도 같은 규칙이 소유자와 ACL을 확인합니다(lablup/mlxcel#2797). `D:\mlxcel` 같은 폴더는 보통 드라이브로부터 Authenticated Users의 쓰기 권한을 상속받으므로, 드라이브 루트 아래의 모든 구성 요소는 본인, SYSTEM, Administrators 또는 TrustedInstaller(SYSTEM이나 NetworkService로 실행되는 서비스의 프로필이 있는 `C:\Windows`를 소유)를 소유자로 가져야 하며, 다른 어떤 계정에도 쓰기 권한을 부여하는 항목이 없어야 합니다. 그렇지 않으면 통지에 `is writable by <SID>`, `is not owned by the current user`, `has no access control list`(FAT 또는 exFAT 볼륨은 이를 보고하지 않습니다), `cannot be inspected`, `is a symbolic link`(mlxcel만 만드는 이름에 있는 symlink 또는 junction: `sm<cc>-nvrtc<version>`, 그리고 임시 디렉터리 fallback의 `mlx`, `<mlx-version>`, `ptx`), `is not a directory`, `cannot be created`, `cannot be resolved` 중 하나의 이유가 표시됩니다. 항목은 `icacls <directory> /remove <account>`로 제거하십시오. 상속받은 상위 디렉터리에서도 제거해야 합니다. 또는 `MLX_PTX_CACHE_DIR`을 프로필 아래의 디렉터리 같은 private 디렉터리로 지정하십시오. 드라이브 루트는 검사하지 않습니다. 기본적으로 드라이브 루트는 Authenticated Users가 모든 드라이브에 하위 디렉터리를 만들 수 있도록 허용하지만, 이것만으로는 다른 계정의 디렉터리를 삭제하거나 이름을 바꿀 수 없습니다. 상속으로 부여되는 권한은 상속을 유지하는 첫 번째 구성 요소에 나타나며, 이 구성 요소는 검사됩니다. Windows에서는 아무것도 복구되지 않습니다. ACL은 여러분이 바꿔야 하는 것이기 때문입니다. 캐시 항목(`.ptx` 또는 `.txt`)은 같은 소유자 및 쓰기 규칙을 통과하는 일반 파일이고 reparse point가 아닌 경우에만 사용되므로, 비공개로 바꾸기 전에 공유되던 디렉터리에 남은 항목은 로드되지 않습니다. 대신 커널이 다시 컴파일되고 항목이 교체됩니다.

lablup/mlxcel#2797 이후 Windows의 `mlxcel`과 `mlxcel-server`는 문서에 설명된 대로, Known Folder API가 보고하는 프로필 디렉터리(`USERPROFILE` 변수로는 이동하지 않습니다) 아래의 `.cache\mlxcel\cuda-ptx\<mlx-commit>` 또는 `MLXCEL_CACHE_DIR`을 사용합니다. 이전 릴리스는 해당 디렉터리를 만들었지만 항목은 `%TEMP%\mlx`에 두었습니다. overlay가 C 런타임의 환경 사본을 통해 변수를 읽었기 때문입니다. 따라서 업그레이드 후 첫 실행에서 한 번 컴파일되며, `%TEMP%\mlx`는 삭제해도 됩니다.

`MLX_PTX_CACHE_DIR`의 `..`은 어떤 디렉터리도 만들어지기 전에 해석됩니다. 캐시 항목(`.ptx` 또는 `.txt`)은 본인 또는 root가 소유하고 group이나 other의 쓰기 권한이 없는 일반 파일이며 symlink가 아닌 경우에만 사용됩니다. 그렇지 않으면 커널이 다시 컴파일되고 항목이 교체됩니다. 따라서 이전 릴리스가 `umask` `002`로 기록한 항목은 한 번 다시 컴파일됩니다.

캐시 루트 바로 아래에 있던 이전의 평평한 레이아웃 항목은 더 이상 읽지 않습니다. 게시용 임시 파일은 `O_EXCL`과 `O_NOFOLLOW`로 만들어지며, 1시간보다 오래된 것은 정리됩니다.

### C++ ISA 기준선 (`MLXCEL_CXX_MARCH`)

릴리스 빌드에서 C++ 브리지는 기본적으로 `-march=native`를 사용하며, 이는 빌드 호스트의
CPU에 맞춰 튜닝되고 그 CPU에서만 실행됩니다. 빌드한 곳에서 실행하는 빌드(개발자
머신, 머신별 GB10/GH200 릴리스 에셋)에는 올바른 설정입니다. 다른 머신에서 실행되어야
하는 바이너리에는 `MLXCEL_CXX_MARCH`를 이식 가능한 기준선으로 설정하십시오. 릴리스
워크플로의 x86-64 에셋은 `x86-64-v3`(AVX2)를 사용합니다.

```bash
# Portable x86-64 build (any AVX2-capable CPU, ~2013+).
MLXCEL_CXX_MARCH=x86-64-v3 cargo build --release --features cuda

# Omit -march entirely (compiler default baseline).
MLXCEL_CXX_MARCH=none cargo build --release --features cuda
```

MSVC에는 `-march=native`가 없으므로, Windows 빌드는 대상의 기준 ISA를 기본값으로 사용하며, 명시적인 값은 `/arch:<value>`로 전달됩니다(예: x64에서는 `MLXCEL_CXX_MARCH=AVX2`, arm64에서는 `armv8.2`).

## NVIDIA CUDA 기반 Windows (실험적)

`cuda` feature를 포함해 MSVC로 빌드한 네이티브 `mlxcel.exe` / `mlxcel-server.exe`입니다(lablup/mlxcel#58). 이것은 단일 머신을 위한 소스 빌드입니다. tensor 병렬화, pipeline 병렬화, disaggregated serving, 클러스터 bring-up은 Windows에서 시작 시점에 거부됩니다([현재 제한 사항](#windows의-현재-제한-사항) 참고). 위에 나온 Linux CUDA 경로를 사용하는 WSL2는 여전히 지원되는 대안입니다.

검증된 구성: NVIDIA RTX Spark N1X(Blackwell, `sm_121`, 드라이버 616.62)에서의 `aarch64-pc-windows-msvc`, Visual Studio 2026(MSVC 14.51), CUDA 13.4.3 및 cuDNN 9.27. 빌드 글루 코드는 `x64`와 `arm64` 툴킷 레이아웃을 같은 방식으로 처리합니다. CUDA 13.4.3 및 cuDNN 9.27을 사용한 `x86_64-pc-windows-msvc` 빌드는 컴파일과 링크가 되지만(lablup/mlxcel#2640), 아직 x64 NVIDIA 호스트에서 실행된 적은 없습니다.

### 사전 요구 사항

- "Desktop development with C++" 워크로드가 포함된 Visual Studio 2022 이상. VS에 번들된 CMake와 Ninja로 충분합니다. nvcc는 호스트 컴파일러로 MSVC `cl.exe`를 필요로 하므로 GNU 툴체인은 동작하지 않습니다.
- 머신에 맞는 MSVC 대상(`x86_64-pc-windows-msvc` 또는 `aarch64-pc-windows-msvc`)의 Rust이며, `-gnu` 대상이 아닙니다.
- CUDA 13용으로 빌드된 CUDA Toolkit 13.x 및 cuDNN 9. CUDA 12.x는 아직 지원하지 않습니다. 이 버전의 Windows 레이아웃은 DLL을 `bin\<arch>\`가 아니라 `bin\`에 두기 때문입니다. x64에서는 CUDA 설치 프로그램이 동작합니다. cuDNN은 반드시 아래의 재배포 아카이브 레이아웃을 사용해야 합니다. cuDNN 설치 프로그램의 `lib\13.x\x64` 레이아웃은 검색되지 않습니다. Arm용 Windows의 경우 NVIDIA는 재배포 아카이브만 제공합니다(`developer.download.nvidia.com/compute/cuda/redist/` 및 `.../cudnn/redist/` 아래의 `windows-arm64` 항목). `cuda_cudart`, `cuda_nvcc`, `cuda_crt`, `cuda_nvrtc`, `cccl`, `libcublas`, `libcufft`, `libcusolver`, `libcusparse`, `libnvjitlink`, `libnvvm` 아카이브와 cuDNN 아카이브의 압축을 하나의 디렉터리에 푸십시오. 관리자 권한은 필요하지 않습니다.
- 툴킷을 지원하는 NVIDIA 디스플레이 드라이버.

빌드가 기대하는 툴킷 레이아웃은 `bin\nvcc.exe`, `include\`, `lib\<arch>\`, `bin\<arch>\`이며, 여기서 `<arch>`는 `x64` 또는 `arm64`입니다. cuDNN은 같은 디렉터리에 있거나, 같은 레이아웃을 가진 자체 루트에 있을 수 있습니다.

### 빌드

대상 아키텍처용 Visual Studio 개발자 프롬프트("x64 Native Tools" 또는 "ARM64 Native Tools Command Prompt")에서 실행하여 `cl.exe`, CMake, Ninja가 `PATH`에 있도록 하십시오:

```bat
set CUDA_PATH=C:\path\to\cuda
rem Only when cuDNN is not unpacked into CUDA_PATH:
set CUDNN_PATH=C:\path\to\cudnn
cargo build --release --features cuda
```

`CUDA_HOME`이 `CUDA_PATH`보다 우선 적용됩니다. `CMAKE_GENERATOR`가 설정되지 않은 한 빌드는 MLX에 Ninja 생성기를 사용합니다. Visual Studio 생성기는 NVIDIA의 MSBuild 통합 기능이 Visual Studio에 설치되어 있지 않으면 `.cu` 파일을 컴파일할 수 없기 때문입니다. 아키텍처 목록은 Linux에서와 같이 `nvidia-smi`로 감지됩니다(`MLX_CUDA_ARCHITECTURES`가 이를 재정의합니다).

Windows 빌드가 Linux와 다르게 수행하는 작업은 다음과 같습니다:

- `lib\<arch>`에서 CUDA import 라이브러리를 링크하고, MLX 자체 hook을 통해 CUDA 및 cuDNN DLL을 delay-load합니다. 이 hook은 하나의 런타임 디렉터리에서 DLL을 해석합니다([아래](#cuda-런타임-디렉터리와-번들) 참고). 번들이 없으면 빌드 시점에 기록된 툴킷의 `bin\<arch>` 디렉터리입니다. mlxcel의 이 hook 오버레이(`overlays/cuda/mlx/backend/cuda/delayload.cpp`)는 또한 각 DLL을 로드하여 그 자체의 import가 같은 디렉터리에서 해석되도록 하며, NVRTC(`nvrtc-builtins`)와 cuDNN(`cudnn_engines_*`, `cudnn_ops`, ...)이 처음 사용할 때 단순 이름으로 여는 DLL을 미리 로드합니다. 따라서 바이너리는 `PATH`에 툴킷이 없고 `CUDA_PATH`가 없는 상태에서도, 첫 JIT 컴파일을 포함하여 실행됩니다. 디스플레이 드라이버의 `nvcuda.dll`은 delay-load되지 않으며 드라이버에서 가져옵니다.
- MLX의 CMake가 CPU 백엔드용으로 내려받는 OpenBLAS를 링크하고, `openblas.dll`을 빌드된 실행 파일 옆에 복사합니다.
- C++ 브리지를 GCC/Clang 표기 대신 MSVC 옵션(`/std:c++20`, `MLX_STATIC`, `/fp:fast`)으로 컴파일합니다.
- NVRTC를 위해 툴킷의 `include\` 디렉터리를 기록합니다. 따라서 런타임 커널 컴파일러는 실행 시점에 `CUDA_PATH`가 설정되어 있지 않아도 CUDA 헤더를 찾습니다.

런타임 번들이 없으면 실행 파일은 실행 시점에 빌드 머신의 툴킷 디렉터리에 의존합니다. 다른 머신에서 실행하려면 다음에 설명하는 번들을 만드십시오.

### CUDA 런타임 디렉터리와 번들

모든 CUDA, cuDNN, NVRTC DLL은 하나의 런타임 디렉터리에서 가져오며, 프로세스마다 한 번 다음 후보 중에서 선택됩니다(lablup/mlxcel#2640, [ROCm 번들](#런타임-디렉터리와-번들)과 같은 설계):

1. `MLXCEL_CUDA_RUNTIME_DIR`. 절대 경로여야 합니다. 이 변수가 설정되면 이것이 유일한 후보가 됩니다. 상대 경로이거나 CUDA 런타임 DLL이 없는 디렉터리이면, 바이너리는 첫 CUDA 호출에서 문제를 명시한 메시지와 함께 중단됩니다.
2. `mlxcel.exe`(또는 `mlxcel-server.exe`)의 디렉터리. 바이너리가 import하는 CUDA 런타임 DLL을 포함하고 있을 때입니다. nvcc는 CUDA 런타임을 정적으로 링크하므로, 확인에 쓰는 DLL은 cuBLASLt DLL(CUDA 13의 경우 `cublasLt64_13.dll`)이며, 이는 바이너리 자체의 delay-import 테이블에서 읽습니다.
3. 빌드 시점에 기록된 툴킷과 cuDNN의 `bin\<arch>` 디렉터리. 툴킷이 빌드 경로에 있는 개발자 빌드는 번들도 변수도 필요하지 않으며, 이전과 같이 로드됩니다.

바이너리는 선택 결과를 한 번 출력합니다(예: `[mlx-cuda] CUDA runtime: C:\mlxcel\bin (executable directory)`). 번들(첫 두 경우)에서 디렉터리에 없는 DLL이 있으면, 바이너리는 기대했던 전체 경로를 명시하며 중단됩니다. 다른 CUDA 설치, 현재 디렉터리, `PATH`에서 DLL을 가져오는 일은 결코 없습니다. 여기에는 import가 아니라 이름으로 라이브러리가 여는 DLL도 포함됩니다(`nvrtc64_*`, `nvrtc-builtins*`, `cublas64_*`, `cublasLt64_*`, `nvJitLink_*`, `cudnn_*` 하위 라이브러리). 바이너리는 CUDA DLL을 하나라도 로드하기 전에 이 모든 DLL을 확인하고 번들에서 미리 로드합니다. cuSOLVER만 import하는 `cusparse64_*`도 확인하므로, cuSPARSE가 없으면 cuSOLVER 로드 실패로 보고되지 않고 이름이 명시됩니다. cuBLASLt DLL 자체가 없는 번들은 번들로 인정되지 않습니다. 이때 빌드 시점의 툴킷도 없으면, 바이너리는 실행 파일 디렉터리에서 기대했던 cuBLASLt 경로를 명시합니다. 디스플레이 드라이버의 `nvcuda.dll`은 번들에 포함되지 않습니다.

CUDA 라이브러리는 번들 밖의 DLL도 참조하며, 이 DLL들은 라이브러리가 직접 엽니다. 드라이버의 `nvcuda.dll`과 `nvcudart_hybrid64.dll`, `zlibwapi.dll`(cuDNN), `cask_profile_interface.dll`(cuDNN 프로파일링 hook), `nvtileiras64.dll`(nvJitLink)입니다. 바이너리는 이 DLL들을 로드하거나 확인하지 않으므로, 이 DLL들에는 Windows의 표준 검색 순서가 적용됩니다. 실행 파일 디렉터리, System32, Windows 디렉터리, `PATH` 순서입니다. 번들에 `cudnn_ext64_9.dll`과 `cudnn_engines_tensor_ir64_9.dll`이 없는 경우에도 같은 방식이 적용됩니다. 확인 과정이 이 DLL들을 요구하지 않기 때문입니다(이전 cuDNN 9 릴리스는 이 DLL들을 포함하지 않으며, 패키징 스크립트는 포함합니다).

현재 디렉터리는 그 검색에 포함되지 않습니다(lablup/mlxcel#2641). 시작 시에는 번들이든 개발자 빌드이든 똑같이 바이너리가 `SetDllDirectoryW(L"")`를 호출하며, 이는 현재 디렉터리만 제거합니다. 따라서 작업 디렉터리에 같은 이름의 DLL이 있어도 로드되지 않습니다. 단, 작업 디렉터리가 실행 파일 디렉터리인 경우(`bin`에서 `.\mlxcel.exe`를 실행하는 경우)에는 예외입니다. 실행 파일 디렉터리는 검색 경로에 계속 남기 때문입니다. `nvcuda.dll`은 여전히 System32에서 해석됩니다. `SetDefaultDllDirectories`는 채택하지 않았습니다. 프로세스와 디스플레이 드라이버가 로드하는 나머지 모든 항목의 검색 방식도 함께 바뀌기 때문입니다.

이것으로 경로가 완전히 닫히는 것은 아닙니다. `PATH`는 검색에 남아 있으므로, `PATH`의 디렉터리에 이 이름 중 하나를 가진 DLL이 있으면 여전히 로드됩니다(`nvcuda.dll`을 포함하는 드라이버가 없는 머신에서도 마찬가지입니다). `.`처럼 상대 경로인 `PATH` 항목은 현재 디렉터리를 다시 검색에 포함시킵니다. `PATH`에는 다른 사용자가 쓸 수 없는 절대 경로 디렉터리만 두십시오. `SearchPathW`는 이 설정을 따르지 않으므로, 그 방식으로 파일을 찾는 라이브러리는 여전히 현재 디렉터리에서 파일을 찾습니다. 자식 프로세스는 이 설정을 상속하지 않으므로(Windows 11에서 측정), `mlxcel`이 시작하는 프로그램은 현재 디렉터리를 포함한 자체의 표준 검색을 유지합니다. 여기에는 비디오 입력용 `ffmpeg`과 `ffprobe`, 그리고 `xla-backend` 빌드에서의 IREE 컴파일러가 포함됩니다. 이들을 사용한다면, 그래도 다른 사용자가 쓸 수 없는 디렉터리에서 `mlxcel`을 시작하십시오. 작업 디렉터리에만 둔 DLL은 더 이상 로드되지 않습니다. 실행 파일 옆에 두십시오.

실행 파일이 직접 import하는 DLL(`openblas.dll` 또는 `libopenblas.dll`, `MSVCP140.dll`, `VCRUNTIME140.dll`)은 위의 모든 과정보다 먼저 로드되므로, 바이너리는 `/DEPENDENTLOADFLAG:0xA00`으로 링크됩니다. Windows는 이 DLL을 실행 파일 디렉터리와 System32에서만 찾으며, 현재 디렉터리나 `PATH`에서는 찾지 않습니다. 빌드와 번들이 그렇게 하듯 OpenBLAS는 실행 파일 옆에 두고, Visual C++ 런타임은 System32에 있도록 설치하거나 실행 파일 옆에 두십시오. DLL이 없으면 바이너리는 작업 디렉터리의 복사본을 가져오지 않고 찾을 수 없음(not-found) 오류와 함께 중단됩니다.

검증된 RTX Spark N1X에서 cuDNN에 도달하는 generate 실행은 `nvcuda.dll`을 System32에서, 드라이버의 `nvcudart_hybrida64.dll`(arm64 이름)을 DriverStore 폴더에서 로드합니다. `zlibwapi.dll`, `cask_profile_interface.dll`, `nvtileiras64.dll`은 로드되지 않으며, 선택적 cuDNN 하위 라이브러리 두 개는 번들에서 로드됩니다.

`scripts/package_windows_cuda_runtime.py`(Python 3, 표준 라이브러리)는 툴킷, cuDNN, CUDA 빌드 트리로부터 번들을 만듭니다:

```bat
python scripts\package_windows_cuda_runtime.py --cuda-path C:\path\to\cuda --out dist
```

이 명령은 `dist\mlxcel-cuda-runtime-<version>-win-<x64|arm64>.zip`을 생성합니다:

- `bin\`에는 `mlxcel`이 로드하는 DLL이 들어 있습니다: cuBLAS, cuBLASLt, cuFFT, cuSOLVER, cuSPARSE, nvJitLink, NVRTC와 그 builtins, cuDNN과 그 하위 라이브러리 아홉 개(CUDA 13.4 및 cuDNN 9.27 기준), 그리고 빌드의 OpenBLAS DLL. `bin\`이 런타임 디렉터리입니다.
- `include\`에는 NVRTC 커널 컴파일러가 필요로 하는 헤더가 들어 있습니다. 즉 JIT 커널이 접근하는 CUDA 헤더(`cuda_fp16.h`, `cooperative_groups.h`, ...), CCCL(`include\cccl`, 빌드가 내려받은 릴리스), CUTLASS(`cute\`, `cutlass\`)입니다. 바이너리는 이 디렉터리가 런타임 디렉터리 안에 있거나 그 옆에 있을 때 이를 사용합니다.
- `licenses\`에는 CUDA EULA, cuDNN 라이선스, CCCL 및 CUTLASS 라이선스가 들어 있습니다.

DLL 목록은 CUDA 13.4.3 및 cuDNN 9.27 `windows-x86_64` DLL의 import 테이블과 내장된 DLL 이름에서 도출했으며, arm64 RTX Spark N1X에서 generate 및 server 실행이 로드한 모듈 목록과 대조했습니다. 목록의 DLL 중 누락된 것은 없었습니다. 그 실행들에서는 cuFFT, cuSOLVER, cuSPARSE가 로드되지 않았으며, cuFFT를 호출하지 않는 실행은 cuFFT가 없어도 중단되지 않습니다. cuDNN은 실행이 어텐션 경로에 도달할 때만 로드됩니다(예: Llama 3.1 8B에서 긴 프롬프트를 prefill하는 경우). 스크립트는 CCCL, CUTLASS, OpenBLAS를 빌드의 `out\build\_deps` 디렉터리에서 읽으며, 이 디렉터리는 `CARGO_TARGET_DIR`(또는 `target`) 아래에서 찾습니다. CUDA 빌드가 둘 이상이면 `--deps-dir`로 이 디렉터리를 지정합니다. 스크립트는 입력에 없는 DLL, 헤더, 라이선스 파일을 모두 명시하며 실패하므로, 이후 CUDA 또는 cuDNN 릴리스에서 이름이 바뀐 DLL은 패키징할 때 검출됩니다. `mlxcel.exe`와 `mlxcel-server.exe`를 압축을 푼 `bin\`에 복사하거나, `MLXCEL_CUDA_RUNTIME_DIR`을 그 `bin\`으로 설정하십시오. `--stage-dir <empty dir>`는 트리를 zip으로 압축하는 대신 복사하며, `--add-binary <file>`(여러 번 지정 가능)은 같은 단계에서 바이너리를 복사합니다:

```bat
python scripts\package_windows_cuda_runtime.py --cuda-path C:\path\to\cuda --stage-dir dist\mlxcel --add-binary target\release\mlxcel.exe --add-binary target\release\mlxcel-server.exe
```

참고 사항:

- 라이선스 파일. 스크립트는 CUDA EULA를 `EULA.txt`(설치 프로그램 레이아웃) 또는 툴킷 루트의 `LICENSE`에서 읽고, cuDNN 라이선스는 cuDNN 루트의 `LICENSE`에서 읽습니다. 또한 각 파일의 내용을 확인합니다. 재배포 아카이브는 모두 `LICENSE`라는 이름의 파일을 포함하므로, 하나의 디렉터리에 풀면 마지막 아카이브의 사본이 나머지를 덮어쓰기 때문입니다. 이 경우 `cuda_cudart`와 `cudnn` 아카이브의 `LICENSE`를 추출하여 `--cuda-license`와 `--cudnn-license`로 전달하십시오. `windows-arm64` `cccl` 아카이브의 `LICENSE`는 CUDA EULA이며, 빌드가 내려받는 CCCL 릴리스에는 라이선스 파일이 없습니다. 따라서 일치하는 `NVIDIA/cccl` 릴리스 태그의 `LICENSE`를 `--cccl-license`로 전달하십시오.
- 헤더와 재배포. CUDA EULA(Attachment A)는 위의 CUDA DLL과 일부 헤더(`cuda_fp16.h`, `cuda_bf16.h`, `cuda_fp8.h`, `cuda_runtime_api.h`, `vector_types.h`, ...)를 재배포 가능하다고 명시하지만, JIT 커널이 포함하는 모든 헤더를 명시하지는 않습니다(`cooperative_groups.h`, `math_constants.h`, `cuda_runtime.h`, `mma.h`, `driver_types.h`, ...). 이 번들은 자신의 머신에서 사용하기 위한 것입니다. 공개하기 전에 NVIDIA와 약관을 확인하십시오.
- `CUDA_HOME`은 번들의 헤더보다 우선합니다. CUDA 설치 프로그램이 시스템 전체에 설정하는 `CUDA_PATH`는 그렇지 않습니다.
- 런타임 디렉터리, 번들의 `include\`, `mlxcel.exe`가 들어 있는 폴더는 관리자만(또는 본인만) 쓸 수 있어야 합니다. DLL은 mlxcel 프로세스 안에서 실행되고, 헤더는 GPU 커널로 컴파일되기 때문입니다. `C:\Program Files`에는 이 보호가 적용되어 있습니다. `C:\` 바로 아래에 만든 폴더라면, [ROCm SDK](#amd-rocm-기반-windows-실험적)에서 설명한 대로 상속된 접근 권한을 제거하십시오.
- 번들에는 Visual C++ 런타임이 포함되어 있지 않습니다. Visual Studio도 설치된 검증 호스트에서는 `MSVCP140.dll`과 `VCRUNTIME140.dll`이 System32에서 로드되었습니다. 따라서 이들이 없는 머신에는 Visual C++ Redistributable이 필요합니다. 디스플레이 드라이버만 있는 머신에서는 실행해 보지 않았습니다.
- `openblas.dll`은 일반 import이므로 `MLXCEL_CUDA_RUNTIME_DIR`이 아니라 실행 파일 폴더에서 가져옵니다. 번들 밖에서 실행되는 바이너리 옆에도 복사본을 두십시오.
- 번들 경로는 짧게 유지하십시오(예: `C:\mlxcel`). 260자를 넘는 DLL 경로는 Win32 긴 경로가 활성화되어 있어야 합니다.
- 다른 애플리케이션의 CUDA DLL이 들어 있는 폴더에는 번들을 풀지 마십시오. 애플리케이션마다 각자의 폴더를 주고, `MLXCEL_CUDA_RUNTIME_DIR`은 mlxcel의 폴더를 가리키게 하십시오.

긴 경로: `target\` 아래의 MLX 의존성 트리는 깊습니다. CMake 또는 nvcc가 열 수 없는 경로를 보고하면 Win32 긴 경로를 활성화하거나 짧은 `CARGO_TARGET_DIR`을 설정하십시오.

### Windows의 현재 제한 사항

| 영역 | 상태 |
|------|--------|
| 단일 GPU 생성과 `mlxcel-server` | 검증된 구성에서 동작합니다. |
| Tensor / pipeline 병렬 처리, disaggregated serving, `--pp-auto` / `--pp-peer` | 시작 시 해당 플래그를 명시한 오류와 함께 거부됩니다. |
| Unix 도메인 소켓 listen (`--host *.sock`) | 거부됩니다. TCP에서 listen하십시오. |
| Router 모드 모델 다운로드와 캐시 제거(WebUI 모델 라이브러리) | 거부됩니다. 이 기능은 디렉터리 fd에 고정된 파일 시스템 작업에 의존하며, 이 작업에는 Windows 구현이 없습니다. |
| CPU device (`MLXCEL_DEVICE=cpu`, 또는 `cuda` 없이 빌드한 경우) | 빌드되고 로드되지만, 양자화된 matmul이 MLX의 스칼라 CPU 루프로 실행됩니다. 검증된 호스트에서 gemma-4-e4b-it-qat-4bit의 24-token `generate`가 30분 뒤에도 끝나지 않았습니다. |
| CUDA 런타임 디렉터리와 번들 | RTX Spark N1X(`aarch64-pc-windows-msvc`, 드라이버 617.74, CUDA 13.4, cuDNN 9.27)에서 툴킷 이름을 바꾸고 `CUDA_PATH`를 설정하지 않은 상태로 검증되었습니다. cold JIT 캐시에서의 `mlxcel generate`와 `mlxcel-server` chat 요청이 번들 폴더에서, 그리고 `MLXCEL_CUDA_RUNTIME_DIR`을 통해 실행되었습니다. 로드된 모든 CUDA, cuDNN, NVRTC DLL은 번들에서 가져왔고, 필수 DLL을 제거하면 실행이 예상 경로와 함께 중단되었습니다. 실행하지 않은 항목: x64 GPU 호스트, 디스플레이 드라이버만 있는 머신, cuFFT 또는 cuSOLVER 호출. |
| 사전 빌드된 릴리스 아카이브 | 아직 없습니다. |
| CI | 컴파일 전용이며 GPU가 없습니다. `ci.yml`의 호스팅된 `Windows build (compile-only)` 작업은 Windows 빌드 스크립트, `build_support/`, 오버레이의 런타임 디렉터리 소스, 번들 패키저 또는 MLX pin이 변경되면 `windows-latest`에서 실행됩니다(또는 PR에 `ci:full` 라벨이 붙은 경우이며, 이 라벨은 저장소가 아직 정의하지 않았습니다). 실행되는 항목은 `mlxcel-mlx-pin` 단위 테스트, `tests/` Python 테스트 모음, 그리고 `cargo check -p mlxcel-core --features rocm`을 ROCm 10.1.0 `gfx1151` TheRock SDK에 대해 실행한 결과입니다. 이 SDK는 MLX와 그 HIP 커널을 빌드합니다. Windows에서 `cuda` feature를 빌드하는 CI 작업은 없습니다. 두 오버레이의 런타임 디렉터리 선택에 대한 호스트 테스트는 `overlay-host-tests`에서 Linux로 실행됩니다. |

## AMD ROCm 기반 Windows (실험적)

MSVC로 빌드한 `rocm` 기능 포함 네이티브 `mlxcel.exe` / `mlxcel-server.exe`입니다
(lablup/mlxcel#2601). [Linux 빌드](#amd-rocm-기반-linux-실험적)와 같은 ROCm
overlay를 사용하며, [Windows CUDA 빌드](#windows의-현재-제한-사항)와 같은 단일 노드
제한을 따릅니다.

검증된 구성: `x86_64-pc-windows-msvc` 대상, AMD Ryzen AI MAX+ 395(Radeon 8060S, `gfx1151`, BIOS에서 UMA carve-out 설정),
Windows 11, Adrenalin 32.0.31041.1004, ROCm Core SDK 10.1.0(TheRock tarball), 함께 번들된 Ninja가 포함된
Visual Studio 18(MSVC 14.51)입니다. 이 구성에서
`mlx-community/Qwen3-0.6B-4bit`은 약 115 tok/s로, `mlx-community/Qwen3-1.7B-bf16`은 약 31 tok/s로
decode됩니다. 빌드는 `x86_64-pc-windows-msvc`만 허용하며, 다른 Windows 대상은 기능 검사에서
실패합니다.

### 사전 요구 사항

- Windows용 ROCm 10.x SDK: GPU 계열에 맞는 TheRock tarball(`gfx1151`의 경우
  <https://stable.repo.amd.com/rocm/core/tarball/>에서 받는
  `therock-dist-windows-gfx1151-10.1.0.tar.gz`)을 `C:\rocm\10.1.0` 같은 디렉터리에 풀거나,
  AMD의 ROCm pip wheel에서 얻은 SDK를 사용합니다. ROCm 10에는 HIP SDK 설치 프로그램이 없습니다.
  빌드에는 `bin\hipcc.exe`, `lib\amdhip64.lib`와 `lib\`의 나머지 import 라이브러리, 그리고 대상
  감지를 위한 `amdgpu-arch.exe`가 필요합니다(ROCm 10.1 tarball은 이를 `lib\llvm\bin\`에 두고,
  다른 SDK는 `bin\`에 둡니다).
- 해당 GPU를 지원하는 AMD 디스플레이 드라이버.
- `cl.exe`, `lib.exe`, CMake, Ninja를 위한 "Desktop development with C++" 워크로드가 포함된
  Visual Studio 2022 이상.
- `x86_64-pc-windows-msvc` 대상이 포함된 Rust.

### 빌드

Visual Studio의 "x64 Native Tools Command Prompt"에서 실행하십시오:

```bat
set ROCM_PATH=C:\rocm\10.1.0
cargo build --release --features rocm
```

`ROCM_PATH`가 설정되지 않은 경우 `HIP_PATH`를 읽습니다. 값은 절대 경로여야 합니다. 상대 경로이면
빌드가 panic합니다. CMake가 상대 `bin` 또는 `include` 디렉터리를 기록하고, 이것이 실행 시점의
작업 디렉터리 기준으로 해석되기 때문입니다. SDK `bin` 디렉터리에 `amdhip64_*.dll`이 없어도
빌드가 panic합니다. 그렇지 않으면 바이너리가 HIP 런타임을 SDK에서 delay-load하는 대신 이름으로
즉시 import하게 되기 때문입니다. HIP 대상은 `amdgpu-arch`로 감지합니다(Windows SDK에는
`rocminfo`가 없습니다). 감지를 건너뛰거나 설치되지 않은 GPU용으로 빌드하려면
`MLX_ROCM_ARCHITECTURES=gfx1151`을 설정하십시오. `CMAKE_GENERATOR`가 설정되지 않은 경우 빌드는
MLX에 Ninja generator를 사용합니다. Visual Studio generator는 백엔드의 HIP 컴파일 명령을 실행할
수 없기 때문입니다. CMake나 `hipcc`가 열 수 없는 경로를 보고하면 `C:\t` 같은 짧은
`CARGO_TARGET_DIR`을 설정하십시오.

Windows 빌드가 Linux와 다르게 하는 일:

- 전체 스택이 MSVC ABI입니다. ROCm의 clang은 `x86_64-pc-windows-msvc`를 대상으로 하고, ROCm DLL은
  MSVC 런타임에 링크됩니다. HIP 커널은 `/MD` 런타임 의미로 `.obj` 파일로 컴파일되고,
  `lib.exe`가 `mlx_rocm_kernels.lib`로 아카이브합니다.
- SDK의 `lib\`에서 import 라이브러리를 링크합니다. hipBLASLt의 import 라이브러리는
  `libhipblaslt.dll.a`로 배포되며, 빌드는 이를 `OUT_DIR` 아래의 `hipblaslt.lib`로 복사합니다.
  rpath는 포함되지 않습니다.
- ROCm DLL을 하나의 런타임 디렉터리에서 delay-load합니다. 기본값은 빌드 시점에 기록된 SDK `bin`
  디렉터리입니다(lablup/mlxcel#2602이며, 다른 후보는
  [런타임 디렉터리와 번들](#런타임-디렉터리와-번들)에 있습니다). 실행 파일은 `/DELAYLOAD`를
  통해 `amdhip64_7.dll`, `rocblas.dll`, `libhipblaslt.dll`, `hipfft.dll`, `hiprtc0716.dll`(ROCm
  10.1.0의 이름)을 import하며, MLX에 컴파일된 notify 훅이 각 DLL을 런타임 디렉터리의 전체 경로로
  로드하고 그 의존 DLL은 옆에서 해석되도록 합니다. HIP 런타임과 hipRTC가 실행 시점에 이름으로
  로드하는 `amd_comgr.dll`과 `hiprtc-builtins*.dll`은 같은 디렉터리에서 미리 로드됩니다. 이들 중
  어느 것도 현재 디렉터리나 `PATH`에서 검색되지 않으므로, 작업 디렉터리에 같은 이름의 DLL이
  있어도 로드되지 않으며, HIP 런타임은 디스플레이 드라이버가 `System32`에 설치한
  `amdhip64_7.dll`이 아니라 SDK의 것입니다. 그 디렉터리에 DLL이 없으면 바이너리는 예상 경로를
  출력하고 중단합니다. HIP 런타임은 또한 `hsa-amd-aqlprofile64.dll`, `hsa-ext-finalize64.dll`,
  `dtif64a.dll`을 이름으로 참조하는데, 이 파일들은 SDK도 드라이버도 제공하지 않습니다. 생성과
  서빙은 이들을 로드하지 않습니다(작업 디렉터리에 decoy DLL을 두어 확인함). 하지만 프로파일링처럼
  이름으로 여는 선택적 경로는 표준 Windows 검색을 사용합니다. 현재 디렉터리는 그 검색에
  포함되지 않습니다(lablup/mlxcel#2641). 첫 ROCm DLL이 로드되기 전에 hook이
  `SetDllDirectoryW(L"")`를 호출하는데, 이는 현재 디렉터리만 제거하므로 작업 디렉터리의 decoy는
  로드되지 않습니다. 단, 작업 디렉터리가 실행 파일의 디렉터리이면(`bin`에서 `.\mlxcel.exe`를
  실행하는 경우) 예외입니다. 실행 파일의 디렉터리는 검색에 남아 있기 때문입니다.
  `SetDefaultDllDirectories`는 프로세스와 디스플레이 드라이버가 로드하는 다른 모든 대상의 검색도
  바꾸기 때문에 채택하지 않았습니다. 이것으로 경로가 완전히 닫히지는 않습니다. System32, Windows
  디렉터리, `PATH`는 검색에 남아 있으므로, 이 이름 중 하나의 DLL이 `PATH`에 있는 디렉터리에
  있으면 여전히 로드되며, `.` 같은 상대 `PATH` 항목은 현재 디렉터리를 다시 검색에 넣습니다.
  `PATH`에는 다른 사용자가 쓸 수 없는 절대 경로 디렉터리만 두십시오. `SearchPathW`는 이 설정을
  따르지 않습니다. 자식 프로세스는 이 설정을 상속하지 않으므로(Windows 11에서 측정함), `mlxcel`이
  시작하는 프로그램은 현재 디렉터리를 포함한 자체의 표준 검색을 유지합니다. 여기에는 동영상 입력용
  `ffmpeg`와 `ffprobe`, 그리고 `xla-backend` 빌드에서는 IREE 컴파일러가 포함됩니다. 이들을
  사용한다면 그래도 다른 사용자가 쓸 수 없는 디렉터리에서 `mlxcel`을 시작하십시오. 작업
  디렉터리에만 두었던 DLL은 더 이상 로드되지 않습니다. 실행 파일 옆에 두십시오.
- 실행 파일이 직접 import하는 DLL(`libopenblas.dll`, `MSVCP140.dll`, `VCRUNTIME140.dll`)은 위의
  모든 과정보다 먼저 로드되므로, 바이너리는 `/DEPENDENTLOADFLAG:0xA00`으로 링크됩니다. 그러면
  Windows는 이들을 실행 파일의 디렉터리와 System32에서만 찾고, 현재 디렉터리나 `PATH`에서는
  찾지 않습니다. 하나라도 없으면 바이너리는 작업 디렉터리의 복사본을 가져오는 대신 찾을 수 없음
  오류와 함께 중단합니다.
- JIT 컴파일된 커널의 디스크 캐시는 사용자별로
  `%USERPROFILE%\.cache\mlxcel\hsaco\<version>\<arch>` 아래에 있습니다(또는 `MLXCEL_CACHE_DIR`
  아래, 또는 `MLX_HSACO_CACHE_DIR\<arch>`). 프로필 디렉터리는 Known Folder API가 보고하는
  디렉터리이며, mlxcel의 다른 캐시들이 쓰는 것과 같으므로 `USERPROFILE` 변수를 바꿔도 어느 쪽도
  옮겨지지 않습니다. `D:\mlxcel` 같은 폴더는 보통 드라이브로부터 Authenticated Users의 쓰기 권한을
  상속받으므로, 드라이브 루트 아래의 모든 구성 요소는 소유자가 본인, SYSTEM, Administrators 또는
  TrustedInstaller여야 하며, 다른 어떤 계정에도 쓰기 권한을 주는 항목이 없어야 합니다. 그렇지
  않으면 캐시는 [커널 디스크 캐시](#커널-디스크-캐시)에서 설명하는 한 줄과 함께 꺼지며, 사유는
  `is writable by <SID>`, `is not owned by the current user`, `has no access control list`,
  `cannot be inspected` 중 하나입니다. 항목은 `icacls`로 제거하십시오. 드라이브 루트는 검사하지
  않습니다. 모든 드라이브의 기본 ACL은 Authenticated Users에게 그곳에 폴더를 만들 권한을 주며, 그
  상속되는 권한은 상속을 유지하는 첫 번째 구성 요소에 나타나기 때문입니다. `hsaco`,
  `<version>`, `<arch>` 위치의 심볼릭 링크나 junction은 거부됩니다. 캐시 항목은 reparse point가
  아닌 일반 파일이어야 하며, 같은 소유자 및 쓰기 규칙을 통과해야 합니다. 그렇지 않으면 커널을 다시
  컴파일하고 항목을 교체하므로, 비공개로 바꾸기 전에 공유되던 디렉터리에 남은 항목은 로드되지
  않습니다.
- 런타임 커널 컴파일러(hiprtc)는 다음 중 적용되는 첫 번째 항목에서 헤더를 가져옵니다
  (lablup/mlxcel#2639):
  1. `ROCM_HOME`, 그다음 `ROCM_PATH`(`<root>\include`). 명시적 오버라이드입니다.
  2. 런타임 디렉터리 안이나 옆에 있는 `include\`. 이 디렉터리가 런타임 번들이고
     (`MLXCEL_ROCM_RUNTIME_DIR`에서 왔거나 실행 파일의 디렉터리인 경우) `hip\hip_runtime.h`를
     가지고 있을 때 사용합니다.
  3. `HIP_PATH`(`<root>\include`).
  4. 런타임 디렉터리가 빌드 시점에 기록된 SDK `bin`인 경우(개발자 빌드), 런타임 디렉터리 옆의
     `include\`.
  5. 빌드 시점에 기록된 SDK의 `include\`.

  런타임에 별도의 변수는 필요하지 않습니다. 번들의 헤더는 `HIP_PATH`보다 우선합니다. AMD HIP SDK
  설치 프로그램이 `HIP_PATH`를 시스템 전체에 설정하며, 다른 ROCm 버전의 헤더로 컴파일된 번들은
  두 릴리스를 섞어 버리기 때문입니다. 환경 변수와 번들이 서로 다른 디렉터리를 제시하면, 바이너리는
  두 가지를 모두 밝히는 한 줄을 출력합니다. 예를 들어
  `[mlx-rocm] hiprtc headers: C:\mlxcel\include (executable directory); ignoring C:\Program Files\AMD\ROCm\6.4\include from HIP_PATH`처럼
  출력하며, `ROCM_PATH`가 번들을 오버라이드하는 경우에는
  `[mlx-rocm] hiprtc headers: C:\rocm\10.1.0/include (ROCM_PATH); ignoring C:\mlxcel\include from executable directory`처럼
  출력합니다. 런타임에 설정한 변수가 우선할 수 있으므로, 그 변수는 SDK와 같은 보호 수준을 가진
  디렉터리를 가리켜야 합니다(다음 항목 참고).
- 빌드 시점에 기록된 SDK 디렉터리는 관리자만(또는 본인만) 쓸 수 있어야 합니다. `bin\`에 쓸 수 있는
  계정은 이 바이너리에서 시작된 모든 mlxcel 프로세스에서 코드를 실행할 수 있고, `include\`에 쓸 수
  있는 계정은 이들이 컴파일하는 GPU 커널을 바꿀 수 있습니다. `C:\rocm`처럼 `C:\` 바로 아래에 만든
  디렉터리는 드라이브 루트로부터 모든 인증된 사용자에 대한 Modify 권한을 상속받으므로, 계정이 둘
  이상인 머신에서는 SDK를 풀고 나서 이 상속된 권한을 제거하십시오:

  ```bat
  icacls C:\rocm /inheritance:r /grant:r *S-1-5-32-544:(OI)(CI)F *S-1-5-18:(OI)(CI)F *S-1-5-32-545:(OI)(CI)RX
  ```

  이렇게 하면 Administrators와 SYSTEM은 전체 제어 권한을, Users는 읽기 및 실행 권한을 갖게
  되며, 이 설정은 `C:\rocm` 아래의 모든 항목으로 전파됩니다. 이후 `icacls C:\rocm\10.1.0\bin`에는
  `Authenticated Users` 항목이 나타나지 않습니다. 같은 원칙이 `mlxcel.exe`가 있는 디렉터리에도
  적용됩니다. Windows는 그곳에서 `libopenblas.dll`을 로드하며, SDK `bin\`에 없는 ROCm DLL의 의존
  대상을 그곳에서 검색합니다.
- host shadow 메모리 모델을 사용합니다. Windows는 APU에서도 GPU 할당을 CPU 주소 공간에 매핑하지
  않으므로, GPU 버퍼에 대한 모든 CPU 읽기나 쓰기는 고정된 host 복사본, 즉 그 버퍼의 shadow를
  거칩니다. CPU 쓰기는 shadow를 dirty로 표시하며, 다음 커널이 버퍼를 읽기 전에 다시 flush됩니다.
  mlxcel 자체의 읽기(샘플링된 token, 바이트 단위 readback)는 shadow를 clean 상태로 두는 읽기 전용
  경로를 사용하므로 flush되지 않습니다(lablup/mlxcel#2643). device 버퍼는 일반 `hipMalloc`
  메모리이며, 검증된 호스트에서는 UMA carve-out 전체(92 GiB 할당)에 도달하는 반면, pinned 호스트
  메모리는 Windows가 인식하는 RAM(약 28 GiB)에서 멈춥니다. 통합 GPU에서 할당기 한도는 HIP이
  보고하는 device 메모리의 80%입니다. 1 GiB 조각으로 할당하면 보고된 107.87 GiB 중 105 GiB에서
  멈추었으므로, Linux와 달리 시스템 RAM은 이 한도를 올리지 않습니다. shadow가 보유하는 바이트는
  `Model loaded` 줄(`host shadows`)에 출력되고, `host_shadow_bytes` 로그 필드에 담기며,
  `mlxcel inspect`와 `--estimate-memory` preflight가 device 한도가 아니라 호스트 RAM과 비교되는
  `Host shadows` 항목으로 예약합니다. `MLX_ROCM_HOST_SHADOW_STATS=1`을 설정하면 종료 시 카운터를
  출력합니다.

### 런타임 디렉터리와 번들

모든 ROCm DLL은 다음 후보 가운데에서 프로세스당 한 번 선택되는 하나의 런타임 디렉터리에서 가져옵니다(lablup/mlxcel#2606):

1. `MLXCEL_ROCM_RUNTIME_DIR`입니다. 반드시 절대 경로여야 합니다. 이 값이 설정되면 이것이 유일한 후보가 되며, 상대 경로이거나 HIP 런타임이 없는 디렉터리이면 바이너리가 첫 ROCm 호출에서 문제를 명시한 메시지와 함께 멈춥니다.
2. `mlxcel.exe` (또는 `mlxcel-server.exe`)가 들어 있는 디렉터리로, 바이너리가 import하는 HIP 런타임(ROCm 10에서는 `amdhip64_7.dll`)을 담고 있을 때 후보가 됩니다.
3. 빌드 시점에 기록된 SDK의 `bin`으로, HIP 런타임을 담고 있을 때 후보가 됩니다. SDK가 빌드 경로에 있는 개발자 빌드에는 번들도 변수도 필요하지 않습니다.

바이너리는 선택 결과를 한 번 출력합니다(예: `[mlx-rocm] ROCm runtime: C:\mlxcel\bin (executable directory)`). 선택된 디렉터리에 없는 DLL이 있으면 바이너리는 기대한 전체 경로를 밝히며 멈추고, 다른 후보에서 DLL을 가져오는 일은 결코 없습니다. 두 ROCm 릴리스의 DLL이 함께 동작한다고 보장되지 않기 때문입니다. 예를 들어 SDK의 `amdhip64_7.dll`을 디스플레이 드라이버의 `amd_comgr.dll` 옆에 두면 device를 찾지 못합니다. 여기에는 HIP 런타임이 이름으로 여는 `amd_comgr*.dll`과 `hiprtc-builtins*.dll`도 포함됩니다. 선택된 디렉터리에 두 패턴 중 어느 것과도 일치하는 파일이 없으면, 바이너리는 어떤 ROCm DLL도 로드하기 전에, 빠진 패턴과 디렉터리를 명시한 메시지와 함께 멈춥니다. Windows가 `System32`에 있는 드라이버의 사본을 찾아 쓰도록 두지 않기 위해서입니다.

`scripts/package_windows_rocm_runtime.py`(Python 3, 표준 라이브러리)는 SDK로부터 런타임 번들을 빌드합니다:

```bat
python scripts\package_windows_rocm_runtime.py --rocm-path C:\rocm\10.1.0 --arch gfx1151 --out dist
```

이렇게 하면 `dist\mlxcel-rocm-runtime-10.1.0-win-x64.zip`이 만들어집니다(`gfx1151`, ROCm 10.1.0 기준 76 MiB, 압축 해제 후 225 MiB). 빌드가 지원하는 GPU 대상마다 `--arch`를 한 번씩 전달하십시오. rocBLAS, hipBLASLt, rocFFT는 데이터를 각자의 DLL을 기준으로 찾으므로, 번들은 SDK의 구조를 그대로 유지합니다:

- `bin\`에는 `mlxcel`이 로드하는 열 개의 DLL(`amdhip64_7`, `amd_comgr`, `rocm_kpack`, `hiprtc0716`, `hiprtc-builtins0716`, `rocblas`, `libhipblaslt`, `origami`, `hipfft`, `rocfft`)과, 요청한 대상만 남도록 줄인 커널 데이터가 `bin\rocblas\library\`와 `bin\hipblaslt\library\`에 들어 있습니다.
- `.kpack\`에는 라이브러리가 `<their directory>\..\.kpack\<library>_<target>.kpack` 경로로 여는 device 코드 아카이브가 들어 있습니다.
- `include\hip\`에는 커널 컴파일러가 필요로 하는 헤더가 들어 있습니다.
- `licenses\`에는 배포되는 구성 요소의 SDK 라이선스 파일이 들어 있습니다. ROCm 10.1.0 tarball에는 HIP 런타임, hipRTC, `rocm_kpack`의 라이선스 파일이 없습니다.

SDK에 이 파일 중 하나라도 없으면 스크립트가 실패하므로, 이후 ROCm 릴리스에서 이름이 바뀐 DLL도 패키징 시점에 잡힙니다. `bin\`이 런타임 디렉터리입니다. 다음 중 하나를 하십시오. 빌드의 `target\release\`에서 `mlxcel.exe`, `mlxcel-server.exe`, `libopenblas.dll`을 풀어 놓은 `bin\`에 복사하거나, `MLXCEL_ROCM_RUNTIME_DIR`을 풀어 놓은 `bin\`으로 설정하고 바이너리는 있던 자리에 그대로 두십시오. `--stage-dir <empty dir>`은 zip으로 묶는 대신 트리를 복사하며, `--add-binary <file>`(반복 가능)은 같은 단계에서 바이너리를 `bin\`으로 복사합니다:

```bat
python scripts\package_windows_rocm_runtime.py --rocm-path C:\rocm\10.1.0 --arch gfx1151 --stage-dir dist\mlxcel --add-binary target\release\mlxcel.exe --add-binary target\release\mlxcel-server.exe --add-binary target\release\libopenblas.dll
```

설치 규칙:

- 다른 애플리케이션의 ROCm 런타임이 들어 있는 폴더에 번들을 풀지 마십시오. 예를 들어 llama.cpp의 Windows ROCm 릴리스는 실행 파일 옆에 ROCm 10.0의 `amdhip64_7.dll`, `amd_comgr.dll`, `rocm_kpack.dll`을 함께 넣어 배포합니다. 두 가지를 한 폴더에 풀면 그 파일들이 대체되어 두 릴리스의 DLL이 짝을 이루게 됩니다. 애플리케이션마다 고유한 폴더를 주고, 둘이 나란히 있어야 할 때에는 `MLXCEL_ROCM_RUNTIME_DIR`이 mlxcel의 폴더를 가리키도록 설정하십시오.
- 런타임 디렉터리, 그 상위 디렉터리(`.kpack\`과 `include\`를 담고 있음), `mlxcel.exe`가 들어 있는 폴더는 위에서 SDK에 대해 든 이유와 같은 이유로 쓰기 권한이 관리자에게만(또는 본인에게만) 있어야 합니다. `C:\Program Files`에는 이 보호가 있으며, `C:\` 바로 아래에 만든 폴더라면 위의 `icacls` 명령으로 상속된 접근 권한을 제거하십시오.
- 번들의 `include\`는 JIT 헤더에 대해 `HIP_PATH`보다 우선하므로, 같은 머신에 설치된 HIP SDK도 변경할 필요가 없습니다. `ROCM_HOME`과 `ROCM_PATH`는 여전히 번들보다 우선합니다. 같은 ROCm 버전을 가리키지 않는 한 mlxcel을 위해서는 이 둘을 설정하지 않은 상태로 두고, `[mlx-rocm] hiprtc headers:` 줄을 확인하십시오. 두 디렉터리가 다르면 이 줄에 두 디렉터리가 모두 표시됩니다.
- 런타임 디렉터리의 경로는 최대 130자로 유지하십시오(예: `C:\mlxcel\bin`). rocBLAS와 hipBLASLt는 그 아래 128자 길이까지의 커널 파일을 엽니다. Windows의 260자 경로 제한 아래에서는 더 긴 위치 때문에 첫 행렬 곱셈이 access violation으로 충돌했습니다(133자 디렉터리, 긴 경로 비활성화 상태). 선택된 디렉터리가 이보다 길면 바이너리가 경고를 출력합니다.
- Adrenalin 32.0.31041.1004와 Radeon 8060S(`gfx1151`)에서 테스트했습니다. 번들이 자체 HIP 런타임을 포함하므로 `System32`에 있는 드라이버의 ROCm DLL은 사용되지 않습니다. 다른 드라이버 버전은 테스트하지 않았습니다.

### 배포용 빌드의 GPU 대상

HIP 커널은 빌드의 대상에 맞춰 미리 컴파일되며, 바이너리는 그 대상 밖의 GPU에서는 시작을 거부합니다. 다른 머신에서 실행할 빌드라면 이 목록을 설정하십시오(lablup/mlxcel#2639):

```bat
set MLX_ROCM_ARCHITECTURES=gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201
cargo build --release --features rocm -j 8
```

RAM이 약 32 GB인 호스트에서는 병렬도를 낮춰 링크하십시오. 일곱 대상 바이너리는 크기가 커서, 메모리가 부족할 때 `-j 8`로 `mlxcel`과 `mlxcel-server`를 나란히 링크하면 `link.exe` `0xC0000142`로 실패했습니다. HIP 오브젝트 컴파일이 끝난 뒤 `-j 2`로 같은 명령을 다시 실행하면 링크가 완료됩니다.

이는 RDNA3(`gfx1100`, `gfx1101`, `gfx1102`), RDNA3.5(`gfx1150`, `gfx1151`), RDNA4(`gfx1200`, `gfx1201`)에 해당합니다. 바이너리는 bound-device 줄 뒤에 목록을 한 번 `[mlx-rocm] compiled HIP targets: gfx1100,gfx1101,gfx1102,gfx1150,gfx1151,gfx1200,gfx1201` 형식으로 출력하며, `mlxcel generate`는 `HIP architecture <gfx>; compiled for [...]`도 출력합니다. `gfx1151`에서만 실행되었습니다(일곱 대상 바이너리는 이 GPU에서 generate를 수행합니다). 나머지 여섯 대상은 ROCm 10.1.0 `gfx1151` tarball의 `hipcc`로 컴파일했지만 하드웨어에서 실행하지는 않았습니다. device 코드는 목록이 늘어날수록 커집니다. ROCm 10.1.0을 사용한 32 GB 검증 호스트에서 측정한 결과는 다음과 같습니다:

| | 일곱 대상 | `gfx1151` 단독 |
|---|---|---|
| `mlxcel.exe` | 745,592,320 바이트 (711.1 MiB) | 193,465,856 바이트 (184.5 MiB) |
| `mlxcel-server.exe` | 735,546,880 바이트 (701.5 MiB) | 183,420,416 바이트 (174.9 MiB) |
| HIP 커널 아카이브 `mlx_rocm_kernels.lib` | 716.6 MiB | 190.1 MiB |
| HIP 커널 단계, CMake configure부터 아카이브까지 (`-j 8`, `CMAKE_BUILD_PARALLEL_LEVEL=8`) | 5 min 8 s (cold tree) | 3 min 6 s |

모든 HIP 오브젝트가 컴파일된 뒤 `-j 2`로 실행한 두 일곱 대상 바이너리의 최종 링크는 7 min 1 s가 걸렸습니다.

계열별 TheRock tarball은 모든 대상에 대해 컴파일되지만(컴파일러와 device 라이브러리가 모든 대상을 포괄합니다), rocBLAS와 hipBLASLt 커널 데이터 및 `.kpack` 아카이브는 자기 계열의 대상만 담고 있습니다. ROCm 10.1.0 `gfx1151` tarball에는 `gfx1151`용만 있습니다. 전체 목록에 대한 번들은 `therock-dist-windows-multiarch-10.1.0.tar.gz`(4.9 GB, 같은 인덱스)로 빌드하십시오. 이 파일의 `bin\rocblas\library\`, `bin\hipblaslt\library\`, `.kpack\`이 일곱 대상을 모두 포괄합니다. 계열별 tarball(`gfx110X-all`, `gfx1150`, `gfx1151`, `gfx120X-all`)은 각각 자기 계열만 포괄합니다. 대상마다 `--arch`를 하나씩 전달하십시오:

```bat
python scripts\package_windows_rocm_runtime.py --rocm-path C:\rocm\10.1.0-multiarch --arch gfx1100 --arch gfx1101 --arch gfx1102 --arch gfx1150 --arch gfx1151 --arch gfx1200 --arch gfx1201 --out dist
```

SDK에 어떤 `--arch`의 커널 데이터가 없으면, 스크립트는 그 GPU에서 첫 행렬 곱셈 때 실패할 번들을 쓰는 대신 상태 1로 종료하면서, 그런 대상 각각과 빠진 파일을 명시합니다.

### 현재 제한 사항

- 런타임 번들이 없으면 바이너리는 빌드할 때 참조한 경로의 ROCm SDK가 필요합니다. 커널은 빌드의 대상에 대해서만 미리 컴파일되므로, 번들은 그 GPU에서만 실행됩니다. [배포용 빌드의 GPU 대상](#배포용-빌드의-gpu-대상)을 참고하십시오. 하드웨어에서 테스트된 것은 `gfx1151`뿐입니다.
- [Windows의 현재 제한 사항](#windows의-현재-제한-사항)의 모든 내용이 적용됩니다: 단일 노드, Unix 소켓 없음, 라우터 모드 다운로드 없음.
- 사전 빌드된 아카이브는 없으며, 번들은 로컬에서 빌드합니다. CI는 GPU 없이 Windows에서 `rocm` feature를 컴파일하고 AMD 하드웨어에서는 아무것도 실행하지 않습니다. [Windows의 현재 제한 사항](#windows의-현재-제한-사항)의 CI 행을 참고하십시오.

## AMD ROCm 기반 Linux (실험적)

**실험적.** Linux의 AMD GPU 지원은 lablup/mlxcel#1802에서 추가되었으며 lablup/mlxcel#1801에서 추적합니다. 소스 빌드만 지원하며, 아직 릴리스 아티팩트도 실행 중인 ROCm CI 작업도 없습니다. 아래에 나열된 공백은 아직 열려 있습니다.

이 백엔드는 upstream MLX의 일부가 아닙니다. mlxcel은 [NripeshN/mlx](https://github.com/NripeshN/mlx/tree/rocm-support)(MIT)의 `rocm-support` 브랜치에서 ROCm 백엔드를 `src/lib/mlxcelverse/overlays/rocm/`로 vendor하고, Metal 및 CUDA 빌드가 사용하는 것과 같은 MLX pin 커밋 위에 이를 적용합니다. overlay의 구성은 [mlxcelverse](architecture.md#mlxcelverse-mlx-측-계층)를, mlxcel이 fork 위에 얹어 유지하는 변경 사항은 `src/lib/mlxcelverse/overlays/rocm/LOCAL_FIXES.md`를 참고하십시오.

테스트된 구성: Radeon 8060S(`gfx1151`, RDNA 3.5)를 탑재한 AMD Ryzen AI MAX+ 395와 96 GiB VRAM carve-out, Debian 13, ROCm 10.0.0 패키지(HIP 7.15, AMD clang 23). 다른 RDNA 3, 3.5, 4 제품도 빌드될 것으로 예상됩니다. CDNA 제품(예: MI300)은 컴파일되지만 튜닝이 없으며, 이들에서는 32-lane wavefront에 걸쳐 reduce하는 fused 커널이 검증될 때까지 MLX 그래프로 fallback됩니다(아래 표의 "Wave64 devices" 행 참고).

### 사전 요구 사항

- Rust 2024 edition과 호환되는 Rust 툴체인, CMake, C++20 컴파일러.
- Linux에서 Rust 의존성이 필요로 하는 `pkg-config`와 OpenSSL 헤더.
- `lapacke.h`를 포함한 BLAS 및 LAPACK 개발 패키지([CUDA 기반 Linux](#cuda-기반-linux) 참고).
- `hip`, `rocblas`, `rocthrust`, `rocprim`, `hiprand`, `rocwmma`, `hipblaslt`, `hipfft`, `hiprtc` CMake 패키지를 제공하는 ROCm 설치(`ls "${ROCM_PATH:-/opt/rocm}"/lib/cmake`로 확인할 수 있음), 그리고 `hipcc` 및 `rocminfo`.
- GPU device 노드에 대한 접근 권한: 빌드 및 실행 사용자가 `video`와 `render` 그룹에 속해 있어야 합니다.

Debian/Ubuntu에서:

```bash
sudo apt-get install -y \
    build-essential cmake git pkg-config libssl-dev \
    libopenblas-dev liblapack-dev liblapacke-dev
# ROCm itself comes from AMD's repository (repo.radeon.com).
rocminfo | grep -E '^\s+Name:\s+gfx'   # should list your GPU target
```

### 빌드

```bash
cargo build --release --features rocm
# or
make release-rocm
```

`rocm`은 `cuda` 또는 `metal`과 함께 사용할 수 없으며, 충돌을 명시한 메시지와 함께 빌드가 실패합니다. 첫 빌드는 `hipcc`로 MLX device 코드를 컴파일하므로, Rust 빌드 시간에 더해 몇 분이 걸립니다.

증분 빌드는 헤더를 추적합니다. `hipcc`가 각 HIP 오브젝트 옆에 의존성 파일을 작성하므로, `src/lib/mlxcelverse/overlays/rocm/` 아래의 헤더를 변경하면 다음 빌드는 그 헤더를 직접 또는 간접적으로 include하는 `.hip` 파일만 정확히 다시 컴파일하며, 변경이 없는 빌드는 아무것도 다시 컴파일하지 않습니다. overlay는 CMake의 `configure_file`을 통해 빌드 트리에 도달하는데, 이 함수는 내용이 다를 때만 복사본을 다시 쓰므로 overlay 헤더를 단순히 `touch`하는 것으로는 아무것도 다시 빌드되지 않으며, 내용이 실제로 바뀌어야 합니다. 의존성 파일에는 ROCm 및 시스템 헤더도 나열되므로, 이들을 업데이트하면 모든 HIP 오브젝트가 별도 조치 없이 다시 빌드됩니다. device 컴파일에서만 include되는 헤더(현재는 rocWMMA의 것뿐)와 컴파일러 자체는 나열되지 않습니다.

그래도 HIP를 깨끗하게 다시 빌드하도록 강제하려면(예: `hipcc`나 rocWMMA를 업그레이드한 뒤, 또는 bisect하는 중에 오래된 오브젝트를 배제하려고 할 때), 사용하는 빌드 프로필의 HIP 오브젝트를 삭제하고 overlay 파일 아무거나 touch하여 Cargo가 빌드 스크립트를 다시 실행하게 하십시오.

```bash
rm -rf target/release/build/mlxcel-core-*/out/build/_deps/mlx-build/mlx/backend/rocm/hip_objs
touch src/lib/mlxcelverse/overlays/rocm/mlx/backend/rocm/CMakeLists.txt
cargo build --release --features rocm
```

`release`는 빌드하는 프로필 디렉터리(`debug`, `test-fast` 등)로 바꾸십시오. `touch`가 없으면 Cargo는 바뀐 입력이 없다고 보고 빌드 스크립트를 건너뛰며 이전 커널을 계속 링크합니다. `cargo clean -p mlxcel-core --release`(또는 `--profile <name>`)도 이를 강제하지만, MLX C++ 라이브러리 전체도 다시 빌드합니다.

### HIP 아키텍처 선택

빌드는 빌드 호스트에서 `rocminfo`가 보고하는 `gfx` 대상에 대해 MLX device 코드를 컴파일합니다. GPU가 없는 호스트에서 빌드하거나 여러 대상을 위해 빌드하는 경우처럼 명시적으로 고르려면 `MLX_ROCM_ARCHITECTURES`를 설정하십시오.

```bash
MLX_ROCM_ARCHITECTURES=gfx1151 cargo build --release --features rocm
MLX_ROCM_ARCHITECTURES="gfx1100;gfx1151" cargo build --release --features rocm
```

각 항목은 HIP 대상 ID(`gfx1151`, `gfx11-generic`, `gfx90a:xnack+`)여야 하며, 그 밖의 값은 configure 단계에서 실패합니다.

변수와 `rocminfo` 모두 대상을 얻지 못하면 빌드가 실패하며 그 변수를 명시합니다. 선택된 목록은 바이너리에 `MLXCEL_ROCM_ARCHITECTURES`로 기록됩니다. ROCm이 `/opt/rocm` 아래에 설치되어 있지 않으면 `ROCM_PATH`를 설정하십시오. 빌드는 CMake 패키지, `hipcc`, `rocminfo`를 그 경로에서 읽습니다.

### 실행

바이너리는 `$ROCM_PATH/lib`를 rpath로 가지므로, ROCm 라이브러리가 동적 로더에 등록되어 있지 않아도 `LD_LIBRARY_PATH`는 필요하지 않습니다. GPU 실행은 시작 시 `Runtime device: GPU`를 출력하며, 해당 프로세스는 `rocm-smi --showpids`에 나타납니다.

```bash
./target/release/mlxcel generate -m models/mlx/Qwen3-0.6B-4bit -p "Hello" -n 50 --temp 0
```

expert-batched `gather_qmm` 커널(bf16 및 f16 affine, bf16 mxfp4에서 기본으로 켜져 있음) 대신 행별 MoE 커널을 사용하려면, 예를 들어 처리량 변화를 A/B로 비교하거나 MoE 회귀를 bisect하려면 `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0`을 설정하십시오. 이 변수는 호출할 때마다 읽힙니다. 이 변수와 백엔드가 실행 시점에 읽는 다른 ROCm 변수는 [ROCm 백엔드 변수](environment-variables.md#rocm-백엔드-변수)에 나열되어 있습니다.

UMA 호스트에서는 GPU가 운영 체제 및 다른 GPU 프로세스와 메모리를 공유하므로, 큰 모델을 로드하기 전에 `rocm-smi --showpids`로 다른 tenant(GPU를 사용하는 다른 프로세스)가 있는지 확인하십시오.

CLI가 아니라 HTTP 서버를 확인하려면, 서버를 시작한 뒤 chat smoke 스크립트를 그에 대해 실행하십시오. dense 체크포인트와 affine MoE 체크포인트 모두 `gfx1151`에서 통과하며, 결과는 [`docs/benchmark_results/rocm-correctness-gfx1151-2026-09-12.md`](benchmark_results/rocm-correctness-gfx1151-2026-09-12.md)에 있습니다. sliding-window 모델, 두 개의 SSM 하이브리드, 그리고 VLM을 추가하고 Apple M5 Max Metal 참조와 비교한 결과는 [`docs/benchmark_results/rocm-correctness-gfx1151-2026-09-30.md`](benchmark_results/rocm-correctness-gfx1151-2026-09-30.md)에 있습니다.

```bash
./target/release/mlxcel-server -m models/mlx/Qwen3-30B-A3B-4bit --port 8080 &
./scripts/server_chat_smoke.sh --port 8080
```

이 스크립트는 `/health`, `/v1/models`, 그리고 `/v1/chat/completions`를 스트리밍과 비스트리밍 모두에 대해 확인합니다. reasoning 채널도 출력으로 계산하므로, 예산 전체를 thinking 블록 안에서 소진하는 thinking 모델은 빈 응답처럼 보이는 대신 `channel=reasoning only`로 보고됩니다.

gfx11의 bf16 및 f16 GEMM(hipBLASLt, 그리고 bf16 출력을 쓰는 rocBLAS)은 `v_wmma_f32_16x16x16_bf16`과 그 f16 쌍둥이 명령에서 실행되며, 이 명령은 부호가 반대인 항을 IEEE 방식으로 더하지 않습니다. 한 명령에서 `1 + -1`은 `-2^-24`를 줍니다. 따라서 정확한 값이 0인 출력이 정수 입력에서도 `-2^-22` 같은 값으로 나올 수 있습니다. 측정된 오차는 표준 f32 내적 한계 `K u sum_k |a_k b_k|`의 2.1% 이내였고, 이 명령을 피하는 hipBLASLt 솔루션이 없으므로 백엔드는 이를 그대로 둡니다. 그래서 ROCm에서 bf16 GEMM을 비교하는 테스트는 정확히 일치하는지가 아니라 이 한계 안에서 비교합니다. 측정, 커널, probe는 [`docs/benchmark_results/rocm-bf16-gemm-residue-gfx1151-2026-10-08.md`](benchmark_results/rocm-bf16-gemm-residue-gfx1151-2026-10-08.md)(lablup/mlxcel#2206)에 있습니다.

`gfx1151`에서 같은 호스트의 mlx-lm과 비교한 decode 및 prefill 처리량은 [`docs/benchmark_results/rocm-baseline-gfx1151-2026-09-30.md`](benchmark_results/rocm-baseline-gfx1151-2026-09-30.md)에 있습니다. `scripts/bench_decode.sh`는 ROCm 호스트를 스스로 인식합니다([벤치마크](benchmarks.md#rocm-호스트-issue-1810) 참고). 이 decode 시간이 커널별로 어디에 쓰이는지, 그리고 #1814 커널 포트가 반영되어야 할 순서는 [`docs/benchmark_results/rocm-decode-profile-gfx1151-2026-09-30.md`](benchmark_results/rocm-decode-profile-gfx1151-2026-09-30.md)에 있습니다. 통합 decode 엔진(epic #2166)은 병합 후 `gfx1151`에서 검증되었습니다. ROCm gate, 모든 ROCm 전용 테스트, CLI 대 서버 parity(dense 및 paged), 그리고 epic 이전 루프와 비교한 decode 처리량은 [`docs/benchmark_results/rocm-unified-engine-gfx1151-2026-10-08.md`](benchmark_results/rocm-unified-engine-gfx1151-2026-10-08.md)에 있습니다.

#### 커널 디스크 캐시

hipRTC로 JIT 컴파일된 커널은 사용자별로 디스크에 캐시되며, 위치는 `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/hsaco/<version>/<arch>`입니다. 새 디렉터리는 mode `0700`으로 생성됩니다(lablup/mlxcel#2641). 이는 공유 경로인 `/tmp/mlx/<version>/hsaco/<arch>`를 대체합니다. 그 경로에서는 다른 로컬 사용자가 디렉터리를 먼저 만들어 커널을 심어 둘 수 있었으며, 그 경로의 항목은 더 이상 읽지 않습니다. `MLX_HSACO_CACHE_DIR`은 위치를 재정의하며, 아키텍처가 뒤에 덧붙습니다. 이 캐시에서는 이 변수와 `MLXCEL_CACHE_DIR`이 절대 경로여야 합니다. 상대 경로는 현재 디렉터리를 기준으로 해석되기 때문입니다.

디렉터리에 쓸 수 있는 사람은 GPU에 로드될 컴파일된 코드를 심을 수 있으므로, 디렉터리는 기본값이든 명시적이든, 디렉터리 자체부터 `/`까지 경로의 모든 구성 요소가 본인 또는 root 소유이고 group이나 other가 쓸 수 없을 때만 사용됩니다(root 소유의 sticky 디렉터리인 `/tmp`는 괜찮습니다). `hsaco`, `<version>`, `<arch>`는 비공개 대상을 가리키더라도 심볼릭 링크가 아닌 실제 디렉터리여야 합니다. `~/.cache/mlxcel`과 지정한 디렉터리는 심볼릭 링크일 수 있으며, 그 경우 대상을 기준으로 판단합니다. 경로에서 존재하는 부분을 먼저 검사하므로, 거부된 실행은 아무것도 만들지 않습니다. 거부된 디렉터리는 캐시를 끄며, 이후 모든 실행이 커널을 다시 컴파일하고, mlxcel은 한 줄을 출력합니다. `mlxcel: ROCm kernel disk cache disabled: <directory> <reason>`(reason은 `is not owned by the current user`, `is group- or other-writable`, `is a symbolic link`, `is not a directory`, `cannot be resolved`, `cannot be inspected`, `cannot be created` 중 하나입니다). 또는 디렉터리를 지정할 수 없을 때는 `no directory: MLXCEL_CACHE_DIR and the home directory are unset` 또는 `MLX_HSACO_CACHE_DIR must be an absolute path, got "<value>"`를 출력합니다(`MLXCEL_CACHE_DIR`에도 같습니다). 지정한 디렉터리에서 group과 other의 쓰기 권한을 제거하거나(`chmod go-w <directory>`), `MLX_HSACO_CACHE_DIR`을 비공개 디렉터리로 지정하여 해결하십시오.

`umask`가 `002`이면 group 쓰기가 가능한 디렉터리가 생성됩니다. 그래서 기본 위치의 경우, 위의 모든 검사를 통과하면 mlxcel은 본인이 소유한 디렉터리 중 자신이 관리하는 것에서 group 쓰기 권한을 제거합니다. 대상은 `<arch>`, `<version>`, `hsaco`, 그리고 `MLXCEL_CACHE_DIR`이 설정되어 있지 않을 때의 `~/.cache/mlxcel` 자체입니다. other가 쓸 수 있는 디렉터리는 복구되지 않으며 거부됩니다. group이 쓸 수 있는 명시적 `MLX_HSACO_CACHE_DIR` 또는 `MLXCEL_CACHE_DIR`, 또는 group이 쓸 수 있는 `~/.cache`는 그대로 두며 여전히 거부됩니다.

캐시 항목(`.hsaco`, `.txt`, `.hip`)은 일반 파일이고, 심볼릭 링크가 아니며, 본인 또는 root 소유이고 group이나 other의 쓰기 권한이 없을 때에만 사용됩니다. 또한 `.hsaco`는 비어 있지 않아야 하며 모든 `.txt` 줄의 형식이 올바라야 합니다. 그렇지 않으면 커널을 다시 컴파일하고 항목을 교체합니다. 새 항목은 mode `0600`을 받습니다.

#### 메모리 사용량

두 가지 기본값이 allocator의 사용량을 가중치 크기에 가깝게 유지합니다(lablup/mlxcel#2062). `gfx1151`에서 pp512/tg128로 측정한 결과, Meta-Llama-3.1-8B-Instruct-4bit의 MLX 최대 사용량은 20.60 GB에서 6.14 GB로, Qwen3-30B-A3B-4bit는 23.56 GB에서 18.58 GB로 줄었습니다. decode 처리량은 변하지 않았고 prefill은 2% 이내였습니다. 세부 내역과 비교한 모든 설정은 [`docs/benchmark_results/rocm-memory-gfx1151-2026-09-30.md`](benchmark_results/rocm-memory-gfx1151-2026-09-30.md)에 있습니다.

- **In-flight 한도, `MLX_ROCM_MAX_INFLIGHT_MB`(기본값 1024).** 이전 최대 사용량의 대부분은 캐시가 아니었습니다. prefill은 연산마다 임시 버퍼(transient)를 할당하고(dequantize 후 GEMM 경로의 f16 4-bit 모델에서는 각 가중치 행렬의 f16 복사본), 각 버퍼는 해당 command batch가 GPU에서 끝나야만 해제되는데, 호스트는 GPU보다 훨씬 앞서 인코딩할 수 있었습니다. 이제 백엔드는 batch가 이 예산의 4분의 1만큼 할당하면 batch를 commit하고, in-flight 양이 예산을 넘는 동안에는 가장 오래된 batch를 기다리므로, 임시 버퍼는 대략 이 MiB 수 이내로 유지됩니다(할당 개수를 세는 것이므로 근사값입니다). `0`은 제한 없는 이전 동작을 복원합니다.
- **버퍼 캐시 한도, `MLXCEL_CACHE_LIMIT`(ROCm의 기본값 2 GiB).** 해제된 버퍼는 재사용을 위해 캐시되는데, ROCm allocator는 요청된 크기와 정확히 같은 버퍼만 재사용하며, 캐시 한도의 기본값은 메모리 한도(여기서는 76.8 GiB)였습니다. 이제 ROCm 빌드는 `MLXCEL_CACHE_LIMIT`이 달리 지정하지 않으면 캐시를 2 GiB로 제한하며, `0` 또는 `none`은 제한을 해제합니다. 이 변경 이전에는 allocator가 캐시 한도를 아예 무시했으므로 `MLXCEL_CACHE_LIMIT`은 ROCm에서 효과가 없었습니다. decode 처리량은 두 모델 모두에서 128 MiB와 제한 없음 사이에서 변하지 않았고, prefill은 8B 모델에서 128 MiB일 때 16% 떨어졌습니다. 캐시가 각 f16 가중치 복사본을 담지 못하면 매번 다시 할당되기 때문입니다. 512 MiB 이상에서는 측정 가능한 차이가 없었습니다. 2 GiB는 그 손실 없는 가장 작은 값의 네 배이므로, 더 큰 모델의 더 큰 가중치 복사본을 위한 공간이 남습니다.

allocator의 최대 사용량은 활성 버퍼만 세므로, allocator가 보유하는 최대량은 최대 사용량에 캐시 한도를 더한 값입니다. 로드 전 추정이 읽는 메모리 한도(`memory_limit()`, 76.80 GiB)는 두 기본값 어느 쪽으로도 바뀌지 않습니다. 추정의 합계에는 ROCm 빌드에서 in-flight 예산이 포함되며, 이는 `mlxcel inspect`의 별도 `Backend in-flight` 줄로 표시됩니다(lablup/mlxcel#2155). 이를 포함하면 추정치는 640 및 4096 token, 그리고 256에서 4096 MiB까지의 예산에서 Llama-3.1-8B, Qwen2.5-7B, Qwen3-30B-A3B의 측정된 최대 사용량보다 높습니다. 1.20 headroom 계수만으로는 dense 모델에서 부족했습니다.

### 현재 상태

| 영역 | ROCm에서의 상태 |
|------|----------------|
| Affine 4-bit / 8-bit 체크포인트 | 네이티브로 실행됩니다. |
| mxfp8 및 mxfp4 체크포인트 | `gather_qmm`을 통한 MoE expert(예: gpt-oss-20b-MXFP4-Q4)를 포함해 네이티브로 실행됩니다. mxfp4 `quantize`, `quantized_matmul`, `gather_qmm`은 `tests/rocm_mxfp4_quant.rs`에서 CPU 참조와 대조하여 검증됩니다(lablup/mlxcel#1808). 벤더 FP8 block 체크포인트(`quant_method: fp8`, 128x128 블록)는 로드 시 mxfp8로 재양자화되어 네이티브로 실행됩니다. dense 0.8B Qwen3.5 FP8 체크포인트에서 end to end로 검증되었으며, mxfp8 `gather_qmm`과 `quantized_matmul`은 `models::switch_layers::mxfp_tests`에서 검증됩니다. 공식 Qwen3.5 FP8 MoE 릴리스는 ROCm에서 실행해 보지 않았습니다(lablup/mlxcel#1807). |
| NVFP4 체크포인트 | 네이티브로 실행됩니다(lablup/mlxcel#2617). dense 레이어와 MoE expert 모두에서, expert별 global scale을 `gather_qmm` 내부에서 적용합니다. ModelOpt NVFP4 체크포인트는 Metal 및 CUDA에서와 마찬가지로 로드 시 MLX 네이티브 NVFP4로 트랜스코딩되며, MLX 네이티브 NVFP4 export(`"mode": "nvfp4"`, 예: `mlx-community/*-nvfp4`)는 그대로 로드됩니다. decode와 짧은 행 수에서는 전용 NVFP4 matrix-vector 커널을 실행하며(여덟 행이 가중치 로드 하나를 공유), prefill 규모의 dense GEMM은 가중치를 dequantize한 뒤 hipBLASLt 또는 rocBLAS로 실행합니다. 정렬된 MoE prefill은 각 expert의 run을 순회하는 expert-batched 커널을 사용합니다(lablup/mlxcel#2665). `quantized_matmul`과 `gather_qmm`은 `tests/rocm_nvfp4_quant.rs`에서 CPU 참조와 대조하여 검증됩니다. 실제 체크포인트 `mlx-community/Nemotron-Cascade-2-30B-A3B-nvfp4`(`nemotron_h`, 128 experts, top-6)로 `gfx1151`에서 검증되었습니다. 일관된 greedy 출력이 나왔고, expert-batched 커널을 사용할 때 512-token prefill은 269~328 tok/s, 2048-token prefill은 237~330 tok/s였습니다(세 번 측정의 중앙값). decode는 약 44~48 tok/s였으며, `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED=0`일 때 greedy 출력이 바이트 단위로 동일했습니다. lablup/mlxcel#2698 이후 NVFP4 `dequantize`(임베딩 조회)는 CPU 스트림 대신 GPU에서 실행됩니다. CPU 스트림은 512-token prefill의 약 3분의 2 동안 fine-grained 메모리에서 가져온 행을 읽고 있었습니다. 정렬된 MoE prefill은 expert 평균 행 수가 16 이상이면 wave32 device에서 rocWMMA 타일 변형을 사용합니다. 그 결과 512-token prefill은 329~1008 tok/s, 2048-token prefill은 331~1106 tok/s, decode는 45~60 tok/s가 되었습니다(세 번 측정의 중앙값). greedy 출력은 GPU `dequantize`에서 동일하며, f32 누적 순서가 다른 WMMA 변형에서는 달라집니다(`MLX_ROCM_NVFP4_GATHER_WMMA=0`은 스칼라 커널을 복원합니다). #2665 이전에는 Nemotron-H가 NVFP4 export를 로드하지 않았습니다. 그 fused Mamba2 decode 단계(`fused_mamba2_forward`)는 affine 투영만 곱하므로, NVFP4 투영은 unfused 단계를 사용합니다. 그 단계와 `fused_moe_forward`를 위한 NVFP4 지원은 #2698에서 구현되었지만 더 빨라지지 않았습니다(512 prompt token 이후 decode 61.0 대 61.9 tok/s). 그래서 유지하지 않았습니다. #2617 이전에는 ModelOpt NVFP4가 로드 시 affine 4-bit로 변환되었고 MLX 네이티브 NVFP4는 거부되었습니다(lablup/mlxcel#1806). |
| Affine MoE 모델(예: Qwen3-30B-A3B) | 네이티브로 실행됩니다. 단일 token decode는 Metal 및 CUDA와 마찬가지로 HIP 포트를 통해 fused 2-kernel MoE 경로(gate-up 다음 down)를 사용합니다(lablup/mlxcel#2065). `gfx1151`에서 Qwen3-30B-A3B-4bit decode는 61.3에서 62.5 tok/s로 올랐으며, 결정된 모든 위치에서 logit이 Metal의 fused 커널과 일치했습니다([결과](benchmark_results/rocm-fused-moe-gfx1151-2026-10-05.md)). `MLXCEL_FUSED_MOE=0`은 대신 `gather_qmm` 그래프 경로를 선택합니다. `MLXCEL_FUSED_MOE_MAX_DFF`(기본값 8192, CUDA와 같으며 Mixtral-8x7B는 14336)보다 넓은 expert는 `gather_qmm`에 남습니다. |
| 모델별 fused 커널(Apertus xIELU, Cohere2 add3 + LayerNorm, Mamba1 selective scan, Nemotron-H의 opt-in `MLXCEL_FUSED_MOE_RELU2`) | HIP 포트로 실행됩니다(lablup/mlxcel#2069). xIELU와 add3 LayerNorm은 ROCm 자체의 fused되지 않은 그래프와 바이트 단위로 동일하며, Mamba1 scan은 Metal과 마찬가지로 상태를 f32로 유지합니다. relu2 경로는 Nemotron-H에서 성능에 중립적이어서 opt-in으로 남습니다. Apertus, Cohere2, Mamba1 계열은 테스트된 호스트에 체크포인트가 없으므로 커널 테스트로만 검증됩니다([결과](benchmark_results/rocm-metal-only-ports-gfx1151-2026-10-05.md)). |
| Fused residual-add RMSNorm 및 RoPE + KV append (`MLXCEL_FUSED_ADD_RMSNORM`, `MLXCEL_FUSED_ROPE_APPEND`) | ROCm에서 기본으로 켜져 있고(lablup/mlxcel#2145), Metal과 CUDA에서는 꺼져 있습니다. 둘 다 HIP 포트(lablup/mlxcel#2063)로 실행되며, 출력이 대체하는 그래프와 바이트 단위로 동일하므로 logit을 바꾸지 않습니다. `gfx1151`에서 decode는 이들을 사용해도 느려진 적이 없습니다(Llama 3.1 8B에서 +0.3%, Qwen2.5 7B에서 +1.7%로 실행 간 편차 이내). 둘 중 어느 변수든 `0`으로 설정하면 그래프로 fallback됩니다. Llama 3.1은 `rope_scaling` 테이블 때문에 그래프 RoPE를 유지하며, Gemma와 IQuest Loop Coder도 norm 포트를 사용하지만 ROCm에서 측정된 체크포인트는 없습니다([결과](benchmark_results/rocm-fused-norm-rope-gfx1151-2026-10-05.md)). |
| Paged attention (서버의 batched paged decode, MLA split-KV, sparse paged decode) | v1 decode, v2 partial, merge 커널이 HIP 포트로 실행되므로(lablup/mlxcel#2068), 이 경로들은 Metal 및 CUDA와 마찬가지로 gather 후 SDPA 대신 fused 커널을 사용하며, 커널 테스트는 `gfx1151`에서 gather 및 호스트 참조와 일치합니다. 같은 바이너리의 gather 경로와 비교한 `gfx1151`의 Meta-Llama-3.1-8B-Instruct-4bit 서버 decode는 batch 4, 약 16K token에서 요청당 decode 1.28x이고, 하나의 sequence에서는 약 16K와 약 32K에서 각각 1.40x와 1.77x이며, batch 4와 약 4K에서는 측정 가능한 변화가 없습니다([결과](benchmark_results/rocm-paged-attention-gfx1151-2026-10-05.md)). `MLXCEL_PAGED_ATTENTION_NATIVE=0`은 gather 경로를 강제합니다. |
| 샘플링 decode (`temperature > 0`) | fused Gumbel-max sampler(필터 없음)와 dual-pivot rejection sampler(top-p 및 라우팅이 허용하는 다른 필터 조합)가 Metal 및 CUDA와 마찬가지로 HIP 커널로 실행됩니다(lablup/mlxcel#2064). `gfx1151`에서 Meta-Llama-3.1-8B-Instruct-4bit로 `--temperature 0.8 --top-p 0.95`의 decode는 33.98에서 37.56 tok/s로 올랐습니다(세 번 측정의 중앙값. 변경 전 실행은 36.26에서 32.81로 떨어졌고, 변경 후 실행은 37.06에서 37.82로 유지되었습니다). `--temperature 0.8`만 사용하면 변화는 노이즈 범위 안입니다([결과](benchmark_results/rocm-samplers-gfx1151-2026-10-05.md)). Llama 3 어휘(128256)와 Qwen의 151936 및 152064에서는 top-p만, top-p와 min-p, 그리고 top-k와 top-p(예: `--top-k 40 --top-p 0.95`)가 rejection 커널을 사용하고, top-k만, min-p만, top-k와 min-p는 기본 chain을 사용합니다. ROCm 빌드는 top-k와 top-p를 어휘 152064까지 라우팅합니다(Metal과 CUDA에서는 32768). 이를 사용하면 `--temperature 0.8 --top-k 40 --top-p 0.95` decode는 Meta-Llama-3.1-8B-Instruct-4bit에서 33.15에서 37.73 tok/s로, Qwen2.5-7B-Instruct-4bit에서 41.91에서 46.92 tok/s로 올랐습니다(세 쌍을 번갈아 측정한 중앙값, [joint-cap 결과](benchmark_results/rocm-rejection-joint-cap-gfx1151-2026-10-07.md), lablup/mlxcel#2157). `MLXCEL_SAMPLING_GUMBEL=0`과 `MLXCEL_SAMPLING_REJECTION=0`은 그래프를 강제합니다. |
| 하이브리드 SSM decode (granite-4.0-h, falcon-h1, plamo-2, Nemotron-H) | 단일 token Mamba2 SSM 업데이트는 약 55개 연산의 SSD 그래프 대신 fused HIP 커널을 실행합니다(lablup/mlxcel#2067). `gfx1151`에서 decode는 granite-4.0-h-tiny-4bit에서 60.4에서 88.4 tok/s로, Nemotron-3-Nano-30B-A3B-4bit에서 51.4에서 74.4 tok/s로 올랐으며, 결정된 모든 위치에서 모델 logit이 그래프 및 Metal의 커널과 일치했습니다([결과](benchmark_results/rocm-ssm-update-kernel-gfx1151-2026-10-04.md)). granite와 Nemotron-H에서 측정되었습니다. falcon-h1과 plamo-2는 같은 gate를 사용하지만 ROCm에서 실행해 보지 않았습니다. `MLXCEL_SSM_KERNEL=0`은 그래프를 강제합니다. |
| GPU fault | 오류로 보고됩니다(lablup/mlxcel#1804). HIP가 거부하는 launch(과도하게 큰 block, device용 code object 없음)는 그것을 발행한 evaluation을 실패시키며, device는 계속 사용할 수 있습니다. 비동기 fault(out-of-bounds 접근)는 멈추거나 NaN을 반환하는 대신, 이를 기다리는 evaluation을 약 1초 안에 실패시킵니다. 서버는 그 요청을 실패시키고, 이후 요청에도 같은 오류로 응답합니다. queue fault 이후에는 HIP 런타임이 프로세스가 끝날 때까지 모든 호출을 거부하므로 프로세스를 다시 시작해야 하며, 종료에 SIGKILL이 필요할 수 있습니다(HIP의 teardown이 fault가 난 queue가 결코 실행하지 않는 callback을 기다리기 때문). 끝나지 않는 커널은 감지되지 않으며, `MLX_ROCM_GPU_WATCHDOG_SECS`(기본값 꺼짐)는 그보다 오래 걸리는 단일 호스트 대기를 실패시킵니다. |
| UMA 호스트에서의 메모리 추정 | 올바릅니다. 테스트된 구성에서 측정한 결과, ROCm allocator는 0이 아닌 상한(96 GiB carve-out 중 76.80 GiB)을 보고합니다. 추정기는 호스트 RAM에 닿기 전에 이 값을 읽으므로, carve-out에 들어가는 것은 그 이유로 거부되지 않습니다(lablup/mlxcel#1805). |
| 진단 | AMD 벤더, device 이름, `gfx` 대상, device 메모리를 보고하며, 더 이상 이에 대해 CUDA compute capability를 출력하지 않습니다(lablup/mlxcel#1805). 컴파일된 `gfx` 목록이 device를 포함하지 않는 바이너리는 첫 번째 커널 launch에서 실패하는 대신 시작을 거부합니다. |
| ROCm 빌드의 CPU device (`MLXCEL_DEVICE=cpu`) | 실행되지만 느립니다. 테스트된 호스트에서 Qwen3-0.6B-4bit decode 한 단계에 약 2분이 걸리므로, 서비스 모드가 아니라 정확성 참조이자 잘못 맞춘 `gfx` 빌드에 대한 탈출구입니다. lablup/mlxcel#1807 이전에는 모든 attention 모델이 첫 token에서 `NYI`로 중단되었습니다. 이 device의 BLAS 작업(f32 matmul, convolution, 선형 대수)은 하나의 OpenBLAS 스레드에서 실행되며, OpenBLAS의 기본 스레드 수에서는 ROCm allocator의 fine-grained 메모리에 잘못된 출력 열을 간헐적으로 기록했습니다(lablup/mlxcel#2072, `overlays/rocm/LOCAL_FIXES.md` 27번 항목). |
| `mlxcel-server` chat completions | dense 및 affine MoE 체크포인트에서 스트리밍과 비스트리밍 모두 동작하며, `scripts/server_chat_smoke.sh`로 검증되었습니다. |
| 오디오 (speech to text, text to speech) | 동작합니다. FFT primitive는 hipFFT에서 실행되며, plan은 `MLX_ROCM_FFT_CACHE_SIZE`까지 캐시됩니다(CUDA와 마찬가지로 기본값 128, lablup/mlxcel#1825, #1876). 양의 정수가 아닌 값은 경고와 함께 무시됩니다(#2051). |
| Wave64 devices (CDNA: MI200 `gfx90a`, MI300 `gfx942`) | 검증되지 않았습니다. 지금까지 유일한 wave64 실행은 MI300X 하나의 bring-up뿐이었으며(`overlays/rocm/LOCAL_FIXES.md` 52번 항목), lablup/mlxcel#2699에서 검증을 추적합니다. lane에 걸쳐 reduce하는 fused HIP 커널(BitLinear, 두 개의 fused MoE 커널과 squared-ReLU 커널, Mamba2 SSM 업데이트, Mamba1 scan, add3 LayerNorm, fused residual-add RMSNorm, paged-attention v1 decode와 v2 partial)은 32-lane wavefront에서만 실행되었고, 이들의 컴파일 시점 wavefront guard는 AMD clang 23에서 아무 효과가 없습니다. 따라서 mlxcel은 device가 32-lane wavefront를 보고할 때만 이들을 선택합니다(`MLX_ROCM_FORCE_WARP_SIZE`로도 바뀌지 않는 하드웨어 값). 64-lane device에서는 그 경로들이 MLX 그래프 fallback을 사용하며, stderr가 프로세스당 한 번 이를 알립니다(lablup/mlxcel#2147). lane 간 연산이 없는 커널(xIELU, RoPE + KV append, paged-attention merge, 두 sampler)은 어떤 폭에서든 선택됩니다. BitLinear에는 그래프 fallback이 없으므로 BitNet 체크포인트는 wave64 device에서 로드 시 거부됩니다. rocWMMA 양자화 GEMM(`qmm_wmma_dense_kernel`과 grouped-prefill gather)도 32-lane warp를 가정하므로, 마찬가지로 wave32 device로만 제한됩니다. 그 전에는 `MLX_ROCM_WMMA_QMM=0`이 설정되지 않은 한 gfx942에서 모든 bf16 양자화 prefill이 쓰레기 출력(`!!!!`)을 냈습니다(`overlays/rocm/LOCAL_FIXES.md` 52번 항목). |
| Windows | 실험적이며, [AMD ROCm 기반 Windows (실험적)](#amd-rocm-기반-windows-실험적)을 참고하십시오. |
| 다중 GPU, 분산 추론 | 지원되지 않습니다. |

테스트된 구성에서 측정한 decode 처리량이며, 참고용으로만 제시합니다: Qwen3-0.6B-4bit 약 250 tok/s(greedy, 짧은 프롬프트), fused MoE 커널을 사용한 Qwen3-30B-A3B-4bit 약 62 tok/s([결과](benchmark_results/rocm-fused-moe-gfx1151-2026-10-05.md)), 512 token 프롬프트 이후 mxfp4 warp-shared gather 커널을 통한 gpt-oss-20b-MXFP4-Q4 약 64 tok/s(`overlays/rocm/LOCAL_FIXES.md` 41번 항목. 그 이전에는 8.3 tok/s였고, 위에 링크된 baseline 페이지에서는 8.07 tok/s였습니다. [결과](benchmark_results/rocm-moe-decode-mxfp4-gfx1151-2026-10-08.md)).

## 런타임 환경 변수

| 변수 | 설명 | 기본값 |
|----------|-------------|---------|
| `CUDA_HOME` | CUDA toolkit 루트, 빌드 시점 및 런타임 NVRTC 헤더용 | 존재하면 `/usr/local/cuda` |
| `MLX_CUDA_ARCHITECTURES` | CUDA SM 대상 목록, 빌드 시점 | `nvidia-smi`로 자동 감지, 이후 `90a` fallback |
| `MLXCEL_CXX_MARCH` | C++ 브리지 `-march` 값, 빌드 시점. `none`은 플래그를 생략 | `native` |
| `MLXCEL_CCCL_DIR` | CUDA NVRTC JIT가 사용하는 번들된 CCCL(libcu++) 헤더 디렉터리 재정의 | 번들된 `<exe-dir>/../include/cccl`, 이후 빌드 시점 fallback |
| `MLXCEL_CUTLASS_DIR` | 양자화 matmul 커널을 위해 CUDA NVRTC JIT가 사용하는 번들된 CUTLASS/CuTe 헤더 디렉터리 재정의 | 번들된 `<exe-dir>/../include`, 이후 빌드 시점 fallback |
| `MLX_PTX_CACHE_DIR` | JIT 컴파일된 CUDA 커널의 디스크 캐시. 커스텀 커널도 포함되며 `sm<cc>-nvrtc<version>` 하위 디렉터리에 저장됩니다. 절대 경로여야 합니다. Linux에서는 모든 경로 구성 요소가 본인 또는 root 소유이고 group이나 other에 대해 쓰기 가능하지 않은 경우가 아니면 거부됩니다(`CUDA kernel disk cache disabled` 줄이 한 번 출력되고 커널은 여전히 컴파일됨). Windows에서는 드라이브 루트 아래의 모든 구성 요소가 본인, SYSTEM, Administrators 또는 TrustedInstaller 소유이고 다른 계정의 쓰기 항목이 없는 경우가 아니면 거부됩니다 | `mlxcel` 및 `mlxcel-server`는 `~/.cache/mlxcel/cuda-ptx/<mlx-commit>`, 그 외에는 `$TMPDIR/mlx-<euid>/<mlx-version>/ptx` (Windows: `%TEMP%\mlx\<mlx-version>\ptx`) |
| `MLXCEL_QUIET_JIT` | 콜드 첫 실행 시 한 번 표시되는 "compiling CUDA kernels" 안내를 숨김 | unset (안내 표시) |
| `MLXCEL_DEVICE` | 런타임 device 힌트 (`gpu`, `metal`, 또는 `cpu`) | `gpu` |
| `MLXCEL_WIRED_LIMIT` | Apple Silicon wired 메모리 상한, 예: `64GB`. `0`/`none`은 비활성화 | `max` |
| `MLXCEL_REQUEST_MEMORY_BUDGET` | [Nemotron-H 요청 admission](request-memory-admission.md)을 위한 하향 전용 양수 바이트/크기 상한, 예: `24GB`. 잘못된 명시 값은 시작을 중단시킴 | 런타임 device/wired/allocator 중 최솟값 |
| `MLX_ROCM_*` 및 기타 ROCm 백엔드 변수 | ROCm 백엔드 튜닝, A/B 및 진단, 예: `MLX_ROCM_GATHER_QMV_EXPERT_BATCHED`, `MLX_ROCM_MAX_INFLIGHT_MB`, `MLX_NO_HIPBLASLT`. [ROCm 백엔드 변수](environment-variables.md#rocm-백엔드-변수) 참고 | 해당 섹션 참고 |
| `LLAMA_ARG_*` | clap이 받아들이는 환경 변수 기반 서버 옵션 | unset |

전체 `MLXCEL_*` 참조는 [환경 변수](environment-variables.md)를 참고하십시오.

## 비디오 입력과 ffmpeg

비디오 프레임 추출은 시스템의 `ffmpeg`와 `ffprobe`를 하위 프로세스로 실행합니다. 둘 다 `PATH`에 있어야 하며, 둘 다 **ffmpeg 5.0(2022) 이상**이어야 합니다. 어느 쪽도 빌드 시점 의존성이 아닙니다. ffmpeg 없이 빌드해도 완전하며, 모든 텍스트, 이미지, 오디오 경로가 동작하고, `--video`(CLI) 또는 `video_url` 콘텐츠 블록(서버)은 모호하게 실패하는 대신 이름이 붙은 오류를 반환합니다.

```bash
# macOS
brew install ffmpeg
# Debian / Ubuntu
sudo apt-get install -y ffmpeg

ffmpeg -version | head -1   # must report 5.0 or newer
```

하한은 플래그 하나로 정해집니다. 추출은 `-fps_mode vfr`을 전달하는데, 이 옵션은 ffmpeg가 이전 옵션인 `-vsync`를 deprecated 처리하던 시점과 같은 5.0에서 추가되었고, ffmpeg 8은 `-vsync`를 완전히 제거했습니다. 4.x 이하에서는 `-fps_mode`가 인식되지 않아 비디오 입력이 지원되지 않으므로, 우회하려 하지 말고 시스템 바이너리를 업그레이드하십시오. 상한은 없으며 9.x까지의 릴리스가 변경 없이 동작합니다.

버전이 맞지 않는 ffmpeg는 프레임이 하나도 디코딩되기 전에 인자 파싱 단계에서 실패하므로, 오류는 비디오가 아니라 옵션을 지목합니다:

```text
Unrecognized option 'fps_mode'.
Error splitting the argument list: Option not found
```

비디오 경로를 수정하는 기여자는 ffmpeg 기반 테스트를 실제로 실행하는 `make verify-test-video`를 실행해야 합니다. 이 테스트들은 일반 스위트에서 `#[ignore]` 처리되어 있으므로, ffmpeg가 없는 머신에서는 조용히 통과하는 대신 ignored로 보고됩니다 (#1172).

NVIDIA 호스트에서는 `make verify-test-cuda-qmm-determinism`이 실제 4-bit Llama-3.2-1B 체크포인트에 대해 양자화 prefill 결정성 테스트를 실행합니다. 이 테스트는 `verify-test-cuda`에서 `#[ignore]` 처리되어 있으며, 체크포인트가 없으면 건너뛰는 대신 실패합니다 (#1558).

## 빌드 확인

```bash
./target/release/mlxcel --version
./target/release/mlxcel-server --version

# `download` defaults to the global store at
# ${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/models/<owner>/<name>.
./target/release/mlxcel download mlx-community/Qwen3-0.6B-4bit
./target/release/mlxcel generate \
    -m ~/.cache/mlxcel/models/mlx-community/Qwen3-0.6B-4bit \
    -p "Hello" -n 1
```

CUDA 호스트에서는 테스트 스위트를 단일 스레드로 실행하십시오. 2026-07 MLX pin 이후 양자화 커널은 처음 사용할 때 JIT 컴파일되고 모듈이 로드되는데, 이 첫 사용 경로는 동시에 실행되는 테스트 스레드에 대해 안전하지 않으므로 기본 병렬 실행은 중단됩니다. 측정된 시그니처는 아래 표에 있습니다. 추론 바이너리는 영향을 받지 않으며, 이는 테스트 병렬성에서 비롯된 현상입니다.

```bash
make verify-test-cuda
# which is:
cargo test --workspace --profile test-fast --features cuda --no-fail-fast -- --test-threads=1
```

MLX pin `2c46b953`의 유휴 GB10(sm_121)에서 `cargo test --release --features cuda -p mlxcel-core --lib`를 세 번 실행하여 이를 수치로 확인했습니다(#1048):

| 스레드 | `MLX_USE_CUDA_GRAPHS` | 결과 |
|---|---|---|
| 기본값 (20) | on | SIGABRT, `cudaStreamEndCapture ... previous error during capture` |
| `--test-threads=1` | on | 판정까지 실행됨, 88초에 1410개 테스트 |
| 기본값 (20) | `0` | SIGABRT, `cuLaunchKernelEx ... invalid argument` |

세 번째 행이 `MLX_USE_CUDA_GRAPHS=0`이 보기와 달리 우회책이 아닌 이유입니다. capture를 비활성화해도 병렬 실행이 구제되지 않고, 실패를 보고하는 CUDA 호출만 바뀝니다. 다른 스레드의 stream capture와 경합하는 모듈 로드에서 커널 configure 경합으로 바뀔 뿐입니다. 직렬화는 원인을 해결하며, gate 아래에서도 capture는 완전히 켜진 상태로 유지되므로 스위트가 계속 이를 검증합니다. `make verify-test-cuda`가 실제로 빌드하는 `[profile.test-fast]`에서도 같은 명령은 동일하게 동작합니다. 1411개 통과, 4개 실패, 89.35초, 중단 없음. 중단 지점과 오류 텍스트는 실행마다 달라지며, 이 때문에 날 것의 SIGABRT는 읽어내는 데 비용이 큽니다. 마치 그때 실행 중이던 어떤 테스트가 깨진 것처럼 보이기 때문입니다.

`mlxcel-core`에는 `the_cuda_test_suite_must_run_single_threaded` guard(`src/lib/mlxcel-core/src/cuda_test_serialization_tests.rs`)가 있어, 플래그를 빠뜨린 호출이 올바른 명령과 함께 이름을 지목하며 실패하도록 합니다. 일반 테스트이므로 이 guard는 필터가 이름과 맞지 않는 범위 제한 실행에서는 걸러지며, 그러한 실행은 병렬로 유지됩니다. 범위가 한정된 부분 집합은 병렬로 통과하고, 중단되는 것은 전체 스위트 실행입니다. 이름과 일치하는 필터, 예를 들어 `--lib cuda`는 안전했을 실행에서도 guard를 작동시키므로, 그 경우 `MLXCEL_ALLOW_PARALLEL_CUDA_TESTS=1`을 설정하여 경고로 낮추십시오.

`make verify-test-cuda`는 `make verify-test`에 대응하는 Linux/NVIDIA 쪽 대상입니다. #1048 이전에는 `mlxcel-core`의 테스트를 실행하는 CUDA 대상이 아예 없었습니다. `verify-test`는 `--features metal,accelerate`로 고정되어 있고, `make test-fast-cuda`는 직렬화하지만 루트 패키지에 머무르므로, 그 아래의 옵션 없는 `cargo test`는 `-p mlxcel`로 해석되어 `mlxcel-core`를 전혀 빌드하지 않습니다. 이는 #1007이 macOS에서 제거한 것과 같은 사각지대가 다른 백엔드에 있었던 것입니다.

## 빠른 반복 빌드

`cargo build --release`(그리고 직접 실행하는 `cargo test --release`)는 `[profile.release]`를 사용합니다. 고정된(locked) 약 439개 crate 전체에 걸친 fat LTO와, 약 39만 줄 규모의 메인 crate에 대한 `codegen-units = 1`입니다. 배포하는 모든 것에는 올바른 trade-off이지만, 일상적인 편집-테스트 루프에는 비용이 큽니다. 증분 리빌드마다 4~6분으로 측정되므로, 편집-테스트 반복이 여러 번인 전형적인 issue 사이클은 순수 컴파일 시간만 20분 이상 씁니다.

로컬 및 에이전트 개발에는 대신 `[profile.test-fast]`를 사용하십시오(crate 간 LTO 없음, `codegen-units = 16`, 증분 컴파일, `strip = false`; `opt-level = 3`은 그대로 유지하여 MLX 위주의 수치 연산이 대표성을 갖도록 합니다):

```bash
# CPU / Metal / Accelerate (macOS adds metal,accelerate automatically)
make test-fast

# Linux / CUDA
make test-fast-cuda

# Linux / ROCm (experimental): no make target yet
cargo test --profile test-fast --features rocm -- --test-threads=1

# Narrow to a subset while iterating
make test-fast-cuda FILTER=server::chat_request
```

또는 cargo를 직접 호출하십시오:

```bash
cargo test --profile test-fast --features cuda -- --test-threads=1
```

Linux/CUDA 개발 머신에서 측정한 결과(2026-07): 콜드 `test-fast` 빌드(모든 의존성과 MLX C++ 트리 포함)는 약 4m53s가 걸리고, 메인 crate 소스 파일 하나를 건드린 뒤의 증분 리빌드는 약 19초가 걸립니다. `[profile.release]`에서의 4~6분과 비교하면 반복 속도가 대략 13배에서 19배 빨라집니다. 대표적인 좁은 테스트 집합(model, server, sampling, cache 모듈에 걸친 139개 테스트)은 두 프로필에서 동일하게 통과합니다.

배포하거나, 벤치마크하거나, 대표 성능으로 인용하는 모든 것에는 `[profile.release]`(`make release*`, 또는 일반 `cargo build --release`)를 사용하십시오. `test-fast`는 링크 시간과 바이너리 크기를 리빌드 속도와 맞바꾸며, 둘 중 어느 쪽에도 맞춰 튜닝되어 있지 않습니다.

*테스트* 실행은 예외입니다. `make verify-test`와, 이를 호출하는 [nightly 워크플로](https://github.com/lablup/mlxcel/blob/main/.github/workflows/nightly-verify.yml)도 테스트 바이너리를 `test-fast`로 빌드합니다. 약 77개 테스트 바이너리를 fat LTO로 링크하느라 그 작업은 테스트가 하나도 실행되기 전에 180분 예산을 전부 써버리고 있었습니다. `opt-level = 3`은 그대로이므로 스위트가 의존하는 최적화된 MLX 수치 연산은 동일하며, 더 이상 검증되지 않는 것은 release LTO나 `codegen-units = 1`에서만 재현되는 결함입니다. 그중 하나를 쫓을 때는 직접 `cargo test --release --features metal,accelerate`를 사용하십시오.

### CPU 전용 link 확인

GPU feature가 없는 Linux 빌드는 GPU 백엔드 없이 MLX를 컴파일하므로, MLX의 GPU 측 helper(예: `copy_gpu_inplace`)가 존재하지 않습니다. `src/lib/mlxcelverse/kernels/`와 `src/lib/mlxcel-core/cpp/` 아래에서 이를 호출하는 브리지 C++는 `MLXCEL_BRIDGE_GPU_BACKEND` 뒤에 있어야 하며, 이 매크로는 MLX가 Metal, CUDA 또는 ROCm을 빌드할 때에만 `mlxcel-core/build.rs`가 정확히 정의합니다. guard를 빠뜨리면 이 구성에서만 드러나는 link 오류가 됩니다. `cargo check`는 link하지 않으며, 모든 GB10 CI 작업이 `--features cuda`를 전달하기 때문입니다. CUDA 캐시를 밀어내지 않도록 별도의 target 디렉터리로 로컬에서 확인하십시오:

```bash
CARGO_TARGET_DIR=target/cpu cargo test -p mlxcel-core --profile test-fast --lib --no-run
```

GB10에서 측정한 결과(2026-10, lablup/mlxcel#2108): CPU 전용 MLX 트리를 포함한 콜드 빌드는 약 2.5분이 걸리고, 브리지 또는 `mlxcel-core`의 파일 하나를 변경한 뒤 리빌드 및 재링크하는 데는 약 6초가 걸립니다. CI는 pull request가 Rust 소스, manifest, 또는 `src/lib/mlxcelverse/`나 `src/lib/mlxcel-core/cpp/` 아래의 브리지 C++를 변경할 때마다 `.github/workflows/ci.yml`의 `CPU-only link` 작업에서 같은 명령을 실행하며, 이 작업은 GB10 runner에서 자체 영구 target 디렉터리(`$HOME/.cargo-target/mlxcel-cpu-link-ci`)를 사용합니다(lablup/mlxcel#2111).

## gate가 `--workspace`를 지정하는 이유

`make verify-clippy`와 `make verify-test`는 `--workspace`를 전달하며, 이를 빼면 실행 속도만이 아니라 다루는 범위가 바뀝니다. 이 저장소의 workspace 루트는 그 자체가 `mlxcel` 패키지이므로, 여기서의 일반 `cargo test`나 `cargo clippy`는 `-p mlxcel`로 해석되어 `mlxcel-core`, `mlxcel-surgery`, `mlxcel-xla`를 빌드하지 않습니다. #1007까지 gate가 정확히 그렇게 동작했고, 그 결과 1754개 테스트가 실행되지 않았으며 그중 1354개가 MLX `cxx` 브리지, `layers.rs`, KV cache, 양자화 로더를 담고 있는 crate인 `mlxcel-core`에 있었습니다. lint 쪽의 구멍은 놓치기 더 쉽습니다. `--workspace` 없는 `--all-targets`는 멤버의 *테스트* 대상도 컴파일하지 않으므로, 테스트 전용 lint 오류와 테스트 전용 컴파일 오류가 모두 gate를 통과했습니다.

이를 대신하는 `default-members`는 `Cargo.toml`에 의도적으로 두지 않았습니다. 그렇게 하면 `release.yml`이 실행하는 `cargo build --release --target aarch64-apple-darwin --locked`를 포함하여 저장소의 모든 일반 cargo 호출의 범위가 한꺼번에 바뀌고, 기본값이 꺼져 있는 `mlxcel-xla`가 모든 릴리스 빌드에 컴파일되기 시작하기 때문입니다.

각 멤버는 루트가 선택한 feature 집합으로 빌드됩니다. `mlxcel-core`는 루트 패키지의 forwarding을 통해 `metal`과 `accelerate`로 해석되므로 모든 멤버가 공유하는 빌드가 하나입니다. `mlxcel-mlx-pin`, `mlxcel-surgery`, `mlxcel-xla`는 비어 있는 기본값으로 해석되며, 특히 `mlxcel-xla`의 `iree` feature는 꺼져 있어 빌드 스크립트가 네이티브 shim을 건너뛰고 gate에 IREE 배포본이 필요 없습니다. 이 때문에 `iree`, `diagnostics`, `micro-oracle` 뒤의 코드는 여전히 gate 밖에 있으며 검사하려면 로컬 IREE dist가 필요합니다(`scripts/iree/setup-macos.sh` 참고). `mlxcel-mlx-pin`은 프로덕션 역할이 없는 leaf로 `mlxcel-core/build_support/mlx_pin.rs`의 MLX-pin 로직 단위 테스트를 담고 있으며, 의도적으로 `mlxcel-core`에 의존하지 않으므로 `cargo test -p mlxcel-mlx-pin`은 MLX C++ 빌드를 일으키지 않고 몇 초 만에 실행됩니다.

`make verify-test-cuda` (#1048)도 같은 이유로 `--workspace`를 지정하며, `metal,accelerate` 대신 `cuda`를 사용하는 것만 다르고 동일하게 해석됩니다. CUDA 경로에서 `mlxcel-surgery`와 `mlxcel-xla`를 포함하는 것은 의도된 것이며 비용이 거의 들지 않습니다. 둘 다 `mlxcel-core`에 의존하고 `cargo test --workspace --features
cuda` 한 번이 이를 `cuda`가 활성화된 단일 빌드로 통합하므로, 어느 쪽도 두 번째 MLX 컴파일을 일으키지 않습니다. `mlxcel-xla`의 `iree` feature도 거기서 꺼져 있어 순수 Rust로 유지되며 IREE 배포본이 필요 없습니다. 이들의 테스트 대상에 들어 있는 것은 백엔드에 독립적인 Rust이므로, 이들을 제외하면 두 crate가 macOS에서만 gate되고 다른 곳에서는 gate되지 않을 뿐입니다. `mlxcel-mlx-pin`은 `mlxcel-core`에 전혀 의존하지 않으며 실행에 몇 초만 더합니다.

`make verify-test`는 `--no-fail-fast`도 전달하는데, 이는 실행이 다섯 멤버를 다루게 된 지금에야 의미가 있습니다. 이것이 없으면 처음 실패한 테스트 바이너리가 실행을 끝내 버리고 나머지 네 개를 가립니다. cargo는 여전히 0이 아닌 값으로 종료하므로 gate가 그만큼 약해지지는 않습니다. `make verify-test-cuda`도 이를 전달합니다.

범위를 넓혀도 #1008의 동시성 위험에 빠지지 않습니다. #1008에서는 두 `mlxcel-core` 스위트가 하나의 Metal device를 공유하여 12번 중 7번의 실행이 중단되었습니다. Cargo는 모든 테스트 바이너리를 빌드한 뒤 하나씩 실행하므로, `mlxcel-core` 스위트는 device에서 루트 스위트와 겹치지 않습니다. 이는 가정이 아니라 측정한 것입니다. cargo 1.97.1에서 세 crate짜리 probe workspace는 첫 테스트 바이너리가 시작되기 전에 전체 빌드를 마치고, 다음 바이너리를 시작하기 전에 각 바이너리를 끝냅니다. 이를 바꾸는 것(예를 들어 `cargo nextest` 같은 병렬 테스트 러너)은 이를 다시 확립해야 합니다. `src/lib/mlxcel-core/src/gpu_exclusivity_tests.rs`의 `no_other_mlxcel_core_test_binary_is_sharing_the_gpu` guard는 같은 device를 두고 경쟁하는 루트 스위트 바이너리가 아니라 두 번째 `mlxcel-core` 바이너리를 감지합니다.

**`make verify-test`는 `--test-threads=1`도 전달합니다(#1092).** 위의 순서 보장은 바이너리 *사이*의 동시성만 제한하며 다른 것은 제한하지 않는데, 2026-08-16에 `main`을 red로 만든 실패는 하나의 바이너리 안에서 발생했습니다. `mlxcel-core` 바이너리가 `signal: 11, SIGSEGV`로 죽었고 panic도 `test result` 줄도 남기지 않았으므로 `--no-fail-fast`가 수집할 것이 없었으며, cargo는 설명 없이 실패한 대상을 보고했습니다. libtest는 기본적으로 논리 CPU마다 테스트 스레드 하나를 사용하며, 18-core M5 Max에서의 로컬 재현에서 얻은 macOS 크래시 리포트는 여기서 그것이 무엇을 의미하는지 보여 줍니다. fault 시점에 18개의 libtest 워커가 살아 있었고 모두 MLX 기반 캐시 테스트를 실행 중이었으며, 두 개는 `iokit_user_client_trap` 안에, 두 개는 allocator 안에 있었고, 매핑된 어떤 영역에도 없는 주소에서 fault가 발생했습니다. 이는
#1048의 CUDA 중단이 다른 백엔드에서 나타난 것입니다. `--jobs 1`은 이를 해결하지 못하는데, `--jobs`는 빌드를 제한하고 어떤 테스트가 실행될 때쯤에는 빌드가 이미 끝나 있기 때문입니다.

직렬화는 거의 비용이 들지 않습니다. 호스트 스레드가 직렬화되든 아니든 작업은 하나의 Metal device에서 직렬화되기 때문입니다. M5 Max의 `5dfcb390`에서 warm 캐시, 전체 workspace, 101개 바이너리와 8128개 테스트로 측정했습니다.

| 테스트 스레드 | 소요 시간 |
|---|---|
| 기본값 (18) | 69.17s |
| `--test-threads=1` | 76.39s |

+7.2s이며, nightly가 180 minutes를 배정한 `cargo test` 단계와 비교하면 미미합니다. 그 단계의 시간은 테스트 실행이 아니라 빌드에 쓰입니다. 두 큰 멤버는 반대 방향으로 작용하여 거의 상쇄됩니다. `mlxcel-core`는 직렬화하면 +23s(10.2s에서 33.2s)가 드는 반면, 루트 스위트는 오히려 12s *단축*됩니다(23.5s에서 11.6s). 5695개 테스트에 걸친 스레드 경합이 순차 실행보다 나쁘기 때문입니다. 이 수치를 재현할 때는 순서가 중요합니다. 콜드 첫 실행은 일회성 Metal 셰이더 컴파일에 대략 50s를 쓰며, 두 조건(arm)이 모두 warm하지 않으면 그 차이만으로도 비교가 뒤집힐 수 있습니다.

`the_cuda_test_suite_must_run_single_threaded` guard에 대응하는 macOS 쪽 guard는 의도적으로 없습니다. CUDA 스위트는 병렬로 실행할 때마다 중단되므로 이름을 지목하며 실패해도 비용이 없지만, Metal 스위트는 드물게 크래시하고, 강한 guard는 병렬로 세 배 빠르며 거의 항상 성공하는 `cargo test -p mlxcel-core --lib`를 깨뜨릴 것입니다. 범위를 좁힌 직접 실행은 병렬로 유지하도록 되어 있습니다. 직렬화하는 것은 전체 스위트 gate입니다.

## 문제 해결

**macOS에서 Metal 툴체인이 없는 경우** — `xcodebuild -downloadComponent MetalToolchain`을 실행한 뒤 다시 빌드하십시오.

**Linux에서 `Cannot find CUDA library directory`** — `CUDA_HOME`을 CUDA toolkit 루트로 설정하고 다시 빌드하십시오.

**빌드 호스트에서 `nvidia-smi`를 사용할 수 없는 경우** — `MLX_CUDA_ARCHITECTURES`를 명시적으로 설정하십시오.

**CUDA/cuDNN 링커 오류** — pin된 MLX 버전이 기대하는 라이브러리가 설치되어 있고 링커가 찾을 수 있는지 확인하십시오. 루트 빌드 스크립트는 CUDA 런타임/수학 라이브러리를 직접 링크하며 `libcuda`는 시스템 드라이버에 의존합니다.

**`qmm_*.cu` 컴파일 중 `gmake: *** Error 137` (SIGKILL)** — 빌드가 메모리 부족으로 종료되었습니다. CUTLASS를 많이 사용하는 양자화 matmul 커널은 병렬 작업당 컴파일러 메모리가 최대 ~4-5 GB에 이르므로, 기본 `-j$(nproc)` 빌드에는 대략 `5 GB × cores`의 메모리가 필요합니다. `cargo build -j N ...`으로 병렬성을 제한하십시오(cargo는 `N`을 CMake 하위 빌드에 전달합니다). `N ≈ available_RAM_GB / 5`로 선택하십시오.

**CMake 오류: `LAPACK_INCLUDE_DIRS ... NOTFOUND`** — `liblapacke-dev`(MLX는 `lapacke.h`가 필요한데 `liblapack-dev`만으로는 제공되지 않습니다)와 `libopenblas-dev`를 설치하십시오.

**ROCm: MLX configure 중 `Could NOT find hip` (또는 `rocblas`, `rocwmma`, ...)**: 해당 CMake config를 제공하는 ROCm 개발 패키지를 설치하거나, `lib/cmake` 아래에 이를 가진 ROCm 루트를 `ROCM_PATH`로 지정하십시오.

**ROCm: `pkg-config`를 찾지 못해 `openssl-sys`가 실패하는 경우**: `pkg-config`와 `libssl-dev`를 설치하거나, 빌드에 `OPENSSL_LIB_DIR`과 `OPENSSL_INCLUDE_DIR`을 설정하십시오.

**ROCm: `rocminfo reported no GPU agent`**: `MLX_ROCM_ARCHITECTURES`를 설정하고, `rocminfo`가 `/dev/kfd`를 열 수 있도록 사용자가 `video` 및 `render` 그룹에 속해 있는지 확인하십시오.

**ROCm: `[cuda_kernel] No CUDA back-end.` 이후 중단**: ROCm fallback이 없는 fused 경로에 도달했습니다. lablup/mlxcel#1803은 MoE를 포함해 그래프 fallback이 있는 fused 경로를 라우팅했으므로, 라우팅된 경로에서는 이것이 더 이상 발생하지 않아야 합니다. 모델과 traceback을 보고해 주십시오. ROCm 포트가 없는 fused 커널 진입점을 직접 호출하면 중단되는 대신 커널 이름과 그 fallback을 밝히는 오류를 반환합니다(lablup/mlxcel#1885). Gumbel-max 및 rejection sampler에는 HIP 포트가 있습니다(lablup/mlxcel#2064).
