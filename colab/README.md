# mlxcel Granite 추론 테스트 기록

Rust 오픈소스 [lablup/mlxcel](https://github.com/lablup/mlxcel)을 Python 래퍼(`mlxcel` 패키지)로 구동해
`granite-4.0-h-350m-8bit` 모델(`mlx-community/granite-4.0-h-350m-8bit`와 같은 형식)의 추론을 Linux에서 성공시킨 과정과,
같은 흐름을 Colab(CPU)에서 돌리도록 만든 `run.py`를 정리합니다.

## 폴더 구성

| 경로 | 설명 |
| --- | --- |
| `colab/model/` | 추론 대상 모델 (safetensors, tokenizer, config) |
| `colab/run.py` | Colab(CPU)용 실행 스크립트 (CUDA 제거, 드라이브 모델/바이너리 캐시) |
| `colab/run.ipynb` | 같은 내용을 셀 단위로 나눈 Colab 노트북 (`run.py`와 동일 동작) |
| `colab/run_colab_cuda_original.py` | 기존 Colab(CUDA, Gemma) 스크립트 백업 |
| `colab/linux_cpu/run_granite.py` | Linux(CPU)에서 Python 래퍼로 추론하는 드라이버 |
| `colab/linux_cpu/setup_linux.sh` | Linux 환경 준비 (시스템 패키지, 저장소 클론, 빌드, 파이썬 래퍼 설치) |

## 테스트 환경

- 클라우드 Linux VM (x86_64, Ubuntu 24.04 계열), CPU 2코어, 메모리 약 7.8GB, **GPU 없음**
- Python 3.13, Rust stable 1.97.0, cmake 3.28, gcc/g++
- mlxcel 저장소: `lablup/mlxcel` (main, commit `d23e290`), 버전 `mlxcel 0.7.0`

## 진행 순서

1. **모델 준비**: Windows의 `colab/model/` 10개 파일을 Linux 작업 공간으로 옮김 (`model.safetensors` 약 362MB 포함).
   `config.json`에서 `GraniteMoeHybridForCausalLM`, `quantization: affine 8bit, group_size 64` 확인.
2. **시스템 패키지 설치**: `libopenblas-dev liblapack-dev liblapacke-dev gfortran pkg-config libssl-dev`.
   MLX CPU 백엔드가 BLAS/LAPACK을 사용합니다.
3. **바이너리 빌드 (CPU 전용)**
   ```bash
   cargo build --release --bin mlxcel --no-default-features --features surgery
   ```
   - `cuda`, `metal` feature는 사용하지 않습니다 (CPU 빌드).
   - `webui`는 제외했습니다. 추론에 필요 없고, 프런트엔드 빌드 산출물(`webui/dist`)이 저장소에 없어 임베드 단계에서 실패할 수 있습니다.
   - 약 28분 소요 (MLX C++ 포함).
4. **바이너리 단독 추론 확인 (CLI)**
   ```bash
   MLXCEL_DEVICE=cpu mlxcel generate -m <model_dir> -p "Explain what Artificial Intelligence (AI) is in one short sentence." -n 60 --temp 0.3
   ```
   결과: 정상 문장 생성 (43 토큰, 약 215초, 0.2 tok/s).
5. **Python 래퍼로 추론 확인**: `pip install ./python` 후 `mlxcel.LLM(model_dir, binary=..., startup_timeout=...)`로
   `mlxcel serve`를 띄우고 `generate()` 호출.
   결과: 서버 준비 4.6초, 40 토큰 생성 113.8초.
   출력: "Artificial Intelligence (AI) is the ability of a computer to perform tasks that normally require human intelligence..."
6. **Colab용 `run.py` 검증**: 로컬 Linux에서 Colab과 같은 흐름을 환경변수(`MLX_WORKDIR`, `DRIVE_ROOT`, `MODEL_DIR`)로 바꿔 테스트.
   결과: 모델 탐색 → 바이너리 로드 → 저장소 클론 → 서버 기동 → 생성 성공 (생성 73.7초, 24 토큰).

## 발견한 문제와 해결

| 문제 | 원인 | 해결 |
| --- | --- | --- |
| `rustup`이 `1.97.1` 툴체인을 받지 못함 | 프록시 환경에서 `static.rust-lang.org` 접속 실패 | 설치된 stable(1.97.0)을 `RUSTUP_TOOLCHAIN=stable`로 사용 (Colab에서는 정상 설치되므로 pin 그대로 사용) |
| 기존 Colab 스크립트의 `build.rs`/`kv_inplace_write.cpp` 패치 | CUDA 빌드 + Turing 환경 우회용 | CPU 빌드에는 불필요하므로 제거 (원본 저장소를 그대로 빌드) |
| `run.py`에서 바이너리가 캐시에 있으면 래퍼 소스를 못 찾음 | 저장소 클론이 빌드 단계 안에만 있었음 | 저장소 준비를 빌드와 분리하여 항상 수행 |
| CPU 추론 속도가 느림 (약 0.2~0.4 tok/s) | 이 프로젝트는 Linux CPU를 공식 지원 대상으로 두지 않음 (`docs/installation.md`) | 기능 검증은 성공. 실사용 속도는 GPU/Apple Silicon 권장 |

## Colab에서 실행하기 (`colab/run.py`)

- 모델: 구글 드라이브 `MyDrive/colab/model` 에 `colab/model` 폴더를 업로드 (없으면 `/content/model`).
- 바이너리 캐시: `MyDrive/colab/mlxcel_cache/mlxcel-cpu`. 첫 실행은 빌드(약 30분), 이후에는 캐시를 복사해 빌드를 생략합니다.
- 실행 순서: 드라이브 마운트 → 모델 탐색 → (캐시 없으면) apt 패키지, rustup, 저장소 클론, CPU 빌드 → `MLXCEL_DEVICE=cpu`로 서버 기동 → 생성.
- 환경변수로 조정: `PROMPT`, `MAX_TOKENS`, `TEMPERATURE`, `STARTUP_TIMEOUT`, `MODEL_DIR`.

- 방법 1: `run.ipynb`를 Colab에서 열고 셀을 순서대로 실행
- 방법 2: `!python run.py` 실행

## Linux에서 실행하기

```bash
cd colab/linux_cpu
bash setup_linux.sh                       # 최초 1회 (빌드 포함)
.venv/bin/python run_granite.py --max-tokens 40
```

## 참고

- 이 기록의 모든 추론 결과는 CPU에서 측정했습니다. GPU 가속은 검증하지 않았습니다.
- `MLXCEL_DEVICE=cpu`를 설정하면 GPU 백엔드가 없어도 CPU로 실행됩니다.
