# 벤치마크

이 페이지는 `mlxcel`의 벤치마크 주장을 어떻게 기록해야 하는지 설명합니다. 의도적으로 보수적인 기준을 따릅니다. 모델별 원본(raw) 행과 측정에 사용한 정확한 소프트웨어/하드웨어 버전이 없다면 종합 속도 향상 수치를 게시하지 마십시오.

## 기록할 항목

모든 벤치마크 실행에는 다음을 포함하십시오.

- 하드웨어 모델과 메모리 크기
- 운영 체제 버전
- `mlxcel` 버전 또는 커밋
- 고정(pin)된 MLX 커밋/버전
- 비교 대상 런타임 버전 (`mlx-lm`, `mlx-vlm` 또는 다른 baseline)
- 모델 체크포인트 이름과 quantization 형식
- 프롬프트 길이, 요청한 decode 길이, batch 크기, warmup 정책
- 캐시 모드와 서버/생성 플래그
- 가능한 경우 모델별 원본 prefill 및 decode 처리량
- op 수준 마이크로벤치마크의 경우, 아래 섹션에 따른 메모리 모드(warm 또는 cold last-level cache)와 rotation 횟수

평균값은 원본 행이 확보된 뒤에만 유용합니다. 비교 가능한 모델 집합과 제외 항목이 명확하지 않다면 "X보다 빠르다"와 같은 표현은 피하십시오.

체크포인트 이름은 무엇을 측정했는지에 대한 주장이며, 그 가치는 뒷받침하는 근거만큼입니다. [`docs/model-catalog.md`](model-catalog.md)는 `models/` 아래 각 디렉터리에 실제로 무엇이 들어 있는지, 어떤 근거에 기반하는지, 그리고 동일한 체크포인트가 이전에 어떤 이름으로 측정되었는지를 기록합니다. 이름이 바뀐 뒤에도 과거 날짜의 CSV 행을 현재 행과 연결(join)할 수 있는 것은 이 때문입니다.

측정하기 전에 바이너리가 소스 트리와 일치하는지 확인하십시오. `bench_decode.sh`는 실행 파일이 아니라 git에서 `mlxcel_commit`을 기록하므로, pull이나 rebase 이후 다시 빌드하지 않고 sweep을 실행하면 바이너리가 갖지 않은 출처 정보가 기록됩니다. 이제 하니스(harness)는 이 상태에서 실행을 거부하며, bisect 용도로는 `BENCH_ALLOW_STALE_BINARY=1`로 이를 우회할 수 있습니다.

## 현재 결과 스냅샷

종합 수치가 문서마다 어긋나지 않도록 공개 결과 요약은 한 곳에 모아 둡니다. 현재 Apple Silicon 벤치마크 보고서는 다음과 같습니다.

- [Benchmark Report - 2026-05-19](benchmark_results/benchmark-report.md)

이 통합 보고서는 하드웨어 간 비교를 위한 자료이지만, 요약 대상인 호스트별 표보다 오래되었습니다. 가장 최신의 전체 sweep은 mlxcel 0.6.0 (2026-09-03/04)에서 측정한 M5 Max 결과로, 텍스트, VLM, speculative 및 batched serving을 다루며 [model_tests_m5max.md](benchmark_results/model_tests_m5max.md)에 있습니다. M1 Ultra와 GB10은 아직 0.4.0-rc.1이므로 현재 하드웨어 간 비율에는 서로 다른 버전이 섞여 있습니다. 다른 호스트를 다시 측정하기 전까지는 호스트별 수치는 하드웨어별 표에서 가져오고, 통합 보고서는 비교의 형태를 파악하는 용도로만 사용하십시오.

릴리스 노트, README 업데이트, 용량 계획에는 이 표들을 사용하십시오. 이 페이지는 방법론, 필수 메타데이터, 주의 사항에 집중합니다.

디코드 격차 조사(mlxcel이 기준 런타임에 뒤처지는 지점에 대한 근본 원인 분석)는 스냅샷과 함께 보관됩니다.

- [MoE decode gap investigation](benchmark_results/moe-decode-gap-investigation.md)
- [Fused decode-MoE kernel: design and roadmap](benchmark_results/fused-moe-decode-kernel-design.md)
- [Gemma3n decode profile: is a compiled fusion justified?](benchmark_results/gemma3n-decode-profile.md)
- [Gemma3n decode profile on M5 Max](benchmark_results/gemma3n-decode-profile-m5max.md)
- [Gemma 4 31B QAT/non-QAT MTP exactness on M5 Max](benchmark_results/gemma4-31b-mtp-exactness-2026-09-26.md)
- [Unified engine baseline: CLI vs server B=1 decode, prefill chunk and KV storage (GB10)](benchmark_results/unified-engine-baseline-gb10-2026-10-07.md)
- [Unified engine final measurement: pre-epic CxxGenerator vs the engine, batched serving, prefill chunk under live decode (GB10)](benchmark_results/unified-engine-final-gb10-2026-10-08.md)
- [Penalty pipelining and server prompt lookup: penalty requests across three builds, prompt lookup at B=1, under concurrency and on long paged prompts (GB10)](benchmark_results/server-prompt-lookup-gb10-2026-10-11.md)

임베딩 및 rerank 처리량(`/v1/embeddings`, `/v1/rerank`)은 `scripts/bench_embeddings.py`로 구동되는 별도의 ladder를 가집니다.

- [Embedding and rerank throughput on GB10 - 2026-08-26](benchmark_results/embeddings-rerank-gb10-2026-08-26.md)
- [Embedding and rerank throughput on M5 Max - 2026-09-04](benchmark_results/embeddings-rerank-m5max-2026-09-04.md)

### 측정이 의존하는 모든 버전을 기록하십시오

벤치마크 행은 세 가지 가변 요소에 의존하는데, 과거에는 하나의 열이 어떤 스크립트가 썼는지에 따라 이 세 가지를 모두 담았습니다. 이제는 분리되어 있습니다.

| 열 | 작성 주체 | 내용 |
|--------|-----------|-------|
| `mlxcel_version` | `bench_decode.sh`, `bench_embeddings.py` | `Cargo.toml`에 기재된 이 저장소의 crate 버전 |
| `mlxcel_commit` | 위 두 스크립트 | 측정한 소스 리비전의 8자리 값. 추적 파일이 수정된 경우 `-dirty` |
| `mlx_commit` | 위 두 스크립트 | 바이너리가 링크하는 고정된 MLX C++ 리비전의 8자리 값 |
| `mlx_version` | 과거 CSV 전용 | 이 열이 MLX 버전으로 하드코딩되어 있던 시절의 MLX 라이브러리 버전 |
| `baseline_version` | `bench_mlxlm.py` | Python baseline. `mlx-lm-<v>` 또는 `mlx-vlm-<v>` 형태이며, ROCm 호스트에서는 뒤에 `+mlx-<v>`가 붙습니다. 해당 환경의 MLX는 버전에 커밋이 포함되는 소스 빌드이기 때문입니다 |
| `mlx_rocm_overlay_commit` | `bench_decode.sh`, ROCm 행 전용 | MLX ROCm overlay(`src/lib/mlxcelverse/overlays/rocm/UPSTREAM`)의 8자리 소스 커밋. `mlx_commit`은 upstream 고정 값 그대로 유지됩니다 |
| `hip_version` | `bench_decode.sh`, ROCm 행 전용 | `hipconfig --version`으로 얻은 HIP 런타임 버전. ROCm 릴리스는 `hardware` 문자열에 있습니다 |
| `decode_path` | `bench_decode.sh` | 러너가 시간을 측정한 decode 루프: `engine`(batch-native 엔진의 raw-completion 클라이언트이며, `mlxcel generate`와 서버의 B=1 행이 실행하는 루프, epic #2166 Phase 6)은 #2176 이후의 모든 행에 해당하고, 그 이전 행은 비어 있으며 퇴역한 `CxxGenerator`의 시간을 측정한 것입니다 |

speculative 또는 batched-serving 표를 수작업으로 옮겨 적을 때 이 열들을 빠뜨리지 마십시오.

각 열은 다른 열이 답할 수 없는 질문에 답합니다. `mlxcel_version`은 릴리스 사이에 변하지 않는 `Cargo.toml`을 읽으므로, 개발 주기 동안 수행한 모든 sweep은 `main`이 얼마나 진행되었든 같은 값을 기록합니다. 시점을 알려 주는 것은 `mlxcel_commit`입니다. MLX 핀(pin) 갱신은 둘 중 어느 것도 바꾸지 않으면서 커널을 바꾸므로, 이를 잡아내는 것은 `mlx_commit`입니다. 하드웨어 간 격차를 하드웨어 탓으로 돌리기 전에, 호스트들이 세 값 모두에서 일치하는지 확인하십시오. 일치하지 않는다면 표를 게시하는 곳에 그 사실을 밝히십시오.

분리 이전의 과거 CSV는 실제로 담고 있는 내용에 맞는 이름을 그대로 유지합니다. 88개는 `mlxcel_version`을, 2026-06-12 이전의 15개 파일은 실제 `mlx_version`(당시에는 이 열이 MLX 릴리스로 하드코딩되어 있었음)을, 26개의 Python baseline 파일은 `baseline_version`을 가집니다.

## 두 CSV 비교하기

join을 직접 작성하지 마십시오. `scripts/compare_bench_csv.py`는 비교가 충족해야 하는 전제 조건을 담고 있으며, 하나라도 실패하면 수치를 반환하는 대신 거부합니다.

```bash
# the runtime against its own earlier sweep
scripts/compare_bench_csv.py --before benchmarks/metal_m5max_vlm_2026-09-06.csv \
    --after benchmarks/metal_m5max_vlm_2026-09-09.csv --allow-commit-change

# the runtime against a reference, which has to be the same host and day
scripts/compare_bench_csv.py --before benchmarks/pylm_m5max_vlm_2026-09-09.csv \
    --after benchmarks/metal_m5max_vlm_2026-09-09.csv --reference
```

이 도구는 텍스트 행과 짝지어진 VLM 행을 거부하고, 버전 변경 자체가 측정 대상이 아닌 한 서로 다른 두 `mlxcel_commit` 값을 거부하며, 하루 넘게 떨어져 측정된 reference를 거부합니다. 또한 `prompt_tokens`가 10% 넘게 다른 쌍은 제외하고, 임베더와 reranker는 생성 대상 목록에서 제외합니다. 이름은 `docs/model-catalog.tsv`를 통해 해석되며, 이 파일의 `aliases` 열은 `;`로 구분됩니다. 또한 동일한 호스트와 하니스에 대해 다른 CSV에 더 새로운 측정값이 있어 baseline 행이 대체된 경우 경고하는데, 이는 변화처럼 보이지만 변화가 아닌 경우입니다.

제외된 모든 항목은 사유와 함께 보고됩니다. 쌍의 개수만으로는 무엇이 빠졌는지 알 수 없기 때문입니다.

이 도구가 존재하는 이유는 2026-09 캠페인에서 공개된 수치 여섯 개가 바로 이런 방식으로 잘못되었고, 각 비교가 서로 다른 일부 검사만 갖춘 채 따로 작성되었기 때문입니다. 이 도구로 다시 확인한 결과, 하드웨어 간 0.35x 열세와 298% 마진은 모두 즉시 거부되고, 61% 동등성 수치는 하니스 불일치로 비정상 종료 코드를 반환하며, 두 호스트 간 쌍 개수 불일치는 대상 목록 정책이 한 곳에 있으므로 해소됩니다. 수정된 마진은 M5 Max에서 105%, M1 Ultra에서 108%로 재현됩니다.

## 권장 벤치마크 명령

저장소에는 `scripts/` 아래에 벤치마크 보조 스크립트가 있습니다. 정확한 인자는 바뀔 수 있으므로 결과를 게시하기 전에 각 스크립트를 직접 확인하십시오.

아래 명령은 2026-09-03/04 M5 Max 캠페인에서 실제로 실행한 것이므로, 다른 호스트에서도 그대로 재현됩니다. `bench_decode.sh`는 모델 경로를 위치 인자(POSITIONAL argument)로 받고, 감지된 하드웨어와 날짜로 CSV 이름을 자동 생성하며, `-m`이나 `--runs`는 없습니다.

```bash
# Single-model decode benchmark. Writes
# benchmarks/{backend}_{hw}_{date}_single_{model}.csv
./scripts/bench_decode.sh --cooldown 30 --big-cooldown 30 models/<checkpoint>

# Full text suite. Writes benchmarks/{backend}_{hw}_{date}.csv
./scripts/bench_decode.sh all --cooldown 30 --big-cooldown 30

# Full VLM suite. Writes benchmarks/{backend}_{hw}_vlm_{date}.csv
./scripts/bench_decode.sh all --vlm --cooldown 30 --big-cooldown 30

# Cap the weight budget so checkpoints that cannot fit are classified
# SKIP:oom_estimate up front instead of being launched and recorded as
# FAIL:bench. The factor is (0.85 * system_memory) / desired_budget; on a
# 128 GB host 1.209 yields a ~90 GB budget.
BENCH_MEM_OVERHEAD_FACTOR=1.209 ./scripts/bench_decode.sh all --cooldown 30 --big-cooldown 30

# Speculative / MTP sweep. Prints a Markdown table on stdout and writes no CSV;
# transcribe it into benchmarks/{backend}_{hw}_spec_{date}.csv.
./target/release/speculative_bench --sweep --max-tokens 128

# Prompt lookup (`generate --prompt-lookup`): plain decoding against both
# drafting policies, arms interleaved per repetition, plus the verify-width
# cost the policies are built on. Method and GB10 results:
# docs/benchmark_results/prompt-lookup-governor-gb10-2026-10-02.md.
CASES=qwen3-8b-4bit:edit_json:400,qwen3-8b-4bit:story:800 REPS=3 BENCH_OUT=out-ab \
    python3 docs/benchmark_results/data/prompt-lookup-governor-gb10-2026-10-02/harness/ab.py
cargo run --release --features cuda --example verify_width_cost -- models/mlx/qwen3-8b-4bit

# Batched serving ladder, against a server started with
# --parallel 4 --max-batch-prefill 4. Writes no CSV; transcribe into
# benchmarks/{backend}_{hw}_batch_{date}.csv.
# --metrics also prints, per level, the batch scheduler's decode steps, mean
# batch occupancy, interleaved prefill chunks and mixed steps, and the mean
# decode step ms from /metrics (issue #2156).
python3 scripts/bench_serving_concurrency.py --port <port> \
    --concurrency 1,2,4 --prompt-tokens 512 --max-tokens 128 --metrics

# Embedding and rerank ladder. One server per checkpoint; writes its own CSV.
python3 scripts/bench_embeddings.py --bin target/release/mlxcel-server \
    --out benchmarks/{backend}_{hw}_embeddings_{date}.csv --repeats 5 --port 18091

# Continuous-batching sweep over every model dir. Takes ONE positional
# argument, the output log path, and no flags.
./scripts/bench_all_models.sh <output_file>

# Single-stream (B=1) decode, CLI path (the engine's raw-completion client) next to the server path
# (the BatchScheduler, driven in-process), at a 256- and an 8192-token prompt.
# One pass is a functional check; for numbers use the interleaved rounds with a
# null arm. ADR 0007 lists the baseline and A/B commands it is decided by.
make bench-engine MODEL=models/mlx/qwen3-1.7b-4bit
scripts/engine_bench_rounds.py --model models/mlx/qwen3-1.7b-4bit --rounds 5 \
    --null-every-arm --arm cli="--path cli" --arm server="--path server" \
    --out baseline-qwen.jsonl
# Each benchmark process is killed after --run-timeout seconds (default 3600,
# 0 disables). Arm names ending in -null are reserved for the generated null
# arms and are rejected.

# CLI vs server token parity (first divergent token per pair), same prompt and
# SamplingConfig, B=1 dense and paged, MLXCEL_SDPA_DETERMINISTIC=1. The prompt
# must render to at least two tokens (one with --no-prompt-cache-case); an
# empty or one-token prompt is rejected before any model loads.
make engine-parity MODEL=models/mlx/qwen3-1.7b-4bit

# Sampling step, no model attached. Gumbel-max (#900) covers the no-filter
# path; the rejection kernel (#901) covers top-k / top-p / min-p.
cargo run --release --features metal,accelerate --example gumbel_sampling_microbench
cargo run --release --features metal,accelerate --example rejection_sampling_microbench
```

두 샘플링 하니스는 표를 출력하기 전에, `mlxcel-server`가 INFO 레벨로 알리는 것과 같은 레코드에서 가져온, 각 arm이 기록한 dispatch 결과를 출력합니다. 수치를 읽기 전에 해당 줄을 먼저 읽으십시오. 이슈 #899에서는 어떤 경로가 실행되었는지 알려 주는 정보가 없었기 때문에, 전체 sweep에 걸쳐 fallback을 자기 자신과 비교하고도 깔끔해 보이는 null 결과를 내놓은 프로덕션 벤치마크가 배포되었습니다. 두 arm이 같은 경로를 보고하는 샘플링 sweep은 아무것도 측정하지 못한 것입니다.

### ROCm 호스트 (issue #1810)

Linux에서 `nvidia-smi`가 GPU를 찾지 못하면, `bench_decode.sh`는 가장 작은 체크포인트(또는 지정한 체크포인트)에서 토큰 하나짜리 `mlxcel generate`를 실행하고, `generate`가 출력하는 줄에서 백엔드, `gfx` 타깃, 장치 이름, 장치 메모리를 읽습니다. 대체 수단은 `rocminfo`입니다. 그러면 ROCm 실행은 `benchmarks/rocm_<hw>_<date>.csv`로 기록되며, `<hw>`는 Radeon 8060S 호스트에서는 `strixhalo-gfx1151`, 그 외에는 `amd-<gfx>`입니다. 85% 메모리 가드는 장치 메모리를 기준으로 적용됩니다. UMA carve-out에서는 GPU가 96 GiB를 가지지만 호스트는 약 31 GiB만 인식합니다. `bench_mlxlm.py`도 같은 방식으로 호스트에 태그를 붙이므로, 두 결과는 `compare_bench_csv.py --reference`로 비교할 수 있습니다. MLX에는 ROCm wheel이 없으므로 Python baseline에는 ROCm 백엔드로 MLX를 소스 빌드한 환경이 필요합니다. `MLXLM_PYTHON`이 그 인터프리터를 가리키도록 하십시오. 다른 GPU 사용자는 먼저 중단하십시오. 공개된 gfx1151 baseline이 이를 어떻게 확인했는지 기록하고 있습니다([rocm-baseline-gfx1151-2026-09-30.md](benchmark_results/rocm-baseline-gfx1151-2026-09-30.md)).

```bash
MODELS_DIR=models/mlx ./scripts/bench_decode.sh all --cooldown 30 --big-cooldown 30
MLXLM_PYTHON=<mlx-rocm-venv>/bin/python LD_LIBRARY_PATH=/opt/rocm/lib \
    MODELS_DIR=models/mlx ./scripts/bench_mlxlm.py all --cooldown 30 --big-cooldown 60 --big-threshold-gb 10
```

`bench_mlxlm.py`의 `--big-cooldown`은 `bench_decode.sh`가 더하는 방식과 달리 큰 모델 이후의 일반 cooldown을 대체하므로, 여기서의 60은 `bench_decode.sh`의 30 + 30과 같습니다.

`bench_decode.sh --temperature T [--top-p P]`는 샘플링 decode를 측정하며, fused sampler가 실행되는 것은 이 경우입니다(greedy는 이를 호출하지 않습니다). CSV에는 샘플링 열이 없으므로 자동 생성되는 파일 이름에 `_t<T>`와 `_p<P>`가 덧붙습니다.

`scripts/rocm_gpu_guard.sh -- <command>`는 그 검사를 스크립트로 만든 것입니다. `/sys/class/kfd/kfd/proc`이 비어 있고 컴파일러 프로세스가 없는 상태를 90초 동안 기다린 뒤, 1 Hz 모니터 아래에서 명령을 실행합니다. 어떤 샘플에서든 다른 GPU 프로세스나 컴파일러가 보이면 명령을 다시 실행하며, 모든 샘플을 `--log`에 추가합니다. 이것이 게시 페이지가 인용하는 증거입니다.

한 호스트의 guard들은 차례로 실행됩니다. 각 guard는 첫 idle 대기 이전부터 종료할 때까지 `ROCM_GPU_GUARD_LOCK`(기본값 `/tmp/mlxcel-rocm-gpu-guard.lock`)에 대한 `flock`을 유지하므로, 동시에 시작된 두 guard는 서로의 명령을 경합이라며 거부하는 대신 하나씩 순서대로 실행됩니다(issue #2146). 병렬 작업에 서로 다른 `--idle-secs` 값을 줄 필요도 없어졌습니다. 어떤 사용자의 guard든 lock 파일을 공유할 수 있습니다. 파일이 없으면 누구나 쓸 수 있도록 생성되며 읽기 전용으로 열립니다. `--max-wait`는 lock 대기와 idle 대기를 합친 하나의 예산이며, 명령이 실행되는 시간을 제외하고 guard 시작 시점부터 계산합니다. lock을 기다리다 이 예산을 소진한 guard는 명령을 실행하지 않고 종료 코드 75로 종료합니다. 명령은 lock 디스크립터 없이 `ROCM_GPU_GUARD_LOCK_HELD`가 설정된 상태로 실행되므로, 그 안에 중첩된 guard(조상이 해당 guard인 경우)는 교착 상태에 빠지지 않고 lock을 건너뜁니다. 같은 바깥 guard 안에 중첩된 guard들끼리는 직렬화되지 않으며, `sudo`를 통한 중첩에는 `--preserve-env=ROCM_GPU_GUARD_LOCK_HELD`가 필요합니다. guard는 `flock`(util-linux)이 필요하며 없으면 종료 코드 2로 종료합니다. 로컬 사용자라면 누구나 기본 lock 파일을 열어 점유할 수 있고, 읽을 수 없게 미리 만들어 둘 수도 있습니다. 신뢰할 수 없는 사용자와 공유하는 호스트에서는 `ROCM_GPU_GUARD_LOCK`을 벤치마크 사용자만 접근할 수 있는 경로로 지정하십시오.

#### 공유 호스트에서의 세션 (issue #2244)

guard는 명령 하나당 lock을 잡으므로, 실행마다 따로 guard를 거는 gate나 측정은 실행할 때마다 다시 대기열에 서게 되고, 다른 작업의 컴파일과 테스트 실행이 그 사이 틈을 채웁니다. 대신 세션 전체에 대해 lock을 한 번만 잡으십시오. `scripts/rocm_gpu_guard.sh --hold [--max-wait SECS] -- COMMAND`는 동일한 대기 절차를 거쳐 lock을 획득하고(`--max-wait`가 소진되면 종료 코드 75), idle 대기와 모니터링은 하지 않으며, `ROCM_GPU_GUARD_LOCK_HELD`를 export하고, COMMAND를 포그라운드로 실행한 뒤 그 종료 상태로 종료하며, 종료, INT, TERM 시점에 lock을 해제합니다. COMMAND 안에서 시작된 guard는 lock은 건너뛰지만 idle 대기와 모니터는 유지하므로 각 실행은 여전히 검사를 받습니다. `--hold`와 함께 `--idle-secs`, `--max-attempts`, `--log`를 쓰면 거부됩니다(종료 코드 2). 바깥 guard 아래에서 `--hold`는 COMMAND를 그냥 실행합니다.

- 먼저 lock 없이 빌드하십시오. 컴파일 전에 hold를 잡으면 다른 모든 사용자가 컴파일이 끝날 때까지 기다리게 되고, 컴파일러가 내부 guard들이 idle 구간을 찾는 것도 방해합니다.
- Gate: `make verify-rocm-held`는 test, release, clippy 산출물을 lock 없이 빌드한 다음 하나의 `--hold` 아래에서 `make verify-rocm`을 실행하므로, 재빌드는 캐시 적중이 됩니다. 공유 호스트에서 `make verify-rocm`을 단독으로 실행하지 마십시오. lock 없이 GPU 테스트를 실행하여 다른 guard들의 idle 구간을 무효화합니다(#2192의 parity guard는 이렇게 28번 거부되었습니다).
- 측정: 세션 전체를 하나의 hold로 감싸고 그 안의 각 실행에 guard를 두십시오. 예: `scripts/rocm_gpu_guard.sh --hold -- bash -c 'for i in 1 2 3; do scripts/rocm_gpu_guard.sh --log run$i.log -- ./scripts/bench_decode.sh ...; done'`.
- `scripts/rocm_gpu_guard.sh --status`는 lock이 점유 중인지, 누가 점유하는지를 `ROCM_GPU_GUARD_LOCK.holder`(pid, 시작 시각, 모드, 명령줄)에서 읽어 출력합니다. 이 파일은 모든 holder가 lock을 획득한 뒤 기록하고 종료 시 제거합니다. 이미 사라진 pid를 가리키는 파일은 stale로 보고됩니다. lock이 비어 있으면 종료 코드 0, 점유 중이면 1입니다. 일반 `flock`으로 잡은 lock에는 holder 파일이 없습니다.

lock은 대기열이 아닙니다. `flock`은 대기자를 순서대로 처리하지 않습니다. `--hold` 세션이 여전히 굶는다면, 측정 결과와 함께 FIFO 티켓 큐 이슈를 등록하십시오.

### ROCm 커널별 decode 프로파일 (issue #2061)

`scripts/rocm_decode_profile.sh`는 ROCm decode 시간이 커널별로 어디에 쓰이는지에 답합니다. 각 모델에 대해 일반 `mlxcel-bench-decode` 실행 한 번과 `rocprofv3 --kernel-trace --hip-graph-trace --stats -f csv` 아래의 실행 한 번을 수행하며, 둘 다 기본 pp512/tg128 형태이고 둘 다 `rocm_gpu_guard.sh`를 거칩니다. bench는 `MLXCEL_BENCH_PHASE_MARKS=1`로 실행되어 warmup, 측정된 prefill, 측정된 decode, 종료 시점의 호스트 클럭 시각을 출력하고(`src/bin/bench_decode/phase_marks.rs`), `scripts/rocm_decode_profile.py`는 측정된 decode 안에서 시작하는 dispatch만 남깁니다. 이로부터 커널별 표(`<run>_decode_kernels.csv`: 호출 수, GPU 시간, decode GPU 시간 중 비중, 커널 클래스, #1814 port unit)와 요약(`<run>_summary.json`: 토큰당 decode GPU 시간과 호스트 gap, 프로파일러 오버헤드, port unit별 비중)을 작성하며, rocprofv3 자체의 프로세스 전체 `<run>_kernel_stats.csv`가 그 옆에 놓입니다.

```bash
cargo build --release --features rocm --bin mlxcel-bench-decode
scripts/rocm_decode_profile.sh models/mlx/Meta-Llama-3.1-8B-Instruct-4bit models/mlx/Qwen3-30B-A3B-4bit
scripts/rocm_decode_profile.sh --temperature 0.7 --top-p 0.95 --no-plain models/mlx/Meta-Llama-3.1-8B-Instruct-4bit
python3 scripts/rocm_decode_profile.py report benchmarks/rocm_profiles/gfx1151_<commit>
```

결과는 `benchmarks/rocm_profiles/gfx1151_<commit>/`에 저장됩니다. 전체 trace는 수백 MB에 달하므로 임시 디렉터리에 저장되며(`--trace-dir`로 보존할 수 있음), 프로파일링 실행에서는 절대 시간이 아니라 비중을 읽으십시오. 트레이서가 decode를 느리게 만들며, 요약에는 일반 실행 대비 얼마나 느려졌는지가 기록됩니다. `--temperature`와 `--top-p`는 프로파일이 sampler를 볼 수 있도록 존재하며, greedy 기본값은 sampler를 호출하지 않습니다.

port unit이 decode GPU 시간에서 차지하는 비중은 `1 / (1 - share)`의 속도 향상 상한(요약의 `ceiling_gpu_share`)을 의미하지만, 이 비중은 dispatch 사이의 호스트 launch gap을 무시하므로 작은 dispatch 여러 개를 하나의 커널로 대체하는 port의 효과를 과소평가합니다(issue #2148). 따라서 요약은 decode 구간의 각 idle 구간을 그것을 끝내는 dispatch에 귀속시키기도 합니다(`scripts/rocm_decode_gaps.py`). role별로는 `role_host_gap_ms_per_token`, port unit별로는 토큰당 및 dispatch당 호스트 gap, decode 벽시계 시간(커널과 그 앞의 gap)에서의 비중, 그리고 그 비중이 의미하는 속도 향상 한계인 `ceiling_wall`을 제공합니다. 일반 실행이 있는 경우 gap은 일반 실행의 호스트 gap에 맞춰 스케일되고(트레이서는 커널 시간이 아니라 호스트 gap을 부풀립니다) 토큰당 벽시계 시간으로 나뉘며, 이것이 `plain_ceiling_wall_est`입니다. port의 실측 속도 향상을 비교해야 하는 기준이 바로 이 한계이며, `report`는 이를 GPU 비중 상한 옆에 출력합니다. 배포 기본값으로 도달하는 role에 대해서도 같은 필드가 있습니다(`reached_default_*`). 공개된 프로파일과 각 커널 이름이 port unit에 어떻게 귀속되었는지는 [rocm-decode-profile-gfx1151-2026-09-30.md](benchmark_results/rocm-decode-profile-gfx1151-2026-09-30.md)에 있습니다.

### op 수준 수치는 decode 수치가 아닙니다 (issue #901)

`examples/rejection_sampling_microbench.rs`는 행마다 두 가지 속도 향상을 보고하며, 읽어야 할 것은 두 번째입니다.

`iso_x`는 고전적인 op 수준 측정입니다. 반복마다 한 번씩 build, `eval`, `synchronize`를 수행합니다. `pipe_x`는 decode 루프가 하는 일, 즉 소프트웨어 파이프라인을 재현합니다. 다음 forward와 다음 sample을 build하고, 둘 다 `async_eval`한 뒤, 이전 step의 토큰을 읽으며, 실행이 끝날 때 한 번만 synchronize합니다.

테스트 대상 연산이 호출자에게 동기화를 강제할 때마다 두 수치는 어긋납니다. `iso`는 매 반복마다 이미 동기화 비용을 지불하고 있어 구조적으로 추가 동기화 한 번을 알아차리지 못하기 때문입니다. #901의 첫 구현은 sampler 안에서 장치 플래그를 호스트로 읽어 왔고, `iso`는 vocab 152064에서 1.14x~1.17x 더 빠르다고 평가했지만 Qwen3-0.6B의 end-to-end decode는 1.7x 더 느리게 측정되었습니다. `iso_x`는 큰데 `pipe_x`가 나쁘다면 그 연산이 동기화를 일으키고 있는 것입니다.

여기서 얻는 일반 규칙은 다음과 같습니다. op 수준 하니스는 연산을 고립된 상태에서 측정하는데, "고립"에는 "호출자의 파이프라인이 이미 비워진 상태"가 조용히 포함됩니다. 호스트 메모리를 건드리거나, 플래그를 읽거나, 장치 상태에 따라 분기하는 모든 연산은 그 수치가 의미를 가지려면 파이프라인 arm이 필요합니다. `src/lib/mlxcel-core/src/sampling_rejection_tests.rs`의 `the_production_sampling_call_never_synchronizes`는 같은 검사를 더 저렴하게 수행하는 형태입니다. 큰 matmul 체인을 enqueue한 뒤 sampler가 그 체인이 비워지기 전에 반환하는지 assert하므로, 회귀는 벤치마크가 아니라 테스트에서 실패합니다.

### `bench_decode.sh all`의 체크포인트 중복 제거 (issue #1615)

`all` 모드는 `"$MODELS_DIR"/*/`를 열거하는데, 하나의 체크포인트가 둘 이상의 디렉터리 이름 아래 있는 경우가 흔합니다. 재다운로드로 생긴 별칭이거나 공유 모델 저장소로의 심볼릭 링크입니다. 중복 제거가 없으면 각 별칭이 따로 로드, prefill, decode, cooldown되어 CSV에 별개의 행으로 남습니다. 이 저장소가 개발되는 M1 Ultra 호스트에서 `models/mlx`에는 중복 그룹이 15개 있고, 불필요한 디렉터리 16개와 불필요한 가중치 124.5 GB를 차지하며, 이러한 중복은 `metal_m5max_2026-04-04.csv`까지 거슬러 올라가 커밋된 모든 sweep에 존재했습니다. 그중 세 그룹은 `docs/benchmark_results/model_tests_m1ultra.md`와 `docs/benchmark_results/model_tests_m5max.md`에 별도의 표 행으로 실렸습니다.

`all` 모드는 이제 첫 모델을 실행하기 전에 체크포인트 동일성으로 중복을 제거합니다. 동일성 키는 `sha256(config.json)`과 모든 `*.safetensors` shard의 `(basename, byte size)` 목록을 정렬한 것의 조합입니다. 작은 파일 하나를 읽고 shard마다 `stat`을 한 번 수행할 뿐 가중치를 해시하지 않습니다. quantization이나 weight dtype만 다른 체크포인트는 `config.json`의 `quantization` 블록이 다르기 때문에 의도적으로 합치지 않습니다. 이 호스트에서 `bitnet-b1.58-2b-4t`와 `bitnet-b1.58-2b-4t-4bit`는 서로 다른 키로 해시됩니다.
`realpath`(실패 시 `cd && pwd -P`)로 해석한 물리 경로가 다른 후보와 같은 디렉터리도 하나로 합쳐지므로, 공유 모델 저장소로의 심볼릭 링크가 콘텐츠 키와 무관하게 행을 두 배로 만드는 일은 없습니다. `config.json`이 없는 디렉터리는 콘텐츠 기준으로 그룹화되지 않으며, 이 중복 제거 단계가 생기기 전과 똑같이 항상 측정됩니다.

중복 그룹 안에서 살아남는 것은 sweep 자체의 열거 순서상 가장 먼저 도달하는 디렉터리입니다. 이 순서는 끝에 슬래시가 붙은 경로에 대한 glob 순서이므로, 한 이름이 다른 이름의 접두사인 경우 더 긴 이름이 이깁니다. `pixtral-12b-4bit`가 `pixtral-12b`보다 우선하고, `qwen2.5-7b-4bit`가 `qwen2.5-7b`와 `qwen2.5-7b-instruct-4bit` 모두보다 우선합니다. 이는 결정적이며 안정적이고, 마침 공개된 비교 표가 이미 사용하는 이름을 유지하게 됩니다. 나머지 구성원은 모두 건너뛰고, 후행 상태 `SKIP:duplicate_of=<survivor-name>`와 함께 각자의 CSV 행으로 기록합니다. 따라서 `all` sweep의 행 수는 여전히 디렉터리 수와 같고, 별칭 집합은 조용히 사라지지 않고 CSV에 그대로 남습니다. 합쳐진 그룹은 첫 모델을 측정하기 전에 stderr로 한 번 출력됩니다. `--no-dedup`은 #1615 이전의 동작을 정확히 복원합니다. 모든 디렉터리를 측정하고, 별칭 행을 내보내지 않으며, sweep 시작 전에 아무것도 출력하지 않습니다. 단일 모델 모드(`bench_decode.sh models/<name>`)는 중복 제거를 하지 않으며, 명시적으로 지정한 경로는 항상 정확히 그 디렉터리를 측정합니다.

## Fused decode 커널: 측정 후 유지 기준 (issue #905)

issue #905의 두 fused decode 커널인 residual-add + RMSNorm, 그리고 q/k RoPE + KV-append 레이아웃은 측정 후 유지(measure-then-keep) 정책에 따라 반영됩니다. 각 커널은 이를 채택하는 백엔드에서 op 수준으로 대체한 그래프보다 빠르고 다른 백엔드에서 손해를 보지 않는 경우에만 연결(wiring)을 유지합니다. `examples/fused_norm_rope_microbench.rs`가 그 증거를 만들어 냅니다. 모델을 로드하지 않고, hidden size 2048 / 4096 / 8192를 batch 1 / 4 / 8과 조합해 sweep하며, 사람이 읽을 수 있는 표와 CSV 블록을 모두 출력합니다.

```bash
caffeinate -i cargo run --release --features metal,accelerate \
    --example fused_norm_rope_microbench

caffeinate -i cargo run --release --features cuda \
    --example fused_norm_rope_microbench
```

어느 op든 속도 향상이 1.00 미만이면 `src/lib/mlxcel-core/src/layers.rs`의 `FUSED_ADD_RMSNORM_DEFAULT` 또는 `FUSED_ROPE_APPEND_DEFAULT`를 뒤집으라는 신호입니다. 그러면 커널을 제거하는 대신 그대로 두되 `MLXCEL_FUSED_ADD_RMSNORM=1` / `MLXCEL_FUSED_ROPE_APPEND=1`로 선택해야 사용하도록 바뀝니다. 두 상수는 모두 `cfg!(feature = "rocm")`입니다. Metal 및 CUDA 빌드에서는 꺼져 있고 ROCm 빌드에서는 켜져 있습니다(#2145).

op 수준 수치는 end-to-end 효과의 하한입니다. 두 fusion은 모두 호출마다 MLX 그래프에서 전체 폭의 중간 결과 하나를 제거하는데, 이는 커널 시간이 아니라 할당자와 의존성 추적 부담으로 나타납니다. 따라서 결정적인 측정은 end-to-end decode sweep(dense 모델 하나와 MoE 모델 하나, batch 1과 4, A/B를 위해 각 kill switch를 뒤집은 상태)입니다. 두 결과를 모두 `docs/benchmark_results/fused-norm-rope-<hw>-<date>.md`에 기록하십시오.

## Warm 대 cold last-level cache (issue #906)

입력을 한 번 할당해 두고 시간을 측정하는 모든 반복에서 재사용하는 op 수준 마이크로벤치마크는 warm cache를 측정합니다. 첫 반복 이후에는 working set이 last-level cache(Apple의 System Level Cache, NVIDIA의 L2)에 상주하므로 나머지 반복은 캐시 대역폭으로 읽습니다. 대역폭에 묶인(bandwidth-bound) 커널의 경우 이는 프로덕션이 겪는 측정과 다릅니다. KV pool은 어떤 last-level cache보다 훨씬 크고 decode step마다 한 번씩만 접근되므로, 대표적인 읽기는 DRAM에서 발생합니다.

이 격차는 이 epic이 다루는 바로 그 커널들에서 가장 중요합니다. Paged KV gather, paged decode attention, rmsnorm은 대역폭에 묶여 있으므로, 이들의 warm-cache 수치는 추정값이 아니라 상한입니다. 연산에 묶인(compute-bound) 커널(큰 M의 quantized GEMM)은 두 모드 사이에서 거의 변하지 않는데, 이것 자체가 유용한 신호입니다. warm 수치와 cold 수치가 일치하는 커널은 메모리에 의해 제한되지 않는다는 뜻입니다.

### 하니스의 처리 방식

`mlxcel_core::bench_rotation`은 입력의 복사본을 여러 개 할당하고 시간 측정 반복마다 하나씩 앞으로 옮깁니다. rotation 횟수는 `ceil(2 * last_level_cache / per_iteration_read_bytes)`이며 64로 제한됩니다. 따라서 rotation 집합 전체가 캐시를 넘고, rotation이 같은 버퍼로 돌아올 때쯤에는 그 버퍼가 이미 축출된 상태입니다. 2배의 여유는 캐시가 시스템의 나머지 부분 및 커널 자체의 출력 트래픽과 공유되는 상황을 감안한 것입니다.

macOS는 `sysctl`로 SLC 크기를 노출하지 않으므로, 캐시 크기는 장치 계열별 추정값입니다. 기본 M 시리즈는 8 MiB, Pro는 24 MiB, Max는 48 MiB, Ultra는 96 MiB(다이 두 개, SLC 두 개)입니다. 추정값은 높게 잡혀 있습니다. 과대 추정은 추가 rotation 버퍼에 쓰이는 메모리 비용만 들지만, 과소 추정은 warm-cache 편향을 조용히 다시 들여오기 때문입니다. CUDA L2 크기를 읽으려면 아직 존재하지 않는 FFI 헬퍼를 통해 `cudaDeviceProp::l2CacheSize`를 읽어야 하므로, CUDA 호스트에서는 `MLXCEL_BENCH_LLC_BYTES`를 장치의 실제 L2 크기로 설정하십시오. 특정 칩의 공개된 SLC 수치를 알고 있다면 Apple Silicon에서도 같은 변수를 설정하십시오.

큰 working set은 rotation이 전혀 필요하지 않다는 점에 유의하십시오. 한 번의 반복이 last-level cache에 담길 수 있는 양보다 더 많이 읽게 되면 rotation 횟수는 1로 줄어들고 cold 모드는 비용이 들지 않습니다. M1 Ultra(96 MiB SLC)에서는 그 교차점이 batch 4 / context 16384에서 나타납니다. 두 모드는 작은 batch와 짧은 context에서 갈라지며, 이는 warm 측정이 가장 오해를 일으키기 쉬운 영역이기도 합니다.

### Apple Silicon에서 실제로 측정된 결과

효과의 크기보다 그 형태가 더 중요한 것으로 드러났기 때문에, 가정하기 전에 측정했습니다. M1 Ultra에서 batch 1 / context 4096(rotation 12)일 때, 각각 다섯 번 반복한 중앙값은 다음과 같습니다.

| 경로 | Warm | Cold | 차이 |
|---|---|---|---|
| `contig_sdpa` | 438.3 us | 433.7 us | -1.0% |
| `gatherA_sdpa` | 509.4 us | 535.0 us | +5.0% |

중앙값은 거의 움직이지 않습니다. 움직이는 것은 분산입니다. warm `gatherA_sdpa`는 425.6~542.1 us(27%, 708 us의 첫 반복 이상치 하나 포함)에 걸쳤지만, cold는 525.1~546.0 us(4.0%)였습니다.

따라서 이 칩에서 cold 모드를 쓰는 이유는 수치에 대한 큰 보정이 아니라 재현성입니다. 대역폭이 매우 높은 통합 메모리는 이 기법의 동기가 되는 캐시 절벽을 무디게 합니다. 이 결론을 CUDA로 확장하지 마십시오. PCIe 뒤에 전용 L2를 둔 이산(discrete) GPU는 절벽이 훨씬 가파르므로, 같은 rotation이 중앙값을 훨씬 더 크게 움직일 것으로 예상해야 합니다. 전체 수치와 측정 당시의 부하 조건은 [autotuner-m1ultra-2026-07-30](benchmark_results/autotuner-m1ultra-2026-07-30.md)에 있습니다.

### 실행 및 기록

```bash
# Warm (historical default, unchanged).
caffeinate -i cargo run --release --features metal,accelerate \
    --example page_gather_microbench

# Cold last-level cache.
caffeinate -i cargo run --release --features metal,accelerate \
    --example page_gather_microbench -- --cold-l2
```

하니스는 헤더에 `memory mode=...`를, 구성마다 `mode=`와 `rotation=`을 출력하고, `CSV:` 행의 마지막 두 열로 `mode`와 `rotation`을 덧붙입니다. 기록되는 모든 결과는 어느 모드에서 나온 것인지 명시해야 합니다. 같은 커널의 warm 수치와 cold 수치는 비교할 수 없으며, 보고서의 같은 열에 함께 나타나서는 안 됩니다.

`examples/qmm_gemv_microbench.rs`는 도입된 이래 이와 비슷한 임시 방식을 자체적으로 가지고 있었습니다(고정된 128 MiB 타깃, 가중치 사본 2~12개를 round-robin). `bench_rotation`은 그 발상을 일반화한 것이며, 캐시 크기를 가정하는 대신 감지합니다.

## Speculative decoding (MTP)

MTP speculative decoding은 decode 타깃을 작은 assistant drafter와 짝지어, drafter가 토큰 블록을 제안하면 타깃이 단일 forward pass로 이를 검증합니다. `temperature 0`에서 가속된 출력은 **런타임 정확성 probe가 그렇다고 판정하는 경우에** 일반 decode와 바이트 단위로 동일하므로, probe를 통과하는 경우 변하는 지표는 decode 처리량뿐입니다. 두 completion을 diff하여 정확성을 확인하십시오. probe는 형식적인 절차가 아닙니다. `T = K` verify 블록이 `K`개의 단일 토큰 step과 비트 단위로 같은지는, 각 quantized projection이 `M = K`와 `M = 1`에서 어떤 MLX 커널로 dispatch되는지에 달려 있으며, 이는 Apple GPU 세대, quantization 모드, 블록 폭에 따라 다릅니다. Qwen 3.5 계열은 probe가 실패하면 일반 decode로 물러나며(#1186), #1188 이후 Gemma 4 arm도 같은 probe를 실행합니다. probe가 실패하면 gate는 먼저 `qmv_wide`를 끄고 재시도하여, 그것으로 정확성이 회복되면 꺼진 상태를 유지하고(이 계열의 verify forward에서 약 23%) 그렇지 않으면 거부합니다. 이 gate가 도입되기 전에 측정된 Gemma 4 행은 바이트 단위로 동일한 커널이 아니라 빠른 커널의 결과이며, 어느 쪽인지는 행의 기록에 나와 있습니다. probe가 물려받는 한 가지 주의점이 있습니다. probe 통과는 증명이 아니라 측정된 증거이며, 2026-08-19(3개 호스트 sweep)에 기록된 M1 Ultra의 prose 프롬프트 불일치가 이에 대해 재검증해야 할 알려진 사례입니다.

각 조합에 대해 baseline(drafter 없음)과 MTP 실행을 모두 기록하십시오.

- 각각의 decode tok/s와 속도 향상 비율 (MTP를 baseline으로 나눈 값)
- 평균 acceptance 길이 (verify당 수락된 draft 토큰 수). `MTP round-loop diagnostics` 로그 줄에서 읽습니다.
- 블록 크기(`--draft-block-size`)와 singleton burst가 작동했는지 거부되었는지 여부

`speculative_bench` 하니스 또는 서버로 측정하십시오.

```bash
# In-process harness: baseline vs MTP on the same target.
./target/release/speculative_bench --target <target_dir> --kind none --max-tokens 256
./target/release/speculative_bench --target <target_dir> --draft <drafter_dir> --kind mtp --max-tokens 256

# Server (production path): time a fixed temperature-0 completion with and
# without the drafter. The server logs decode tok/s and acceptance per request.
mlxcel serve -m <target> --draft-model <drafter> --draft-kind mtp
```

오프라인 `mlxcel generate` 명령도 Gemma 4 텍스트, VLM, Unified 타깃에 대해 MTP round-loop 경로를 지원합니다(issue #166). `--draft-model <drafter> --draft-kind mtp`를 명시적으로 전달하십시오. `--draft-kind mtp`가 없으면 drafter가 MTP로 자동 감지되더라도 이 명령은 하위 호환성을 위해 기존 speculative 경로를 유지합니다.

동등성(Parity): `temperature 0`이고 repetition, frequency, presence, DRY 페널티가 없는 경우, 오프라인 MTP 출력은 문서화된 f16 / #203 batched-kernel 지터 범위 내에서 비 speculative 경로와 일치합니다(일부 하드웨어, 특히 M1 Ultra에서는 거의 동점인 토큰 선택이 달라질 수 있습니다). 샘플링 페널티가 활성화되면 첫 번째 보너스 토큰만 페널티가 적용된 분포에서 샘플링되고, 각 verify 윈도우의 이후 토큰은 greedy하게 수락되거나 거부되므로, 비 greedy 또는 페널티가 적용된 요청은 비 speculative 경로와 바이트 단위로 동일하지 않습니다. 이는 서버 burst 경로의 알려진 제한과 일치합니다.

### 수치를 바꾸는 변경을 판단하기

바이트 동일성은 한 가지 질문, 즉 이 산술 경로가 기준과 비트 단위로 같은가에는 잘 답하지만 그 밖의 것에는 답하지 못합니다. 답이 아니오가 되고, Apple GPU 세대 15 이상에서는 호출자가 선택하지 않은 이유로 그렇게 되는 순간, 이 도구는 소진됩니다. Perplexity는 다른 질문, 즉 코퍼스에서 모델의 예측 분포가 나빠졌는지에 답하며, 커널 재배열은 사용자가 보는 greedy 토큰의 수 퍼센트를 뒤집으면서도 이 값은 그대로 둘 수 있습니다.

`examples/logit_trace`가 그 틈을 메웁니다. 이 도구는 teacher-forced 방식이므로 두 arm이 같은 토큰 스트림에서 평가되어 divergence 때문에 비교가 사라지지 않습니다. free-running 비교는 첫 번째로 뒤집힌 토큰에서 무너지며, 실제 조합에서는 250개 중 비교 가능한 위치가 16개만 남았습니다. 각 구성이 자체 trace를 기록하므로 `MLXCEL_QMV_WIDE`와 같은 프로세스 전역 스위치도 비교할 수 있고, `scripts/compare_logit_traces.py`가 두 trace를 읽습니다.

```bash
cargo build --release --features metal,accelerate --example logit_trace
./target/release/examples/logit_trace  MODEL CORPUS.txt 5 60 8 512 > a.tsv
MLXCEL_QMV_WIDE=0 \
./target/release/examples/logit_trace  MODEL CORPUS.txt 5 60 8 512 > b.tsv
python3 scripts/compare_logit_traces.py b.tsv a.tsv
```

Gemma 4 verify 변경에는 실제 MTP 경로가 필요합니다. `MLXCEL_TRACE_GEMMA4=chain|block|verify`를 설정하십시오. `chain`은 추적되는 각 토큰을 M=1 decode로 실행하고, `block`은 일반적인 블록 산술을 재현하며, `verify`는 MTP 검증 보정을 활성화합니다. 두 block arm은 모두 rotating-cache 버퍼를 포함한 실제 타깃 어댑터를 통해 prefill합니다. `MLXCEL_TRACE_START_TOKEN`은 고정된 코퍼스 context 뒤에서 측정 대상 continuation을 시작합니다. `MLXCEL_TRACE_CONTIGUOUS=1`은 한 번 prefill한 뒤 같은 강제 토큰 스트림을 chunk들에 걸쳐 이어 가며, context가 커질 때의 버퍼링된 full-accept 진행을 검증합니다(rejection/rollback은 generation 테스트가 다룹니다). 이 설정은 Gemma 4 모드와 0이 아닌 start/prefill이 필요합니다.

```bash
# Same 256 teacher-forced tokens in all arms, behind 1536 tokens of context.
cargo build --profile test-fast --features metal,accelerate --example logit_trace
for arm in chain block verify; do
  MLXCEL_QMV_WIDE=0 MLXCEL_TRACE_GEMMA4="$arm" \
  MLXCEL_TRACE_START_TOKEN=1536 MLXCEL_TRACE_CONTIGUOUS=1 \
  ./target/test-fast/examples/logit_trace MODEL docs/architecture.md 4 64 8 1536 > "$arm.tsv"
done
python3 scripts/compare_logit_traces.py chain.tsv block.tsv --decided 2
python3 scripts/compare_logit_traces.py chain.tsv verify.tsv --decided 2
```

기준으로 삼을 지표는 **decided 위치에서의 불일치**입니다. 기준(reference) 자신의 상위 두 후보가 명시된 격차보다 더 크게 벌어져 있었는데도 후보(candidate)가 다른 것을 선택한 위치의 비율입니다. 기준이 어느 쪽이든 상관하지 않았던 위치에는 틀릴 만한 정답이 없으며, 이를 decided 위치와 합쳐 버리면 유일하게 중요한 구분이 가려집니다. 바이트 동일성은 모든 격차에서 불일치가 0인 극한의 경우입니다.

네 가지가 그 답이 의미를 갖는지를 결정합니다.

**테스트 대상 코드가 실행되는 폭으로 trace하십시오.** `N`개 위치에 대한 forward는 quantized projection을 `M = N`으로 실행하고, MLX는 `M`마다 다른 커널을 선택하므로, chunk 폭은 얼마나 많이 측정하는지가 아니라 무엇을 측정하는지를 결정합니다. gemma-4-12b-it-4bit에서 같은 `MLXCEL_QMV_WIDE` 비교는 폭 8에서 top-1 불일치 20.6%, 16에서 19.5%, 32와 256에서는 둘 다 0.0%로 읽힙니다. `use_qmv_wide`는 `M >= 2`에서 갈라지는 반면, batch 한계가 더 큰 `M`을 두 arm이 공유하는 matrix-matrix 커널로 보내기 때문입니다. MTP verify 블록은 블록 크기이고, decode step은 1이며, prefill은 프롬프트 길이입니다.

**context 길이와 forward 폭을 분리하십시오.** 여섯 번째 인자는 행이 추적되지 않는 context를 prefill하므로, 좁은 forward를 현실적인 이력에 대해 측정할 수 있습니다. 이것은 중요합니다. context가 없는 폭 5의 비교와 512 토큰 뒤의 폭 5 비교는 서로 다른 측정이며, verify가 실제로 실행되는 형태는 두 번째뿐입니다. 512 토큰 뒤에서는 두 커널이 전체 위치의 4.0%에서 불일치하지만 decided 위치에서는 0.585%이고, 불일치의 4분의 3은 기준의 차순위 후보입니다.

**변경 없는 arm을 실행하여 실패하는 것을 확인하십시오.** 수정된 arm만 측정한 비교는 코드가 작동한다는 것을 보여 줄 뿐이며, 작동하지 않았다면 측정이 그것을 알아챘을지는 보여 주지 못합니다. 이 둘은 서로 다른 주장입니다. 변경을 되돌리고 나머지는 모두 동일하게 유지한 뒤 지표가 움직이는지 확인하십시오. 이 두 번째 arm이 존재하기 전까지는 비교가 아니라 하나의 판독값만 있는 것입니다.

**입력이 변경된 분기에 도달하는지 확인하십시오.** 조건부 경로에는 임계값이 있고, 그 임계값은 보통 체크포인트 자체의 config에 적혀 있으므로, probe가 통과한 뒤가 아니라 probe를 작성하기 전에 알 수 있습니다. `nemotron_nas` rope-scaling 수정이 실제 사례입니다. config에 `original_max_position_embeddings: 8192`가 있으므로, 1300 토큰짜리 probe는 교체된 scaling을 전혀 건드리지 않고 통과했고, 그 통과는 다른 커밋에서 수정된 head 레이아웃과 GQA 경로에서 비롯된 것이었습니다. 9500 토큰짜리 probe는 실제로 그 분기에 도달했지만 수정된 빌드에서만 실행되었으므로, 코드가 작동한다는 것은 입증했지만 probe 자체의 판별력은 검증되지 않은 채로 남았습니다. 증거는 해당 파일 하나를 되돌린 상태의 14737 토큰에서 나왔습니다. needle을 찾지 못했습니다. 길이가 어떤 주파수 대역이 활성인지를 결정한 것은, 앞서 forward 폭이 어떤 커널이 dispatch되는지를 결정한 방식과 정확히 같으며, 두 경우 모두 같은 실수를 할 수 있습니다.

### A/B의 출력 절반

처리량 arm은 변경이 더 빠르다는 것을 알려 줍니다. 모델이 여전히 같은 답을 내는지는 알려 주지 않으며, 그 절반은 결과가 아닌데도 결과처럼 읽힐 수 있는 경우가 세 가지 있습니다. `scripts/ab_output_equality.sh`는 그중 어느 것도 가능하지 않도록 이를 실행합니다.

```bash
git stash && cargo build --release --features metal,accelerate --bin mlxcel
/bin/cp target/release/mlxcel target/release/mlxcel.before
git stash pop && cargo build --release --features metal,accelerate --bin mlxcel
./scripts/ab_output_equality.sh --baseline target/release/mlxcel.before \
                                --arm target/release/mlxcel \
                                --model models/mlx/granite-4.0-h-tiny-4bit
```

**샘플링은 비교를 양방향으로 무의미하게 만듭니다.** 체크포인트의 `generation_config.json`은 호출자가 플래그를 주지 않아도 샘플링을 켤 수 있으며, 그러면 비교되는 두 파일은 두 구현이 아니라 두 샘플입니다. 건드리지 않은 arm이 실패로 읽히고, 실제로 망가진 arm이 통과할 수 있습니다. 스크립트는 두 arm 모두에 `--temp 0`을 전달하며 호출자로부터 받지 않습니다.

**빈 content 채널은 빈 generation이 아닙니다.** `mlxcel generate`는 기본적으로 `<think>` 채널을 숨기므로, 채널이 닫히기 전에 generation이 끝나는 reasoning 모델은 아무것도 출력하지 않고, 빈 것과 빈 것을 비교하면 토큰을 전혀 비교하지 않으면서도 통과합니다. 반대로 읽으면 더 나쁩니다. 빈 출력이 망가진 체크포인트나 테스트 대상 arm이 일으킨 문제처럼 보입니다. 이 해석은 RMS-norm A/B에서 `glm-4.1v-9b-thinking-4bit`와 `nvidia-nemotron-3-nano-30b-a3b-4bit`에 대해 하루에 두 번이나 코앞까지 왔고, 판정을 뒤집었을 것입니다. 스크립트는 두 arm 모두에 `--show-reasoning`을 전달하므로 생성된 모든 토큰이 비교에 포함됩니다. CLI도 이제 이 경우를 명시합니다. 토큰이 생성되었는데 content 채널에 도달한 것이 없으면 `generate`와 chat REPL은 빈 줄 대신 `[All N generated tokens went to the reasoning channel ...]`을 출력합니다.

**baseline이 자기 자신과 일치할 때만 출력 차이를 arm의 탓으로 돌릴 수 있습니다.** 일부 계열은 실행마다 비트 단위로 안정적이지 않으며(f16 reduction 순서 지터 부류), 그런 계열에서 arm 간 차이는 변경에 대해 아무것도 말해 주지 않습니다. 스크립트는 baseline을 두 번 실행하여 대조군으로 삼고, 두 baseline 실행이 일치하지 않으면 arm이 다르다고 보고하는 대신 종료 상태 2와 함께 `INCONCLUSIVE`를 보고합니다. 그런 체크포인트에서는 이 도구가 아니라 위의 teacher-forced logit trace가 사용할 도구입니다.

### Gemma 4 Unified (12B) + 4-bit assistant

타깃은 `mlx-community/gemma-4-12b-it-4bit`, drafter는 `mlx-community/gemma-4-12B-it-assistant-4bit`이며, `temperature 0`, warm 상태로, warm-up은 버리고 arm을 번갈아 실행했고, arm당 샘플 여덟 개를 사용했으며, 호스트의 백그라운드 인덱서는 일시 중지했고, 분산은 중앙값 대비 M5 Max에서 1.9% 이하, M3 Ultra에서 2.9% 이하, M1 Ultra에서 1.9% 이하입니다.

| 호스트 | 출력 | 프롬프트 | 토큰 | 블록 | acceptance | classic | MTP | 속도 향상 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M5 Max (128 GB) | 열거 | "Count from 1 to 200, one number per line, with no other text." | 400 | 4 | 0.997 | 43.1 | 135.4 | **3.14x** |
| M5 Max (128 GB) | 소스 코드 | "Write a Python function that computes the nth Fibonacci number, with a docstring and type hints." | 300 | 5 | 0.784 | 43.5 | 121.0 | **2.79x** |
| M5 Max (128 GB) | 산문 | "Explain how speculative decoding accepts or rejects draft tokens." | 400 | 5 요청, 4 유효 | 0.489 | 43.3 | 82.4 | **1.90x** |
| M3 Ultra (512 GB) | 열거 | "Count from 1 to 200, one number per line, with no other text." | 400 | 4 | 0.980 | 63.4 | 165.5 | **2.61x** |
| M3 Ultra (512 GB) | 소스 코드 | "Write a Python function that computes the nth Fibonacci number, with a docstring and type hints." | 300 | 5 | 0.733 | 64.2 | 138.5 | **2.16x** |
| M3 Ultra (512 GB) | 산문 | "Explain how speculative decoding accepts or rejects draft tokens." | 400 | 5 요청, 4 유효 | 0.544 | 64.0 | 111.2 | **1.74x** |
| M1 Ultra (128 GB) | 열거 | "Count from 1 to 200, one number per line, with no other text." | 400 | 4 | 0.997 | 34.2 | 50.4 | **1.48x** |
| M1 Ultra (128 GB) | 소스 코드 | "Write a Python function that computes the nth Fibonacci number, with a docstring and type hints." | 300 | 5 | 0.815 | 34.9 | 43.5 | **1.25x** |
| M1 Ultra (128 GB) | 산문 | "Explain how speculative decoding accepts or rejects draft tokens." | 400 | 5 요청, 4 유효 | 0.525 | 34.5 | 32.7 | **0.95x** |

호스트 안에서는 acceptance가 나머지를 설명하는 열입니다. 열거 행은 거의 모든 draft를 수락하고(블록 4에 대해 verify당 방출되는 토큰이 M5 Max에서 3.990, M3 Ultra에서 3.912, M1 Ultra에서 3.990), 산문 행은 약 절반을 수락합니다(2.463, 2.625, 2.574). 이 행들 사이에서 다른 것은 프롬프트뿐입니다.

속도 향상을 가로로 비교하지 말고 호스트를 열 방향으로 읽으십시오. M3 Ultra의 세 행은 모두 M5 Max의 대응 행보다 아래에 있는데, 두 arm 모두 절대값으로는 더 빠릅니다. classic decode가 거기서는 약 63~64 tok/s로 43보다 높으므로, 비율의 분모인 baseline이 MTP arm보다 더 많이 향상된 것입니다. 속도 향상은 호스트의 순위가 아니라 짝(pair)의 속성이며, 다른 실리콘에서 측정한 비율과 비교하는 것은 아무것도 말해 주지 않습니다. 아래의 Qwen 조합은 같은 두 호스트에서 반대 방향으로 움직이는데, 바로 그 점이 핵심입니다. 호스트 효과의 크기는 조합 사이에도 이전되지 않습니다.

M1 Ultra 행은 위의 산술이 메커니즘이 아님을 보여 줍니다. 두 arm 모두 M5 Max보다 빠르기는커녕 *느리고*, 비율은 M3 Ultra에서보다 더 떨어져 산문에서는 명백한 퇴보가 됩니다. 세 호스트가 모두 공유하는 것은 단 하나의 양이며, 측정할 가치가 있는 것은 바로 그것입니다. 해당 호스트 자신의 classic decode step을 단위로 한 verify round의 비용, 즉 round의 벽시계 시간을 `1 / classic tok/s`로 나눈 값입니다.

| 호스트 | 블록 | round 비용 (classic step 단위) | 손익분기를 위한 verify당 방출 토큰 |
| --- | ---: | ---: | ---: |
| M5 Max (128 GB) | 4 | 1.27 (열거), 1.29 (산문) | ~1.28 |
| M5 Max (128 GB) | 5 | 1.31 (소스 코드) | ~1.31 |
| M3 Ultra (512 GB) | 4 | 1.50 (열거), 1.51 (산문) | ~1.51 |
| M3 Ultra (512 GB) | 5 | 1.61 (소스 코드) | ~1.61 |
| M1 Ultra (128 GB) | 4 | 2.70 (열거), 2.72 (산문) | ~2.71 |
| M1 Ultra (128 GB) | 5 | 3.15 (소스 코드) | ~3.15 |

각 호스트에서 공통점이 전혀 없는 두 프롬프트가 round 비용에서 1% 이내로 일치하는데, 이는 이것이 프롬프트가 아니라 호스트와 블록 폭의 속성이라는 대조군입니다. 그 두 프롬프트는 열거 행과 산문 행으로, 둘 다 유효 블록 4에서 실행됩니다. 코드 행은 expansion gate를 통과해 5에서 실행되므로 따로 나열했으며, 블록을 넓히면 그래야 하듯 그곳에서는 round당 비용이 더 큽니다. 블록 5 항목은 별도로 시간을 잰 것이 아니라 도출한 값입니다. verify당 방출 토큰은 아래의 각 호스트별 폭 sweep에서, 비율은 공개된 행에서 가져왔으며 두 호스트 모두 0.4% 이내로 일치합니다. 또한 이 값은 세 호스트를 속도 향상과 정확히 같은 순서로 정렬하며, 어느 arm이 더 많이 향상되었는지를 끌어들이지 않고도 위의 M3 Ultra 해석을 설명합니다. verify가 단순히 M5 Max에서보다 거기서 상대적으로 더 비쌉니다.

손익분기 열이 퇴보의 원인입니다. round는 classic step으로 환산한 비용보다 더 많은 토큰을 방출해야 하므로, M5 Max는 verify당 1.28개의 토큰이 필요하고 여기의 모든 프롬프트에서 큰 차이로 이를 넘으며, M3 Ultra는 1.51개, M1 Ultra는 블록 4에서 2.71개가 필요합니다. M1 Ultra의 산문 행은 2.574를 방출하여 그 바로 아래에 놓이는데, 이것이 0.95x의 전부입니다.

메커니즘은 `src/models/speculative_exactness.rs`에 문서화된 `use_qmv_wide` 분기입니다. Apple GPU 세대 15부터는 `M >= 2`의 quantized projection이 하나의 wide pass로 실행되지만, 세대 13에는 그런 경로가 없어 블록을 폭에 따라 비용이 커지는 narrow pass들로 실행합니다. M1 Ultra는 세대 13이며, 다른 두 칩이 분할 상환(amortise)하는 verify의 비용을 거의 위치 수에 비례하여 부담합니다.

동일한 프롬프트와 조합에서도 acceptance는 호스트마다 달라집니다(코드 행에서 0.784, 0.733, 0.815). 다만 항상 그런 것은 아니어서, 열거 행은 M5 Max와 M1 Ultra 모두에서 세 자리까지 0.997로 읽힙니다. 샘플링은 원인이 아닙니다. `temperature 0`에서는 두 arm 모두 greedy이기 때문입니다. 이 페이지의 나머지 내용과 일치하는 설명은, 호스트마다 타깃 자신의 거의 동점인 위치를 다르게 해소하여 continuation이 달라지고, 따라서 수락할 대상 자체가 달라진다는 것입니다. 이는 아래에서 정확성 probe가 보고하는 divergence를 acceptance 쪽에서 바라본 것입니다. 호스트별로 추적하지는 않았으므로, 측정된 원인이 아니라 수치에 대한 해석으로 받아들이고, acceptance는 행 사이에 이월되는 값이 아니라 호스트별 수치라고 생각하십시오.

표를 재현하거나 확장하려면 `scripts/bench_speculative.sh`를 사용하십시오. 이 스크립트는 프롬프트, 블록 폭, 프로토콜을 담고 있으며, 호스트를 감지하고, 위와 같은 형태로 행을 출력하며, 다른 어떤 것도 GPU를 사용하지 않을 때까지 시작을 거부합니다. 진입 gate는 시작 시점만 지키므로 측정하는 동안에도 계속 감시합니다. 그렇지 않으면 나중에 들어온 부하는 분산 검사에만 맡겨지는데, 일정한 부하는 모든 샘플을 똑같이 낮춰 그 검사를 피해 갑니다. 분산이 중앙값의 4%를 넘거나 시간의 5분의 1 넘게 경합 상태였던 실행은 평균을 내지 않고 신뢰할 수 없는 것으로 보고합니다. 오염된 중앙값은 문서에 실리는 순간 진짜 퇴보와 구별할 수 없기 때문입니다.

누군가의 데스크톱이기도 한 Mac에서 이 검사들을 실패시킬 가능성이 가장 높은 것은 기계 자체의 유지 관리 작업입니다. Spotlight 인덱싱, Photos 분석, 클라우드 동기화는 유휴 상태에서 트리거되므로, 호스트를 무언가 측정하도록 내버려 둔 바로 그때 시작되고, `pmset -g therm`이 아무것도 보고하지 않는 상태에서도 CPU를 수백 퍼센트까지 사용합니다. sweep을 `scripts/with_indexers_paused.sh`로 실행하십시오. 이 스크립트는 하나의 명령이 실행되는 동안 이들을 일시 중지하고, 명령이 어떻게 끝나든 재개합니다.

```bash
./scripts/with_indexers_paused.sh ./scripts/bench_speculative.sh --reps 4
```

SIGSTOP과 SIGCONT만 사용하므로 일시 중지된 작업은 중단된 지점에서 이어지며, 래퍼 자체가 SIGKILL을 받아도 살아남는 경로를 포함해 세 가지 별도 경로가 이를 재개합니다. `INDEXER_EXTRA_NAMES`는 이 특정 호스트에서 조용히 해야 하는 다른 것들의 이름을 한 줄에 하나씩 나열한 목록을 받습니다. 어떤 채팅 및 메일 클라이언트가 인덱서 위에서 동작하는지는 기계마다 다르기 때문입니다. Time Machine은 의도적으로 건드리지 않는 유일한 경쟁자입니다. 백업 세션을 한 시간 동안 얼리는 대신, sweep 전에 `tmutil stopbackup`으로 실행 중인 백업을 끝내십시오. 그러면 나중에 증분 방식으로 재개됩니다.

M1 Ultra 행은 2026-08-19에 이 방식으로 측정되었습니다. 기록은 `benchmark_results/speculative-decoding-m1ultra-2026-08-19.md`에 있으며, guard가 거부하여 재측정한 두 행과 위 round 비용 수치의 유도 과정을 포함합니다.

아래쪽의 블록 폭 표는 `scripts/bench_block_width.sh`에서 나온 것이며, 같은 방식으로 실행합니다.

```bash
./scripts/with_indexers_paused.sh ./scripts/bench_block_width.sh gemma
./scripts/with_indexers_paused.sh ./scripts/bench_block_width.sh qwen
```

이 스크립트는 round마다 모든 폭을 한 번씩 방문하고 round를 시작하는 폭을 돌려 가며 바꿉니다. 한 폭을 끝까지 측정한 뒤 다음 폭을 측정하면 실행 중의 drift가 늦게 측정된 폭에 몰리게 되고, 이는 그 폭이 더 느린 것과 구별되지 않기 때문입니다. 그런데 sweep이 묻는 질문이 바로 그것입니다.

**호스트와 프롬프트를 기록하십시오.** 둘 다 대부분의 코드 변경보다 비율을 더 크게 움직입니다. 프롬프트는 acceptance를 결정하고, 호스트는 각 quantized projection이 어떤 커널로 dispatch되는지, 따라서 verify 블록의 비용이 얼마인지를 결정하며, 같은 조합이 한 세대에서는 이득이고 다른 세대에서는 퇴보하는 이유가 이것입니다. 둘 중 하나라도 없는 행은 재현하거나 비교할 수 없으며, 다른 프로토콜의 행은 같은 표에 속하지 않습니다.

블록 폭은 그리 큰 의미가 있는 튜닝 값은 아니지만, 어디에서 정점을 찍는지는 상수가 아니라 호스트별 사실입니다. 아래 두 sweep은 모두 코드 행을 `scripts/bench_block_width.sh`로 실행하며, 이 스크립트는 round마다 모든 폭을 한 번씩 방문하고 시작 폭을 돌려 가며 바꾸므로 실행 중의 drift가 마지막에 측정된 폭에 몰릴 수 없습니다.

M5 Max에서는 정점이 5입니다.

| 폭 | decode tok/s | 분산 | acceptance | verify당 방출 | 정점 대비 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 98.4 | 1.6% | 0.847 | 2.694 | -19.0% |
| 4 | 114.8 | 1.2% | 0.790 | 3.360 | -5.5% |
| 5 | 121.5 | 0.2% | 0.784 | 3.646 | **정점** |
| 6 | 117.6 | 0.3% | 0.752 | 3.785 | -3.2% |

이 순서에는 모호한 것이 없습니다. 5는 4보다 5.8%, 6보다 3.3% 높으며, 분산은 0.2~1.6%입니다. 이 sweep의 첫 실행은 폭 5와 6을 각각 7.4%와 5.1%의 분산으로 거부했으며, 재측정하자 그 실행의 121.4와 117.5에 대해 121.5와 117.6이 나왔습니다. 이는 guard가 스스로의 설명대로 동작한 것입니다. 경합이 중앙값을 움직이지 않고 분산만 넓혔습니다. 폭 8, 10, 12는 이 호스트에서 스크립트로 실행하지 않았으므로, 이 구간에서 처리량이 계속 떨어진다는 기존 주장은 여기서 검증되지 않은 상태입니다.

그 결과 이 호스트에서는 배포 기본값이 정점이 아닙니다. Gemma assistant 체크포인트는 4토큰 verify 블록으로 구성되어 있고 `mtp`는 `--draft-block-size`의 기본값을 4로 두므로, 폭을 전혀 전달하지 않은 사용자는 유효 블록 4로 실행되어 114.6 tok/s를 측정하는 반면, 5를 명시하면 121.2입니다. 5.8%의 격차이며, 분산은 0.1%와 0.3%이고 두 샘플 집합은 겹치지 않습니다. 아래 M3 Ultra 결과에서는 같은 기본값이 그 호스트의 정점에 정확히 놓이는 것과 반대이므로, 어느 호스트의 답도 일반화되지 않으며 이 플래그는 sweep을 실행해 본 곳에서만 전달할 가치가 있습니다.

M3 Ultra에서는 정점이 4이고, 그 뒤에는 하락이 아니라 고원(plateau)이 이어집니다.

| 폭 | decode tok/s | 분산 | acceptance | verify당 방출 | 정점 대비 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 129.1 | 0.7% | 0.843 | 2.670 | -10.1% |
| 4 | 143.6 | 0.6% | 0.803 | 3.398 | **정점** |
| 5 | 138.0 | 0.9% | 0.733 | 3.477 | -3.9% |
| 6 | 138.8 | 0.5% | 0.738 | 3.737 | -3.3% |
| 8 | 135.1 | 0.6% | 0.683 | 3.934 | -5.9% |
| 10 | 126.2 | 2.7% | 0.628 | 4.041 | -12.1% |
| 12 | 127.7 | 2.8% | 0.658 | 4.333 | -11.1% |

폭 5~8은 정점의 6% 이내에 있으며, 이 대역 안의 순서는 이 샘플들로는 해소되지 않습니다. 6은 5보다 0.6% 높게 읽히는데, 이는 둘 중 어느 쪽의 측정 분산보다도 작습니다. 10과 12 행은 2.7%와 2.8%의 분산에 대해 1.2% 차이로 읽히며, 이는 아래의 후속 sweep이 해결합니다. 여기서 깔끔하게 구분되는 것은 세 가지뿐입니다. 대역 위의 폭 4, 그리고 대역 아래의 3, 10, 12입니다. 따라서 이 호스트의 결론은 M5 Max가 주는 깔끔한 순위가 아니라 "정점은 4이고 그 뒤는 고원"입니다. 여기서 폭 5는 138.0으로 읽히는데, 위 표가 블록 5에서 측정한 138.5와 분산 안에서 0.4% 일치하므로, 이 sweep과 공개된 행은 같은 측정을 두 번 한 것입니다.

메커니즘은 마지막 두 열에 있습니다. verify당 방출은 `1 / (1 - acceptance)`를 향해 오르다가 평평해지는데(범위 전체에서 2.670, 3.398, 3.477, 3.737, 3.934, 4.041, 4.333), verify forward는 위치마다 비용이 계속 늘어나므로 정점을 지나면 폭을 넓힐 때마다 얻는 것보다 치르는 것이 더 큽니다.

폭 4는 배포 기본값이기도 하며 우연이 아닙니다. Gemma assistant 체크포인트는 4토큰 verify 블록으로 구성되어 있고 `mtp`는 `--draft-block-size`의 기본값을 같은 4로 두기 때문입니다. 따라서 폭을 전혀 전달하지 않은 사용자는 이 호스트의 정점에 정확히 놓이며, sweep의 143.6에 대해 143.5 tok/s를 측정합니다.

위 코드 행과 산문 행의 5는 런타임의 선택이 아니라 이 벤치마크 자체의 요청입니다. `scripts/bench_speculative.sh`가 5를 전달하는 이유는 M5 Max가 5에서 정점을 찍었기 때문이며, 프로토콜에 고정된 폭이 호스트 간 행을 비교 가능하게 만듭니다. `effective_mtp_block_size`는 drafter의 구성된 깊이를 넘는 요청을 설정값이 아니라 *상한*으로 취급합니다. 최소 8개의 round가 완료되고 최근 32개 중 적어도 65%에서 구성된 prefix가 완전히 수락될 때까지는 4에 머물다가, 그 뒤에 요청값으로 확장됩니다. 코드 행이 "5"로, 산문 행이 "5 요청, 4 유효"로 읽히는 이유가 이것이 전부입니다. 코드 프롬프트는 expansion gate를 통과하고 산문은 통과하지 못합니다.

이 결과는 일반적인 주의와 반대 방향으로 작용하므로 분명히 말해 둘 만합니다. M3 Ultra에서 프로토콜의 폭 5는 배포 구성이 사용했을 기본값보다 3.9% *느리므로*, 위 표의 138.5와 2.16x는 아무것도 튜닝하지 않는 사용자가 이 호스트에서 얻는 값을 과소평가합니다. 같은 classic arm에 대해 약 143.5 tok/s와 2.24x입니다. 표는 호스트 간 행의 비교 가능성을 위해 5를 유지하고, 그 격차는 한 행에서 튜닝으로 없애지 않고 여기에 기록합니다.

M1 Ultra는 세 번째 답을 주는데, 이 호스트에서는 폭이 해로워지기 전까지는 중요하지 않다는 것입니다.

| 폭 | decode tok/s | 분산 | acceptance | verify당 방출 | 정점 대비 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 43.3 | 1.9% | 0.876 | 2.743 | **정점** |
| 4 | 42.9 | 3.9% | 0.799 | 3.398 | -0.9% |
| 5 | 43.0 | 3.0% | 0.815 | 3.934 | -0.9% |
| 6 | 41.4 | 1.0% | 0.760 | 4.153 | -4.3% |
| 8 | 38.7 | 1.2% | 0.669 | 4.530 | -10.6% |
| 10 | 35.9 | 1.3% | 0.614 | 5.155 | -17.1% |
| 12 | 33.8 | 3.2% | 0.519 | 5.155 | -21.9% |

폭 3, 4, 5는 서로 0.9% 이내에 있고 분산은 1.9, 3.9, 3.0%이므로 "정점은 3"은 이 샘플들이 뒷받침하는 주장이 아닙니다. 정직한 해석은 동률인 세 폭의 대역과 6부터의 하락이며, 그 이후의 모든 격차는 분산을 훨씬 벗어납니다. 기본값은 그 대역 안에 놓입니다. 폭을 전달하지 않으면 네 번의 실행에서 42.85 tok/s를 측정하고 `block_size=4`를 보고하며, acceptance와 verify당 방출은 세 자리까지 폭 4 행과 같고, 프로토콜의 명시적 5보다 1.5% 낮은데 이는 어느 쪽의 측정 분산 안에 있습니다. 호스트 세 곳에 답이 세 가지이며, 이 플래그는 sweep을 실행해 본 곳에서만 전달할 가치가 있습니다.

여기서 폭 5는 43.0으로 읽히는데, 위 표가 블록 5에서 측정한 43.5와 1.1% 차이이며 분산 안에 있습니다. 폭 12는 이 페이지에서 코드 프롬프트를 **퇴보**(0.97x)로 바꾸는 유일한 항목입니다. 10과 12 행의 verify당 방출이 5.155로 같은 이유는 300 토큰에서 둘 다 58 round에 도달하기 때문이며, 따라서 그 꼬리는 나머지보다 거칩니다.

위 섹션의 round 비용을 각 폭에 대입하면 정성적 주장이 기울기로 바뀝니다. 이는 블록 폭에 대해 아핀(affine)이며, 각 sweep에 걸쳐 `cost = a + b K`로 피팅하면 이 조합에서 M3 Ultra는 `1.14 + 0.090 K`, M1 Ultra는 `1.35 + 0.346 K` classic step이 되고 가장 큰 잔차는 각각 0.08과 0.20입니다. 고정 부분은 비슷하지만, 위치당 부분은 3.8배 차이가 납니다. 그 배수가 `use_qmv_wide` 분기에 값을 매긴 것입니다. 세대 15+는 블록 위치 하나가 늘어도 하나의 wide pass에 흡수하고, 세대 13은 narrow pass 하나를 더 치릅니다. 호스트가 오래될수록 사용 가능한 대역이 계속 왼쪽으로 이동하고, 가장 오래된 것만이 명백히 손해를 보는 폭에 도달하는 이유이기도 합니다. M5 Max의 sweep은 폭 3~6만 다루어 기울기와 절편을 분리하기에는 지렛대가 너무 짧으므로 피팅하지 않았습니다.

꼬리 부분만 0.4~0.6%의 분산으로 다시 측정하면 위 sweep이 열어 둔 질문이 해결됩니다. 값은 단조롭게 증가하며 12에서의 꺾임은 없습니다.

| 폭 | decode tok/s | 분산 | acceptance | verify당 방출 |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 126.6 | 0.6% | 0.628 | 4.041 |
| 11 | 127.2 | 0.4% | 0.649 | 4.211 |
| 12 | 128.0 | 0.4% | 0.658 | 4.333 |
| 13 | 128.8 | 0.4% | 0.650 | 4.397 |

10과 12의 측정값은 위 sweep과 0.3% 이내로 일치하므로, 12는 실제로 10보다 1.1% 높고 11은 그 사이에 있습니다. 넷 모두 4의 정점보다 11% 이상 낮으므로 어느 것도 튜닝할 가치가 없습니다.

그럼에도 이 호스트에는 커널 경계가 12에 실제로 존재하며, 처리량이 이를 가립니다. `get_qmv_batch_limit`은 아키텍처 세대 *와* 크기 문자를 읽고, 이 호스트는 `applegpu_g15d`를 보고합니다. 세대 15와 세대 17 분기는 모두 `arch_size != 'd'`를 요구하므로 `d` 부품은 이들을 통과해 세대 13의 `case 'd'` 표로 떨어지며, 그 표는 projection 형태에 따라 32, 18, 12를 읽는 반면 `d`가 아닌 세대 15 부품은 13, 15, 13을 읽을 것입니다. `use_qmv_wide`에는 그런 문자 검사가 없으므로 같은 호스트가 어느 경우든 세대 15 wide 경로를 탑니다. 게이트가 둘이고 답이 서로 다릅니다.

이 타깃에서는 경계가 정확히 12에 놓입니다. gate, up, down, `lm_head` projection(3840x15360, 15360x3840, 3840x262144)은 마지막 분기에 속해 `M = 12`에서 qmv 계열을 벗어나고, q, k, v, o는 4096 분기에 속해 18까지 남습니다. verify는 `M = K`로 실행되므로, 폭 12를 요청하면 가장 큰 세 projection이 qmm으로 옮겨 갑니다. verify forward만 `verify_forward_ms / rounds`로 측정하되 `MLXCEL_QMV_WIDE`를 켜고 끈 결과는 다음과 같습니다.

| 폭 | verify forward, qmv_wide 켬 | qmv_wide 끔 | 끔 / 켬 |
| ---: | ---: | ---: | ---: |
| 10 | 25.49 ms | 46.98 ms | 1.84x |
| 11 | 26.47 ms | 50.69 ms | 1.92x |
| 12 | 27.03 ms | 45.81 ms | **1.70x** |
| 13 | 27.15 ms | 45.34 ms | **1.67x** |

셀당 샘플 세 개, 분산은 0.2~0.5%입니다. 끔 열은 폭 11까지 오르다가 9.6% 떨어지는데, 이는 qmm이 도착한 것입니다. 경계를 넘으면 플래그가 더 이상 큰 세 projection에 영향을 주지 못하고 attention만 바꿀 수 있게 됩니다. 켬 열은 같은 경계를 부드럽게 지나는데, 그곳에서 qmm과 qmv_wide의 비용이 거의 같기 때문이며, 폭 sweep이 12에서 아무것도 보여 주지 않는 이유가 이것입니다. 두 열은 한 쌍으로 읽으십시오. 각각 자신의 플래그 설정으로 측정되었고, 플래그를 끄면 생성되는 텍스트가 달라지므로 처리량과 acceptance 수치는 두 열 사이에서 비교할 수 없습니다.

두 가지가 따릅니다. 이 호스트의 폭 sweep 꼬리는 `d`가 아닌 세대 15 부품이었다면 경계가 13에 놓였을 것이므로 다른 dispatch를 측정하고 있습니다. 그리고 아래의 경험 규칙은 하나가 아니라 서로 독립적인 두 gate에 관한 것입니다. `qmv_wide`는 세대만으로 결정되고, qmv가 qmm에 자리를 내주는 지점은 세대와 크기 문자가 함께 결정합니다.

가속된 출력은 시작 시 정확성 probe가 그렇다고 판정하는 경우에 일반 decode와 바이트 단위로 동일하며, 이는 런타임이 가정하지 않고 측정합니다. Apple GPU 세대 13/14의 affine 4-bit 타깃은 블록 폭 12 미만에서 바이트 단위로 동일한 반면, 세대 15 이상(M3, M4, M5)은 MLX가 `M >= 2`의 quantized projection을 다른 reduction으로 보내기 때문에 블록 폭 2부터 diverge합니다. 12는 세대 상수가 아니라 이 타깃의 가장 큰 projection에 대한 `d` 항목입니다. 같은 표는 `d`가 아닌 세대 13 부품에서 이 형태들에 6을, 이 타깃의 attention projection에는 어느 쪽이든 18을 읽습니다. `qmv_wide`에서 diverge하는 probe는 이를 끄고 재시도하며, 그것으로 정확성이 회복되면 narrow 커널을 유지합니다(#1199). 지금까지 측정된 모든 세대 15+ 호스트에서 일어나는 일이 이것이며, 두 경우 모두 diverge하는 probe만이 `MLXCEL_MTP_ALLOW_INEXACT=1`이 설정되지 않은 한 일반 decode로 물러납니다. B=1(단일 요청) MTP는 모든 MTP 타깃에서 기본으로 실행되며, Gemma 4 Unified 타깃은 아예 batch할 수 없으므로 B=1이 유일한 decode 경로이기도 합니다. batch가 가능한 31B + bf16 assistant는 M5 Max에서 약 1.2~1.4x, M3 Ultra에서 1.95x~2.65x로 측정되며(아래 참조), #1217 이후 Apple GPU 세대 15부터 기본으로 실행됩니다. B=1 verify forward가 비용을 회수하지 못하는 하드웨어에서는 `MLXCEL_ENABLE_MTP_B1=0`을 설정하여 끄십시오.

위의 Gemma 4 행은 #1188 gate가 도입되기 전에 측정되었으므로 바이트 단위로 동일한 커널이 아니라 빠른 커널의 결과입니다. gate가 적용된 지금 세대 15+의 기본값은 바이트 단위로 동일한 커널이며, 빠른 행을 재현하려면 `MLXCEL_QMV_WIDE=1`과 `MLXCEL_MTP_ALLOW_INEXACT=1`을 함께 지정해야 합니다(고정(pin)은 gate의 재시도가 `qmv_wide`를 끄지 못하게 하고, override는 그래도 MTP를 작동시킵니다). `MLXCEL_MTP_ALLOW_INEXACT=1`만으로는 이에 도달하지 못합니다. 재시도가 override를 확인하기 전에 실행되어 프로세스를 narrow로 고정하고, 기본 환경과 바이트 단위로 동일한 출력을 내며, 그 처리량은 바이트 단위로 동일한 행 자체의 처리량으로 측정됩니다(2026-08-22 M3 Ultra에서 실시간 검증, 12B 조합에서 117 대 139 tok/s. [qmv-wide-pin-tax-m3ultra-2026-08-22](benchmark_results/qmv-wide-pin-tax-m3ultra-2026-08-22.md) 참조). `qmv_wide`를 끄는 방식으로 코드 행의 바이트 동일성을 유지하면, M5 Max에서는 121.0 대신 93.2 tok/s(2.79x 대신 2.14x)를, M3 Ultra에서는 138.5 대신 117.5 tok/s(2.16x 대신 1.83x)를 측정합니다. 한 호스트에서 처리량의 23%, 다른 호스트에서 15%이며, 이는 probe가 Qwen 조합에 대해 제시하는 17~20%와 같은 양이 아닙니다. probe는 verify forward에 드는 비용이고 이 수치는 end-to-end decode이며, 후자에서는 drafter step과 수락된 토큰의 방출이 영향을 받지 않습니다. 질문이 어느 쪽에 관한 것인지에 맞는 것을 인용하고 다른 쪽은 인용하지 마십시오.

M1 Ultra에는 그런 비용이 없습니다. 세대 13은 애초에 `qmv_wide`를 사용하지 않기 때문입니다. 그러나 arm들이 probe되지 않은 동안에도 divergence는 통과했으며, #1188이 이 arm들을 probe를 거치도록 한 지금 probe를 시험하는 사례가 바로 이것입니다. 그곳의 메커니즘은 `qmv_wide`일 수 없으므로, 그 호스트에서 probe가 무엇을 읽는지(그리고 통과한 뒤에도 산문 행이 여전히 diverge하는지)는 별도의 실행이 필요합니다. 이제 위의 세 프롬프트에 대해 `temperature 0`에서 두 arm을 diff했으며, 세 GPU 세대에서 각각 한 호스트씩, probe되는 Qwen 조합을 대조군으로 나란히 실행했습니다.

| 호스트 | 조합 | Probe | 소스 코드 | 열거 | 산문 |
| --- | --- | --- | --- | --- | --- |
| M1 Ultra (gen 13) | Gemma 4 12B + 4-bit assistant | 없음 (#1188) | 동일 | 동일 | **diverge** |
| M1 Ultra (gen 13) | Qwen 3.8 27B + 4-bit MTP head | 통과, fallback 없음 | 동일 | 미실행 | 미실행 |
| M3 Ultra (gen 15) | Gemma 4 12B + 4-bit assistant | 없음 (#1188) | **diverge** | 동일 | **diverge** |
| M3 Ultra (gen 15) | Qwen 3.8 27B + 4-bit MTP head | `qmv_wide`를 끈 뒤 통과 | 동일 | 미실행 | 미실행 |
| M5 Max (gen 17) | Gemma 4 12B + 4-bit assistant | 없음 (#1188) | **diverge** | 동일 | **diverge** |

M1 Ultra의 divergence는 산문에서 일어나며, 1755바이트 generation의 892바이트 지점입니다.

```
classic: ...tokens in parallel (as long as they are provided as inp
MTP:     ...tokens in parallel (the attention mechanism allows this
```

M3 Ultra도 산문에서 1802바이트 중 581바이트 지점에서, 그리고 소스 코드에서 1078바이트 중 22바이트 지점(여섯 단어째)에서 diverge합니다.

```
classic: Here are two ways to write this function. The first is
MTP:     Here are two ways to implement this. The first is the
```

M5 Max는 같은 프롬프트에서 1025바이트 중 54바이트 지점에서 갈라지며, M3 Ultra의 두 arm이 불일치했던 시작 부분에서는 두 arm이 일치합니다.

```
classic: ...ways to implement this. The first is the standard **iterative** ap
MTP:     ...ways to implement this. The first is the **efficient** approach (u
```

각 arm은 각 호스트에서 세 번의 실행에 걸쳐 자기 자신과 바이트 단위로 동일하며, M3 Ultra와 M5 Max에서는 diverge한 모든 행에서 arm 간 조합 아홉 개가 모두 다릅니다. 따라서 이는 실행 간 노이즈가 아니라 block 대 chain 경로의 차이입니다. 두 호스트의 실행은 모두 `--show-reasoning`을 사용했는데, Gemma 4는 기본적으로 그 채널을 숨기기 때문입니다. 이 프롬프트들은 그 채널에 아무것도 내보내지 않으므로, diff는 보이는 답변뿐 아니라 방출 전체를 다룹니다. 동일한 행은 조용한 fallback이 아니라 결과입니다. MTP arm이 M3 Ultra의 행을 만들기 위해 acceptance 0.980으로 유효 블록 4에서 102 verify round를, M5 Max의 행을 만들기 위해 acceptance 0.997로 100 round를 실행했기 때문입니다.

블록 폭은 행을 가르는 요인이 아닙니다. 열거는 M3 Ultra와 M5 Max 모두에서 블록 4와 블록 5에서 동일하게 유지되고, 소스 코드는 각각 두 블록 모두에서 diverge합니다.

행을 정렬하는 것은 acceptance이며, 지금까지 측정한 모든 셀을 정렬합니다.

| acceptance | 호스트 | 프롬프트 | 결과 |
| ---: | --- | --- | --- |
| 0.997 | M1 Ultra | 열거 | 동일 |
| 0.997 | M5 Max | 열거 | 동일 |
| 0.980 | M3 Ultra | 열거 | 동일 |
| 0.815 | M1 Ultra | 소스 코드 | 동일 |
| 0.784 | M5 Max | 소스 코드 | **diverge** |
| 0.733 | M3 Ultra | 소스 코드 | **diverge** |
| 0.544 | M3 Ultra | 산문 | **diverge** |
| 0.525 | M1 Ultra | 산문 | **diverge** |
| 0.489 | M5 Max | 산문 | **diverge** |

세 GPU 세대에 걸친 아홉 개 셀에서 뒤바뀜이 없고, 경계는 0.784와 0.815 사이에 있습니다. M5 Max 소스 코드 행은 이 순서가 제안된 *뒤에* 측정되었는데, 그 0.784가 당시 0.733과 0.815 사이의 공백에 속했기 때문입니다. 이 행은 diverge했고, 이는 순서가 예측한 바이며 공백을 좁혔습니다.

acceptance는 프롬프트가 대리로 작용한다는 뻔한 반론도 무너뜨립니다. 소스 코드는 하나의 프롬프트인데 양쪽 결과가 모두 나옵니다. M1 Ultra에서는 0.815로 동일하고, M5 Max에서는 0.784로, M3 Ultra에서는 0.733으로 diverge합니다. 판정은 프롬프트 텍스트가 아니라 그 조합이 해당 호스트에서 우연히 도달하는 acceptance를 따릅니다.

메커니즘은 이 페이지 앞부분에서 측정한 것과 같습니다. 두 커널을 위치별로 비교하면 전체 위치의 4.0%에서 불일치하지만 **decided 위치에서는 0.585%**입니다. 커널 차이는 타깃이 상위 두 후보 사이에서 거의 무관심했던 곳에서만 방출 토큰을 바꿉니다. 낮은 acceptance는 거의 무관심한 위치들의 흐름을 drafter 쪽에서 본 모습이므로, acceptance가 낮은 generation은 두 경로가 갈라질 수 있는 바로 그 위치들을 계속 지나가는 것입니다. 열거가 세대 17에서 살아남는 것은 커널이 거기서 일치하기 때문이 아니라, 불일치할 위치를 거의 제공하지 않기 때문입니다.

이를 임계값이 아니라 아홉 개 점의 순서로 받아들이십시오. 경계는 위치를 특정한 것이 아니라 구간으로 좁힌 것이며, 열 번째 측정이 그 구간 안의 어느 쪽에도 놓일 수 없다고 말해 주는 것은 여기에 없습니다.

어떤 호스트도 위의 경험 규칙이 예측하는 대로 동작하지 않습니다. 세대 13은 폭 12 미만에서 바이트 단위로 동일해야 하는데 프롬프트 하나는 그렇지 않고, 세대 15와 17은 폭 2부터 diverge해야 하는데 프롬프트 하나는 두 호스트 모두에서 그렇지 않습니다. 이 규칙은 op 수준에서는 성립하지만, acceptance 순서가 제시하는 이유로 모델 수준으로는 어느 방향으로도 이어지지 않습니다. generation이 두 커널이 불일치할 수 있는 위치에 도달하는지는 continuation의 속성이고, 하드웨어는 거기에 도달했을 때 무슨 일이 일어나는지만 결정합니다. 이것이 #1188의 논거입니다. probe는 하드웨어로 예측하는 대신 시작 시 조합에서 그 속성을 측정하며, probe되는 Qwen 조합은 확인한 두 호스트 모두에서 동일하게 나오고, M3 Ultra에서는 `qmv_wide`에서 실패한 뒤 프로세스에서 이를 비활성화하는 방식으로 그렇게 됩니다. arm당 generation 한 번으로 이 모든 것이 재현되며, M3 Ultra의 소스 코드 행은 22바이트 지점에서 갈라집니다.

프로세스를 narrow로 고정했을 때 나머지 부분이 치르는 비용은 [qmv-wide-pin-tax-m3ultra-2026-08-22](benchmark_results/qmv-wide-pin-tax-m3ultra-2026-08-22.md)(issue #1261)에서 측정되었습니다. 두 MTP 계열 모두 batch를 행별 `M = 1` forward로 decode하여 `qmv_wide`에 도달하지 않으므로, B = 2~8의 batched decode는 최대 1%만 손해를 봅니다. 실제 부수적 비용은 prompt-cache-hit 요청당 약 15 ms 하나이며, 채택된 짧은 접미사의 prefill이 qmv 구간에 놓이기 때문입니다.

처리하지 않으면 그 diff가 거짓말을 하게 만드는 것이 두 가지 있습니다. MTP arm은 drafter 로더 줄을 `Generating...` *뒤에*, 에코된 프롬프트 바로 다음에 출력하므로, 단순한 diff는 1바이트 지점의 divergence를 보고하지만 이는 배너일 뿐입니다. 비교 전에 제거하십시오. 그리고 Gemma 4는 `--show-reasoning`을 전달하지 않으면 reasoning 채널을 숨기므로, 예산을 거기에 쓰는 조합은 빈 문자열 둘로 비교되어 공허하게 통과합니다. Qwen 검사에는 바로 이 이유로 그 플래그가 필요합니다. 이 세 Gemma 프롬프트는 그 채널을 전혀 사용하지 않는데, 이것 또한 가정하지 않고 확인해야 할 사항입니다.

### 모델 자체 MTP head를 사용하는 Qwen 3.5 / 3.6 / 3.8

Qwen은 MTP head를 별도의 동반 체크포인트가 아니라 계열의 일부로 제공하며, 분리된 형태(`Qwen3.8-27B-MTP-bf16`, `-4bit`)이거나 타깃 안에 포함됩니다. 같은 프로토콜로, `qwen3.8-27b-4bit`와 `qwen3.8-27b-mtp-4bit`, 위의 코드 프롬프트, 300 토큰, arm당 샘플 여덟 개입니다.

| 호스트 | 경로 | decode tok/s | 속도 향상 |
| --- | --- | ---: | ---: |
| M5 Max (128 GB) | classic decode (drafter 없음) | 32.7 | 1.00x |
| M5 Max (128 GB) | MTP, 블록 3 (drafter가 선언한 폭) | 53.4 | **1.63x** |
| M3 Ultra (512 GB) | classic decode (drafter 없음) | 35.7 | 1.00x |
| M3 Ultra (512 GB) | MTP, 블록 3 (drafter가 선언한 폭) | 59.5 | **1.67x** |
| M1 Ultra (128 GB) | classic decode (drafter 없음) | 23.8 | 1.00x |
| M1 Ultra (128 GB) | MTP, 블록 3 (drafter가 선언한 폭) | 23.4 | **0.98x** |

M5 Max에서 MTP 수치는 50.8~55.2에 걸친 여덟 샘플의 중앙값으로, 같은 실행의 classic arm이 0.3%인 데 비해 8.2%의 분산입니다. 경합이었다면 두 arm이 모두 움직였을 것이므로, 이는 호스트가 아니라 조합에 속한 것입니다. `temperature 0`에서 acceptance와 유효 블록은 매 실행 동일하게 나오며, 달라지는 것은 적응형 B=1 컨트롤러(#333)가 초기 burst를 프로파일링할 때 어디에 정착하느냐입니다. 그럼에도 중앙값은 안정적이어서 세 번의 독립 sweep에서 1.61x, 1.64x, 1.63x로 읽힙니다. 그 호스트에서 이 조합의 단일 실행은 대략 1.55x~1.69x의 가치가 있으므로, 그보다 작은 변화는 결과가 아닙니다.

그 분산은 이 조합에서 이어지지 않은 유일한 부분입니다. M3 Ultra의 같은 sweep은 두 arm 모두 0.7%로 측정되었으므로, 위 구간의 폭은 조합만의 속성이 아니라 *그 호스트에서의* 조합의 속성입니다. M3 Ultra 행은 표면 그대로 읽고, 제3의 호스트에 대한 구간을 인용하기 전에 분산을 다시 측정하십시오. 컨트롤러는 호스트마다 프로파일링할 초기 burst 집합이 다릅니다.

그 두 비율에 대해 두 가지가 있습니다. 이 값은 바이트 동일성 보장을 **포함한 채로** 측정되었습니다. 정확성 probe가 M5 Max와 M3 Ultra에서 작동하여 `qmv_wide`를 끄고, 이는 verify forward에 약 17~20%의 비용을 부과합니다. 따라서 두 Qwen 행은 위의 Gemma 행이 치르지 않는 것을 이미 치르고 있으며, 이것이 그 행들이 더 낮은 주된 이유입니다. M3 Ultra에서 바이트 단위로 동일한 Gemma 코드 행은 1.83x로 Qwen의 1.67x와 비교되고, 빠른 커널의 Gemma 행은 2.16x로 읽힙니다. 그리고 블록 폭은 단순히 기본값이어서가 아니라 실제로 3~4에서 최적입니다. 타깃의 64개 레이어 중 48개가 GatedDeltaNet, 즉 토큰을 순서대로 처리하는 recurrence이므로, verify 비용은 attention 전용 타깃처럼 분할 상환되지 않고 블록에 거의 선형으로 증가합니다. 폭 5, 6, 8은 M5 Max의 이전 실행에서 48.0, 46.8, 35.7로 측정되었으며, 그곳에서 폭 3과 6 사이의 격차는 그 호스트의 실행 간 범위 안에 있었고 폭 8만 명확히 그 밖에 있었습니다.

M3 Ultra sweep은 그 범위가 삼켜 버린 것을 분리해 냅니다.

| 폭 | decode tok/s | 분산 | acceptance | verify당 방출 | 정점 대비 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 54.6 | 0.5% | 0.875 | 1.869 | -8.2% |
| 3 | 59.5 | 0.9% | 0.753 | 2.492 | **정점** |
| 4 | 57.4 | 0.7% | 0.689 | 3.051 | -3.5% |
| 5 | 52.1 | 0.6% | 0.607 | 3.398 | -12.4% |
| 6 | 49.4 | 0.4% | 0.546 | 3.691 | -17.0% |
| 8 | 40.7 | 0.3% | 0.412 | 3.833 | -31.6% |

"3~4에서 최적"은 여기서 선언된 폭을 되풀이하는 것이 아니라 측정입니다. 3이 정점이고 4가 3.5% 뒤에 있으며, 표의 모든 격차는 측정 분산 0.9%를 훨씬 벗어납니다. M5 Max가 해소하지 못한 3~6 사이의 격차도 포함합니다. 하락폭도 같은 폭 범위에서 Gemma 조합보다 가파른데, 이는 GatedDeltaNet recurrence가 예측하는 바입니다. verify 비용이 블록 전체에 분할 상환되지 않고 블록에 따라 커집니다.

M1 Ultra는 이 조합에서 선언된 폭에서는 본전이고 그 외의 모든 폭에서는 손해입니다. acceptance는 이 페이지의 어떤 것보다 높지만(블록 3에서 0.855, verify당 2.694 토큰 방출) 그 호스트의 round 비용을 넘지 못합니다.

| 폭 | decode tok/s | 분산 | acceptance | verify당 방출 | 정점 대비 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 23.6 | 1.2% | 0.935 | 1.929 | **정점** |
| 3 | 23.1 | 1.3% | 0.855 | 2.694 | -2.3% |
| 4 | 21.9 | 1.1% | 0.781 | 3.322 | -7.1% |
| 5 | 20.1 | 1.7% | 0.673 | 3.691 | -14.8% |
| 6 | 17.7 | 0.7% | 0.567 | 3.785 | -25.3% |
| 8 | 14.2 | 2.1% | 0.424 | 3.934 | -40.0% |

classic arm 23.79에 대해 정점에서 0.99x, 폭 8에서 0.60x입니다. "3~4에서 최적"은 이 호스트에 대한 설명이 아닙니다. 정점은 측정된 가장 좁은 폭이고, 폭을 올릴 때마다 자체 분산 밖에서 명확한 손해이며, `--draft-block-size`를 전달하지 않으면 3이 되어 폭 3 행의 acceptance와 verify당 방출로 23.3 tok/s를 측정합니다. 따라서 위 표의 0.98x가 튜닝하지 않은 사용자가 얻는 값입니다.

이 여섯 폭에 걸쳐 피팅한 round 비용은 `0.46 + 0.771 K` classic step이며 가장 큰 잔차는 0.06입니다. 같은 호스트에서 Gemma 조합의 `1.35 + 0.346 K`와 비교하면, 위치당 비용은 두 배를 약간 넘게 늘었고 고정 부분은 줄었는데, 이는 GatedDeltaNet 주장을 단언하는 대신 값을 매긴 것입니다. 토큰을 순서대로 처리하는 recurrence는 블록 위치가 하나 늘 때마다 거의 전액을 부과하는 반면, attention 전용 타깃은 이를 분할 상환합니다. 호스트 효과와 타깃 효과는 합쳐지며 이 조합은 세대 13 호스트와 recurrent 타깃 둘 다를 지니고 있으므로, 이 페이지에서 모든 폭에서 손해를 보는 유일한 항목입니다.

세대 15+ 호스트가 지닌 두 주의 사항은 여기에는 해당하지 않습니다. 세대 13은 `qmv_wide`를 사용하지 않으므로 probe는 그대로 통과하고 17~20%의 비용은 치르지 않으며, 두 arm의 분산은 M5 Max의 8.2%가 아니라 0.2%와 1.3%로 측정되었습니다. 이것은 `benchmark_results/qwen38-mtp-m1ultra-2026-08-16.md`에서 bf16 drafter로 그 호스트에서 0.59x~0.70x를 측정한 것과 같은 조합입니다. drafter를 4-bit로 quantization한 것(#1185 Phase 3)이 손익분기까지 끌어올렸지만 그 너머로 옮기지는 못했습니다. 여기서 측정된 어떤 것도 세대 13에서 이 조합을 활성화해야 한다고 주장하지 않습니다.

### Gemma 4 12B 장문 context 정확성

[M1 Ultra 12B 조사](benchmark_results/gemma4-12b-mtp-exactness-m1-ultra-2026-10-15.md)는 짧은 시작 draw는 통과하는데도 긴 buffered verify 불일치가 재현됨을 보입니다. sliding-ring 순서, mask 없는 단일 query full attention, classic prefill chunking을 맞추면 측정된 폭 4 경로가 복원됩니다. 256개의 teacher-forced 위치(그중 155개 decided)에서 불일치가 0이고, 1,238~1,638 토큰 프롬프트 뒤의 256 토큰 completion 세 개가 바이트 단위로 동일합니다. active-Metal 12B 시작 guard는 이제 긴 buffered draw를 포함하며, 지원되지 않는 레이아웃은 정확성을 우회하지 않고 여전히 거부하고, 기존 CUDA/31B 정책은 유지됩니다. 세 번의 counterbalanced release 반복은 27개 출력 해시를 모두 유지하지만, 강제 폭 MTP는 M1 Ultra에서 더 느립니다(짝지은 중앙값 비율 0.837x/0.903x/0.815x). scheduler-setup에서 probe 통과까지의 간격은 1.245 s에서 7.041 s로 늘어납니다. 적격성/수익성 기본값은 변경되지 않았습니다. 이것은 M1 Ultra 증거이며, M5나 CUDA 검증이 아닙니다. 고정된 검증 스위치, 산출물 출처, 분산, 측정 한계는 해당 기록을 참조하십시오.

### Gemma 4 31B + bf16 assistant

**현재 정확성 (2026-09-26, #1983/#1279):** QAT와 non-QAT 31B 모두 이제 M5 Max에서 자동 narrow 커널 재시도 후 `MLXCEL_MTP_ALLOW_INEXACT` 없이 시작 probe를 통과합니다. non-QAT의 폭 2, 3, 4가 모두 통과합니다. 이 수정은 full-attention query/mask 기하, sliding-cache 물리적 ring 순서, classic prefill chunking을 일치시킵니다. singleton 선형 verify로 제한되며, batched MTP는 classic으로 fallback하고, 비선형 tree round는 비활성화되어 dispatch는 지원되는 선형 경로를 사용합니다. 장문 context logit 지표, greedy 동등성, 현재 test-fast 처리량은 [매칭된 QAT/non-QAT 측정](benchmark_results/gemma4-31b-mtp-exactness-2026-09-26.md)을 참조하십시오. 정적 하드웨어 선호는 측정된 정확성과 적응형 수익성에 계속 종속됩니다.

현재 M5 Max / 128 GiB, test-fast 프로파일, 폭 4, 256 토큰 출력 세 개 (시나리오 및 arm당 관측 하나):

| 타깃 | Classic 이후 (tok/s 범위) | MTP (tok/s 범위) | 짝지은 MTP/classic 범위 | Greedy 동등성 |
| --- | ---: | ---: | ---: | --- |
| 31B QAT 4-bit | 8.97–17.28 | 10.87–19.76 | 0.85–1.21x | 3/3 바이트 동일 |
| 31B non-QAT 4-bit | 12.20–13.77 | 15.99–17.80 | 1.18–1.38x | 3/3 바이트 동일 |

실행 순서에 따른 drift와 열 압박이 처리량 결론을 제한합니다. 이것들은 관측이며 보장된 속도 향상이 아닙니다. 개별 샘플, baseline 비교, 호스트 증거는 링크된 보고서를 참조하십시오.

아래 M3 Ultra 수치는 **정확성 gate와 이 수정 이전의 과거 측정**입니다. 그 구현의 처리량을 설명할 뿐 오늘의 정확한 경로의 속도를 입증하지 않습니다.

31B 텍스트 타깃은 batch가 가능하며, 이는 B=1 정적 gate(`mtp_b1_default`)가 관장하는 경우입니다. issue #1217 이전에는 이 gate가 `has_neural_accelerator`가 성립하는 곳에서만 singleton 경로를 실행했는데, 이 조합의 속도 향상이 batched(B>1) verify 윈도우에서 나오고 bf16 assistant의 단일 스트림 acceptance는 추가 drafter forward를 상쇄하기에 너무 낮다는 해석에 근거한 것이었습니다. 그 해석의 두 절반 모두 #1194, #1199, #1203, #1208 및
#1215 이전에 측정된 것이며, 재측정에서 어느 쪽도 살아남지 못했습니다.

M3 Ultra, 2026-08-20, 블록 4, greedy, 위의 프로토콜(`scripts/bench_speculative.sh gemma31b`)에 따름:

| 호스트 | 출력 | 토큰 | 블록 | acceptance | 방출/verify | classic | MTP | 속도 향상 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M3 Ultra (512 GB) | 열거 | 400 | 4 | 1.000 | 3.990 | 31.5 | 83.6 | **2.65x** |
| M3 Ultra (512 GB) | 소스 코드 | 300 | 4 | 0.882 | 3.646 | 31.8 | 76.6 | **2.41x** |
| M3 Ultra (512 GB) | 산문 | 400 | 4 | 0.656 | 2.956 | 31.7 | 61.9 | **1.95x** |

단일 스트림 acceptance는 너무 낮지 않은 0.66~1.00이며, singleton 경로는 모든 프롬프트에서 이득을 봅니다. 세 행 모두 verify round를 classic decode step 1.51~1.52개로 측정하고 토큰 2.96~3.99개를 방출하므로, 손익분기를 대략 두 배로 넘깁니다. 폭 sweep은 `0.83 + 0.170 K` classic step으로 피팅되며 가장 큰 잔차는 0.06이고, 폭 5에서 정점을 찍으며 폭 4는 분산 안에서 동률입니다.

같은 호스트에서 12B 조합의 `1.14 + 0.090 K`와 비교하면, bf16 drafter는 블록 위치가 하나 늘 때마다 약 1.9배의 비용이 들고 두 직선은 K = 4 근처에서 교차하는데, 블록 4의 round 비용이 일치하는 유일한 이유가 이것입니다. 이 두 조합 사이에서 다른 폭의 round 비용을 옮겨 쓰지 마십시오.

그때 도입된 정적 gate는 Apple GPU 세대를 읽습니다. 세대 15(M3, M4, M5)부터는 켜지고, 세대 13(M1, M2)은 classic decode입니다. M4는 공유하는 `use_qmv_wide` dispatch로 묶었을 뿐 측정하지는 않았습니다. 세대 13은 호스트가 없어 재측정하지 못했으며, 위의 기울기 비율을 그 `1.35 + 0.346 K`에 적용하면 블록 4 round는 classic step 약 3.6개가 되어 방출 토큰이 겨우 충당하는 수준이므로, 당초의 0.75~0.96x는 낡은 것이 아니라 타당한 것으로 읽히며 계속 거부됩니다. 전체 기록과 방법은 `benchmark_results/mtp-b1-gate-m3ultra-2026-08-20.md`에 있습니다.

이 조합은 `speculative_bench`(`REACHABLE_PAIRINGS`)에도 연결되어 있으며, `gemma-4-31b-it-4bit`와 `gemma-4-31B-it-assistant-bf16` 체크포인트가 모델 저장소에 있으면 실행됩니다. #1613 이전에는 체크포인트의 존재만으로는 충분하지 않았습니다. `run_mtp`가 `LoadedModel::Gemma4Unified`만 매칭했기 때문에 31B 타깃(`Gemma4VLM`으로 로드됨)은 두 체크포인트가 디스크에 있어도 로드 후에 거부되었습니다. 하니스는 이제 `src/server/batch/speculative_burst.rs`와 같은 방식으로 변형별로 어댑터를 선택하므로, Gemma 4 text, VLM, Unified 래퍼와 Qwen 3.5 text, MoE, VLM 래퍼를 모두 벤치할 수 있고, 카탈로그에는 `qwen3_5_mtp` head에 대한 Qwen 3.8 27B 조합도 포함됩니다.

### 적응형 B=1 MTP 정책

issue #333 이후 서버는 B=1 MTP 경로를 정적인 하드웨어별 gate만으로 결정하지 않습니다. 각 (타깃, drafter, 하드웨어) 조합의 처음 몇 번의 B=1 burst(acceptance 길이, verify 지연, drafter 지연, batch 크기, 프롬프트 형태)를 프로파일링하고 데이터 기반 판정으로 정착합니다. 명백히 유리한 프로파일은 정적 gate가 거부했을 곳에서도 MTP를 활성화하고, 명백히 불리한 프로파일은 거부하며, 모호한 프로파일은 위의 정적 하드웨어별 기본값을 유지합니다. 정착된 판정(활성화/거부와 대략적인 acceptance 비율이며 프롬프트 데이터는 절대 포함하지 않음)은 `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy/` 아래에 캐시되므로 프로파일링은 조합당 한 번의 비용이며 재시작 후에도 유지됩니다. MTP는 수학적으로 정확한 상태를 유지합니다. 정책은 언제 실행할지만 선택하므로, 위의 정확성 probe가 확립하는 바이트 동일성을 만들지도 없애지도 않습니다. `MLXCEL_ENABLE_MTP_B1`은 어느 방향으로든 결정을 고정하며(프로파일링은 억제됨), `MLXCEL_MTP_ADAPTIVE=0`은 #333 이전의 정적 gate를 복원합니다. 벤치마크 수치를 기록할 때는 프로파일링 구간을 버리고 정착된 판정의 정상 상태(steady state)를 보고하십시오.

## 권장 출력 레이아웃

릴리스를 게시하기 전에 전용 디렉터리 아래에 벤치마크 산출물을 추가하십시오. 예:

```text
benchmarks/
  2026-05-08_m1-ultra_text.csv
  2026-05-08_m1-ultra_vlm.csv
  README.md
```

각 CSV는 기계가 읽을 수 있어야 하며, 방법론, 제외 항목, 알려진 실패를 설명하는 짧은 Markdown 노트가 함께 있어야 합니다.

## 주의 사항

- **열 조건이 중요합니다.** Apple Silicon의 decode 처리량은 지속 부하에 따라 달라집니다. cooldown과 실행 순서를 기록하십시오.
- **MLX pin이 중요합니다.** 고정된 MLX 커밋이 바뀌면 커널 선택이 달라질 수 있습니다.
- **VLM 비교는 텍스트 비교와 별개입니다.** 비전 전처리, 이미지 해상도, 프롬프트 구성은 계열마다 다릅니다.
- **CUDA 수치는 GPU 간에 호환되지 않습니다.** 결과와 함께 SM 타깃과 드라이버/툴킷 버전을 게시하십시오.
- **`models/`의 중복 체크포인트는 호스트별 산물입니다.** `bench_decode.sh
  all`은 이제 체크포인트 동일성으로 중복을 제거하므로(위 참조), 심볼릭 링크나 재다운로드된 별칭이 속하는 그룹은 로컬 `MODELS_DIR`에 실제로 무엇이 있는지에 달려 있습니다. CSV 행의 survivor 이름은 어느 별칭이 upstream에서 정식(canonical)인지에 대한 주장이 아니라, 이 특정 sweep이 어느 디렉터리를 측정했는지에 대한 것일 뿐입니다.
