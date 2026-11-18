# macOS 메모리 가드레일

Mac에서 모델이 통합 메모리의 대부분을 차지할 때 mlxcel이 어떻게 동작하는지, 그리고 어떤 스위치가 이를 제어하는지 설명합니다(이슈 #2883).

## 왜 총 RAM 용량이 잘못된 수치인가

버퍼의 GPU wiring(고정 할당)은 마지막 커맨드 버퍼 이후 몇 초 뒤에 끝납니다. 서버가 유휴 상태로 있는 동안 다른 프로그램이 메모리를 필요로 하면 macOS는 모델의 메모리를 압축하거나 스왑으로 내보낼 수 있으며, 다음 요청이 그 지연 비용을 치르게 됩니다. 총 RAM 용량과 Metal의 권장 작업 셋(working set)은 머신이 최대로 담을 수 있는 용량을 의미할 뿐, 압축 없이 지금 당장 내어줄 수 있는 용량을 의미하지 않습니다.

## 현재 사용 가능한 메모리와 예약량

macOS에서만 `mlxcel inspect` 및 `--estimate-memory`가 다음 내용을 출력합니다:

| 라인 | 의미 |
|------|------|
| 총 RAM | `hw.memsize` |
| Metal 작업 세트 | Metal 권장 작업 셋 |
| 현재 사용 가능 | `host_statistics64`의 `free + purgeable + min(inactive + speculative, file-backed)` 페이지 (`free`에서 speculative 페이지 제외, 커널의 `free_count`에는 포함되어 있음). 다른 프로그램의 비활성 익명 메모리(inactive anonymous memory)는 제외됩니다. macOS는 이를 압축하거나 스왑해야만 해제할 수 있기 때문입니다. |
| macOS 예약 | `max(7 GiB, 15% of RAM)` |
| 이번 로드 | 가중치 + KV cache + 활성화(activations) |

로드량이 '현재 사용 가능량 - 예약량'을 초과하면 `WARNING: memory available now`로 시작하는 줄이 관련 수치를 안내합니다. 이는 로드를 절대 거부하지 않으며 `FITS` 판정을 변경하지도 않습니다.

이 수치와 예약량은 요청 시 동적으로 증가하는 모든 스토어를 위해 `mlxcel_core::host_memory`(`host_memory_status`, `HostMemoryStatus`, `AvailabilityGate`)에 구현되어 있습니다.

## 풋프린트 보고

로드 후 서버는 `Process footprint after load`를 로그에 남깁니다. `--metrics`를 지정하면 macOS에서 `/metrics`에 다음 게이지가 추가됩니다:

| 메트릭 | 의미 |
|--------|------|
| `mlxcel_process_phys_footprint_bytes` | 프로세스의 `phys_footprint` |
| `mlxcel_process_compressed_bytes` | 압축기 또는 스왑에 들어 있는 프로세스의 바이트 수. 유휴 상태 중 증가는 유휴 압축(idle compression)을 의미합니다. |
| `mlxcel_mlx_active_bytes`, `mlxcel_mlx_cache_bytes` | MLX 할당자 카운터 |
| `mlxcel_weights_wired_bytes` | 마지막 로드 시 `MLXCEL_WIRE_WEIGHTS`에 의해 배선(wired)된 바이트 수 |

## 스위치

- `MLXCEL_WIRED_LIMIT`는 변경되지 않았습니다(기본값 `max`).
- `MLXCEL_WIRE_WEIGHTS=1`은 로드 후 평가된 가중치를 `mlock`으로 배선합니다(macOS 전용, 기본값 off). 가중치는 1 GiB 청크 단위로 배선됩니다. 첫 번째 청크 전과 각 청크 처리 후 시스템 배선 총합을 'RAM - 예약량'과 비교하며, 이를 초과하는 청크는 배선이 해제되고 패스가 중단됩니다. 실패를 포함한 결과가 기록됩니다. 기본 설정인 즉시 평가(eager) `MLXCEL_WEIGHT_MATERIALIZATION`이 필요합니다. 멀티모달 타워는 배선되지 않으며, 첫 번째 토큰에서 실행되는 리패킹(전문가 스태킹, dtype 변환)은 배선된 원본을 배선되지 않은 결과로 교체합니다.

## 증가하는 스토어

`qwen4_exp`의 외부 PLE 행 캐시는 각 64 MiB 확장 단계 전에 가용성을 확인하고, macOS를 예약량 미만으로 밀어붙이는 대신 확장을 멈추고 축출을 수행합니다. 이 확인은 250 ms 동안 하나의 쿼리를 재사용합니다. KV 프롬프트 캐시 스토어는 게이트되지 않습니다. 엔트리의 배열이 삽입 시점에 이미 존재하고 스토어가 바이트 예산으로 제한되어 있으므로 거부할 확장 단계가 없기 때문입니다.

프롬프트 캐시 스냅샷 스토어도 동일한 방식으로 확장됩니다. 바이트 예산 미만인 동안, 상주 크기를 늘리게 될 각 스냅샷 삽입은 예약량 대비 가용성을 다시 확인하며 호스트 여유가 없을 때 저장 대신 건너뜁니다. 예산 한도에 도달한 스토어는 엔트리만 교체하므로 거부되지 않습니다.

## 측정된 내용

아직 없습니다. soak 테스트는 아직 실행되지 않았으며, 이들 가드레일은 증거를 바탕으로 한 기본값으로 채택되지 않았습니다. 즉, `MLXCEL_WIRE_WEIGHTS`는 꺼져 있고 `MLXCEL_WIRED_LIMIT`는 기본값을 유지하며 로드는 거부되지 않습니다. 조용한 머신에서 측정하려면 다음과 같이 실행하십시오:

```bash
cargo build --release --features metal,accelerate
scripts/macos_memory_soak.sh models/Qwen3.8-Flash-Next-OptiQ-2bit 40G
scripts/macos_memory_soak.sh --wire-weights --wired-limits default models/Qwen3.8-Flash-Next-OptiQ-2bit 40G
```

첫 번째 명령은 기본값, `none`, `60G`에서의 `MLXCEL_WIRED_LIMIT`를 다루며, 설정당 하나의 CSV와 쓰기 전후의 첫 토큰 생성 시간(TTFT)을 담은 `summary.md`를 `docs/benchmark_results/` 아래에 작성합니다. `--dry-run`은 모델을 로드하거나 파일을 작성하지 않고 계획을 출력하며, `--allocate 16G`는 쓰기 작업 옆에 할당을 유지합니다. 64 GB Mac에서는 비례하여 더 작은 모델과 더 낮은 명시적 한도(예: `--wired-limits default,none,30G`)를 사용하십시오.
