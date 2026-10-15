# Cascade attention (공유 프롬프트 접두사)

Cascade attention은 여러 동시 시퀀스가 공유하는 프롬프트 접두사를 시퀀스마다가 아니라 **decode step마다 한 번만** 계산합니다. 이는 mlxcel이 이미 수행하는 저장소 측 중복 제거에 대응하는 연산 측 기법입니다.

상태: opt-in이며 기본값은 꺼짐입니다. 정확성은 Metal에서 f32 및 f16 풀을 사용해 호스트 f64 레퍼런스와 flat fused launch 모두에 대해 검증되었습니다. 실제 서빙 워크로드에서의 처리량은 아직 측정되지 않았으며, 이것이 꺼져 있는 이유입니다. [측정하기](#measuring-it)를 참고하십시오.

## 현재 중복되는 것

paged 풀은 블록의 참조 횟수를 세며, `CachePool::clone_detached_paged_prefix`는 동일한 APC 엔트리를 채택하는 모든 시퀀스에 *동일한* 블록 id를 넘겨줍니다. 따라서 하나의 8K 시스템 프롬프트 뒤에 있는 클라이언트 네 개는 해당 프롬프트의 KV를 네 개가 아니라 하나의 물리적 사본으로 보유합니다.

그러나 연산은 중복 제거되지 않습니다. Batch decode에서는 모든 시퀀스가 공유 구간에 대해 독립적으로 attention을 수행하므로, 읽는 모든 바이트가 동일하더라도 해당 구간에 소비되는 attention 대역폭은 batch 크기에 비례해 늘어납니다. 8K 공유 접두사와 batch 4에서는 attention이 읽는 KV 바이트의 4분의 3이 동일한 페이지를 다시 읽는 것입니다.

## 분해

Softmax 상태는 합성이 가능합니다. 키 범위를 서로소인 부분들로 나누고 각 부분이 자신의 정규화된 출력 `V_i`와 log-sum-exp `LSE_i`를 산출한다면,

```
m = max_i LSE_i
w_i = exp2(LSE_i - m)
V = (sum_i w_i V_i) / (sum_i w_i)
```

는 전체 범위에 대한 attention과 정확히 같습니다. 이는 CTA 간 split-KV decode가 사용하는 것과 같은 대수이며 같은 커널에서 실행됩니다.

따라서 cascade decode step은 다음과 같습니다.

1. **Level 0**, 공유 구간: 하위 그룹 전체에 대해 한 번만 attention을 수행합니다.
2. **Level 1**, 각 요청의 비공개 접미사: 요청별로 attention을 수행합니다.
3. 각 멤버의 두 상태를 짝지어 한 번 병합(merge)합니다.

두 level이 flat launch가 읽는 키 범위를 정확히 분할하므로, 결과는 f32 반올림 오차를 제외하면 flat 경로와 비트 단위로 비교 가능합니다.

## Level 0이 여러 query head를 가진 하나의 요청인 이유

이 부분이 동일한 작업을 하는 두 번의 launch가 아니라 실제 이득이 되게 만드는 핵심입니다.

Fused v2 partial 커널은 `(request, page tile)` 쌍 하나에 대해 하나의 KV head에 속한 모든 query head를 하나의 threadgroup에 할당합니다. 그 threadgroup은 각 K와 V 원소를 한 번만 로드하여 자신이 맡은 모든 query head에서 재사용합니다. 동일한 페이지 목록을 `M`개의 별도 요청에 넘기면 구간을 `M`번 읽게 되지만, `M`배 많은 query head를 가진 **하나의** 요청에 넘기면 한 번만 읽습니다.

커널은 query head `h`를 KV head `h / NRep`에 매핑하므로 쌓는 순서가 결정적입니다. `G = Hq / Hkv`, 멤버 `m`, 그룹 슬롯 `g`일 때 level-0 head 인덱스는 다음과 같아야 합니다.

```
h0 = kv_head * (M * G) + m * G + g
```

이는 입력 시 `[M, Hkv, G, D] -> [Hkv, M, G, D]` 전치이고 출력 시 그 역변환입니다. 대신 멤버 우선(member-major)으로 쌓으면 모든 query head가 조용히 잘못된 KV head를 읽게 됩니다. shape는 같고 launch도 성공하지만 답이 틀립니다. `paged_v2::cascade_launch_tests::member_major_head_stacking_reads_the_wrong_kv_head`가 이를 고정하는 음성 대조군(negative control)입니다.

이 기능을 위해 작성되거나 수정된 커널은 없습니다. 두 level 모두 일반적인 v2 launch이며, 병합은 이슈 #898의 `paged_attention_merge_states`입니다. 그 계약(`v_in [N, H, D]`는 이미 정규화됨, `lse_in [N, H]`는 **log2** 단위, `o_indptr [M + 1]`은 연속된 행을 묶음)은 조정되지 않고 그대로 준수됩니다.

## 공유 구간을 찾는 방법

탐지는 pool 참조 횟수가 아니라 CSR 페이지 테이블의 `indices`를 요청 간에 위치별로 비교합니다. 하나의 `PagedCsrView` 안에서 살아 있는 모든 블록 id는 정확히 하나의 물리적 pool 행으로 해석되며 두 개의 살아 있는 블록 id가 하나의 행을 공유하지 않습니다. 따라서 페이지 `i`가 같은 행인 두 요청은 같은 블록을 보유하며, 이는 `refcount > 1`이고 바이트가 동일함을 의미합니다. 페이지 테이블에서 읽어내면 판단이 `mlxcel-core` 안에 머물고, 스케줄러 배관이 필요 없으며, 서버가 만들었다고 아는 공유뿐 아니라 어떤 경로로 발생했든(APC 채택, 포크된 시퀀스, 향후 블록 해시 중복 제거) 공유를 잡아낼 수 있습니다.

후보는 처음 방출되는 페이지별로 그룹화되며, 페이지 읽기를 가장 많이 제거하는(`shared_pages * (members - 1)`) 하위 그룹이 선택됩니다.

## 거부되는 경우와 그 이유

| 조건 | 이유 |
|---|---|
| 윈도우가 페이지 중간에서 시작하는 요청(`first_page_offset != 0`) | sliding window가 첫 페이지 중간까지 잘라냈으므로 구간을 더 이상 전체 페이지 단위로 표현할 수 없습니다. 이러한 요청은 하위 그룹에서 제외되며, 나머지는 여전히 cascade할 수 있습니다. |
| 멤버의 마지막 페이지에 도달하는 구간 | 요청의 마지막 페이지만 부분적으로 채워질 수 있는데 level 0은 자신의 페이지가 가득 찼다고 선언합니다. 구간은 모든 멤버에 대해 `pages - 1`로 제한됩니다. |
| 공유 페이지 수가 `MLXCEL_CASCADE_MIN_SHARED_PAGES`보다 적음 | 추가 launch 비용이 그것으로 제거되는 중복 읽기보다 큽니다. |
| 멤버 수가 `MLXCEL_CASCADE_MIN_MEMBERS`보다 적음 | 멤버가 하나이면 중복이 없습니다. |
| Logit soft-cap, 다중 token step, pool 기반이 아닌 캐시, 다중 slab 레이어 | cascade를 확인하기 전에 #899 whole-batch 경로에서 상위 단계에서 거절됩니다. |

선택된 하위 그룹 밖의 시퀀스도 해당 step에서 제외되지 않습니다. 이들은 level 1에서 전체 페이지 범위를 유지하며 병합 그룹은 단일 행이고, 병합 커널은 이를 항등 연산으로 처리합니다. 하나의 launch로 혼합 batch를 처리합니다.

## 플래그

| 변수 | 기본값 | 의미 |
|---|---|---|
| `MLXCEL_CASCADE_ATTENTION` | off | `1`/`true`/`on`/`yes`는 cascade를 활성화하고, `0`/`false`/`off`/`no`는 kill switch입니다. 설정되지 않은 경우를 포함해 그 외의 값은 배포된 기본값을 따릅니다. |
| `MLXCEL_CASCADE_MIN_SHARED_PAGES` | `16` | 하위 그룹이 공유해야 하는 전체 페이지 수. 기본 블록 크기 32에서 16페이지는 512 token입니다. |
| `MLXCEL_CASCADE_MIN_MEMBERS` | `2` | 구간을 공유해야 하는 시퀀스 수. |

세 변수 모두 프로세스당 한 번만 읽습니다. `MLXCEL_CASCADE_ATTENTION`이 설정되지 않으면 decode 경로는 페이지 테이블 스캔을 전혀 수행하지 않으며 flat launch는 #903 이전 동작과 바이트 단위로 동일합니다.

## 어떤 경로가 실행되었는지 확인하기

whole-batch decode는 서로 다른 결과 각각의 첫 launch를 `info` 수준으로 알립니다.

```
INFO ...: paged decode v2: fused v2 cascade launch (batch 4, 4 of them sharing 256 pages / 8192 KV tokens read once; 8 shared-span chunks, 4 suffix chunks)
INFO ...: paged decode v2: flat v2: the cascade decomposition was planned but its launch failed (...)
```

그리고 `/metrics`는 그 분할을 제공합니다.

```
mlxcel_paged_decode_launches_total{path="fused_v2"} 12288
mlxcel_paged_decode_launches_total{path="gather"}   0
mlxcel_paged_decode_launches_total{path="cascade"}  12288
mlxcel_cascade_shared_tokens_total     100663296
mlxcel_cascade_member_sequences_total  49152
mlxcel_cascade_failures_total          0
```

`cascade_shared_tokens_total / cascade_launches`는 평균 공유 구간 길이이고 `cascade_member_sequences_total / cascade_launches`는 평균 하위 그룹 크기입니다. `cascade`는 `fused_v2`의 부분집합이므로, `fused_v2`가 증가하는데 cascade 횟수가 0이면 기능이 활성화되었지만 한 번도 작동하지 않는다는 뜻입니다. 공유되는 것이 없거나, 임계값이 워크로드가 실제로 공유하는 구간보다 높은 경우입니다.

## 측정하기

레이어 수준 하니스는 공유 접두사 길이를 batch 크기에 대해 스윕하고, 각 셀마다 해당 방식을 설명하는 launch 통계를 출력합니다.

```bash
cargo run --release --features metal,accelerate --example cascade_attention_bench -- \
    --shared 2048,8192 --batches 4,8 --tail 256 --steps 200 --warmup 40 --reps 5
```

또한 공유가 없는 경우의 오버헤드 게이트도 실행합니다. 아무것도 공유되지 않는 동일한 shape를 flat launch 앞에 탐지 스캔을 둔 경우와 두지 않은 경우로 시간을 측정합니다.

종단 간으로는 prompt cache를 켠 상태로 실행 중인 `mlxcel-server`에 대해 다음을 수행합니다.

```bash
python3 scripts/bench_serving_concurrency.py \
    --shared-prefix-tokens 2048 --prompt-tokens 32 \
    --concurrency 4,8 --max-tokens 256 --metrics
```

`--shared-prefix-tokens`는 모든 클라이언트가 동일한 시스템 프롬프트를 보내고 짧은 사용자 질문만 다르게 하여, 참조 횟수가 있는 공유 블록을 만들어냅니다. `--metrics`는 각 level 전후로 `/metrics`를 스크레이프하여 paged decode 경로 분할을 보고하므로, 실제로 cascade를 실행하지 않은 level은 가정되지 않고 그렇다고 표시됩니다. 같은 서버 빌드에서 `MLXCEL_CASCADE_ATTENTION=1`을 설정한 경우와 설정하지 않은 경우로 동일한 명령을 실행하여 전/후 쌍을 얻으십시오.

결과는 [`benchmarks.md`](benchmarks.md)의 형식에 따라 `docs/benchmark_results/cascade-attention-<hw>-<date>.md` 아래에 기록하십시오. 그러한 파일이 존재하기 전까지 `paged_v2::cascade::DEFAULT_CASCADE_ENABLED`의 기본값은 `false`로 유지됩니다.

## 제한 사항

- 공유 level은 하나입니다. 중첩된 공유(공통 시스템 프롬프트와 테넌트별 하위 접두사)는 세 번째 level이 필요하며 범위 밖입니다.
- 탐지는 decode step마다 레이어마다 실행됩니다. 이 탐지가 읽는 페이지 테이블은 이미 step 간에 캐시되어 있지만 스캔 자체는 메모이제이션되지 않습니다. 그 비용을 제한하는 것은 위의 오버헤드 게이트입니다.
- Level-0 launch는 멤버 수에 대해 JIT 특수화되므로, 새로운 하위 그룹 크기마다 첫 decode step은 커널 컴파일을 한 번 치릅니다.
- 두 level의 청크 크기는 휴리스틱에서 가져옵니다. #906 autotuner는 어느 cascade 지오메트리에도 연결되어 있지 않습니다.
- CUDA는 테스트되지 않았습니다. 두 level 모두 기존 v2 커널을 재사용하는데, 그 CUDA 본문은 이미 #898에서 검증되지 않은 채로 배포되었습니다. 여기서 새로운 CUDA 코드를 추가하지는 않지만, 이를 검증하지도 않습니다.
