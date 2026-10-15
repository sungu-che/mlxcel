# 페이지 간접 참조를 통한 fused sparse decode

Sparse-attention 모델은 쿼리가 어떤 KV 위치를 볼 수 있는지 결정하지만, 과거에는 그럼에도 dense 비용을 그대로 지불했습니다. 이 문서는 mlxcel이 그 선택 결과를 fused v2 decode 커널을 위한 페이지 테이블로 바꾸는 방법, 어떤 패밀리에 연결되어 있고 어떤 패밀리에는 연결되어 있지 않은지와 그 정확한 이유, 그리고 벤치마크 출력으로 실제로 어떤 경로가 실행되었는지 확인하는 방법을 설명합니다.

Issue: #904, epic #909의 일부. #898(CSR 페이지 테이블과 v2 커널) 및 #899(프로덕션 디스패치 정책)에 의존합니다.

## 환원(reduction)

v2 decode 커널은 요청 `r`의 가시 토큰 `i`를 `indices[indptr[r] + (first_page_offset[r] + i) / page_size]`를 통해 해석합니다. 이 행들이 인접하거나, 정렬되어 있거나, 완전하다고 가정하지 않습니다. 이들은 물리적 풀(pool) 행이며 커널은 단순히 그것을 읽을 뿐입니다. 따라서 sparse 선택은 새로운 커널이 아니라 **더 짧은 페이지 목록**입니다.

`page_size = 1`에서는 각 페이지가 토큰 하나를 담고, `indices`는 선택된 위치 목록 그 자체이며, 커널의 페이지 간접 참조는 attention 루프 안에서 실행되는 토큰별 gather가 됩니다. 어느 시점에도 gather된 복사본은 만들어지지 않으며, 선택되지 않은 위치에 대해서는 아무것도 계산되지 않습니다.

## 연속 캐시를 페이지 풀로 주소 지정하기

이 issue가 대상으로 하는 두 패밀리 모두 `PagedBlockPool`을 사용하지 않습니다. 둘 다 저장소가 `[B, H, Cap, D]` row-major 할당인 dense `KVCache`를 사용합니다. `ContiguousCacheLayout`(`src/lib/mlxcel-core/src/cache/sparse_csr.rs`)이 둘을 연결합니다.

- `page_size = 1`이고 풀 KV 헤드가 하나이면 커널의 주소 계산은 `row * D + d`로 축약되므로, 할당의 풀 뷰는 `[B * H * Cap, 1, 1, D]`이며 **순수한 reshape**입니다.
- reshape 대상은 `update_and_fetch`가 반환하는 윈도우가 아니라 **할당**입니다. 그 윈도우는 step 단위로 패딩된 버퍼의 토큰 축 슬라이스이므로, 이를 reshape하면 캐시 전체를 복사하게 되어 목적이 무의미해집니다. `KVCache::raw_kv_allocations`는 할당과 활성 길이를 돌려주며, 행이 raw fp16 K/V가 아닌 모든 모드, 풀 기반 캐시, 헤드가 잘린 윈도우에 대해서는 `None`을 반환합니다.
- 헤드 간 행 stride는 활성 길이가 아니라 **예약된 용량** `Cap`입니다. 활성 길이를 사용하면 버퍼에 여유가 조금이라도 있는 순간(거의 항상 그렇습니다) 헤드끼리 겹치게 됩니다.
- KV 헤드 축은 *요청* 축으로 접힙니다. 요청 `r = b * H_attn + h`입니다. 쿼리도 이에 맞게 reshape됩니다(`[B, Hq, 1, D]`에서 `[B * Hkv, n_rep, 1, D]`로, 역시 순수한 reshape입니다. MLX GQA는 쿼리 헤드 `i`를 KV 헤드 `i / n_rep` 아래에 번호 매기기 때문입니다). 하나의 CTA는 여전히 하나의 KV 헤드와 그에 속한 쿼리 헤드 그룹을 소유하므로, CTA당 작업량은 변하지 않습니다.

v2 커널은 K와 V를 *동일한* `indices` 항목으로부터 해석하므로, 두 할당이 `(b, h, t)`를 같은 행에 배치할 때에만 선택을 표현할 수 있습니다. `shared_row_mapping`이 이를 강제합니다. 두 할당의 shape이 같으면 자동으로 성립합니다. 헤드 수가 다른 경우에도 `B == 1`에 한해서는 성립하는데, 이것이 MiniMax-M3의 경우입니다. 인덱스 키가 K 할당의 헤드 `H_attn`에 실려 있어 K가 V보다 헤드가 하나 더 넓으며, `b == 0`에서는 두 base가 모두 `h * Cap`으로 축약됩니다. 배치 1을 넘으면 시퀀스 stride가 어긋나므로, 잘못된 주소로 실행하는 대신 launch를 거부합니다.

## 블록 단위가 아닌 토큰 단위 정확성

Issue #904는 두 가지 granularity를 제시하며 하나를 구현하고 다른 하나는 문서화하라고 요구합니다.

**토큰 단위(`page_size = 1`)가 구현되었습니다.** 인덱서가 선택한 집합에 정확히 attention하므로, 측정된 품질 차이를 받아들이는 대신 greedy decode가 mask 구현과 정확히 일치하도록 요구할 수 있습니다.

**블록 단위(`page_size = C`)는 여기에서 구현되지 않았습니다.** 이를 위해서는 요청 `r`의 블록 `j`의 풀 행이 `base(r) / C + j`여야 하며, 즉 요청별 행 stride가 블록 크기의 배수여야 합니다. 이는 행이 곧 블록인 `PagedBlockPool`에서는 성립하며 풀 기반 캐시에는 자연스러운 경로입니다. 선택된 블록 id를 `indices`로 그대로 전달하고 풀 자체의 `block_size`를 `page_size`로 사용하면 확장 단계가 전혀 필요 없습니다. 그러나 연속 할당에서는 성립하지 않습니다. 연속 할당의 예약 용량은 `C`와 무관한 `step` 단위로 증가하기 때문입니다. 대상 두 패밀리가 모두 연속 캐시 모델이므로, 블록 단위 페이지는 현재 어느 쪽에도 적용되지 않습니다.

따라서 block-sparse 모델도 여전히 토큰 단위의 정확한 페이지 테이블에 도달합니다. `selection_from_blocks`가 블록 id를 `O(selected)` 디바이스 패스 한 번으로 토큰 행으로 확장하며, 이는 그것이 대체하는 attention 트래픽의 1퍼센트에도 못 미치는 비용입니다.

## decode 경로에서의 동기화 없음

선택된 *내용*은 디바이스 값(`argpartition` 출력)입니다. 선택된 *개수*는 그렇지 않습니다. top-`k` 선택은 구조상 매 step마다 무언가 중 `k`개를 고르기 때문입니다. 이 구분이 이것을 critical path에서 벗어나게 합니다. `SparseCsrStructure`(`indptr`, `last_page_len`, `first_page_offset`, `seq_lens`)는 `(requests, per_request)`만으로 호스트에서 도출되고, `indices`는 lazy MLX 배열로 남으며, 어떤 것도 다시 읽어 오지 않습니다.

`SparseSelection::materialize`는 실제로 값을 읽어 오며, 테스트와 각 요청의 선택된 행을 출력하는 `MLXCEL_SPARSE_PAGED_DUMP=1`용으로 예약되어 있습니다. 시간 측정을 하는 실행에서는 절대 dump를 켜지 마십시오.

## MiniMax-M3 block-sparse decode

`src/models/minimax_m3_layers.rs`는 인덱서가 sparse 영역(`kv_len > 2 * topk_blocks * block_size`)에 있을 때마다 `l == 1`에서 fused 경로를 시도하고, 거부되면 additive-mask 경로로 폴백합니다.

점수 계산 대상 집합에는 마지막 블록이 제외되며, 마지막 블록은 무조건 추가됩니다. decode에서 이 블록은 쿼리 자신의 블록이므로 `local_blocks >= 1` 강제 유지가 이미 고정해 둡니다. 이를 제외하면 선택된 개수가 점수가 아니라 `kv_len`의 함수로 유지되고, 선택된 블록이 활성 윈도우를 넘어선 행을 가리킬 수 있는 유일한 경로(풀 뷰에서는 범위 초과 오류가 아니라 다른 요청의 토큰이 됨)가 사라집니다.

`minimax_m3_tests::the_decode_selection_names_exactly_the_blocks_the_mask_path_keeps`는 부분적인 마지막 블록을 포함한 네 가지 컨텍스트 길이에 걸쳐, 페이지 테이블이 attention하는 위치와 mask 경로가 유지하는 열이 집합으로서 동일함을 검증합니다. 이는 두 attention 출력을 비교하는 것보다 강한 진술입니다. 잘못된 블록 선택은 그럴듯한 출력 텐서 안에서는 쉽게 숨겨지지만 집합 비교 안에서는 전혀 숨을 수 없습니다.

실제 427B 체크포인트는 개발 머신에 올라가지 않으므로, 여기의 모든 것은 합성 데이터로 검증됩니다. 이 점은 여기뿐 아니라 코드에도 명시되어 있습니다.

## DeepSeek Sparse Attention은 이 경로로 라우팅되지 **않습니다**

Issue #904는 DSA(DeepSeek-V3.2, GLM-MoE DSA와 공유)를 가장 먼저 나열합니다. 이것은 구현되지 않았으며, 그 이유는 선택 방식이 아니라 커널의 특성에 있습니다.

DSA decode는 선택된 위치 `t`에 대해 다음을 계산합니다.

```
score = scale * (q_latent . kv_latent[t])  +  scale * (q_pe . k_pe[t])
```

두 번째 항은 **(헤드, 위치)별 additive mask**(`pe_scores`)로 `mlxcel_core::layers::attention_from_ptr`에 도달합니다. v2 커널은 `scale * (q[h] . k_pool[row])`를 계산하며, 이런 항이 들어올 수 있는 입력이 없습니다. 이 문서의 선택 배관은 DSA에 그대로 적용되지만, 커널은 그렇지 않습니다.

세 가지 해결 방안이 검토되었습니다.

- **bias를 커널 입력으로 materialize합니다.** CSR 슬롯으로 인덱싱되어야 하므로 `[B, Hq, S]`이며, 이를 만들려면 gather된 `k_pe`가 필요한데 그 gather가 바로 제거하려는 대상입니다. 대신 전체 컨텍스트에 대해 만들면 `[B, Hq, kv_len]`(`Hq = 128`, 16K에서 8 MB)이고 `Hq * kv_len * qk_rope_head_dim` MAC이 들며, 이는 16K에서 gather 버전의 여덟 배입니다. 대체하려는 경로보다 엄밀히 더 나쁩니다.
- **위치 항을 내적에 접어 넣습니다.** 풀 행을 `concat(kv_latent, k_pe)`로 만들면 `q_cat . row`가 점수 전체가 되어 bias가 없어집니다. 수치적으로 정확하고 커널 변경이 필요 없지만, DSA 캐시는 `concat(kv_latent, indexer_key)`를 K 버퍼로, `k_pe`를 V 버퍼로 저장하는 별개의 두 할당이므로 연결을 만드는 데 레이어당 step당 `O(kv_len * 576)` 복사(16K에서 약 19 MB)가 들며, 이는 대체하려는 2 MB gather와 비교됩니다. 캐시가 `concat(kv_latent, k_pe, indexer_key)`를 저장하도록 순서를 바꾸면 비용이 없어지지만, 그러면 Int8 및 Turbo4 KV 모드가 그룹화하고 패킹하는 버퍼의 폭과 필드 순서가 바뀌며, DeepSeek-V3.2(약 685B)는 이를 검증하기 위해 로드할 수 없습니다.
- **커널을 일반화합니다.** 행 stride와 두 번째 K 스트림을 추가합니다. 입력 `k_extra_pool`과 `q_extra`가 `dot(q_extra, k_extra[row])`를 점수에 기여하며, 템플릿 상수는 `RowStride`, `Dim`, `ExtraDim`입니다. 이는 FlashInfer의 paged-MLA 커널(`head_dim_ckv`와 `head_dim_kpe`)이 사용하는 형태이며, 위의 선택 배관과 **캐시 변경 없이** 결합됩니다. `k_pool`은 `RowStride = kv_lora_rank + index_head_dim`, `Dim = kv_lora_rank`인 K 할당이고, `k_extra_pool`은 `ExtraDim = qk_rope_head_dim`인 `k_pe` 할당입니다. 둘 다 비용 없이 reshape되고 헤드가 하나이므로 어떤 배치에서도 행 매핑을 공유합니다.

세 번째가 실제 해답이며 권장되는 후속 작업입니다. 여기서 시도하지 않은 이유는, #899가 방금 프로덕션 decode 경로에 출시한 커널을 수정해야 하고, 그 대상 패밀리의 유일한 체크포인트는 사용 가능한 어느 머신에서도 실행할 수 없으며, 조용히 활성화되지 않는 fused decode 경로가 이 epic이 만들어낸 가장 비싼 실패 유형이기 때문입니다.

후속 작업이 마주해야 할 또 하나의 사실이 있습니다. `Hkv = 1`이면 MLA의 헤드 접기가 시퀀스당 **하나의** CSR 요청을 만들므로, 배치 1 DSA decode는 배치 환경의 floor가 아니라 단일 요청 디스패치 floor인 4096에 직면합니다. 기본 `index_topk` 2048에서는 해당 launch가 *floor 아래*에 있어 커널이 갖춰져 있어도 거부됩니다. DSA가 디스패치되기를 기대하려면 먼저 `Hkv = 1` sparse launch에 대한 floor를 다시 도출해야 합니다.

## 디스패치 floor의 위치

#899는 fused 커널을 gather와 비교 측정하여 손해를 보는 shape를 정확히 하나, 즉 1024 가시 토큰의 배치 1(0.91x)을 발견했고, 이로부터 `paged_v2::dispatch`의 두 영역 floor가 나왔습니다. 단일 요청에는 가시 토큰 4096, 배치 시에는 요청당 512입니다. sparse launch에서는 두 가지가 서로 반대 방향으로 이 floor를 움직입니다.

- **floor를 넘어야 하는 수는 컨텍스트가 아니라 선택된 개수입니다.** launch는 요청당 `S`개의 행을 읽으므로, 32K 컨텍스트에서 4K를 선택하면 4K 토큰 launch입니다. floor를 `kv_len`에 적용하면 디스패치를 정당화한 수치보다 실제 작업이 여덟 배 작은 launch까지 디스패치하게 됩니다.
- **요청 수는 KV 헤드 수만큼 곱해집니다.** 헤드 축을 요청 축으로 접으면 단일 시퀀스가 `Hkv`개의 요청이 되므로, `Hkv > 1`인 배치 1 sparse decode는 *배치* floor에 직면합니다. 이것은 허점이 아닙니다. 손해를 본 측정은 청크당 두어 페이지로 퇴화하여 아무 이득 없이 merge 비용을 지불하는 계획 때문이었고, 각각 `S` 페이지인 `Hkv`개의 요청은 0.91x로 측정된 배치 1 shape가 아니라 1.41x 및 1.47x로 측정된 배치 shape입니다. 위에서 언급했듯이 `Hkv == 1`(MLA)에는 적용되지 **않습니다**.

MiniMax-M3의 출시 구성(`Hkv = 4`, `topk_blocks = 16`, `block_size = 128`)에서 단일 시퀀스 decode는 약 2048 행짜리 요청 4개이며, 이는 2048행 floor를 여유 있게 넘습니다.

이 부분은 도출된 것이지 측정된 것이 아닙니다. #899의 표는 dense 페이지 목록에서 측정되었습니다. floor는 동일한 코드와 동일한 환경 변수 재정의(`MLXCEL_PAGED_V2_MIN_KV_TOKENS`, `MLXCEL_PAGED_V2_MIN_KV_TOKENS_PER_REQUEST`)이므로, 재측정하면 둘이 함께 움직입니다.

## 토큰 floor만으로는 충분하지 않습니다: sparsity 게이트

토큰 floor는 "이 launch가 충분히 큰가"를 묻습니다. 그러나 *sparse* launch를 좌우하는 질문, 즉 "건너뛰는 것이 건너뛰는 데 쓰는 커널의 비용만큼 가치가 있는가"는 묻지 않습니다. `mlx::fast::scaled_dot_product_attention`은 고도로 튜닝된 dense 커널이고, v2 partial 커널은 스칼라 레인별 스윕입니다. 데이터의 절반을 상수 계수가 두 배인 커널로 읽는 것은 손해입니다.

`examples/sparse_paged_decode_bench`가 정확히 이를 측정합니다. MiniMax-M3 geometry(쿼리 헤드 64, KV 헤드 4, head_dim 128, `block_size` 128, `topk_blocks` 16), 유휴 상태의 Apple Silicon 호스트에서 40 step을 1회 반복한 결과이므로 **기록된 결과가 아니라 참고용**입니다.

| 컨텍스트 | sparsity | fused vs mask | 일시적 메모리 |
|---|---|---|---|
| 4096 | 2.0x | **0.67x** | -1.6 MB |
| 8192 | 4.0x | **0.77x** | -3.2 MB |
| 16384 | 8.0x | 1.22x | -16.2 MB |
| 32768 | 16.0x | 1.17x | -8.7 MB |
| 65536 | 32.0x | 2.06x | -16.4 MB |

예상대로 모든 지점에서 일시적 메모리가 감소합니다. mask 경로는 `[B, 1, 1, kv_len]` mask를 할당하고 SDPA가 전체 윈도우를 stage하게 하는 반면, fused 경로는 계획이 크기를 정한 workspace만 할당하기 때문입니다.

따라서 launch는 `MIN_SPARSITY_RATIO`(기본값 **8**)도 넘어야 합니다. 활성 윈도우가 선택된 개수의 최소 그 배수여야 합니다. 이 기본값은 측정되지 않은 4x~8x 구간에 보간하는 대신 **측정된 이득 지점 바로 위**에 놓입니다. #899는 토큰 floor에 대해 반대로 논증하여 가장 약한 측정 지점보다 의도적으로 아래에 두었는데, 그 지점 아래에 있는 것은 놓친 기회였기 때문입니다. 여기서 그 지점 아래에 있는 것은 *측정된 퇴보*이며, 좁은 구간에서 퇴보를 출시하는 것은 그 구간의 적당한 이득을 거부하는 것보다 나쁩니다.

그 표에 대한 단서가 둘 있으며 모두 같은 방향을 가리킵니다. 하네스는 시간 측정 영역 밖에서 호스트에 mask를 만드는 반면 실제 mask 경로는 매 step마다 여러 번의 `O(kv_len)` 패스로 디바이스에서 이를 다시 만듭니다. 또한 하네스는 sparse 쪽에 미리 만들어 둔 블록 목록을 넘기지만 실제 경로는 `O(selected)` 패스 한 번으로 선택을 확장합니다. 둘 다 mask 쪽에 유리하므로, 실제 교차점은 8x보다 낮은 sparsity에 있을 것입니다. 따라서 기본값은 어떤 것도 퇴보시킬 수 없는 방향으로 보수적이며, 제대로 된 측정이 가장 먼저 재검토해야 할 수치입니다.

MiniMax-M3의 출시 구성에서 게이트는 16K 컨텍스트에서 열리며, 이는 issue #904가 decode의 개선을 요구하는 바로 그 지점입니다. `should_apply_sparse`가 작동하는 지점(4097 토큰)부터 16K 사이에서는 mask 경로가 실행되며 그 사실을 알립니다.

## 환경 변수

| 변수 | 기본값 | 효과 |
|---|---|---|
| `MLXCEL_SPARSE_PAGED_ATTENTION` | 활성 | `0`, `false`, `off`, `no`는 모든 호출자를 #904 이전의 gather 또는 mask 경로로 되돌립니다. 그 외의 값은 fused 경로를 활성화하므로, 오타가 있어도 측정된 경로가 유지됩니다. 프로세스당 한 번 읽습니다. |
| `MLXCEL_SPARSE_PAGED_DUMP` | 꺼짐 | `1`, `true`, `on`, `yes`는 각 요청의 선택된 풀 행을 출력합니다. **동기화를 유발**하므로 디버깅 전용입니다. |
| `MLXCEL_SPARSE_PAGED_MIN_SPARSITY` | `8` | fused 경로를 디스패치하기 전 최소 `live_len / selected` 비율. `0`은 게이트를 비활성화하며, 벤치마크가 정책이 거부하는 영역을 측정하는 방법입니다. 정수가 아닌 값은 기본값으로 폴백합니다. |

## 출력으로 어떤 경로가 실행되었는지 읽기

모든 호출은 `SparseDecodeOutcome`을 반환하며, `report_sparse_outcome_once`는 각 **종류**의 첫 발생을 `tracing`뿐 아니라 stderr로도 알립니다. 전역 플래그 하나가 아니라 종류별 플래그 하나입니다. 하나의 one-shot은 가장 먼저 일어난 일만 보고하는데, 실제 서버에서는 그것이 짧은 warmup 요청이며, 이후 다른 이유의 영구적인 거부는 드러나지 않게 됩니다.

`tracing`뿐 아니라 stderr도 사용하는 이유는 `mlxcel` CLI가 tracing subscriber를 설치하지 않기 때문에, CLI 전용 decode 경로의 `tracing::info!`는 어떤 `RUST_LOG`에서도 아무것도 출력하지 않기 때문입니다. 프로세스 수명 동안 이러한 줄은 최대 여덟 개입니다.

fused launch는 다음과 같이 표시됩니다.

```
sparse paged decode: fused sparse launch (4 request(s) x 2048 selected rows, 256 chunks, merge on)
```

그리고 거부는 그 원인이 된 수치를 명시합니다.

```
sparse paged decode: fallback: 128 selected rows across 4 request(s) is below the 2048-row dispatch floor
```

`mlxcel_core::paged_v2::sparse_decode_stats()`는 첫 발생 줄이 아니라 횟수가 필요한 하네스를 위해 누적 합계를 반환합니다.

## 벤치마크 하네스

`examples/sparse_paged_decode_bench.rs`는 MiniMax-M3의 헤드 geometry와 sparse 구성에서 디코더 레이어 하나의 attention을 시간 측정하며, **동일한** 선택에 대해 additive-mask 쪽과 fused sparse 쪽을 비교합니다.

```bash
export DEVELOPER_DIR=/Applications/Xcode-26.6.0.app/Contents/Developer

# Default sweep: 8K / 16K / 32K.
cargo run --release --features metal,accelerate --example sparse_paged_decode_bench

# Wider, more repetitions.
cargo run --release --features metal,accelerate --example sparse_paged_decode_bench -- \
    --contexts 16384,32768,65536 --reps 5 --steps 400

# Confirm the kill switch restores the mask path.
MLXCEL_SPARSE_PAGED_ATTENTION=0 cargo run --release --features metal,accelerate \
    --example sparse_paged_decode_bench
```

불가피하게 합성 데이터입니다. DeepSeek-V3.2와 MiniMax-M3는 각각 685B와 427B이며 어느 쪽도 개발 머신에 올라가지 않습니다. 이 수치는 "attention 연산에 무슨 일이 일어나는가"로 읽고, forward pass를 추론하려면 sparse 레이어 수를 곱하며, end-to-end 초당 토큰 수로 읽지 마십시오.

두 가지 가드가 출력을 스스로 설명하게 합니다. 하네스는 sparse 쪽이 fused 커널을 디스패치했는지 폴백했는지를 `SparseDecodeStats` 델타와 함께 이름으로 출력하며, sparse 쪽이 fuse되지 않았다면 시간 비교 결과를 **아예 보고하지 않습니다**. 또한 시간 측정에 앞서 두 쪽이 수치적으로 일치하는지 확인합니다. 서로 일치하지 않는 두 쪽의 시간 비교는 비교가 아니기 때문입니다. #899는 조용히 활성화되지 않는 fused decode 경로를 출시한 뒤 폴백을 자기 자신과 벤치마크했고, 그 결과 깔끔해 보이는 null 결과가 실제 발견으로 받아들여질 뻔했습니다.

다른 작업이 없는 조용한 머신에서 직렬화하여 실행하고 반복하십시오. 이 크기에서 decode attention step은 1밀리초에 훨씬 못 미치므로, GPU에 상주하는 다른 모든 것이 결과를 지배합니다.

## phi3-small과 phimoe도 라우팅되지 **않습니다**

Issue #904는 phi3-small과 phimoe를 MiniMax-M3와 함께 "block-sparse"로 묶습니다. 이들은 block-sparse이지만, 이 페이지 테이블이 표현할 수 있는 형태는 아닙니다.

`phi3small::build_blocksparse_mask`는 `(qb >= kb) && ((qb - kb < local_blocks) || ((kb + h + 1) % vert_stride == 0))`일 때 헤드 `h`의 쿼리 블록 `qb`에 대해 키 블록 `kb`를 선택합니다. 선택은 정적(점수 기반이 아니라 위치 기반)이어서 문제가 없지만, `+ h` 항 때문에 **쿼리 헤드마다** 달라집니다.

CSR 요청은 하나의 KV 헤드와 그 아래에 묶인 `n_rep`개의 쿼리 헤드를 소유하며, 커널은 `indptr[request]`로부터 요청의 페이지 범위를 도출합니다. 따라서 한 요청의 `n_rep`개 쿼리 헤드는 반드시 하나의 페이지 목록을 공유하고, 쿼리 헤드별 패턴은 `n_rep == 1`일 때만 표현할 수 있습니다. 일반적으로 지원하려면 커널에서 쿼리 헤드별 페이지 범위가 필요한데, 이는 #904의 전제("전용 sparse 커널이 필요 없다")가 허용하는 것보다 큰 변경입니다.

`kv_len <= local_blocks * block_size`는 해당 패밀리에서 이미 일반 causal attention으로 퇴화하므로 변경되지 않습니다.

## 다루지 않는 범위

- **Prefill.** 두 패밀리 모두 기존 prefill 경로(DSA sparse additive mask, MiniMax-M3 block mask)를 유지합니다. 다중 토큰 step은 #904의 범위 밖이며 자연스러운 후속 작업입니다.
- **CUDA.** 검증되지 않았습니다. v2 partial 커널의 CUDA 본문은 이미 #898에서 검증되지 않은 것으로 표시되었으며, 여기에서는 `nvcc`로 컴파일하거나 실행한 것이 없습니다. 호스트 측 선택 코드는 백엔드와 무관하지만, 그것이 공급하는 커널은 그렇지 않습니다.
- **DSA decode.** 위를 참조하십시오.
- **실제 체크포인트 검증.** 이 작업에서 사용할 수 있었던 어떤 머신에도 두 대상 체크포인트가 존재하지 않습니다.
