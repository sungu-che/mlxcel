# 분산 추론

`mlxcel`에는 세 가지 분산 또는 멀티 디바이스 방식이 있습니다. 이들은 `src/distributed/` 아래의
코드를 공유하지만, 모드와 모델 패밀리에 따라 성숙도가 다릅니다.

| 모드 | 목적 | 성숙도 |
|------|---------|----------|
| Tensor parallelism (TP) | 프로세스 내 rank들에 걸쳐 텐서 연산을 샤딩합니다. | 선택된 dense 텍스트 패밀리에 구현되어 있으며, 모델별로 검증해야 합니다. |
| Pipeline parallelism (PP) | 레이어 범위를 스테이지별로 분할합니다. | Llama 계열 텍스트 모델과 2-스테이지 토폴로지에서 가장 잘 검증되었습니다. |
| Disaggregated inference (DI) | 각 노드가 모델을 보유한 채 prefill/decode 역할을 분리합니다. | 인프라는 존재하며, 사용자의 토폴로지에서 검증하기 전까지는 실험적인 것으로 취급하십시오. |

## 모드 선택

```text
Model fits on one device?
├── yes
│   ├── latency-sensitive single-user serving → single device
│   └── many concurrent users                 → consider DI after validation
└── no
    ├── high-bandwidth local devices          → TP or PP
    └── multi-host / uneven memory            → PP with explicit layer ranges
```

## Tensor parallelism

TP는 트랜스포머 레이어 내부의 가중치를 샤딩하고 row-parallel 출력을 동기화합니다.
공개된 옵션은 다음과 같습니다.

```bash
mlxcel generate -m models/<checkpoint> \
    --tp-size 2 \
    -p "Hello" -n 100

mlxcel-server -m models/<checkpoint> \
    --tp-size 2 \
    --port 8080
```

관련 옵션으로 `--tp-moe-mode`, `--tp-embedding-mode`, `--tp-lm-head-mode`가 있습니다.
현재 런타임은 많은 패밀리에서 복제된(replicated) embedding 및 LM head 모드를 요구합니다.

현재 공개된 TP 패밀리 목록의 기준은 `src/main.rs`와 `src/bin/mlx_server.rs`의 도움말 텍스트입니다.
이 문서를 작성한 시점에는 dense Llama, Qwen 2/2.5/3/3.5 텍스트, Gemma 3/4 텍스트,
ERNIE 4.5, Hunyuan v1 Dense가 포함되어 있으며, 다른 패밀리를 위한 추가 구현 요소도 있습니다.

제한 사항:

- 모델은 선택한 rank 수로 깔끔하게 샤딩되어야 합니다.
- 일부 서버 배칭 및 VLM 경로는 TP에서 의도적으로 보수적입니다.
- 실행하려는 모든 모델 패밀리와 rank 수에 대해 벤치마크와 정확성 검증을 반복해야 합니다.

## Pipeline parallelism

PP는 모델을 레이어 범위로 분할합니다. 모델이 단일 디바이스의 메모리를 초과하거나
호스트 간 메모리 용량이 고르지 않을 때 유용합니다.

Kimi K3(`kimi_k3`)는 이것을 아예 필수로 요구하게 될 패밀리입니다. 93개 레이어, 2.8T 파라미터의 전체
체크포인트는 4비트에서 약 1.4 TB를 차지하며(mxfp4 experts와 bf16 attention, shared experts, embeddings, `lm_head`),
이는 가장 큰 단일 Apple Silicon 호스트 메모리의 두 배가 넘으므로 어떤 단일 호스트도 이를 담을 수 없고,
512 GB 노드 최소 세 개가 가능한 가장 작은 토폴로지입니다. **PP는 아직 이 패밀리를 실행할 수 없습니다.**
현재 트리에 있는 것은 파티션 프로필(`src/distributed/pipeline/partition_profile_heuristics.rs`)로,
레이어를 유형별로 가격 산정하여 자동 분할이 균일한 레이어당 비용을 가정하는 대신 dense 레이어 0,
네 번째 위치마다 있는 더 저렴한 q-LoRA MLA 레이어, 그 외 모든 곳의 15~16 GB MoE 레이어를 반영하게 합니다.
이는 계획용 입력일 뿐이며, `StageFamily` variant도 스테이지 실행기도 없으므로
`--pp-size`와 `--pp-layers`는 `kimi_k3` 모델을 거부합니다. Issue #1734가 실행기와 전체 모델 멀티 노드 실행을
추적하며, 여기에는 Attention Residual 블록이 스테이지 경계를 넘는 방식도 포함됩니다. 그때까지 이 패밀리는
단일 프로세스 전용이며, 레이어를 잘라낸 로컬 사본(`config.json`의 `text_config.num_hidden_layers`를 낮춤.
sanitizer가 그 이후 레이어를 버립니다)이 로딩, shape, 유한한 logit을 확인하는 단일 호스트 점검 방법입니다.
그 출력 텍스트는 구조상 의미가 없습니다.

### 프로세스 내 CLI 경로

```bash
mlxcel generate -m models/<checkpoint> \
    --pp-size 2 \
    --pp-micro-batch-size 4 \
    -p "Hello" -n 100
```

자동 파티셔닝에 의존하는 대신 명시적인 레이어 범위를 지정할 수 있습니다.

```bash
mlxcel generate -m models/<checkpoint> \
    --pp-layers 0-15,16-31 \
    --pp-micro-batch-size 4 \
    -p "Hello" -n 100
```

### 서버 / 멀티 호스트 경로

서버는 TOML 클러스터 설정과 함께 `--distributed-config`를 사용합니다. 저장소에는
`examples/distributed/`와 `scripts/benchmark_pipeline_remote_rollout.sh` 아래에 헬퍼 스크립트와 예제가 포함되어 있으므로,
실제 클러스터를 운영하기 전에 이들을 확인하십시오.

최소 구성은 다음과 같습니다.

```bash
# Stage process.
mlxcel-server -m models/<checkpoint> \
    --distributed-config examples/distributed/pipeline_remote_2node_tcp.toml \
    --node-id stage-1 \
    --host 0.0.0.0 --port 18081 --no-warmup

# Coordinator / serving process.
mlxcel-server -m models/<checkpoint> \
    --distributed-config examples/distributed/pipeline_remote_2node_tcp.toml \
    --node-id coordinator \
    --host 0.0.0.0 --port 18080 \
    --parallel 2 --max-batch-size 2 --pp-micro-batch-size 2
```

`--pp-auto N`은 설정이 필요 없는 pipeline 계획을 생성할 수 있으며 `--distributed-config`와 함께 사용할 수 없습니다.
프로덕션에서는 토폴로지가 확정되면 명시적인 TOML을 체크인하는 것이 좋습니다.

## Transport

| Transport | 참고 |
|-----------|-------|
| TCP | 기본 IP transport입니다. |
| Thunderbolt | 공유 TCP 코어 위에서의 macOS Thunderbolt Bridge 선택입니다. |
| RDMA | capability probing과 폴백 동작을 갖춘 백엔드가 있습니다. 가속에 의존하기 전에 대상 OS/하드웨어에서 검증하십시오. |

설정이 필요 없는 시작을 위해 mDNS/정적 탐색 옵션을 사용할 수 있습니다. 서브넷을 넘나들거나
통제가 엄격한 네트워크에서는 정적 설정이 더 안전한 선택입니다.

## Disaggregated inference

DI는 prefill과 decode 역할을 분리합니다. PP와 달리 노드당 모델 메모리를 줄이지 **않습니다**.
각 역할이 여전히 모델을 로드해야 합니다. 의도된 사용 사례는 용량 초과 모델을 맞추는 것이 아니라 처리량 튜닝입니다.

이 코드는 PP와 동일한 클러스터 설정, registry, transport, 메트릭 인프라를 공유합니다.
토폴로지에 종속된 기능으로 취급하십시오. 성능 주장을 게시하기 전에 사용자의 트래픽 형태로 실제 테스트를 수행하십시오.

모델 지원은 PP보다 좁습니다. paged prefill/decode 핸드오프는 풀 기반 Fp16 패밀리
(qwen3, llama3 같은 dense-natural 백엔드)에서만 동작합니다. 모델이 직접 소유하는 paged 패밀리(gemma3, gemma4, llama4, qwen3_5,
qwen3_next)는 KV 상태를 공유 블록 풀 밖에 보관하므로 핸드오프 과정에서 직렬화할 수 없습니다.
이러한 패밀리에 대한 요청은 prefill 작업이 실행되기 전에 prefill 노드에서 요청 단위의 깔끔한 오류로 거부되며,
노드는 이후 요청을 계속 처리합니다(issue #708). 전체 범위와 제한 사항 목록은
[continuous batching 및 disaggregated serving](CONTINUOUS_BATCHING.md#scope-and-limitations)을 참조하십시오.

`--node-role router` 프런트는 독립적인 prefill 풀과 decode 풀의 균형을 맞춥니다. 요청마다 prefill 노드와 decode 노드를 모두 선택하고
(registry가 온라인으로 표시한 노드에 대한 round-robin), 선택된 decode 노드를 요청 프레임에 담아 prefill 노드에 전달하여
prefill이 라우터가 균형을 맞춘 decode 노드로 핸드오프하게 하며, 도달할 수 없는 피어를 (전송 오류 감지와 백그라운드 liveness probe를 통해)
다운으로 표시하고 정상 노드로 failover하며, 요청을 받을 수 있는 노드가 없으면 HTTP 503을 반환하는 admission control을 적용합니다.
`GET /router/stats`는 노드별 부하 분포와 노드 상태를 보고합니다.
플래그와 동작은 [continuous batching 및 disaggregated serving](CONTINUOUS_BATCHING.md#multi-node-routing-load-balancing-and-failover)을 참조하십시오.

### 보안 및 신뢰 모델

disaggregated 역할 transport 연결(라우터에서 prefill로, prefill에서 decode로)에는 인증이나 TLS가 없으며, 이는 pipeline-parallelism transport가 이미 취하고 있는 것과 동일한 가정입니다. 라우터의 클라이언트용 HTTP 포트와 모든 내부 역할 transport 포트를 포함한 클러스터 전체는, 신뢰할 수 없는 호출자가 내부 포트에 접근하지 못하도록 막는 신뢰할 수 있는 네트워크 세그먼트 안이나 네트워크 격리 경계(VPC 보안 그룹, 방화벽 규칙, Kubernetes NetworkPolicy) 뒤에서 실행되어야 합니다.

다중 노드 라우팅 인터페이스에 대한 두 가지 구체적인 운영 참고 사항이며, 둘 다 신뢰 세그먼트 기준선을 넘어서는 심층 방어(defense-in-depth)로 강화되었습니다(issue #389).

- `GET /router/stats`는 `/v1/chat/completions`와 동일한 인터페이스인 라우터의 클라이언트용 HTTP 포트에 마운트됩니다. 기본적으로 등록된 각 피어의 원시 host:port는 가려지고, 라우터가 할당한 안정적인 노드 id(`prefill-<i>` / `decode-<i>`), 역할, 상태, 노드별 디스패치 수, 라우터 메트릭 스냅샷만 보고합니다. 이것만으로도 부하 분포를 읽고 실패한 노드가 도달 불가로 표시되었는지 확인하기에 충분하며, 추론 포트에 접근할 수 있는 클라이언트에게 내부 클러스터 토폴로지(원시 주소)를 노출하지 않습니다. 이 모드에서는 응답에 `"addresses_redacted": true`가 포함됩니다. `MLXCEL_ROUTER_STATS_VERBOSE=1`(`true` / `yes` / `on`도 가능, 대소문자 구분 없음)을 설정하면 신뢰 세그먼트 배포에서 전체 주소 보기(`"addresses_redacted": false`)로 되돌릴 수 있습니다. 이 가림 처리는 심층 방어이지 네트워크 격리의 대체물이 아닙니다. 그와 무관하게 라우터의 클라이언트 포트를 내부 포트와 동일한 신뢰 세그먼트에 두십시오.

- prefill 노드는 KV cache 핸드오프(프롬프트를 인코딩한 것)를 전달하기 위해 수신 요청 프레임의 `decode_target` 필드에 있는 주소로 연결합니다. 라우터는 이 필드를 자신이 설정한 노드 registry로부터만 채우므로, 신뢰 세그먼트 가정 하에서 주소는 운영자가 통제합니다. 선택적 심층 방어로서 prefill 노드는 연결하기 전에 `decode_target`을 allowlist에 대해 검증할 수 있습니다. allowlist 밖의 대상은 연결 대신 거부되어(요청은 명확한 오류로 실패하고 KV cache는 알 수 없는 주소로 나가지 않음), 침해된 세그먼트에서 위조된 프레임이 핸드오프를 클러스터 밖으로 리디렉션할 수 없습니다. allowlist의 출처는 전용 `MLXCEL_DECODE_ALLOWLIST` 환경 변수이며, 라우터가 선택할 수 있는 decode 노드의 전체 집합(공유 클러스터 설정)으로 설정한 `host:port` 주소의 쉼표 구분 목록입니다.

  `--decode-peers`가 아닌 전용 변수를 쓰는 이유: 라우터는 자신의 풀에 있는 어떤 decode 노드에 대한 `decode_target`이든 임의의 prefill 노드에 넘길 수 있으므로 allowlist는 라우터가 선택할 수 있는 모든 decode 노드를 포괄해야 하는 반면, prefill의 `--decode-peers`는 정적 핸드오프 폴백(라우터가 `decode_target`을 생략했을 때 사용하는 첫 번째 항목)일 뿐이며 정당하게 피어 하나만 나열할 수 있습니다. allowlist를 `--decode-peers`에서 가져오면 그 정적 집합 밖의 라우터가 균형을 맞춘 모든 대상이 거부되어 issue #201에서 추가된 다중 노드 decode 부하 분산이 깨집니다. 둘을 독립적으로 유지하십시오. 각 prefill 노드에서 `MLXCEL_DECODE_ALLOWLIST`는 전체 풀로 설정하고, `--decode-peers`는 로컬 폴백으로 남겨 두십시오. `MLXCEL_DECODE_ALLOWLIST`가 설정되지 않았거나(또는 유효한 항목이 하나도 파싱되지 않으면) prefill은 거부하는 대신 허용 상태를 유지하고 경고를 기록하므로, allowlist 출처가 없다고 해서 부하 분산이 조용히 비활성화되는 일은 없습니다. 항목은 숫자 `IP:port` 값이어야 하며(`SocketAddr`로 파싱됨), hostname:port 항목은 파싱되지 않고 경고와 함께 건너뛰므로 라우터가 선택한 대상과 절대 일치하지 않습니다. 파싱되지 않는 항목은 시작을 실패시키는 대신 경고와 함께 건너뜁니다. `decode_target`이 없는 요청 프레임(이전 버전의 라우터)은 여전히 prefill의 첫 번째 정적 `--decode-peers` 항목으로 폴백하며, 이는 와이어로 전달된 주소가 아니라 운영자 설정이므로 검증이 필요 없습니다.

## 공통 제한 사항

- 분산 지원은 모델 패밀리 전반에 걸쳐 균일하지 않습니다.
- VLM 파티셔닝은 부분적이며, 텍스트 전용 경로가 더 잘 커버됩니다.
- 멀티 호스트 CI 커버리지는 단일 호스트 단위 테스트에 비해 제한적입니다.
- Transport 성능은 물리적 인터커넥트와 OS 네트워크 설정에 크게 좌우됩니다.

유지 관리되는 지원 요약은 [지원 모델](supported-models.md)을 참조하십시오.
