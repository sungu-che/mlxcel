# 분산 추론

`mlxcel`에는 세 가지 분산 또는 멀티 디바이스 방식이 있습니다. 이들은 `src/distributed/` 아래의
코드를 공유하지만, 모드와 모델 패밀리에 따라 성숙도가 다릅니다.

| 모드 | 목적 | 성숙도 |
|------|---------|----------|
| Tensor parallelism (TP) | 프로세스 내 rank들에 걸쳐 텐서 연산을 샤딩합니다. | 선택된 dense 텍스트 패밀리에 구현되어 있으며, 모델별로 검증해야 합니다. |
| Pipeline parallelism (PP) | 레이어 범위를 스테이지별로 분할합니다. | Llama 계열 텍스트 모델과 2-스테이지 토폴로지에서 가장 잘 검증되었습니다. |
| Disaggregated inference (DI) | 각 노드가 모델을 보유한 채 prefill/decode 역할을 분리합니다. | 인프라는 존재하며, 사용자의 토폴로지에서 검증하기 전까지는 실험적인 것으로 취급하십시오. |

세 모드 모두 네이티브 Windows에서는 실행되지 않습니다. `mlxcel generate`와 `mlxcel-server`는
모델을 resolve하기 전인 시작 단계에서 플래그 이름을 포함한 오류와 함께 종료됩니다. 두 명령 모두에서
`--tp-size`가 1보다 크거나 `--pp-layers`가 주어진 경우, `mlxcel generate`에서 `--pp-size`가 1보다 큰
경우, 그리고 서버에서 `--pp-auto`, `--pp-peer`, `--distributed-config`, `--node-role`, `--peers`,
`--prefill-peers`, `--decode-peers`, `--serving-bind`가 주어진 경우가 해당합니다(lablup/mlxcel#58).
이 모드들에는 Linux, macOS 또는 WSL2를 사용하십시오.

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

`mlxcel-server --parallel N`에서는 TP Llama, Qwen 3, Qwen 3.5, Gemma 3, ERNIE 4.5, Hunyuan v1 Dense
런타임이 rank와 레이어마다 하나의 네이티브 multi-row decode 그래프를 실행하며, 시작 시
`decode path: native batched forward`를 보고합니다. TP Gemma 4는 issue #2546이 batched rank-layer
경로를 제공할 때까지 행별(per-row) fallback에 머뭅니다.

제한 사항:

- 모델은 선택한 rank 수로 깔끔하게 샤딩되어야 합니다.
- 일부 서버 배칭 및 VLM 경로는 TP에서 의도적으로 보수적입니다.
- 실행하려는 모든 모델 패밀리와 rank 수에 대해 벤치마크와 정확성 검증을 반복해야 합니다.

## Pipeline parallelism

PP는 모델을 레이어 범위로 분할합니다. 모델이 단일 디바이스의 메모리를 초과하거나
호스트 간 메모리 용량이 고르지 않을 때 유용합니다.

Kimi K3(`kimi_k3`)는 PP가 전적으로 필요해질 패밀리입니다. 93개 레이어, 2.8T 파라미터의 전체
체크포인트는 4비트에서 약 1.4 TB를 차지합니다(mxfp4 experts와 bf16 attention, shared experts, embeddings,
`lm_head`). 이는 가장 큰 단일 Apple Silicon 호스트 메모리의 두 배가 넘으므로 어떤 단일 호스트도 이를
담을 수 없고, 512 GB 노드 최소 세 대가 이를 담을 수 있는 가장 작은 토폴로지입니다.
**PP는 아직 이 패밀리를 실행할 수 없습니다.** 현재 트리에 있는 것은 파티션 프로필
(`src/distributed/pipeline/partition_profile_heuristics.rs`)입니다. 이것은 레이어를 유형별로 비용 산정하여,
자동 분할이 레이어마다 균일한 비용을 가정하지 않고 dense 레이어 0, 네 번째 위치마다 있는 더 가벼운
q-LoRA MLA 레이어, 그 밖의 모든 곳의 15~16 GB MoE 레이어를 반영하도록 합니다. 이는 계획용 입력일 뿐이며,
`StageFamily` variant도 스테이지 실행기도 없으므로 `--pp-size`와 `--pp-layers`는 `kimi_k3` 모델을
거부합니다. Issue #1734가 실행기와 전체 모델 멀티 노드 실행을 추적하며, 여기에는 Attention Residual
블록이 스테이지 경계를 넘는 방식도 포함됩니다. 그때까지 이 패밀리는 단일 프로세스에서만 실행할 수 있으며,
레이어를 잘라낸 로컬 사본(`config.json`의 `text_config.num_hidden_layers`를 낮춤. sanitizer가 그 이후의
레이어를 버립니다)이 로딩, shape, 유한한 logits를 확인하는 단일 호스트 점검 수단입니다. 그 출력 텍스트는
구조상 의미가 없습니다.

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
    --cluster-secret-file /etc/mlxcel/cluster.secret \
    --node-id stage-1 \
    --host 0.0.0.0 --port 18081 --no-warmup

# Coordinator / serving process.
mlxcel-server -m models/<checkpoint> \
    --distributed-config examples/distributed/pipeline_remote_2node_tcp.toml \
    --cluster-secret-file /etc/mlxcel/cluster.secret \
    --node-id coordinator \
    --host 0.0.0.0 --port 18080 \
    --parallel 2 --max-batch-size 2 --pp-micro-batch-size 2
```

위 두 명령 모두 [클러스터 인증](#cluster-authentication) 섹션에서 설명하는 클러스터 시크릿이 필요합니다.
시크릿이 없으면 각 프로세스는 시작 시 플래그, 환경 변수, TOML 필드 이름을 포함한 오류와 함께 종료됩니다.

### Zero-config bring-up

`--pp-auto N`과 `--pp-peer`는 N개 스테이지 파이프라인을 손으로 작성한 TOML 대신 사용합니다. 코디네이터
(`--pp-auto N`)는 N개의 피어 프로세스를 찾아내고, 클러스터를 계획하며, 인증된 클러스터 핸드셰이크를 통해
각 피어에 스테이지 할당 하나씩을 전송합니다. 피어(`--pp-peer`)는 그 할당을 기다렸다가 자신의 스테이지를
제공합니다. 코디네이터는 스테이지를 직접 실행하지 않으므로 `--pp-auto 2`에는 `--pp-peer` 프로세스 두 개가
필요합니다. `--pp-auto`는 `--distributed-config`와 상호 배타적입니다. `--distributed-config`와
`--node-id`로 시작한 피어는 할당 대기를 건너뛰고 TOML을 그대로 사용합니다.

피어는 할당을 수락하기 전에 다음 항목을 이 순서대로 확인합니다. 할당 프로토콜 버전, mlxcel 버전, 클러스터 이름
(`--cluster-name`, 기본값 `mlxcel-cluster`), 모델 식별자(`config.json`의 SHA-256, 모든 `*.safetensors` 파일의
이름과 크기, 레이어 수), 코디네이터가 해당 스테이지에 대해 계산한 레이어 범위와 피어가 자신의 체크포인트에서
도출한 레이어 범위의 일치 여부, 제시된 node id가 제시된 스테이지에 대해 `stage-<i>`인지, 제시된 클러스터가
이 피어를 자신의 리스너 포트에 배치하는지, 그리고 코디네이터의 control 주소와 제시된 클러스터의 코디네이터
노드가 핸드셰이크가 연결에 바인딩한 identity와 일치하는지입니다. 프로토콜, mlxcel 버전, 모델 또는 레이어
범위가 맞지 않으면 피어 프로세스가 `stage assignment refused`와 함께 종료됩니다. 같은 코디네이터에 재시도해도
성공할 수 없기 때문입니다. 클러스터 이름, 주소 또는 클러스터 TOML 문제는 피어를 다시 대기 상태로 돌려보냅니다.
코디네이터는 모든 피어가 수락한 뒤에만 commit하고, 모든 피어가 commit한 뒤에만 activate합니다. 피어는
`Activate`를 받으면 자신의 클러스터 TOML을 쓰고 스테이지를 시작합니다. 다른 피어가 commit 단계에서 실패하는
동안 commit에 성공한 피어는 실패 요약과 함께 `Abort`를 받고 다시 대기 상태로 돌아갑니다. 따라서 실패한
기동은 어디에서도 스테이지를 시작하지 않으며, 오류 메시지는 commit하지 못한 피어를 각각 이름으로 알려 줍니다.
코디네이터가 두 `Activate` 쓰기 사이에서 죽으면, 이미 activate된 피어는 기동이 성공한 직후 충돌한 경우와
같이 계속 실행됩니다. 제한 시간은 다음과 같습니다. 핸드셰이크당 5 s, 프레임당 10 s, 코디네이터의 push 전체에
60 s(offer 단계는 10 s 일찍 끝나므로 commit 단계는 항상 온전한 10 s 프레임 창을 가집니다), 피어가
`Accepted`와 `Commit` 사이, 그리고 `Committed`와 `Activate` 사이에서 각각 기다리는 시간은 60 s, 피어가 어떤
할당이든 기다리는 시간은 600 s입니다. 할당 프로토콜 버전은 2입니다. 버전 1의 코디네이터나 피어는
`assignment protocol mismatch`로 거부됩니다.

아래 명령은 세 가지 사실을 전제로 합니다. 코디네이터의 control 주소는 기본적으로 포트 19000의 `--host` IP이며
스테이지들은 이 주소로 응답합니다. 따라서 `127.0.0.1`에서 서빙하는 코디네이터는 다른 호스트의 피어를 구동할 수
없습니다(탐색 결과 loopback이 아닌 피어가 확인되면 시작을 거부합니다). 모델 디렉터리는 모든 호스트에서 동일한
체크포인트여야 합니다. 모델 identity는 `config.json`을 바이트 단위로 비교하기 때문입니다. 그리고 모든 호스트에
같은 시크릿 파일이 있어야 합니다.

시크릿 배포는 한 번만 합니다.

```bash
openssl rand -hex 32 > /etc/mlxcel/cluster.secret
chmod 600 /etc/mlxcel/cluster.secret
scp /etc/mlxcel/cluster.secret host-b:/etc/mlxcel/cluster.secret
scp /etc/mlxcel/cluster.secret host-c:/etc/mlxcel/cluster.secret
```

**정적 탐색**(코디네이터가 피어 주소를 안다). 각 피어는 `--cluster-control-addr`로 자신의 리슨 주소를
지정하고, 코디네이터는 `--cluster-peers`에 정확히 그 주소들을 나열합니다. 피어는 코디네이터보다 먼저 또는
나중에 시작할 수 있습니다. 코디네이터는 60 s push 창 안에서 500 ms마다 연결과 핸드셰이크를 재시도합니다
(시크릿 또는 identity 불일치는 재시도하지 않습니다).

```bash
# host-b (peer, LAN IP 192.168.1.11)
mlxcel-server -m models/<checkpoint> --pp-peer \
    --cluster-control-addr 192.168.1.11:19001 \
    --cluster-secret-file /etc/mlxcel/cluster.secret

# host-c (peer, LAN IP 192.168.1.12)
mlxcel-server -m models/<checkpoint> --pp-peer \
    --cluster-control-addr 192.168.1.12:19001 \
    --cluster-secret-file /etc/mlxcel/cluster.secret

# host-a (coordinator, LAN IP 192.168.1.10)
mlxcel-server -m models/<checkpoint> --pp-auto 2 \
    --cluster-peers 192.168.1.11:19001,192.168.1.12:19001 \
    --cluster-control-addr 192.168.1.10:19000 \
    --host 0.0.0.0 --port 8080 \
    --cluster-secret-file /etc/mlxcel/cluster.secret
```

**mdns 탐색**(코디네이터가 비콘으로부터 피어 주소를 알아냅니다). 각 피어는 할당이 commit될 때까지 매초
`--cluster-discovery-port`(기본값 49555)의 `--cluster-discovery-target`으로 HMAC 서명된 UDP 비콘을
보냅니다. 코디네이터는 모든 인터페이스에서 그 포트를 30 s 동안 수신합니다. 피어는 `--cluster-control-addr`를
생략할 수 있으며, 그 경우 `0.0.0.0`의 OS가 할당한 포트에서 리슨하고 그 포트를 알립니다. 따라서 방화벽이 피어의
포트를 허용해야 할 때는 (아래처럼) 고정 주소를 지정하십시오. 기본 target은 제한된 브로드캐스트
`255.255.255.255`이며, 이는 기본 경로(default route) 인터페이스로만 나갑니다. 클러스터 트래픽이 두 번째 NIC나
Thunderbolt 브리지를 쓰는 호스트는 해당 서브넷의 directed broadcast(`192.168.1.255`)를 넘기고, 브로드캐스트를
버리는 네트워크에서는 코디네이터의 유니캐스트 IP를 사용합니다. 피어는 코디네이터가 시작될 때 비콘을 보내고
있어야 하며, 그렇지 않으면 코디네이터의 30 s 탐색 창 안에서 보내야 합니다.

```bash
# host-b and host-c (peers); the control port is OS-assigned only when
# --cluster-control-addr is omitted
mlxcel-server -m models/<checkpoint> --pp-peer \
    --cluster-discovery=mdns --cluster-discovery-target 192.168.1.255 \
    --cluster-control-addr 0.0.0.0:19001 \
    --cluster-secret-file /etc/mlxcel/cluster.secret

# host-a (coordinator)
mlxcel-server -m models/<checkpoint> --pp-auto 2 \
    --cluster-discovery=mdns \
    --cluster-control-addr 192.168.1.10:19000 \
    --host 0.0.0.0 --port 8080 \
    --cluster-secret-file /etc/mlxcel/cluster.secret
```

방화벽: mdns를 위해 코디네이터에서 UDP 49555(또는 `--cluster-discovery-port`)를 인바운드로 열고, 코디네이터에서
TCP 19000(또는 `--cluster-control-addr`의 포트)을 인바운드로 열며, 각 피어에서는 `--cluster-control-addr`의
TCP 포트를 해당 피어에 인바운드로 여십시오. 코디네이터는 할당을 위해, 그리고 이후 스테이지 트래픽을 위해 이
포트에 연결하고, 이웃 스테이지는 activation을 위해 이 포트에 연결합니다. HTTP 포트는 클라이언트만 필요로
합니다.

`--dry-run`은 `--pp-auto` 코디네이터에서만 사용합니다. 이 옵션은 클러스터를 계획한 뒤 스테이지를 할당하지 않고
종료합니다. `--pp-peer`와 `--dry-run`을 함께 쓰면 소켓을 바인딩하기 전 시작 시점에 거부됩니다.

코디네이터는 계획을 `--cluster-config-out`(기본값 `.mlxcel/cluster.toml`)에 기록하고, 각 피어는 받은 클러스터를
자신의 `--cluster-config-out`(기본값 `.mlxcel/cluster-<node-id>.toml`)에 기록합니다. 두 파일의 본문은 같지만
헤더는 다릅니다. 코디네이터의 파일은 명령을 다시 실행하면 바이트 단위로 똑같이 재현되지만, 피어의 파일은 다음에
commit하는 할당에 의해 덮어쓰이기 때문입니다. 이 파일들은 나중에 `--distributed-config`와 `--node-id`로
수동 실행할 때 입력으로 씁니다. 스테이지 id는 피어 주소의 사전순으로 `stage-0`, `stage-1`, ...입니다. 두 모드
중 어느 쪽이든 단일 호스트 loopback 리허설은 `tests/zero_config_cluster_init.rs`(`zero_config_cli_*`)이며, 이
테스트는 `--cluster-discovery-target 127.0.0.1`로 실제 바이너리를 구동합니다.

### Cluster authentication

모든 분산 역할은 클러스터별로 미리 공유된 시크릿 하나로 피어를 인증합니다. 파이프라인 역할은 issue #2637부터,
disaggregated의 prefill, decode, router 역할은 issue #2655부터 적용됩니다. 코디네이터, 모든 원격 스테이지,
모든 `--pp-peer` 프로세스, 그리고 모든 `--node-role prefill|decode|router` 프로세스가 같은 값을 가져야 합니다.
한 번 생성한 뒤 소유자만 접근할 수 있는 권한으로 각 호스트에 복사하십시오.

```bash
openssl rand -hex 32 > /etc/mlxcel/cluster.secret
chmod 600 /etc/mlxcel/cluster.secret
```

다음 출처 중 하나로 제공합니다. 우선순위가 높은 순서부터 나열합니다.

1. `mlxcel-server`(및 `mlxcel serve`)의 `--cluster-secret-file <PATH>`.
2. `MLXCEL_CLUSTER_SECRET_FILE=<PATH>`.
3. `MLXCEL_CLUSTER_SECRET=<VALUE>`(시크릿 값 자체이며, 공유 호스트에서는 파일 형식을 권장합니다). 두 환경
   변수를 모두 설정하면 오류입니다.
4. 클러스터 TOML의 `[cluster]` 아래 `secret_file = "<PATH>"`. 상대 경로는 TOML 파일 자신의 디렉터리를 기준으로
   resolve됩니다. 시크릿 값 자체는 TOML에 들어가지 않습니다. 코디네이터가 생성된 TOML을 디스크에 쓰고 호스트
   사이에 복사하기 때문입니다. 시크릿이 파일에서 왔다면 `--pp-auto`가 코디네이터의 시크릿 파일을 그곳에 절대
   경로로 기록합니다.

시크릿은 끝의 줄바꿈 하나를 제거한 뒤 최소 16바이트여야 합니다. 파일이 그룹 또는 모든 사용자에게 읽기
가능하면 서버가 경고하며, 값은 절대 로그에 남기지 않습니다.

시크릿이 보호하는 대상:

- **스테이지 연결.** 파이프라인 역할 사이의 모든 TCP 연결(일반 TCP, Thunderbolt, RDMA 래퍼는 모두 같은 코어를
  공유합니다)은 프레임이 하나라도 교환되기 전에 상호 HMAC-SHA256 challenge-response를 수행합니다. 잘못된
  시크릿을 가진 피어는 어느 쪽이 연결을 열었든 거부되며, 코디네이터는 첫 요청이 아니라 시작 시점에 실패합니다.
  인증되지 않은 연결의 프레임은 스테이지 서비스나 코디네이터에 도달하지 않습니다. 수락된 피어의 리스너 주소는
  TCP 소스 IP에 바인딩되며, 모든 프레임이 귀속되는 identity가 됩니다.
- **핸드셰이크 이후의 모든 프레임.** 각 연결은 핸드셰이크로부터 자체 방향별 키를 도출합니다(시크릿과 두 nonce에
  대한 HKDF-SHA256). 연결 위의 모든 프레임은 암묵적인 방향별 시퀀스 번호 아래 AES-256-GCM으로 암호화되고
  인증됩니다. Activation, logits, KV handoff, 제어 명령은 와이어에서 읽을 수 없으며, 변조, 재생, 순서 변경되었거나
  다른 연결에서 기록된 프레임은 인증에 실패하여 연결이 닫힙니다. [위협 모델과 프레임별 보호](#threat-model-and-per-frame-protection)를
  참조하십시오.
- **명령과 응답의 인가.** 스테이지는 명령(`RunEntry`, `CancelRequest`, `BeginDrain`, `Shutdown` 등)을 자신의
  클러스터 설정의 코디네이터 노드로부터만 받고, forward activation은 상위(upstream) 스테이지로부터만 받습니다.
  응답은 인증된 소스로 보내며, 페이로드 안에 적힌 주소로는 절대 보내지 않습니다. 시퀀스는 그것을 준비한 identity에
  묶입니다. `PrepareSequence`의 인증된 소스가 스테이지에서 그 요청 id의 소유자가 되며, 다른 identity로부터 온
  해당 id의 `PrepareSequence`, `RunEntry`, `CancelRequest`, `ReleaseSequence`는 코디네이터 검사를 통과하더라도
  (`0.0.0.0:<port>`처럼 지정되지 않은 IP로 설정된 코디네이터는 해당 포트의 모든 identity를 수락합니다)
  `Error`로 응답되며 아무것도 바꾸지 않습니다. 어떤 코디네이터도 스테이지에 준비하지 않은 id에 대한 `RunEntry`는
  텐서를 디코드하기 전에 `Error`("sequence <id> was not prepared on this stage")로 응답합니다. 그런 id에 대한
  forward activation은 실행 없이 거부되므로, 소유되지 않은 id에 대해서는 어떤 명령이나 activation도 상태를
  만들지 않습니다. 스테이지는 최대 4096개의 준비된 시퀀스를 보유합니다. 그 이상의 새 id에 대한 `PrepareSequence`는
  `Error`("pipeline stage already holds 4096 prepared sequences")로 응답되며 아무것도 바꾸지 않습니다. 반면
  소유자가 이미 보유한 id를 다시 준비하는 것(re-prepare)은 여전히 수락됩니다. 충돌한 코디네이터가 남긴 소유
  시퀀스는 누수되지 않습니다. 상한에 도달한 스테이지에서 새 id의 `PrepareSequence`가 들어오면, 먼저 소유자가
  10분 동안 명령이나 activation을 보내지 않은 소유 시퀀스를 모두 해제합니다(해제 개수는 `warn` 한 건이
  알려 줍니다). 그 뒤에도 스테이지가 여전히 가득 차 있을 때만 새 id를 거부합니다. 상한 아래에서는 아무것도
  회수하지 않으며, 그만큼 유휴 상태인 살아 있는 요청은 다음 `RunEntry`에서 준비되지 않은 것으로 실패하고, 그
  요청 하나만 실패합니다. 빈 id이거나 256바이트를 넘는 id(코디네이터가 만들 수 없는 id)에 대한
  `PrepareSequence`도 마찬가지로 id를 되돌려 적지 않는 `Error`로 응답되며 아무것도 저장하지 않습니다. 코디네이터의
  요청 id에는 임의의 64비트 접미사(`pp-seq-<n>-<16 hex digits>`)가 붙으므로, 시크릿의 다른 보유자가 코디네이터가
  다음에 사용할 id를 미리 준비할 수 없습니다. Forward activation은 코디네이터의 exchange nonce를 담고, 최종
  스테이지는 logits를 그 nonce의 `RunResult`로 한 번만 보내며, 이는 시퀀스를 준비한 identity에게 곧바로
  전달됩니다(디코드된 토큰당 프레임 3개: `RunEntry`, forward activation, `RunResult`. 스테이지에 도달한 reverse
  activation은 버려집니다). 설정된 코디네이터 주소로 대체하는 경로는 없습니다. 그 스테이지에서 아무도 준비하지
  않은 시퀀스에 대한 logits나 오류는 기록되고 버려집니다. 코디네이터는 `Ack`, `State`, `Error`를 자신이 주소를
  지정한 스테이지로부터만, `RunResult`는 최종 스테이지로부터만 수락하며, 그때도 자신이 보낸 교환별 새 nonce가
  붙은 것, 일치하는 요청과 마이크로 배치에 대한 것, 그리고 반환된 logits가 모델의 shape(`[1, seq_len, vocab_size]`)를
  가질 때만 수락합니다. 그 밖의 것은 버려지며, 교환은 절대 데드라인까지 계속 기다립니다.
  스테이지가 받는 텐서도 페이로드를 복사하기 전에 같은 방식으로 제한됩니다. hidden states는 `[1, n, width]` float이어야
  하며, `n`은 1에서 2^20 사이, `width`는 모델의 inbound activation 폭입니다. 토큰 id는 `[1, n]` int32이며 `n`은
  1에서 2^20 사이입니다. Attention mask는 rank 2에서 4이고 모든 축이 1에서 2^20 사이입니다. 코디네이터는
  `[1, n, vocab_size]` logits를 받아들이며, `n`은 요청 입력 길이 이하 1 이상입니다. Activation 경로는 압축 또는
  양자화된 텐서를 거부합니다. 이 범위를 벗어난 프레임은 주장된 크기를 할당하지 않고, 시퀀스를 준비한 코디네이터에게
  `Error`로 응답합니다. 이때 activation의 exchange nonce를 담으므로 코디네이터는 그 요청을 즉시 실패시킵니다.
  Forward activation의 그 밖의 모든 실패(실행 오류, 뒤에 스테이지가 없는 hidden states, forward 실패)도 같은
  방식으로 보고됩니다. 프레임이 버려지는 경우는 아무것도 그 프레임을 연결할 수 없을 때뿐입니다. 즉 reply nonce가
  없거나 기록된 소유자가 없는 경우입니다.
- **스테이지 실패는 코디네이터가 아니라 요청 하나만 실패시킵니다**(#2727). Forward를 실패시킨 스테이지(`Error`
  응답, 전송 오류, 또는 드롭되거나 손상된 프레임이 시크릿 없이 일으키는 stage timeout 초과 교환)는 그 forward의
  요청만 실패시킵니다. 코디네이터는 모든 스테이지에서 시퀀스를 해제하고, 그 요청들에는
  `model pipeline forward failed; see the server log for the stage error`로 응답하며, 계속 서빙합니다. 스테이지
  자체의 오류(주소와 원격 시퀀스 id 포함)는 `warn`으로 기록되며 클라이언트에는 절대 전송되지 않습니다. `PrepareSequence`를
  거부한 스테이지는 그 요청을 `model pipeline could not prepare this request; see the server log for the stage error`로
  실패시킵니다. `Server busy: ...`는 코디네이터 자체의 용량 한도에만 예약되어 있습니다. 한 스테이지가 응답하지
  않더라도 teardown은 모든 스테이지에 도달합니다. Cancel과 release는 스테이지마다 차례로 보내지므로, 다운된
  스테이지가 다른 스테이지에 시퀀스를 남기는 일이 없습니다. 실패 경로는 제한됩니다(#2762). 실패한 forward는
  시퀀스를 teardown하기 전에 코디네이터에서 그 시퀀스의 상태를 버립니다. 따라서 요청의 나중 close는 아무것도
  보내지 않으며, 각 스테이지는 release를 한 번만 받습니다(직렬화할 수 없는 텐서처럼 아무것도 보내기 전에 발생한
  오류는, 그 뒤의 close가 한 번 release하도록 매핑을 남겨 둡니다). 코디네이터는 스테이지가 cancel이나 release를
  확인할 때까지 최대 2초(또는 더 짧은 경우 stage timeout)만 기다립니다. 프레임은 어느 쪽이든 전송되며, 늦은
  확인 응답은 nonce 매처가 버리기 때문입니다. 따라서 응답하지 않는 스테이지 하나가 단일 모델 워커에 드는 비용은
  고정된 30초 stage timeout에서 약 2분 대신 stage timeout 하나와 짧은 확인 대기 두 번(약 34초)이며, 이는 그것을
  감지한 요청에 대해서만 발생합니다. 코디네이터는 가용성 게이트(#2782)를 유지합니다. 이 게이트는 스테이지가 stage
  timeout 안에 응답하지 않거나 명령을 그 스테이지에 보낼 수 없을 때(#2790) 열리며, 열려 있는 동안 모든 새 prepare와
  forward를 즉시 위의 prepare 또는 forward 메시지로 거부합니다. 그래서 각 요청이 stage timeout만큼 기다리지 않습니다.
  이 게이트는 고정 쿨다운이 아닙니다. 응답 하나가 떨어졌다고 그 뒤의 요청까지 거부되어서는 안 되기 때문입니다.
  Timeout 이후 첫 요청은 모든 스테이지를 2초 확인 대기로 탐색하며, 모두 응답하면 정상적으로 처리됩니다. 스테이지가
  계속 침묵하면 10초마다 최대 한 번의 탐색만 보내고, 그 사이의 요청은 아무것도 보내지 않고 거부됩니다. Forward의
  응답은 모든 스테이지가 실행한 뒤 마지막 스테이지에서 오므로, forward가 타임아웃되면 게이트는 스테이지 이름 없이
  열리며, 다음 탐색이 침묵한 스테이지를 지목합니다. 그 전까지는 모든 스테이지가 확인 대기를 유지합니다. 코디네이터
  로컬 오류(직렬화할 수 없는 응답, 닫힌 수신 채널), `Error` 응답(응답이므로), 그리고 놓친 teardown 확인 응답은
  게이트를 절대 열지 않습니다. 게이트가 열려 있는 동안에도 시퀀스의 teardown은 지목된 침묵 스테이지에 여전히
  전송되며, 그 확인 응답은 기다리지 않습니다. 열림과 닫힘 전환은 각각 한 번씩 `warn`과 `info`로 기록되며, 스테이지
  (아직 지목되지 않았다면 원인)와 중단 기간을 포함합니다. 거부 자체는 `debug`로 기록됩니다. Cancel이나 release
  확인이 느리지만 정상인 스테이지(긴 forward를 마무리하는 중인 스테이지)는 여전히 프레임을 적용합니다. 이런 누락에
  대해 코디네이터가 남기는 "did not acknowledge ... within ...; the frame was sent" 줄은 분당 5회 경고 예산을 가진
  자체 `teardown-ack` 예산에서 나오며, 나머지는 `debug`입니다. 스테이지의 `Error` 응답 텍스트와 상위 스테이지가 보내는
  요청 id는 1024바이트로 제한되고, 제어 문자를 이스케이프한 채 양쪽 로그에 기록되기 전에 처리되므로, 인증된 피어가
  로그 줄을 위조할 수 없습니다. 준비되지 않은 시퀀스 이름을 가진 activation이나 activation 실패에 대해 스테이지가
  기록하는 경고(인증된 상위 스테이지만 보낼 수 있습니다)는 분당 5회 경고의 자체 `stage-drop` 예산에서 나오며,
  나머지는 `debug`입니다. Row들이 서로 다른 logits 폭을 반환하는 배치 forward는 전체가 거부되며, 그 배치의 모든
  row를 같은 forward 메시지로 실패시킵니다.
- **Discovery 비콘과 스테이지 할당.** `--cluster-discovery=mdns`를 쓰면 모든 비콘은 타임스탬프, nonce, 그리고 필드
  전체에 대한 HMAC을 담습니다. 코디네이터는 유효한 MAC이 없는 비콘, 자신의 시계와 60초 이상 차이 나거나 그 창 안에서
  재생된 비콘, 다른 mlxcel 버전의 비콘, 광고된 주소가 데이터그램의 소스 IP가 아닌 비콘을 거부합니다. 정적 탐색은
  영향을 받지 않습니다. `--pp-auto` 코디네이터가 각 `--pp-peer`에 보내는 스테이지 할당은 같은 연결 핸드셰이크를
  거칩니다. 코디네이터는 피어가 offer를 읽기 전에 시크릿을 증명하고, 피어는 코디네이터가 offer를 보내기 전에
  증명합니다. 따라서 다른 시크릿을 가진 코디네이터는 스테이지나 모델을 할당할 수 없으며, 피어는 진짜 코디네이터를
  계속 기다립니다. 피어의 할당 리스너는 인증되지 않은 소켓을 대기 풀 8개(소스당 4개. IPv4 주소 또는 IPv6 /64
  단위)에서 받으며, 초과분은 accept 시점에 닫습니다.

- **Disaggregated 역할 연결.** issue #2655 이후 router, prefill, decode 역할의 transport도 같은 연결 핸드셰이크를
  수행합니다. Router의 요청 프레임, prefill 노드가 decode 노드로 보내는 KV handoff, 그리고 모든 결과 프레임은
  인증된 연결에서만 전송되며, 핸드셰이크를 건너뛴 raw TCP 클라이언트는 어떤 역할의 receive loop에도 프레임을
  전달하지 못합니다. Router는 시크릿 없이 설정된 transport에서는 시작을 거부하며, 결과 demux는 발신자의 핸드셰이크로
  바인딩된 identity가 `--prefill-peers` 또는 `--decode-peers` 주소(정규화된 IP와 포트) 중 하나일 때만 프레임을
  전달합니다. 인증되는 것은 소스 IP뿐입니다(핸드셰이크가 TCP 연결과 대조하여 검증합니다). 포트는 발신자가 광고한
  리스너 포트이며 검증되지 않습니다. 따라서 이 검사는 포트 수준의 증명이 아니라 소스 IP와 주장된 포트로 클러스터
  멤버를 구분합니다. 포트는 설계상 증명되지 않은 상태로 둡니다. 모든 멤버가 같은 시크릿을 가지므로, 어떤 핸드셰이크
  증명도 한 IP에 있는 두 시크릿 보유자를 구분할 수 없습니다(리스너 포트를 증명하려면 프로세스별 키 자료가 필요하며,
  모든 핸드셰이크 호출자에 걸친 새로운 identity 계층이 필요합니다). 라우터가 연결한 연결에서만 결과를 수락하려면
  전송이 절대 하지 않는 풀링된 아웃바운드 연결의 read가 필요합니다. 얻는 이득은 작습니다. Demux는 설정된 어떤
  피어든 어떤 요청 id에 대해서도 수락하므로, 남은 위험은 설정된 피어와 IP를 공유하는 설정되지 않은 시크릿 보유자뿐이며,
  위협 모델은 이미 이를 멤버로 취급합니다. 요청 id는 순차적이므로, 시크릿을 공유하는 파이프라인 스테이지 같은 다른
  보유자가 그렇지 않으면 클라이언트의 응답에 토큰을 주입할 수 있습니다. 다른 멤버의 프레임은 경고
  (`router: dropping result frame from ...`)와 함께 버려지고, 요청은 handoff 타임아웃으로 실패합니다. 바인딩된
  identity는 발신자의 `--serving-bind` 주소이며, unspecified bind IP는 연결의 소스 IP로 대체됩니다. 따라서 피어는
  자신이 연결하는 주소로 목록에 있어야 합니다(multi-homed 노드, 또는 `0.0.0.0`에 바인딩되었지만 다른 인터페이스로
  router에 도달하는 노드는 그 인터페이스의 주소를 router의 피어 목록에 넣어야 합니다). Decode 역할의 handoff 수신과
  prefill 역할의 요청 수신은 인증된 멤버를 계속 신뢰합니다. `MLXCEL_DECODE_ALLOWLIST`는 변경되지 않습니다.

Fail-closed 범위: `--pp-auto`, `--pp-peer`, `pipeline_parallel_size`가 2 이상인 모든 `--distributed-config`,
그리고 `--node-role prefill`, `decode`, `router`는 시크릿을 요구하며, 없으면 시작을 거부하고 오류에 모든 출처를
명시합니다. 기본 단일 노드 서빙(`--node-role` 없음 또는 `--node-role hybrid`)과 한 프로세스 안의 텐서 병렬화는
시크릿을 읽지 않으며 이전과 정확히 같이 동작합니다.

핸드셰이크 실패는 운영자가 로그에서 구별할 수 있도록 각각 다르게 보고됩니다.

- *연결하는 쪽의 잘못된 시크릿*: 연결하는 프로세스는 "its authenticator does not verify under the configured cluster secret (secret mismatch)"로
  실패합니다. 수락하는 쪽은 "client closed the connection after receiving this server's authenticator; it most likely holds a different cluster secret"를
  기록합니다. 서버의 authenticator 이후의 reset이나 abort는 정상 종료와 같은 방식으로 분류됩니다.
- *광고된 identity 소유 아님*: 광고된 리스너 IP가 수락하는 쪽이 보는 소스 IP와 다른 노드(잘못된 바인드 또는 광고
  주소, 또는 노드 사이의 NAT이나 포트 포워드)는 서버 nonce나 authenticator를 보내기 전에 명시적인 거부 레코드와
  함께 거부됩니다. 연결하는 쪽은 "server refused this node's advertised identity `<host:port>`: the advertised IP must be the address the server sees this connection come from"로
  실패하고, 수락하는 쪽은 "refusing to bind a listener address the peer does not own"을 기록합니다. 소켓 주소조차 아닌
  광고된 identity는 어떤 mlxcel 클라이언트에서도 오지 않으므로, 연결은 거부 레코드 없이 닫힙니다.
- *핸드셰이크 전에 닫힘*: 연결 후 바로 닫는 liveness probe(router health monitor, `probe_peers`)는 `debug`에만
  기록됩니다.
- *용량 초과*: 수락하는 노드가 이미 `max_connections`개의 인증된 연결을 보유한 상태에서 인증된 피어는 핸드셰이크 안에서
  용량 초과 판정을 받습니다. 연결하는 쪽은 "server authenticated this node but refused the connection: it already holds its maximum number of authenticated connections; retry later"로
  실패합니다. 이것은 핸드셰이크 거부처럼 치명적으로 취급되지 않습니다. 연결 풀은 이를 거부된 연결처럼 다루어,
  같은 지수 백오프(`reconnect_base_delay`에서 시작해 `max_reconnect_attempts`번)로 재시도합니다. 각 대기 시간은
  현재 지연의 절반에서 전체 사이에서 균등하게 뽑히므로, 함께 거부된 클라이언트들이 동시에 재시도하지 않습니다. 그래서
  그 예산 동안 슬롯을 비운 피어는 같은 `acquire`로 도달합니다. 예산이 소진되었을 때만 오류가 반환되며, 그 체인에는
  `ServerAtCapacity`가 남아 있고, 그 후 `Transport::connect`는 경고를 기록하고 계속합니다. 수락하는 쪽은
  "max connections (N) reached"를 분당 처음 몇 번의 거부에 대해서는 warn으로, 그 이후에는 debug로 기록합니다(연결하는
  쪽의 at-capacity 줄과 공유하는 자체 예산 클래스입니다). 다른 모든 핸드셰이크 실패는 백오프 없이 종료 상태로
  남습니다. 판정은 클라이언트의 authenticator가 검증된 뒤에만 보내지므로, 시크릿이 없는 주체는 용량 상태를 알 수
  없습니다. 판정은 자체 authenticator(전체 transcript와 판정 바이트에 대한 HMAC)를 지니므로, 경로상의 주체가
  수락을 용량 초과 판정으로, 또는 거부를 수락으로 바꿀 수 없습니다. 연결하는 쪽은 변조된 판정을
  "server's handshake verdict does not verify under the cluster secret"로 보고하며 키를 도출하지 않습니다.

이 경고들은 다섯 개의 클래스로 속도 제한되어, 잘못된 연결(잘못된 magic, 형식이 잘못된 hello, 5초 timeout까지의
정체, 대기 풀 거부. `noise` 클래스)이 구체적인 거부(probable secret mismatch, identity 소유 아님, suite 불일치,
인증에 실패한 프레임, 알 수 없는 종류의 인증된 프레임. `diagnostic` 클래스)를 debug로 밀어내지 못하도록 합니다.
자체 클래스를 가진 세 가지 운영 경고도 마찬가지입니다. 발신자가 설정된 피어가 아니어서 disaggregated router가
버린 결과 프레임(`router-drop`), 연결 풀이 재시도 중인 용량 거부(`at-capacity`), 그리고 그 시퀀스가 거기에
준비되지 않았거나 reply nonce가 없어서 파이프라인 스테이지가 버린 activation 또는 activation 실패(`stage-drop`)입니다.
각 클래스는 분당 경고 5회까지 기록할 수 있으며, 나머지는 debug로 갑니다. 줄이 억제되었던 창 뒤의 해당 클래스 첫
경고는 억제된 개수와 그 창이 끝난 시점을 보고합니다(`N cluster warnings (diagnostic) were logged at debug during the 60s window that ended Ms ago`).
요약은 그 클래스의 다음 이벤트가 도착할 때 기록되므로, 조용한 기간 뒤에는 `M`이 60을 크게 넘을 수 있습니다.

핸드셰이크 와이어 프로토콜은 버전이 붙어 있습니다(`PROTOCOL_VERSION` 5. 3은 #2656에서 프레임 보호 suite와
프레임별 파생 키를 추가했고, 4는 #2692에서 용량 초과 판정을 추가했으며, 5는 #2714에서 판정 authenticator를
추가했습니다). 서로 다른 버전으로 빌드된 프로세스 쌍은 혼란스러운 I/O 오류 대신 명시적인 버전 메시지와 함께
실패합니다. 한 클러스터 안의 혼합 버전은 지원되지 않습니다. 모든 노드를 함께 업그레이드해야 합니다. 지금까지의
모든 버전 상향은 v0.8.0-beta.1 이후에 들어갔으므로, v0.8.0과 그 베타는 같은 클러스터를 공유할 수 없습니다.

잔여 위험:

- 프레임별 보호는 경로상의 공격자가 인증된 연결의 프레임을 읽고, 변경하고, 주입하고, 재생하고, 순서를 바꾸는
  것을 막습니다. 프레임을 버리거나 지연시키는 것은 막지 못합니다. 버려진 프레임은 끊어진 연결과 정확히 같이
  기존의 stage timeout 또는 절대 데드라인으로 드러납니다.
- 프레임 길이와 종류(`TensorData`, `Control`, RPC 요청 또는 응답)는 와이어에서 보이며, 모든 decode step의
  타이밍도 보입니다. 관찰자는 토큰 내용을 읽을 수 없더라도 클러스터가 몇 개의 토큰을 만들고 있는지는 알 수
  있습니다.
- 인증된 멤버의 프레임은 파이프라인 프로토콜 안과 disaggregated 프로토콜의 prefill 및 decode 수신 경로에서 계속
  신뢰됩니다. 시크릿을 가진 호스트는 그곳에서 완전한 멤버입니다. Router의 결과 demux가 예외입니다(설정된 피어로부터의
  프레임만 수락합니다). 멤버를 폐기할 때는 시크릿을 교체하십시오.
- 핸드셰이크를 끝내지 못한 연결은 512개의 인증된 연결 상한에 포함되지 않습니다. 이들은 별도의 대기 풀(동시 64개,
  소스당 최대 8개, 각각 최대 5초의 핸드셰이크 timeout 동안)에서 슬롯을 받습니다. 어느 한도든 넘는 소켓은 accept
  시점에 닫힙니다. 소스는 IPv4 주소 또는 IPv6 /64입니다(/64를 소유한 호스트는 그렇지 않으면 주소마다 대기 슬롯을
  하나씩 얻게 됩니다). 이는 하나의 IPv6 /64 안에 있는 클러스터 노드들이 소스별 예산을 공유한다는 뜻이기도 하며,
  단일 /64에 있는 대규모 IPv6 클러스터는 더 높은 `max_pending_handshakes_per_ip`가 필요할 수 있습니다. 충분히 많은
  소스(기본값에서 8개 이상)에서 온 인증되지 않은 연결의 홍수는 대기 풀을 채울 수 있으며, 그러면 홍수가 지속되는
  동안 실제 피어의 재연결을 포함해 어떤 소스에서 오는 새 연결도 슬롯당 최대 5초 timeout만큼 지연될 수 있습니다.
  이것이 이미 성립된 인증 연결의 슬롯을 빼앗거나 프레임을 주입할 수는 없습니다. 실패한 인바운드 핸드셰이크와 대기
  풀 거부는 분당 처음 몇 번은 warn으로(클래스별, 위 참조), 그 이후에는 debug로 기록됩니다.
- 시크릿이 없는 주체도 "probable secret mismatch" 경고를 만들 수 있습니다(자신의 IP를 적은 올바른 hello를 보낸 뒤,
  서버의 authenticator 이후에 닫습니다). "identity not owned"와 "suite mismatch" 거부는 클라이언트가 인증하기 전
  hello만으로 결정되므로, 그 주체는 이것들도 유발할 수 있습니다. 어느 것이든 diagnostic 경고 예산을 포화시킬 수
  있습니다. 그럴 때 운영자에게 정보를 계속 알려 주는 것이 억제된 경고의 분당 요약입니다.
- 수신자는 헤더만으로 지나치게 큰 `len`을 거부합니다. 하지만 `len`이 상한(기본 256 MiB) 안에 있는 프레임은 태그가
  검사되기 전에 전체 본문을 할당하고 읽게 만듭니다. 이 경로에 도달할 수 있는 것은 핸드셰이크를 완료한 피어뿐이므로,
  이는 열린 네트워크에 대한 노출이 아니라 인증된 멤버에 대한 자원 상한입니다. Transport `rpc_call`(지연 벤치마크가
  사용)의 응답은 대신 1 MiB 상한으로 읽으므로, 응답이 호출당 그 이상을 강제할 수 없습니다. 더 큰 응답은 호출을
  실패시키고 연결을 끊습니다.
- `MLXCEL_CLUSTER_SECRET`은 서버의 자식 프로세스에 상속됩니다. 값이 환경에 남지 않도록 `--cluster-secret-file` 또는
  `MLXCEL_CLUSTER_SECRET_FILE`을 권장합니다.
- 연결하는 쪽에서 서버 identity는 소켓 IP에 바인딩되지 않으므로, NAT이나 포트 포워드를 통해 도달하는 스테이지 주소도
  시크릿만으로 인증됩니다.

### Threat model and per-frame protection

클러스터 transport는 두 가지 네트워크 중 하나를 가정합니다.

- **신뢰 세그먼트**(사설 VLAN, Thunderbolt 브리지, 실험실 스위치): 와이어에 있는 주체는 클러스터 자체의 호스트뿐입니다.
  핸드셰이크는 잘못 설정된 프로세스나 무관한 프로세스를 막기 위해 존재하며, 프레임별 보호는 특정 공격자가 없는
  상황에서도 약간의 지연 비용을 치릅니다.
- **신뢰할 수 없는 네트워크**(공유 사무실 LAN, 다른 테넌트가 있는 클라우드 VPC, 운영자가 통제하지 않는 호스트에서
  도달할 수 있는 모든 것): 공격자가 경로상에 있을 수 있으며(ARP 또는 DHCP 스푸핑, 침해된 스위치, 같은 세그먼트의
  호스트), 패킷을 읽고, 주입하고, 변경하고, 버리고, 재생하고, 순서를 바꿀 수 있으며, 한 피어의 연결을 실제 서버로
  중계할 수 있습니다.

각 계층이 다루는 범위:

| 계층 | 보호 범위 | 보호하지 않는 범위 |
|-------|--------|----------------|
| 연결 시점 핸드셰이크 (`PROTOCOL_VERSION` 5) | 시크릿 보유자만 연결을 열거나 수락할 수 있습니다. 수락된 identity는 TCP 소스 IP에 바인딩됩니다. 프레임 suite는 인증된 transcript 안에서 합의됩니다. 연결 상한 도달은 치명적이지 않은 타입 판정으로 인증된 피어에게 보고됩니다. 판정 자체가 인증되므로, 경로상의 주체가 판정을 바꿀 수 없습니다. | 핸드셰이크 이후 TCP 스트림에서 일어나는 모든 일. |
| 프레임별 보호 (#2656) | 핸드셰이크 이후의 모든 프레임: 페이로드의 기밀성, 헤더와 페이로드의 무결성, 암묵적 시퀀스 번호를 통한 재생 및 순서 변경 거부, 핸드셰이크가 키를 도출한 연결에 대한 바인딩. 시크릿 없이는 키를 도출할 수 없는 중계를 무력화합니다. 풀링된 아웃바운드 연결, `send_stream`, RPC 요청과 응답, Thunderbolt 및 RDMA 래퍼(같은 TCP 코어), disaggregated router, prefill 및 decode transport, `--pp-auto` 스테이지 할당 채널을 포함합니다. | 프레임 드롭 또는 지연, 프레임 길이, 종류, 타이밍. |
| Discovery 비콘 (`--cluster-discovery=mdns`) | 각 UDP 비콘은 HMAC, 타임스탬프, nonce를 담습니다. 코디네이터는 유효하지 않은 MAC, 60 s 이상의 시계 차이, 그 창 안의 재생, 다른 mlxcel 버전, 데이터그램의 소스가 아닌 광고 주소를 거부합니다. | 클러스터가 형성되고 있다는 사실의 은닉. 비콘 페이로드는 읽을 수 있습니다. |

설계(아래 측정 결과로 선택). 핸드셰이크 후 각 측은 HKDF-SHA256으로 32바이트 키 두 개를 도출합니다. IKM은
클러스터 시크릿이고, salt는 클라이언트 nonce 뒤에 서버 nonce를 이은 값이며, info 문자열은
`mlxcel-cluster-frame-v1/<channel>/c2s`와 `mlxcel-cluster-frame-v1/<channel>/s2c`입니다. 여기서 `<channel>`은 TCP
transport 코어의 경우 `transport`, `--pp-auto` 스테이지 할당 채널의 경우 `assignment`입니다. 따라서 핸드셰이크
입력이 같더라도 두 채널은 키를 절대 공유하지 않습니다. 프레임은 `[kind][len][ciphertext][tag]`입니다. 페이로드는
해당 방향의 키로 AES-256-GCM을 사용해 제자리에서 암호화되고, 96비트 nonce는 프레임의 시퀀스 번호입니다(0에서
시작하며, 전송되지 않고, 절대 되감기지 않습니다. 끝에 도달한 송신자는 연결을 닫습니다). 평문 `kind || len` 헤더는
associated data이므로, 수신자는 어떤 복호화 작업보다 먼저 지나치게 큰 `len`을 거부합니다. 복호화에 실패한 프레임은
연결을 닫습니다. 이 이벤트는 실패한 핸드셰이크와 같은 속도 제한기의 diagnostic 클래스를 통해 기록되며, 페이로드
바이트는 절대 포함하지 않습니다. 키는 연결마다 따로 있으므로, 한 연결에서 기록된 프레임은 다른 모든 연결에서
실패합니다. Suite는 핸드셰이크 hello에서 알려지며 두 authenticator가 모두 덮습니다. 다른 suite로 설정된 노드는
명시적 레코드와 함께 거부됩니다. 새로 설정할 것은 없습니다. 위의 시크릿 출처가 유일한 입력이며, opt-out은
없습니다.

기각된 후보:

- *같은 키와 시퀀스를 쓰는 프레임별 HMAC-SHA256(후보 1)*: 무결성, 재생 및 순서 변경 거부, 연결 바인딩은 주지만,
  activation, KV handoff의 프롬프트, 명령이 와이어에서 계속 읽을 수 있게 둡니다. 측정 결과 AES-256-GCM이 결정 규칙의
  여유 범위 안에 들었으므로, 기밀성이 없는 변형은 필요하지 않았습니다.
- *ChaCha20-Poly1305(후보 2의 다른 AEAD)*: 같은 보장을 제공합니다. 대체 AEAD로 측정되어 벤치마크를 위해 코드에
  남아 있습니다. GB10(AES와 PMULL 확장을 가진 ARM Neoverse)과 Apple silicon은 모두 AES 명령을 가지므로
  AES-256-GCM이 기본값입니다.
- *`tokio-rustls`를 통한 TLS 1.3, 노드마다 일회성 인증서를 사용하고 HMAC 핸드셰이크를 TLS exporter에 바인딩(후보 3)*:
  프로토타입을 만들지 않았습니다. 그 record layer는 후보 2와 같은 AEAD 프리미티브에 16 KiB record framing을 더해
  실행하며, 연결마다 두 번째(TLS) 핸드셰이크를 하고 시작 시 인증서를 생성하므로, 후보 2보다 빠를 수 없습니다.
  결정 규칙은 3이 느리지 않은 경우가 아니라면 2를 선호합니다. 또한 런타임 의존성으로 `rcgen`을 추가하게 됩니다.

측정(NVIDIA GB10 호스트 한 대에서 loopback, 2026-10-27, 다른 작업 없음. 단방향 지연은 피어가 두 번째 인증 연결을
통해 되돌려 보내는 `TensorData` 메시지 왕복 시간의 절반이므로, 양쪽 끝의 프레임 작업이 모두 포함됩니다):

| 모드 | 페이로드 | 반복 횟수 | p50 | p95 | p99 | 단방향 대역폭 |
|------|--------:|-----------:|----:|----:|----:|------------------:|
| 보호 없음 (기준선, #2656 이전의 main) | 8 KiB | 4000 | 5.5 us | 5.6 us | 6.2 us | 1.38 GiB/s |
| 보호 없음 | 64 KiB | 2000 | 11.1 us | 11.2 us | 12.0 us | 5.51 GiB/s |
| 보호 없음 | 1 MiB | 400 | 411 us | 418 us | 419 us | 2.37 GiB/s |
| 보호 없음 | 2 MiB | 200 | 1.040 ms | 1.059 ms | 1.096 ms | 1.87 GiB/s |
| 보호 없음 | 16 MiB | 60 | 16.39 ms | 16.52 ms | 16.62 ms | 0.95 GiB/s |
| 후보 1: HMAC-SHA256 | 8 KiB | 4000 | 12.2 us | 12.4 us | 12.9 us | 0.63 GiB/s |
| 후보 1: HMAC-SHA256 | 64 KiB | 2000 | 64.1 us | 64.8 us | 65.3 us | 0.95 GiB/s |
| 후보 1: HMAC-SHA256 | 1 MiB | 400 | 1.008 ms | 1.016 ms | 1.025 ms | 0.97 GiB/s |
| 후보 1: HMAC-SHA256 | 2 MiB | 200 | 2.115 ms | 2.150 ms | 2.216 ms | 0.92 GiB/s |
| 후보 1: HMAC-SHA256 | 16 MiB | 60 | 27.22 ms | 27.56 ms | 28.29 ms | 0.57 GiB/s |
| 후보 2: AES-256-GCM (선택) | 8 KiB | 4000 | 7.4 us | 7.6 us | 8.2 us | 1.02 GiB/s |
| 후보 2: AES-256-GCM (선택) | 64 KiB | 2000 | 27.4 us | 27.9 us | 28.5 us | 2.23 GiB/s |
| 후보 2: AES-256-GCM (선택) | 1 MiB | 400 | 422 us | 426 us | 434 us | 2.31 GiB/s |
| 후보 2: AES-256-GCM (선택) | 2 MiB | 200 | 0.926 ms | 0.944 ms | 0.954 ms | 2.10 GiB/s |
| 후보 2: AES-256-GCM (선택) | 16 MiB | 60 | 17.93 ms | 18.09 ms | 18.38 ms | 0.87 GiB/s |
| 후보 2: ChaCha20-Poly1305 | 8 KiB | 4000 | 14.0 us | 14.4 us | 14.9 us | 0.56 GiB/s |
| 후보 2: ChaCha20-Poly1305 | 64 KiB | 2000 | 75.4 us | 76.2 us | 76.8 us | 0.81 GiB/s |
| 후보 2: ChaCha20-Poly1305 | 1 MiB | 400 | 1.192 ms | 1.204 ms | 1.220 ms | 0.82 GiB/s |
| 후보 2: ChaCha20-Poly1305 | 2 MiB | 200 | 2.479 ms | 2.489 ms | 2.504 ms | 0.79 GiB/s |
| 후보 2: ChaCha20-Poly1305 | 16 MiB | 60 | 30.16 ms | 30.55 ms | 30.65 ms | 0.52 GiB/s |
| 임의의 모드 | 임의 | 두 노드 링크 (Thunderbolt, 10/25 GbE) | 이번 실행에서 측정하지 않음 | | | |

1 MiB와 2 MiB 행은 main의 JSON 숫자 배열 인코딩에서 토큰당 logits 응답을 둘러싸는 구간입니다. 아래의 base64
인코딩에서는 그 응답이 약 0.4 MB이며, 이는 64 KiB 행과 1 MiB 행 사이에 해당합니다. 이 크기에서 AES-256-GCM 행은
보호되지 않은 행과 loopback 노이즈 범위 안에 있습니다(하드웨어 AES와 PMULL이 loopback 복사보다 빠르게 동작하기
때문입니다). 반면 HMAC-SHA256은 단방향 전송마다 0.6 ms와 1.2 ms를 더하고, ChaCha20-Poly1305는 0.8 ms와 1.4 ms를
더합니다.

결정 규칙(#2656): 8 KiB에서 추가되는 p50 단방향 지연이 후보 1의 20 us 또는 5 percent 중 큰 값 이내이고,
end-to-end decode 저하가 최대 2 percent일 때 기밀성을 제공하는 옵션을 선택합니다. 8 KiB에서 AES-256-GCM은 HMAC
후보보다 4.8 us *빠르며*(ring의 GCM은 AES와 PMULL 명령을 사용하고, 이 코어에서 SHA-256 경로는 더 느립니다), 보호되지
않은 기준선보다 1.9 us 높습니다. 따라서 지연 조항은 여유를 두고 충족됩니다. ChaCha20-Poly1305는 HMAC보다 1.8 us
느려서 선호되지 않았습니다.

두 스테이지 원격 파이프라인(qwen3-0.6b-4bit, 128 토큰, 두 스테이지와 코디네이터를 모두 loopback 위의 한
프로세스에서 실행, `tests/pipeline_remote_decode_bench.rs`)의 end-to-end greedy decode입니다. 기준선과 후보를
교차 실행했습니다:

| 바이너리 | 실행 횟수 | 실행별 중앙값의 중앙값 | 실행별 중앙값 (최소 ~ 최대) |
|--------|-----:|--------------------------:|------------------------------|
| main 4144ecfb, 보호 없음, 배열 인코딩 텐서 | 5 | 68.3 tok/s | 65.4 ~ 72.7 tok/s |
| main cfa14e7b (HEAD), 보호 없음, 배열 인코딩 텐서 | 5 | 68.7 tok/s | 60.7 ~ 69.7 tok/s |
| 이 브랜치, 코덱 비활성화(귀속 확인용): 압축 텐서, 보호 없음 | 5 | 138.8 tok/s | 136.5 ~ 146.7 tok/s |
| 이 브랜치: 압축 텐서, 모든 프레임에 AES-256-GCM | 5 | 133.2 tok/s | 122.2 ~ 145.1 tok/s |

main 기준선과 비교하면 보호된 파이프라인은 94 percent 더 빠르게 decode하지만, 이 이득은 함께 배포되는 압축
인코딩에서 나옵니다. 보호만 따로 보면 2 percent 조항은 충족되지 않습니다. 두 압축 행에서 읽은 보호 자체의 비용은
7.3 ms step의 4 percent입니다(405 KB 프레임 두 개와 작은 프레임들을 합쳐 토큰당 약 0.3 ms). 측정 범위는
겹칩니다. 이는 토큰마다 고정으로 드는 비용이며, 파이프라인 병렬화가 필요한 모델에서는 비례하여 줄어듭니다. 규칙의
fallback인 후보 1은 위의 모든 페이로드 크기에서 AES-256-GCM보다 비용이 더 크고(1.5배에서 2.4배), 페이로드를 평문으로
남기므로, fallback하면 처리량이 더 떨어집니다. 결정: AES-256-GCM, opt-out 없음.

귀속 분석(일회성 계측, 커밋되지 않음): 벤치의 프레임 추적은 디코드된 토큰당 프레임 네 개를 보여 주었습니다. 275바이트
`RunEntry`, 7.5 KB forward activation, 그리고 각 1.1 MB인 프레임 두 개가 있었는데, 이 둘은 숫자 JSON 배열로 된
304 KB의 f16 logits를 담고 있었습니다(최종 스테이지가 이를 entry 스테이지에 반환하고, entry 스테이지가 코디네이터에
전달합니다). 이 2.2 MB에 대한 프로세스 내 AES-256-GCM은 토큰당 0.54 ms(seal 275 us, open 268 us. HMAC-SHA256이라면
1.77 ms)가 들었습니다. 코덱의 프레이밍만 바꾼 변형은 보호된 버전보다 빠르게 디코드되지 않았습니다. 따라서 처음
측정된 차이(배열 인코딩에서 71.3 대 63.5 tok/s)는 프레이밍이 아니라 AEAD 비용과 실행 간 잡음이었습니다. 아래의
압축 인코딩은 main 자체의 스텝 시간 대부분을 제거했습니다. 그 시간은 JSON 숫자 배열을 인코딩하고 파싱하는 데
쓰였으며, 그 결과 보호된 파이프라인은 보호되지 않은 기준선을 크게 앞질렀습니다.

같은 릴리스의 와이어 변경: 스테이지 프로토콜의 대용량 바이트 필드(`RunEntry.input_ids`와 `attention_mask`,
`RunResult.logits`, 그리고 activation 메시지의 `tensor_data`, `attention_mask`, `position_ids`)는 JSON 문서 안에서
숫자 JSON 배열 대신 base64 문자열이 되었습니다(바이트당 약 3.6 문자 대신 1.33 문자). qwen3-0.6b-4bit의 디코드
토큰은 1.1 MB 프레임으로 와이어를 두 번 통과했습니다(최종 스테이지가 logits를 entry 스테이지에 반환하고, entry
스테이지가 코디네이터에 전달). 이제 양방향 각각 약 0.4 MB가 되었습니다. 이로써 기준선 스텝 시간과 AEAD가 덮는
바이트가 모두 줄었습니다. 이 인코딩을 쓴 빌드는 쓰지 않은 빌드와 상호 운용되지 않습니다. 혼합 버전은 이미 지원되지
않았습니다.

후속 변경(#2694): 최종 스테이지가 이제 `RunResult`를 코디네이터에 곧바로 보냅니다. 따라서 디코드된 토큰은 4개 대신
프레임 3개가 되며, logits는 와이어를 한 번만 통과합니다. 송신자는 또한 봉인하기 전에 페이로드를 두 번째 버퍼에
복사하는 대신 메시지를 제자리에서 봉인합니다. Forward activation은 새 `reply_nonce` 필드에 교환 nonce를 담습니다.
이 필드 없이 activation을 받는 최종 스테이지(필드 이전의 entry 스테이지에서 온 경우)는 교환을 추측하지 않고 경고와
함께 logits를 버립니다.

두 노드 링크 행(Thunderbolt 또는 10/25 GbE)은 이번 실행에서 측정하지 않았습니다. 벤치마크(`cargo test --test cluster_frame_bench -- --ignored --nocapture`)는
두 transport를 모두 127.0.0.1에 바인딩하므로, 두 노드 측정에는 두 번째 호스트의 transport를 대상으로
`measure_activation_transfer`(왕복 모드)를 실행해야 합니다.

## Transports

| Transport | 참고 |
|-----------|-------|
| TCP | 기본 IP transport입니다. |
| Thunderbolt | 공유 TCP 코어 위에서의 macOS Thunderbolt Bridge 선택입니다. |
| RDMA | capability probing과 폴백 동작을 갖춘 백엔드가 있습니다. 가속에 의존하기 전에 대상 OS/하드웨어에서 검증하십시오. |

정적 탐색과 mdns 탐색 모두 [zero-config 기동](#zero-config-bring-up)에 사용할 수 있습니다. 서브넷을 넘나들거나
UDP 브로드캐스트를 버리는 네트워크에서는 정적 탐색을 선택하십시오. mdns는 피어의 비콘이 코디네이터의
`--cluster-discovery-port`에 도달해야 합니다.

## Disaggregated inference

DI는 prefill과 decode 역할을 분리합니다. PP와 달리 노드당 모델 메모리를 줄이지 **않습니다**.
각 역할이 여전히 모델을 로드해야 합니다. 의도된 사용 사례는 용량 초과 모델을 맞추는 것이 아니라 처리량 튜닝입니다.

이 코드는 PP와 동일한 클러스터 설정, registry, transport, 메트릭 인프라를 공유합니다.
토폴로지에 종속된 기능으로 취급하십시오. 성능 주장을 게시하기 전에 사용자의 트래픽 형태로 실제 테스트를 수행하십시오.

모델 지원은 PP보다 좁습니다. paged prefill/decode 핸드오프는 풀 기반 Fp16 패밀리
(qwen3, llama3 같은 dense-natural 백엔드)에서만 동작합니다. 모델이 직접 소유하는 paged 패밀리(gemma3, gemma4, llama4, qwen3_5,
qwen3_next)는 KV 상태를 공유 블록 풀 밖에 보관하므로 핸드오프 과정에서 직렬화할 수 없습니다.
이러한 패밀리에 대한 요청은 prefill 작업이 실행되기 전에 prefill 노드에서 요청 단위의 깔끔한 오류로 거부되며,
노드는 이후 요청을 계속 처리합니다(issue
#708). 전체 범위와 제한 사항 목록은
[continuous batching 및 disaggregated serving](CONTINUOUS_BATCHING.md#범위와-제한-사항)을 참조하십시오.

`--node-role router` 프런트는 독립적인 prefill 풀과 decode 풀의 균형을 맞춥니다. 요청마다 prefill 노드와 decode
노드를 모두 선택하고(registry가 온라인으로 표시한 노드에 대한 round-robin), 선택된 decode 노드를 요청 프레임에 담아
prefill 노드에 전달하여 prefill이 라우터가 균형을 맞춘 decode 노드로 핸드오프하게 합니다. 도달할 수 없는 피어는
다운으로 표시하고(전송 오류 감지와, 시작 시 첫 주기가 실행되고 그 후 3 s마다 돌아가는 백그라운드 liveness probe를
통해), 정상 노드로 failover합니다. 요청을 받을 수 있는 노드가 없으면 HTTP 503을 반환하는 admission control을
적용합니다. 일부 prefill 노드가 다운으로 표시된 상태에서 사용 가능한 prefill 노드가 없다는 이유로 거부하기 전에,
router는 다운으로 표시된 노드를 다시 probe합니다(500 ms connect timeout, router 전체에서 250 ms당 최대 한 번의
재-probe, 동시 요청들이 공유). 그리고 되돌아온 노드가 있으면 요청을 처리합니다. 따라서 probe 주기 후에 시작을
마친 피어는 다음 주기까지 거부되지 않습니다(issue #2763). `GET /health`는 router 프로세스 자체의 liveness 확인으로
남습니다. `GET /router/stats`는 노드별 부하 분포와 노드 상태를 보고합니다.
플래그와 동작은 [continuous batching 및 disaggregated serving](CONTINUOUS_BATCHING.md#다중-노드-라우팅-부하-분산-failover)을
참조하십시오.

### 보안 및 신뢰 모델

disaggregated 역할 transport 연결(router에서 prefill로, prefill에서 decode로, 그리고 router로 돌아오는 모든 결과
프레임)은 파이프라인 역할과 같은 [클러스터 시크릿](#cluster-authentication)으로 인증합니다(issue #2655). 모든
`--node-role prefill`, `decode`, `router` 프로세스는 시크릿을 받아야 하며 없으면 시작을 거부합니다. 다른 시크릿을
가진 피어는 연결 시점에 거부되며, 핸드셰이크를 건너뛴 raw TCP 클라이언트는 어떤 역할의 receive loop에도 프레임을
전달받지 못합니다. 이후 이 연결들의 모든 프레임은 핸드셰이크에서 파생된 키로 암호화되고 인증됩니다(issue #2656,
[위협 모델과 프레임별 보호](#threat-model-and-per-frame-protection) 참조). TLS는 사용하지 않습니다. 라우터의
클라이언트용 HTTP 포트와 모든 내부 역할 transport 포트를 포함한 disaggregated 클러스터 전체는 여전히 신뢰
세그먼트 안이나, 신뢰할 수 없는 호출자가 내부 포트에 닿지 못하도록 막는 네트워크 격리 경계(VPC 보안 그룹, 방화벽
규칙, Kubernetes NetworkPolicy) 뒤에서 실행되어야 합니다. 시크릿은 외부인이 역할 포트에 접근하지 못하게 하지만,
격리를 대체하지는 않습니다.

시크릿을 포함한 실행 형태입니다(세 호스트 모두 같은 파일을 사용합니다. 전체 플래그 목록은
[continuous batching 및 disaggregated serving](CONTINUOUS_BATCHING.md#예시-localhost의-프로세스-세-개)을 참조하십시오):

```bash
mlxcel-server -m models/<ckpt> --port 8302 --node-role decode \
    --serving-bind 10.0.0.2:8304 --cluster-secret-file /etc/mlxcel/cluster.secret ...
mlxcel-server -m models/<ckpt> --port 8301 --node-role prefill \
    --serving-bind 10.0.0.1:8305 --decode-peers 10.0.0.2:8304 \
    --cluster-secret-file /etc/mlxcel/cluster.secret ...
mlxcel-server -m models/<ckpt> --port 8300 --node-role router \
    --serving-bind 10.0.0.3:8306 --prefill-peers 10.0.0.1:8305 --decode-peers 10.0.0.2:8304 \
    --cluster-secret-file /etc/mlxcel/cluster.secret
```

`--serving-bind`는 피어가 이 노드의 연결 소스로 보는 IP를 지정해야 합니다(multi-homed 호스트에서는 `0.0.0.0`이
아닙니다). 수락하는 쪽이 광고된 identity를 TCP 소스 IP에 바인딩하고, 불일치는 클러스터 인증 항목에서 설명한
identity 거부로 거부하기 때문입니다.

다중 노드 라우팅 인터페이스에 대한 두 가지 구체적인 운영 참고 사항이며, 둘 다 신뢰 세그먼트 기준선을 넘어서는
심층 방어(defense-in-depth)로 강화되었습니다(issue #389).

- `GET /router/stats`는 `/v1/chat/completions`와 동일한 인터페이스인 라우터의 클라이언트용 HTTP 포트에 마운트됩니다.
  기본적으로 등록된 각 피어의 원시 host:port는 가려지고, 라우터가 할당한 안정적인 노드 id(`prefill-<i>` / `decode-<i>`),
  역할, 상태, 노드별 디스패치 수, 라우터 메트릭 스냅샷만 보고합니다. 이것만으로도 부하 분포를 읽고 실패한 노드가
  도달 불가로 표시되었는지 확인하기에 충분하며, 추론 포트에 접근할 수 있는 클라이언트에게 내부 클러스터 토폴로지(원시
  주소)를 노출하지 않습니다. 이 모드에서는 응답에 `"addresses_redacted": true`가 포함됩니다.
  `MLXCEL_ROUTER_STATS_VERBOSE=1`(`true` / `yes` / `on`도 가능, 대소문자 구분 없음)을 설정하면 신뢰 세그먼트 배포에서
  전체 주소 보기(`"addresses_redacted": false`)로 되돌릴 수 있습니다. 이 가림 처리는 심층 방어이지 네트워크 격리의
  대체물이 아닙니다. 그와 무관하게 라우터의 클라이언트 포트를 내부 포트와 동일한 신뢰 세그먼트에 두십시오.

- prefill 노드는 KV cache 핸드오프(프롬프트를 인코딩한 것)를 전달하기 위해 수신 요청 프레임의 `decode_target` 필드에
  있는 주소로 연결합니다. 라우터는 이 필드를 자신이 설정한 노드 registry로부터만 채우므로, 신뢰 세그먼트 가정 하에서
  주소는 운영자가 통제합니다. 선택적 심층 방어로서 prefill 노드는 연결하기 전에 `decode_target`을 allowlist에 대해
  검증할 수 있습니다. allowlist 밖의 대상은 연결 대신 거부되어(요청은 명확한 오류로 실패하고 KV cache는 알 수 없는
  주소로 나가지 않음), 침해된 세그먼트에서 위조된 프레임이 핸드오프를 클러스터 밖으로 리디렉션할 수 없습니다.
  allowlist의 출처는 전용 `MLXCEL_DECODE_ALLOWLIST` 환경 변수이며, 라우터가 선택할 수 있는 decode 노드의 전체 집합
  (공유 클러스터 설정)으로 설정한 `host:port` 주소의 쉼표 구분 목록입니다.

  `--decode-peers`가 아닌 전용 변수를 쓰는 이유: 라우터는 자신의 풀에 있는 어떤 decode 노드에 대한 `decode_target`이든
  임의의 prefill 노드에 넘길 수 있으므로 allowlist는 라우터가 선택할 수 있는 모든 decode 노드를 포괄해야 하는 반면,
  prefill의 `--decode-peers`는 정적 핸드오프 폴백(라우터가 `decode_target`을 생략했을 때 사용하는 첫 번째 항목)일 뿐이며
  정당하게 피어 하나만 나열할 수 있습니다. allowlist를 `--decode-peers`에서 가져오면 그 정적 집합 밖의 라우터가 균형을
  맞춘 모든 대상이 거부되어 issue #201에서 추가된 다중 노드 decode 부하 분산이 깨집니다. 둘을 독립적으로 유지하십시오.
  각 prefill 노드에서 `MLXCEL_DECODE_ALLOWLIST`는 전체 풀로 설정하고, `--decode-peers`는 로컬 폴백으로 남겨 두십시오.
  `MLXCEL_DECODE_ALLOWLIST`가 설정되지 않았거나(또는 유효한 항목이 하나도 파싱되지 않으면) prefill은 거부하는 대신
  허용 상태를 유지하고 경고를 기록하므로, allowlist 출처가 없다고 해서 부하 분산이 조용히 비활성화되는 일은 없습니다.
  항목은 숫자 `IP:port` 값이어야 하며(`SocketAddr`로 파싱됨), hostname:port 항목은 파싱되지 않고 경고와 함께
  건너뛰므로 라우터가 선택한 대상과 절대 일치하지 않습니다. 파싱되지 않는 항목은 시작을 실패시키는 대신 경고와 함께
  건너뜁니다. `decode_target`이 없는 요청 프레임(이전 버전의 라우터)은 여전히 prefill의 첫 번째 정적 `--decode-peers`
  항목으로 폴백하며, 이는 와이어로 전달된 주소가 아니라 운영자 설정이므로 검증이 필요 없습니다.

## 공통 제한 사항

- 분산 지원은 모델 패밀리 전반에 걸쳐 균일하지 않습니다.
- VLM 파티셔닝은 부분적이며, 텍스트 전용 경로가 더 잘 커버됩니다.
- 멀티 호스트 CI 커버리지는 단일 호스트 단위 테스트에 비해 제한적입니다.
- Transport 성능은 물리적 인터커넥트와 OS 네트워크 설정에 크게 좌우됩니다.

유지 관리되는 지원 요약은 [지원 모델](supported-models.md)을 참조하십시오.
