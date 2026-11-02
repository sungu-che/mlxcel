# Continuous batching과 disaggregated serving

`mlxcel-server`(및 `mlxcel serve`)는 continuous-batching 스케줄러를 실행합니다. 동시에 들어온 요청을 실행 중인 batch에 받아들이고, prefill과 decode를 교차 실행하며, 각 요청의 토큰이 생성되는 대로 스트리밍합니다. 이 문서는 batching 스케줄러, 스케줄러가 paged KV cache를 사용하는 방식, 그리고 prefill과 decode를 프로세스 간에 분리하는 disaggregated serving 역할을 다룹니다.

## Continuous batching

스케줄러는 최대 `--parallel`개의 시퀀스를 동시에 활성 상태로 유지합니다. 매 step마다 대기 중인 프롬프트를 받아들여 prefill하거나(다른 시퀀스가 decode 중이 아닐 때는 2048 토큰, decode batch가 살아 있을 때는 512 토큰 단위로 chunk 처리하며, 재정의하지 않은 경우), 활성 batch를 decode 토큰 하나만큼 전진시킨 다음 새 토큰을 스트리밍합니다. 무엇을 실행할지는 스케줄러가 결정하고, 실행은 엔진이 합니다. `mlxcel_core::engine::Engine`이 모델과 KV pool을 소유하며, 스케줄러가 필요로 하는 모든 모델 forward, sampler draw, finish 단계는 `Engine::prefill`, `Engine::step`(모든 행 수에 대한 단일 진입점이며, 요청 하나도 크기 1의 batch입니다), 그리고 파이프라인 방식의 `submit` / `finish_rows` 쌍을 거칩니다([ADR 0007](adr/0007-unified-batch-native-engine.md), [ADR 0009](adr/0009-engine-step-api.md)). 크기 1의 batch는 행별 체인으로 샘플링하고, fused 호환 파라미터를 공유하는 행들로 이루어진 더 큰 batch는 `[B, vocab] -> [B]` fused draw 한 번을 사용합니다. 어느 쪽이든 샘플링된 토큰은 읽히기 전에 fallible MLX 경계를 거쳐 평가되므로, MLX가 예외를 던지면 서버를 중단시키는 대신 영향받은 요청을 `inference backend error`로 종료합니다(#822). Admission, tick 정책, preemption, prompt cache, disaggregated handoff, speculative burst 생성기는 스케줄러에 남아 있습니다. 관련 플래그는 다음과 같습니다.

| 플래그 | 기본값 | 용도 |
|------|---------|---------|
| `--parallel N` | 4 | 최대 활성(진행 중) 시퀀스 수. 동시 decode batch의 상한입니다. |
| `--max-batch-size N` | (= `--parallel`) | 한 번의 batched step에서 함께 decode되는 최대 시퀀스 수. |
| `--max-batch-prefill N` | 4 | 하나의 prefill forward pass로 묶는 요청 수(지원하는 계열에 한함). |
| `--max-batch-prefill-tokens N` | (유도됨) | batched prefill 하나의 일시적 메모리를 제한하는 패딩된 토큰 예산. 설정하지 않으면 `2 * max_batch_prefill * min(prefill_chunk_size, 512)`로 유도되며, `0`은 상한을 비활성화합니다. |
| `--max-queue-depth N` | 32 | 대기 중인(아직 받아들여지지 않은) 요청의 최대 수. |
| `--prefill-chunk-size N` | 단독 2048, 활성 decode가 있으면 512 | 명시적으로 설정한 경우 두 스케줄러 상태 모두에 적용되는 토큰 chunk 크기. `--batch-size` / `LLAMA_ARG_BATCH`와 `MLXCEL_PREFILL_CHUNK`는 동등한 재정의 수단입니다. 설정하지 않으면 스케줄러가 tick마다 선택하여, 단일 스트림 TTFT를 유지하면서 admission이 활성 decode 지연에 미치는 영향을 제한합니다(#2228). |
| `--prefill-grant-interval N` | 16 | 보류된(parked) chunked prefill이 다음 tick을 grant받기 전까지 양보하는 decode tick 수. 받아들여진 긴 프롬프트의 첫 토큰까지 걸리는 시간을 제한합니다. `0`은 grant를 비활성화합니다(무한 대기). |
| `--kv-admission-watermark F` | 0.01 | 행들이 decode 중일 때 prefill admission이 비워 두는 paged KV block 예산의 비율(`0.0`~`0.5`). 행들이 preemption되지 않고 늘어날 수 있게 합니다. `0`은 비활성화합니다. 아래를 참조하십시오. |
| `--enable-preemption` | off | 대기 중인 시퀀스를 받아들이기 위해 우선순위가 더 낮은 시퀀스를 축출하도록 허용합니다. |
| `--no-batch` | off | batching을 끄고 순차적으로 서비스합니다(기존의 단일 worker). |
| `--no-prompt-cache` | off | prompt-prefix KV cache를 끕니다(기본값은 켜짐). |

### 서빙 처리량 기본값

출하되는 기본값은 다중 클라이언트 서빙에 맞춰 조정되어 있습니다. batched decode는 step을 공유하는 시퀀스 수와 관계없이 step마다 각 가중치를 한 번만 읽으므로, batch가 compute roofline에 도달할 때까지 총 처리량은 동시성에 따라 확장되고, 요청별 decode 속도는 그에 비례해 떨어집니다. 기본값은 다음과 같습니다.

- `--parallel 4`: decode batch를 최대 4개까지 받아들입니다. `supports_batching()`이 false인 계열에 대해서는 worker가 이를 1로 제한하므로, 이 기본값은 모든 아키텍처에서 안전합니다. 명시적인 분할 구성이 이렇게 제한되는 경우, worker는 구성된 전체 context를 하나의 유효 요청 윈도우로 복원하고 로드 후의 그 값을 `/props`, `/slots`, `/v1/models`에 게시합니다. 나머지 clamp는 epic의 class D 제외 항목입니다. diffusion 생성(`diffusion_gemma`, `llada2_moe`), 문서/encoder-decoder 런타임(`florence2`, `nemotron_parse`, Falcon OCR text/VLM 및 `unlimited_ocr`), session이 소유하는 `nemotron_voicechat` 음성 파이프라인, Phi4MM-VL의 요청별 LoRA 선택, 모델 소유 cache 세트를 하나만 유지하는 NemotronNAS와 LongCat Flash/Flash Ngram(#2544. NemotronNAS는 `num_layers()`마다가 아니라 attention block마다 cache를 하나씩 할당하므로, batch 가능한 레이아웃은 후속 작업입니다), 그리고 `TensorParallelGemma4Model`의 로드된 모델 fallback입니다. rank 분할 Gemma 4 경로와 `Gemma4UnifiedModel`은 이미 batching을 지원하므로 clamp되지 않습니다. 이 제외 항목들은 #2574에서 batched-decode 처리량에 대한 주장을 하지 않습니다.
  `supports_batching()`이 true라고 해서 그 계열이 tick마다 forward 한 번을 실행한다는 뜻은 아닙니다. native decode 경로를 가진 계열은 Llama 3, Llama 4, Qwen 3, Gemma 3, Gemma 3n text/VLM, Gemma 4 text/VLM/Unified, Helium, Qwen3-MoE, Ministral 3, ExaOne 4, Mellum, Laguna, Step 3.5, Step 3.7 VLM text decode, AFMoE (Trinity), DeepSeek V4, IQuestLoopCoder, Inkling text/VLM입니다. 뒤의 네 모델 소유 계열은 시퀀스별 attention cache를 유지하면서 projection과 feed-forward 연산을 batch로 처리합니다. Inkling은 convolution tail을 쌓고, DeepSeek V4는 행별로 pooling을 갱신합니다. AFMoE는 해당 커널이 활성화되어 있을 때 행마다 full-fused B=1 expert 구현을 유지하며, 지원되지 않거나 비활성화된 구성은 native whole-batch expert graph를 유지합니다.
  그 밖의 native 계열은 Apertus, Cohere, Cohere 2, Ernie 4.5 dense, ExaOne, Gemma, Gemma 2, GLM-4, Granite, Hunyuan v1 dense, GPT-2, GPT-BigCode, GPT-NeoX, GPT-OSS, InternLM 2/3, MiMo, MiniCPM, MiniMax-M3, NanoChat, Nemotron, OLMo, OLMo 2/3, OpenELM, Phi, Phi-3, Phi-3-small, StableLM, BitNet, Seed-OSS, SmolLM3, StarCoder2, TeleChat3, Mixtral, PhiMoE, Phixtral, Qwen2-MoE, OLMoE, DBRX, DeepSeek v1, GLM-4 MoE, ERNIE 4.5 MoE, Hunyuan MoE, Bailing MoE, Dots1, MiniMax, Klear, Solar Open, Exaone MoE, Cohere 2 MoE이며, 여기에 MLA backbone인 DeepSeek V2/V3/V3.2, MiniCPM3, Mistral 4, GLM-4 MoE Lite, Youtu와 Qwen2-VL/Qwen2.5-VL/Qwen3-VL/Qwen3-VL-MoE/Qwen3-Omni의 text-decode 경로가 더해집니다. Qwen VL은 식별된 단일 토큰 행을 행 순서를 보존하면서 plain-RoPE 그룹과 MRoPE 그룹으로 나눕니다. prefill, 누락된 sequence id, 저장된 위치 행은 정확한 행별 dispatch를 유지합니다. Qwen3 DeepStack 상태는 sequence id로 키가 지정됩니다(#2567). image feature를 가진 행은 plain 그룹을 벗어나며, decode는 절대 주입하지 않으므로 whole-batch fallback은 남지 않습니다. GLM-4V(GLM-OCR에서도 사용), GLM-4V MoE, Hunyuan-VL, ERNIE-4.5-MoE-VL, PaddleOCR-VL, Cohere Compass도 마찬가지로 식별된 decode 행을 plain/MRoPE 위치 모드별로 묶고, 그룹마다 decoder graph 하나를 실행하며, logit을 scheduler 순서로 되돌려 배치합니다. 서로 다른 시퀀스별 delta는 native로 유지됩니다. 저장된 위치 행, prefill, 누락된 sequence id는 정확한 행별 dispatch를 유지합니다. Compass는 sliding/full attention 분할을 보존하며, 그 DeepStack 상태도 sequence id로 키가 지정됩니다(#2567). ERNIE decode는 항상 text expert bank를 사용하고, image-token expert dispatch는 prefill에서만 남습니다. tensor-parallel Llama, Qwen 3, Qwen 3.5, Gemma 3, ERNIE 4.5, Hunyuan v1 dense 런타임도 각 rank와 layer에서 활성 행들에 걸쳐 하나의 native decode graph를 공유합니다. Phi-3V, Phi-4-SigLIP, Granite Vision, DeepSeek-OCR, OCR-2, GLM MoE DSA, DeepSeek-VL2, Kimi-VL, MiniMax-M3-VL, Youtu-VL은 text 전용 decode를 native text tower에 위임합니다. Molmo, Molmo 2, Molmo-Point, Moondream 2/3, Jina VLM은 활성 decode 행들에 걸쳐 text tower를 native로 실행하며, VLM wrapper는 vision tower에 들어가지 않고 그 token 전용 경로를 위임합니다. DotsOCR와 MiniCPM-o도 이미 native인 Qwen 2 및 Qwen 3 tower에 위임합니다. Mllama는 요청별 이미지 상태(cross-attention 상태와 그로부터 유도된 layer별 이미지 K/V)를 sequence id 아래에 보관합니다(#2567). 이미지가 없는 행은 cross-attention layer가 항등인 상태에서 native로 batch되고, 이미지가 있는 행은 자기 항목에 대해 하나씩 실행되어 scheduler 순서로 되돌려 배치됩니다. Baichuan-M1은 layer별 K/V convolution 상태도 sequence id로 키를 지정하며(#2567), sequence 인식 dispatch를 통해 행별로 decode하므로 행끼리 상태를 공유하지 않습니다. 다만 batch 전체에 대해 하나의 fused graph를 실행하지는 않습니다. conv 상태는 최근 append의 제한된 이력을 유지하여(prefill 토큰도 포함하므로 이력이 연속적으로 남습니다), pipelined decode lookahead를 정확하게 되돌릴 수 있습니다. Baichuan-M1은 pipelined lookahead를 유지하고, KV trim과 함께 `rewind_decode_appends`를 통해 conv 상태를 되감습니다. padding 토큰이 conv 상태의 "마지막 토큰"이 되고 scheduler의 KV trim이 그 상태에 닿지 않으므로 padded prefill은 없습니다. conv 상태는 scheduler의 KV slice에 포함되지 않으므로(`LanguageModel::kv_cache_captures_sequence_state()`가 `false`), prompt-cache adopt/donate, Automatic Prefix Caching, disaggregated KV handoff에서 제외됩니다. 복원된 KV prefix는 마지막 토큰의 conv 상태 없이 도착하게 되기 때문입니다.
  GPT-OSS는 시퀀스 id 아래에 혼합된 full/sliding KV cache를 유지합니다. 그 native decode는 projection, norm, expert, output head를 batch로 처리하지만, attention은 학습된 sink를 보존하고 독립된 sliding-window 오프셋을 유지하기 위해 행마다 실행합니다. rotating 상태는 padding을 trim할 수 없으므로 padded prefill을 비활성화합니다. 이 모델이 소유한 상태에 대해서는 prompt-cache 재사용과 decode lookahead도 비활성화된 상태로 남습니다.
  Mamba2, Mamba1/Falcon-Mamba, Jamba, LFM2/LFM2-MoE, RecurrentGemma, RWKV7, Nemotron-H, GraniteMoeHybrid, Falcon-H1, PLaMo2는 시퀀스 id 아래에 recurrent 상태를 유지하며, 활성 행들에 걸쳐 native width-one decode를 실행합니다(#2571, #2573). 기존 recurrence는 쌓인 상태에 대해 한 번 실행되고, 이후 각 행이 독립적인 상태 복사본을 받습니다. attention KV는 행 단위로 유지되며, RecurrentGemma의 독립적으로 감싼 rotating 윈도우도 여기에 포함됩니다. 초기화된 recurrent 상태와 없는 상태가 섞인 batch는 그 step에 대해 식별된 행별 forwarding으로 fallback하며, 다중 토큰 또는 마스크된 호출도 마찬가지입니다. LFM2-VL과 Nemotron-H Nano Omni는 text decode와 모델 소유 저장소 계산을 자기 tower에 위임합니다. Granite 4 Vision은 `supports_batching()`을 활성화한 채로 두지만, 식별된 행들을 의도적으로 따로 실행합니다. Apple M3 Ultra에서 `mlx-community/granite-4.0-3b-vision-4bit`는 native B=3 양자화 decode에서 자유 형식 greedy 출력 세 개 중 두 개를 바꿨습니다. 3/3 exact constrained smoke와 이미지 조건부 B=1 대 B=4 trace가 있었음에도 그랬습니다. 그 trace의 top-1 변경 4/64는 모두 낮은 margin이었습니다(가장 큰 기준 gap 0.125, 결정된 위치의 변경은 0). 행별 wrapper는 동시 admission을 비활성화하지 않으면서 이슈가 요구한 정확한 단독 출력을 보존하며, Granite text tower는 여전히 native입니다. Padded-prefill, snapshot-reuse, paged-attention 적격성은 변하지 않습니다. 커지는 attention KV 길이는 서버의 paged 계정으로 동기화되며, recurrent 및 stateless layer는 가상의 KV block을 예약하지 않습니다.
  Qwen3-Next, Qwen 3.5, Bailing MoE Linear, Kimi Linear, Kimi K3도 마찬가지로, 쌓인(stacked) gated-delta, GLA, KDA 상태에 대해 native 단일 토큰 decode를 실행하면서 시퀀스별 full-attention 또는 MLA cache는 유지합니다(#2572). 새로 생성된(fresh) 상태와 초기화된 상태가 섞인 경우와 지원되지 않는 다중 토큰 또는 마스크된 호출은 식별된 행별 경로를 유지합니다. Qwen 3.5는 기존의 batched-prefill 정책을 보존하고 MRoPE 위치를 시퀀스별로 해석합니다. 저장되었거나 멀티모달 위치를 포함하는 batch는 행별로 full attention을 유지하며, recurrent 및 feed-forward 작업은 여전히 batch로 처리됩니다. Qwen 3.5 VLM은 이미 식별된 decode를 text tower에 위임하고 있으며, Kimi K3 VLM도 이제 같은 방식으로 위임합니다.
  그 밖의 모든 batching 계열은 `LanguageModel` 기본 구현을 실행합니다. 이는 행마다 단일 시퀀스 `forward()`를 한 번씩 호출하고 B개의 graph를 함께 평가하므로, 총 처리량의 확장은 독립적인 단일 토큰 graph가 겹쳐 실행되는 데서만 나옵니다. `docs/benchmark_results/moe-batched-decode-m1ultra-2026-09-04.md`의 2026-09-04 M1 Ultra 프로파일은 override가 들어가기 전의 `qwen3-30b-a3b-4bit`에서 이 fallback을 측정했습니다.
  MiniMax-M3 sparse attention은 추가 index-key head를 cache에 유지하지만, attention에는 모델의 value head만 전달합니다. 이는 이전 단일 행 경로도 바로잡습니다. 그 경로에서는 cache 할당이 value를 index가 덧붙은 key-head 수로 패딩하여 K/V head 형상 실패를 일으킬 수 있었습니다. Batched decode는 index key와 query를 batch 전체에 대해 한 번 projection한 다음, 각 행의 cache된 prefill 이후에 행 단위로 잘라냅니다.
  시작 시 worker는 batching이 가능한 모든 모델에 대해 `decode path: native batched forward` 또는 `decode path: per-row fallback (aggregate decode throughput will not scale with --parallel)` 중 하나를 보고합니다. Qwen 4 Exp와 Muse Glimmer는 서빙 decode-width 경로가 여전히 식별된 행을 순차적으로 dispatch하므로 현재 fallback을 보고합니다. anonymous-batch 또는 batched-prefill override로는 서빙 decode가 native가 되지 않습니다.
- `--max-batch-prefill 4`: 대기 중인 프롬프트를 최대 4개까지 하나의 prefill pass로 묶습니다. `supports_batched_prefill()`에 동의한 계열(Llama 3, Qwen 3, Qwen 3.5, 그리고 Qwen 2.5 같은 별칭)만 이를 사용하며, 나머지는 자동으로 순차 prefill로 fallback합니다.
- Prompt-prefix cache를 켭니다. 2 GiB KV 예산으로 제한되며(`--prompt-cache-*`로 재조정), 반복되는 공유 prefix(예: 긴 system prompt)는 한 번만 prefill되어 재사용됩니다.

Exaone MoE에는 의도적으로 남겨 둔 역사적 B=1 예외가 있습니다. 이전의 `LanguageModel` adapter는 호출마다 K/V를 버리고 offset만 복사해 돌려주었습니다. 수정된 경로는 요청 이력을 보존하며 sliding window 전체에 걸쳐 지속형 `AnyKVCache` 참조와 일치하므로, 요청이 하나뿐이더라도 출력이 그 깨진 기준선과 다를 수 있습니다(#2559).

이전의 단일 클라이언트 동작을 복원하는 escape hatch는 다음과 같습니다. `--parallel 1`(단일 decode 슬롯), `--no-batch`(기존의 순차 worker. 같은 스케줄러를 `max_batch_size=1`로 실행), `--max-batch-prefill 1`(순차 prefill), `--no-prompt-cache`입니다.

### 토큰 정확성: `mlxcel run`과 서버

`mlxcel run`(`-p` 유무와 관계없이)과 그 chat REPL은 epic #2166 Phase 5(issue #2173, [ADR 0007](adr/0007-unified-batch-native-engine.md)) 이후 서버 엔진의 클라이언트입니다. 이들은 슬롯 하나로(`--parallel 1`, `--max-batch-size 1`이므로 decode 저장소는 dense로 결정됨) 모델 worker를 프로세스 안에서 시작하고, 각 turn을 `/v1/chat/completions` 요청 경로로 제출합니다. 즉 같은 chat-template 렌더링, 같은 worker 옵션, 같은 `StreamFilter` reasoning 분리, tool 파싱, stop 문자열을 사용합니다. HTTP 리스너는 열리지 않습니다. CLI의 샘플링 플래그는 그 서버의 옵션이 되며(`mlxcel run --temp 0.7`은 `mlxcel-server --temp 0.7`과 같은 방식으로 해석됨), `generation_config.json`에 명시가 없으면 CLI는 greedy를 기본 sampler로 유지합니다.

이 계약은 공유 코드가 아니라 동등성 불변식(equivalence invariant)입니다. `MLXCEL_SDPA_DETERMINISTIC=1`에서 같은 chat 요청에 대해 `mlxcel run -p`와 서버의 greedy 출력은 동일합니다. `make engine-parity MODEL=<dir>`은 이를 arm `e:run`(`run -p --temp 0`이 만드는 설정)과 arm `f:server`(`mlxcel-server` 기본 구성에서의 같은 요청. 슬롯 네 개이며, 모델이 지원하면 paged 저장소 사용)를 비교하여 검사하며, 양쪽 모두 prompt cache는 꺼 둡니다. GB10에서 Qwen3-1.7B 4-bit와 Llama-3.2-1B 4-bit는 두 쌍이 모두 동일합니다. 같은 실행은 B=1 서버를 dense와 paged 각각에 대해, raw-completion 클라이언트(arm `d:engine`, `mlxcel_core::engine::DirectEngine`)와 `mlxcel generate`가 그 클라이언트를 직접 호출하는 방식(arm `a:cli`, `generate`가 실행하는 1토큰 warmup 포함)으로 비교하므로, divergence가 스케줄러의 정책, 저장소, 엔진 중 어디에서 생기는지 가려낼 수 있습니다. Phase 6(#2176) 이후에는 모든 front가 chat template을 하나의 함수(`server::chat_front`)로 렌더링하므로, 하니스의 렌더링을 쓰는 arm과 `e:run` / `f:server` arm은 `enable_thinking` 기본값을 포함해 같은 프롬프트를 봅니다.

서버 기능 두 가지는 여전히 거의 동점인 greedy 토큰을 바꾸므로, 불변식이 다루는 대상이 아니라 측정된 divergence로 추적됩니다. 하나는 prompt-cache hit로, 캐시된 prefix의 접미사만 forward합니다(chat REPL은 캐시를 켠 채 실행하며 이것이 서버 기본값이고, `run -p`는 요청 하나에 재사용할 것이 없으므로 캐시를 끄고 실행합니다). 다른 하나는 다른 활성 시퀀스와 함께하는 continuous batching입니다. 서버에서 단일 요청 경로를 재현하려면 기본 admission 폭을 유지한 채 `--max-batch-size 1`(dense 저장소) 또는 `--decode-storage-backend dense`와 `--no-prompt-cache`를 사용하십시오.

ROCm에서는 `MLXCEL_SDPA_DETERMINISTIC` 없이도 불변식이 성립하며(이 변수는 CUDA overlay에서만 읽힙니다), dense 대 paged 쌍은 gate가 아니라 기록된 divergence 지표입니다([ADR 0007](adr/0007-unified-batch-native-engine.md), "What parity means on ROCm and Metal"). epic 이후 gfx1151에서 측정한 결과(#2192, [결과](benchmark_results/rocm-unified-engine-gfx1151-2026-10-08.md))는 다음과 같습니다. Qwen3-0.6B 4-bit와 Llama-3.1-8B 4-bit에서는 하니스가 비교하는 모든 쌍이 greedy와 seed 지정 경우 모두에서 동일했습니다(paged arm 포함. HIP paged v2 커널을 실행했고 실행당 launch 1764~2080회). 다만 문서화된 seed 지정 prompt-cache miss 대 hit 행은 예외입니다. 128 토큰 teacher-forced decode-step trace에서 dense 서버와 paged 서버는 TODO_PAGED 위치와 TODO_PAGED_DECIDED 위치(decided 위치)에서 어긋납니다. Mamba hybrid인 granite-4.0-h-tiny 4-bit에서는 CLI, `run`, dense 서버가 동일하며, paged 저장소는 `n/a`(worker가 dense로 fallback)입니다. prompt cache의 history-boundary 분할(`prefill[0..45)+prefill[45..48)`과 캐시를 끈 `prefill[0..48)`)은 다른 분할이므로, 캐시 off arm과 cache-miss arm은 거기서 greedy는 토큰 47에서, seed 지정은 토큰 0에서 diverge합니다.

> 백엔드 참고 (CUDA / Blackwell, 예: GB10): CUDA에서 batched decode는 한때 처리량 면에서 이득이 없었습니다. `M*B < 8` quantized matmul이 행별 qmv로 fallback하여 시퀀스마다 가중치를 다시 읽었기 때문입니다(llama-3.1-8b-4bit에서 B=1/2/4에 걸쳐 총합이 약 45-50 tok/s로 평평했음). multirow qmv 경로(issue #725, `MLXCEL_QMV_MULTIROW`, 기본적으로 켜짐)는 가중치 읽기를 batch 전체에 분할 상환하므로, 이제 CUDA에서도 총 decode 처리량이 동시성에 따라 증가합니다. GB10의 llama-3.1-8b-4bit에서 클라이언트 1/2/4개일 때 40.4 / 68.5 / 74.2 tok/s입니다(kill switch를 끈 경우 40.5 / 49.8 / 49.6). attention KV 읽기가 여전히 B에 따라 늘어나므로 Metal에 비해 확장은 선형 이하로 유지됩니다. 공유된 2D 가중치 위의 decode batch `[B, 1, K]`는 행으로 접히기도 합니다(#2690). 행이 8개일 때부터, 그리고 compute capability 12.1에서는 2^24개 이상의 원소를 가진 가중치에 대해 행이 5개일 때부터, 가중치를 한 번만 읽는 GEMM 하나를 실행합니다. RTX Spark N1X에서 이것은 네 모델에 걸쳐 8-client 총 decode 처리량을 1.2배에서 3.2배로 끌어올렸습니다(`docs/benchmark_results/cuda-qmm-fold-batch-n1x-2026-10-31.md`). `--parallel`을 7 이하로 유지하라는 이전의 조언은 이 접기(fold) 이전의 것이며, GB10은 이 접기를 적용한 상태로 다시 측정되지 않았습니다. `docs/benchmark_results/qmv-multirow-gb10-2026-07-11.md`를 참조하십시오. `--max-batch-prefill`(large-M, 타일 튜닝됨, `docs/benchmark_results/qmm-sm121-tile-tuning-gb10-2026-07-10.md`)과 prompt cache는 CUDA에서도 여전히 독립적인 이득입니다.

메모리 산정: KV 사용량은 활성 batch에 따라 늘어나므로 최대 `--parallel`개의 동시 시퀀스 KV를 예산에 반영하십시오. 자동 `--parallel -1`에서는 mlxcel이 b10621의 통합 기본값과 맞춰집니다. 슬롯 네 개가 노출되고, 모든 슬롯이 `--ctx-size` 윈도우 전체를 사용할 수 있으며, 스케줄러는 살아 있는 모든 prompt+생성 토큰의 합이 그 공유 예산 안에 머물도록 강제합니다. 명시적인 양수 `--parallel`(또는 명시적 `--max-batch-size`)을 주면, `--kv-unified`를 함께 설정하지 않는 한 context가 활성 슬롯에 분할됩니다(슬롯당 `ctx_size / parallel`). `--no-kv-unified`는 분할 윈도우를 강제합니다. 모델 기본값을 쓰려면 `--ctx-size`를 0으로 두십시오. Admission control(`--kv-cache-budget`, `--max-kv-size`)은 여전히 부하가 걸리면 요청을 거절하므로, 큰 `--parallel`은 OOM이 아니라 대기열 적체로 성능이 떨어집니다.

측정된 확장성(Apple M1 Ultra, Metal. `meta-llama-3.1-8b-instruct-4bit`, 512 토큰 프롬프트, 새 토큰 128개. `scripts/bench_serving_concurrency.py`). baseline 열은 이전 기본값인 `--parallel 1`입니다. batched 열은 `--max-batch-prefill 4`와, 각 동시성 수준을 드러낼 만큼 높은 batch 상한(`--parallel 8`)으로 측정했으므로, 8-client 행은 `--parallel 8` 서버를 반영합니다. 출하 기본값은 `--parallel 4`이며 그 batch는 4로 제한됩니다(4-client 행이 포화 상태에서의 기본값이며, 8 client에서는 4개가 batch로, 4개가 대기열에 들어갑니다). 총합(Aggregate)은 동시 client 전체의 tok/s 합계이고, TTFT는 time-to-first-token입니다.

| 클라이언트 | 총합 tok/s (`-p1`) | 총합 tok/s (batched) | TTFT 평균 ms (`-p1`) | TTFT 평균 ms (batched) |
|--------:|------------------------:|--------------------------:|---------------------:|-----------------------:|
| 1 | 56.6 | 56.8 | 889 | 783 |
| 2 | 71.5 | 99.9 | 1150 | 114 |
| 4 | 65.6 | 107.9 | 3257 | 189 |
| 8 | 62.8 | 105.1 (`-p8`) | 7514 | 348 (`-p8`) |

동시 client 4개에서 기본값은 단일 client 총합 처리량의 1.90배를 제공하고, 부하 시 평균 TTFT를 약 17배 줄이며(3257 ms에서 189 ms로), 단일 client 처리량은 변하지 않습니다. 대역폭이 더 높은 decode 타깃(예: GB10)에서는 가중치 읽기 분할 상환의 여지가 더 크지만, 그 수치는 CUDA 측정 세션을 기다리고 있습니다.

batched-decode 기본값은 `auto` paged KV 예산(`--kv-cache-budget auto`, 기본값)과 짝을 이룹니다. #122 block-budget admission이 동시 batch의 KV를 제한하며, 전체 context의 시퀀스 네 개가 OOM abort로 이어지게 두는 대신 backpressure를 반환합니다. dense decode 백엔드에서는 예산이 작동하지 않습니다. `--kv-cache-budget none`으로 guard를 끌 수 있습니다. 메모리가 제한된 호스트에서는 `--parallel`을 낮추거나 `--ctx-size`를 제한할 수도 있습니다([environment-variables.md](environment-variables.md)의 context 크기 산정 참고 사항 참조).

### 하나의 prefill 계획, 그리고 prompt-cache hit가 miss를 재현하는 경우

issue #2170 이후 CLI 생성기와 스케줄러는 하나의 `PrefillPlan`(`mlxcel_core::prefill_plan`)을 통해 prefill합니다. 채택된 prompt-cache prefix는 건너뛰고, snapshot 계열의 chat 프롬프트는 history boundary에서 한 번 나뉘며(#1143, 패딩 없는 forward 하나. prompt cache가 snapshot을 찍는 지점), 나머지는 chunk 조각으로 잘리고, 조각은 모델이 선택한 경우에만 M5+ 하드웨어에서 타일 패딩됩니다. direct engine과 그 밖의 유휴 서버는 2048을 사용합니다. 다른 시퀀스가 decode 중이면 서버는 512를 사용하며, 명시적인 `--prefill-chunk-size`(또는 `--batch-size` / `LLAMA_ARG_BATCH`)나 `MLXCEL_PREFILL_CHUNK`는 두 상태를 모두 고정합니다. 스케줄러는 tick마다 조각 하나를 실행하고 살아 있는 커서에서 continuation을 다시 구성하므로, decode batch가 비거나 활성화될 때 chunk가 안전하게 바뀔 수 있습니다. Gemma 4 MTP burst는 단독(alone) 값을 그대로 쓰며, 행별 prefill 범위를 같은 계획에서 가져옵니다.

같은 KV 상태에서 같은 조각들을 forward하는 같은 프롬프트의 prefill 두 개는 `MLXCEL_SDPA_DETERMINISTIC=1`에서 비트 단위로 동일합니다. 프롬프트를 다르게 나누는 두 prefill은 그렇지 않습니다. CUDA quantized matmul은 8행 미만의 forward를 행별 `qmv` 커널로, 더 큰 forward는 타일 `qmm` 커널로 보내며, 짧은 query 블록에 대한 attention은 전체 causal pass와 다르게 타일링되고, 이 둘은 reduction 순서가 달라서 KV와 logit이 마지막 비트에서 달라지며 greedy 스트림이 거의 동점인 토큰에서 뒤집힐 수 있습니다. history boundary에서의 분할, `--prefill-chunk-size`, prompt-cache hit(채택된 prefix 이후의 접미사만 forward함)는 모두 분할입니다.

따라서 prompt-cache hit는 채택된 prefix가 cold 계획의 분할 지점 중 하나(snapshot 계열의 history boundary 또는 chunk 경계)에서 끝날 때, 같은 프롬프트의 cold(miss) 실행을 정확히 재현합니다(`PrefillPlan::reproduces`). 이것이 조건의 계획 쪽 절반이며, 나머지 절반은 채택된 행이 같은 조각들로 작성된 것이어야 한다는 것입니다. 이전 turn의 boundary snapshot이나 한 조각으로 forward된 priming prefix가 그러합니다. donor가 decode step으로 작성했거나 더 긴 forward 안에서 작성한 행은, 채택된 길이가 분할 지점이더라도 다른 분할입니다. 스케줄러 상태에 따라 chunk가 바뀌는 prefill도 마찬가지로 다른 분할입니다. 같은 스케줄러 상태에서 실행되는 prefill은 기존의 재현 불변식을 유지합니다. `make engine-parity`는 GB10에서 두 형태를 모두 측정합니다(2026-10-07, greedy 및 seed 지정 토큰 64개, hit는 프롬프트의 history prefix로 priming).

| 모델 | miss 분할 | hit 분할 | miss 대 hit |
|---|---|---|---|
| lfm2-350m-8bit (snapshot 계열) | `prefill[0..48)+prefill[48..51)` | `adopted[0..48)+prefill[48..51)` | 두 경우 모두 동일 |
| qwen3-1.7b-4bit (dense KV) | `prefill[0..52)` | `adopted[0..45)+prefill[45..52)` | greedy는 토큰 0에서 diverge(4792 대 785), seed 지정은 토큰 11에서 diverge |
| qwen3-1.7b-4bit, `--server-prefill-chunk 45` | `prefill[0..45)+prefill[45..52)` | `adopted[0..45)+prefill[45..52)` | 두 경우 모두 동일 |
| llama-3.2-1b-instruct-4bit (dense KV) | `prefill[0..75)` | `adopted[0..71)+prefill[71..75)` | greedy 동일, seed 지정은 토큰 25에서 diverge |
| llama-3.2-1b-instruct-4bit, `--server-prefill-chunk 71` | `prefill[0..71)+prefill[71..75)` | `adopted[0..71)+prefill[71..75)` | 두 경우 모두 동일 |

dense-KV 행이 diverge하는 이유는 cold 계획에 45나 71이라는 분할 지점이 없기 때문입니다. hit의 7행 및 4행 접미사 forward는 `qmv`를 거치고, cold 실행의 52행 및 75행 forward는 `qmm`을 거칩니다. 모든 계열에 history-boundary 분할을 주면 이 두 행은 동일해지겠지만, 모든 cold chat prefill에 두 번째 forward launch 비용이 들고, prompt cache가 단일 turn 요청의 출력을 바꾸게 됩니다(off 대 miss는 오늘 dense-KV 계열에서 동일함). Phase 5의 `mlxcel run` 대 `mlxcel generate` 비교가 이에 기대고 있으므로, #2170에서는 이를 기각했습니다. 따라서 의존해야 할 불변식은 다음과 같습니다. history-boundary 분할이 없는 모든 계열에서 캐시 off와 캐시를 켠 cold prefill은 동일하고, 분할 지점에서 시작한 hit는 miss와 동일하며, 조각 내부에서 시작한 hit는 #203 / #325 / #326 거의 동점 부류의 분할 변경입니다. 이는 `mlxcel_core::prefill_plan` 테스트와 `scheduler_prompt_cache_plan_tests::cache_hit_reproduces_miss_exactly_from_a_plan_split_point`로 고정되어 있습니다(조각 내부 hit의 divergence는 테스트가 수치로 단언하는 것이 아니라 위 표에서 측정됩니다).

epic의 실행 종료 시점 측정을 위해 기록한, 계획의 알려진 제한 사항은 다음과 같습니다.

- snapshot 계열 prefill의 history 구간(history boundary 이전 구간, #1143)은 `--prefill-chunk-size`가 무엇이라고 하든 항상 chunk 없는 forward 하나입니다. 이는 계획보다 앞선 것이며(#1143 이후 줄곧 그래 왔음), 이 구간이 prompt cache가 모델 상태를 snapshot하는 지점이므로 계획도 이를 유지합니다. 따라서 snapshot 계열에서 긴 chat history는 동시 decode tick과 교차되지 않는 forward 하나로 prefill되며, 경계 이후의 조각만 교차됩니다.
- PR #2226은 적응형 기본값이 해결하는 절충을 측정했습니다. 4-bit Llama-3.2-1B 스트림 네 개가 decode 중이고 8192 토큰 요청이 받아들여진 상황에서, 2048은 admission 중 스트림 ITL p95 127.9–128.9 ms와 받아들여진 요청의 TTFT 1903–2060 ms를 냈고, 512는 ITL p95 55.8–56.4 ms와 TTFT 3768–4067 ms를 냈습니다(세 라운드). 따라서 스케줄러는 `active_batch`가 비어 있지 않은 동안에만 512를 사용하고, decode 스트림이 끝나면 2048로 돌아갑니다(#2228).

출하된 정책은 새로 릴리스한 빌드의 GB10 승인 실행을 통과했습니다. 조용한 호스트에서의 admission 세 라운드는 라이브 스트림 ITL p95를 54.1, 54.8, 55.2 ms로 측정했으며, 이는 62.0 ms 한도보다 낮습니다. 유휴 서버에서의 TTFT 비교(다섯 라운드)에서는 정책의 중앙값이 Qwen3-1.7B와 Llama-3.2-1B 모두에서 명시적 2048의 null 범위 안에 들었습니다. [원시 결과와 프로토콜](benchmark_results/adaptive-prefill-chunk-gb10-2026-10-11.md)을 참조하십시오.

### 빠듯한 예산에서의 decode 여유 (`--kv-admission-watermark`)

예산이 동시 작업에 비해 작으면, 정확히 빈 block만큼 요청을 받아들였을 때 실행 중인 행이 자랄 곳이 없어집니다. 그러면 다음 decode tick들이 진행이 가장 덜 된 행을 preempt하고, 그 행은 다시 받아들여질 때 처음부터 re-prefill됩니다. `--kv-admission-watermark F`는 admission 시점에 block 예산의 비율 `F`를 비워 둡니다. 요청은 빈 block 수에서 `F * budget`을 뺀 값이 그 요청을 감당할 때만 받아들여집니다. 그 공간을 만들기 위해 차가운 prompt-cache prefix를 축출하고, 더 높은 우선순위의 요청에 대해서는 대기하기 전에 엄격히 더 낮은 우선순위의 행을 preempt합니다(admission이 요청 자체에 대해 이미 하는 것과 같음). watermark는 decode batch가 살아 있는 동안에만 적용되므로, 예산에 맞는 요청이 거부되는 일은 없으며 대신 실행 중인 행이 끝나기를 기다립니다. `0`은 이를 비활성화하며, block 예산이 없으면 작동하지 않습니다.

기본값 0.01은 측정 격자에서 turn을 잃지 않으면서 preemption을 줄인 가장 작은 값입니다. GB10에서 meta-llama-3.1-8b-instruct-4bit, `--kv-cache-budget 600000000`(4444 block), `--max-batch-size 4`, 600 토큰 생성으로 이뤄진 3-turn 대화 세 개를 동시에 실행했으며, 각 값은 동일한 결과로 두 번씩 실행했습니다.

| Watermark | Preemption 횟수 | Decode reclaim pass 횟수 | 완료된 turn | 벽시계 시간 |
|-----------|-------------|-----------------------|-----------------|-----------|
| 0 | 10 | 43 | 9 / 9 | 103-104 s |
| 0.01 | 6 | 6 | 9 / 9 | 90-91 s |
| 0.02 | 6 | 6 | 9 / 9 | 89-90 s |
| 0.05 | 5 | 5 | 9 / 9 | 89-90 s |
| 0.10 | 1 | 2 | 9 / 9 | 86 s (1회 실행) |
| 0.15 | 1 | 3 | 9 / 9 | 88 s (1회 실행) |

예산이 빠듯해서 preemption이 자주 일어나고 새 요청의 추가 대기 지연을 감당할 수 있다면 값을 올리십시오.

예산 회계를 정직하게 유지하는 규칙이 두 가지 더 있습니다(issue #2088). 캐시된 prefix를 채택한 요청은, 채택된 block이 이미 사용 중이므로 접미사가 새로 만들 block에 대해서만 비용을 지불합니다. 실행 중인 행이 block을 필요로 하는데 빈 block이 없으면 회수 순서는 다음과 같습니다. 차가운 prompt-cache prefix를 축출하고, 그다음 대기 중인 요청의 채택된 prefix를 버리며(받아들여질 때 cold로 re-prefill됨), 그다음 실행 중인 행을 preempt합니다.

### chunked prefill이 decode와 tick을 공유하는 방식 (`--prefill-grant-interval`)

위 플래그 표는 `--prefill-chunk-size`가 "prefill이 decode 지연에 미치는 영향을 제한한다"고 말합니다. 이는 사실이지만, chunk 크기가 보통 암시하는 모양과는 다르므로, tick 정책이 실제로 무엇을 하는지 밝혀 둘 가치가 있습니다.

`--prefill-chunk-size`보다 긴 프롬프트는 받아들여져 chunk 0을 실행한 뒤 보류(parked)됩니다. 그 시점부터 활성 시퀀스가 하나라도 있는 동안 tick 정책은 decode 쪽으로 결정됩니다. issue #1011 이전에는 이것이 *영원히* 그렇게 결정되었습니다. 정책은 스케줄러 상태의 순수 함수이고, decode를 실행해도 정책이 읽는 상태는 전혀 바뀌지 않으므로, 보류된 프롬프트는 마지막 decode 시퀀스가 끝날 때까지 전혀 진행하지 못했습니다. M1 Ultra에서 측정한 결과, decode가 162에서 344 step으로 진행되는 동안 prefill은 19개 중 chunk 1에서 20초 동안 머물러 있었습니다. 요청이 계속 도착하면 이 대기에는 상한이 없으며, 받아들여진 요청의 첫 토큰까지의 시간에도 상한이 없습니다.

`--prefill-grant-interval N`(기본값 16)이 이를 제한합니다. 연속된 decode tick이 `N`번 지나면 다음 tick이 보류된 prefill에 GRANT되며, 이 prefill은 chunk 하나를 실행하고 카운트를 초기화합니다. 따라서 다음과 같습니다.

- **상한.** 보류된 프롬프트는 `N + 1` tick마다 최소 한 chunk씩 전진하므로, `C`개 chunk로 된 프롬프트는 batch가 아무리 오래 decode를 계속하더라도 admission 후 `C * (N + 1)` tick 안에 첫 토큰에 도달합니다.
- **대가.** grant 주기 하나 동안 decode 스트림은 decode step `D`와 chunk forward `P`에 대해 벽시계 시간 `N * D + P`당 `N`개의 토큰을 얻으므로, admission 구간 동안의 평균 토큰 간 지연(inter-token latency)은 `D + P / N`입니다. p95는 모양이 다릅니다. `N + 1`개 간격 중 하나에 chunk가 실려 있으므로, `N < 19`이면 chunk가 p95에 나타나고 그 이상이면 p95 위로 숨습니다. 어느 경우든 hiccup의 크기는 같고 빈도만 바뀌므로, 백분위를 깨끗하게 유지하려고 `N`을 조정하는 것은 스트림을 더 빠르게 만드는 것과 같지 않습니다.

따라서 `N`은 `--prefill-chunk-size`가 반대 방향에서 움직이는 것과 같은 frontier 위의 TTFT 대 ITL 다이얼입니다. 더 작은 chunk는 hiccup 하나하나를 작게 만들고, 더 작은 간격은 hiccup을 더 자주 일으키면서 받아들여진 요청을 더 빠르게 합니다. `--prefill-grant-interval 0`은 grant를 비활성화하여 무한 대기를 포함한 #1011 이전의 중재 방식을 복원합니다. 측정된 frontier: [`docs/benchmark_results/prefill-fairness-m1ultra-2026-08-03.md`](benchmark_results/prefill-fairness-m1ultra-2026-08-03.md).

지연 수치를 읽을 때 알아 둘 만한 것이 두 가지 더 있습니다.

- batch가 이미 `--parallel`에 차 있으면 대기 중인 요청은 전혀 받아들여지지 않으므로 prefill에 기여하지 않으며, 위의 어떤 현상도 일어나지 않습니다. "다른 요청이 decode하는 동안 긴 프롬프트가 prefill된다"는 시나리오에는 `--parallel`이 활성 스트림 수보다 엄격히 커야 합니다.
- 보류되는 chunked prefill은 동시에 최대 하나입니다. 스케줄러는 이를 단일 슬롯에 보관하고, chunked 분기가 admission 분기보다 먼저 short-circuit되므로, 프롬프트 하나가 보류되어 있는 동안에는 batch의 점유율과 관계없이 더 이상의 요청이 받아들여지지 않습니다. 따라서 보류된 프롬프트의 체류 시간을 줄이면 그 뒤에 대기 중인 모든 것의 head-of-line 대기 시간도 줄어듭니다.

issue #908은 같은 분기를 분석했으며, fused ragged forward로 고치지 않기로 결정했습니다. [ADR 0005](adr/0005-mixed-prefill-decode-step-execution.md)를 참조하십시오. 실험적인 `MLXCEL_MIXED_STEP=1`은 frontier의 극단으로 남아 있습니다. 모든 tick이 두 작업 부하를 모두 전진시켜, 보류된 프롬프트의 TTFT를 최소화하는 대신 decode 스트림이 치르는 비용을 최대화합니다. 이는 측정 도구이며, 지원되는 운영자용 설정이 아닙니다. `scripts/bench_mixed_step_admission.py`가 이 시나리오를 구동하고, 각 실행을 `mlxcel_batch_mixed_steps_total`(#908 프로토타입)과 `mlxcel_batch_prefill_grants_total`(#1011 grant) 카운터로 귀속시킵니다. `scripts/bench/starvation_probe.sh`는 같은 카운터를 시간에 따라 샘플링합니다.

### batched-prefill 일시적 메모리 제한 (`--max-batch-prefill-tokens`)

위의 `--kv-cache-budget` guard는 정상 상태(steady-state)의 KV를 제한할 뿐, batched prefill이 실행되는 동안 할당하는 일시적 메모리는 제한하지 않습니다. cold 프롬프트 둘 이상이 하나의 패딩된 batched prefill을 공유하면, 이 경로는 모든 행을 윈도우에서 가장 긴 프롬프트 `L`까지 패딩하고, 쌓인 `[B, L, L]` FP32 attention mask와 `[B, L, hidden]` forward를 구체화합니다. 제한을 두지 않으면, 긴 프롬프트 `B`개가 함께 도착할 때 `--prefill-chunk-size`를 무시하는 `O(B*L^2)` mask가 할당됩니다. 8k 프롬프트 네 개는 약 1 GiB의 `[4, 8192, 8192]` FP32 mask를 만들며, 이는 batched prefill이 꺼져 있을 때 그 프롬프트들이 택하는 순차 chunked prefill의 working set을 훨씬 넘는 급증입니다. 서빙 경로에서는 이 eval에서의 할당 실패가 이제 fallible `try_eval` FFI 경계에서 잡혀, 서버 전체를 중단시키는 대신 해당 cohort의 요청만 실패시킵니다(issue #822). cohort에 걸친 연속 eval 실패가 계속되면 여전히 shutdown guard가 작동하므로, 일시적 메모리 자체는 KV 예산이 모델링하지 않는 가용성의 경계로 남습니다.

`--max-batch-prefill-tokens N`은 drain된 batched 윈도우를 총 패딩 토큰 수(`rows * L`)로 제한합니다. `rows * L`을 `N` 너머로 밀어내게 될 행 앞에서 drain을 멈추고, 남은 행은 대기열에 남아 이후 tick에서 prefill됩니다(짧은 것은 다시 batch되고, 긴 것은 chunked 단일 시퀀스 경로를 택합니다). 두 행 batch에 합류하기에도 너무 긴 선두 프롬프트(`2 * head_len > N`)는 batched 경로를 완전히 건너뛰고 chunked로 prefill됩니다. 이 한계는 토큰 예산에서 따라 나옵니다. `L`로 패딩된 `B >= 2`개 행의 cohort는 `B*L <= N` 토큰이 들며, `L <= (B*L)/2`이므로 mask는 `N^2 / 2`개 원소, 즉 FP16에서 `~N^2` 바이트, FP32에서 `~2*N^2` 바이트 이내에 머뭅니다.

기본 예산은 고정값이 아니라 유도됩니다. `2 * max_batch_prefill * min(prefill_chunk_size, 512)`입니다(출하값은 `2 * 4 * 512 = 4096`. 2배의 여유는 chat template 때문에 512를 약간 넘는 실제 512 토큰 프롬프트의 패딩 오차를 흡수합니다). 따라서 짧은 프롬프트로 가득 찬 batch는 batching 대상으로 남고, 더 긴 프롬프트의 윈도우는 chunked 경로로 흘러갑니다. 행당 몫은 512로 제한되어, ADR 0007의 chunk 기본값(2048)이 예산을 네 배로 키우고 최악의 mask를 열여섯 배로 만드는 일이 없도록 합니다(#2170). 기본값에서 FP32 mask는 `2 * 4096^2` 바이트, 약 34 MiB로 제한되며 모델 activation 메모리에 비하면 무시할 만합니다. `0`(플래그 또는 `MLXCEL_MAX_BATCH_PREFILL_TOKENS=0`)은 #715 이전의 제한 없는 동작을 위해 상한을 비활성화합니다. 플래그는 `MLXCEL_MAX_BATCH_PREFILL_TOKENS`보다 우선하고, 이 환경 변수는 유도된 기본값보다 우선합니다([environment-variables.md](environment-variables.md) 참조).

`meta-llama-3.1-8b-instruct-4bit`(Apple M1 Ultra, Metal)에서 8k 토큰 프롬프트 네 개가 동시에 들어올 때의 분석적 예측은 다음과 같습니다.

| 구성 | prefill mask 윈도우 | mask 일시적 메모리 (분석적) | 경로 |
|--------|--------------------:|--------------------------:|------|
| 제한 없음 (`--max-batch-prefill-tokens 0`) | `[4, 8192, 8192]` FP32 | `4 * 8192^2 * 4 B` = 1024 MiB | chunk되지 않은 단일 batched forward |
| 기본 상한 (4096) | `[≤2048, 8192]` chunk mask 네 개 | `2048 * 8192 * 4 B` = 64 MiB (한 번에 하나) | 8k 프롬프트는 chunked 단일 시퀀스 경로로 흘러감 |

경험적 A/B(서버의 phys-footprint 최대치. Apple Silicon에서는 RSS가 MLX Metal 버퍼를 포착하지 못하므로 `phys_footprint_peak`에는 `/usr/bin/footprint -p <pid>`를 사용하십시오)는 측정 세션을 기다리고 있습니다. `--max-batch-prefill-tokens 0`으로 시작한 서버와 기본값 서버를, 8k 토큰 요청 네 개를 동시에 보내는 방식으로 구동하여 재현하십시오(`scripts/bench_serving_concurrency.py --concurrency 4 --prompt-tokens 8192`, prefill을 분리하기 위해 `--max-tokens`는 작게).

짧은 프롬프트의 동시성(#714 기본값의 동기)은 구조상 영향을 받지 않습니다. `--prompt-tokens 512 --concurrency 4`에서 네 행 모두 `4 * 512 = 2048` 패딩 토큰이 들며, 이는 기본 예산의 절반이므로, chat template이 각 프롬프트를 512 조금 위로 밀어 올리더라도 여유를 두고 batch됩니다. `2 * head_len <= budget` admission도 성립합니다(`2 * 512 <= 4096`). TTFT와 총 처리량은 위의 #714 표와 일치합니다(경험적 재실행은 같은 세션을 기다리고 있습니다).

### Paged decode와 prompt-prefix cache

decode 상태와 요청 간 prompt-prefix cache는 하나의 참조 카운트 방식 copy-on-write block pool을 공유합니다(batch가 가능한 pool 기반 계열의 기본값이며, `--decode-storage-backend dense`로 옵트아웃합니다). 프롬프트 prefix를 공유하는 동시 요청은 그 prefix의 KV를 한 번만 저장하고 re-prefill을 건너뜁니다. 채택은 소비하지 않으므로(clone-and-pin), 저장된 prefix 하나가 임의의 수의 동시 차용자에게 서비스합니다. Automatic Prefix Caching(기본적으로 켜져 있으며 `--apc-enabled=false`로 끌 수 있음)은 공유 prefix 이후에 갈라지는 요청이 공통 부분을 재사용하게 합니다. 메커니즘, 측정된 메모리 및 prefill 토큰 절감, decode 처리량, `--kv-cache-budget`은 [turbo-kv-cache.md](turbo-kv-cache.md#통합-paged-kv-cache)에 문서화되어 있습니다. paged 백엔드는 B=1/B=2에서 스케줄러 형태의 qwen3/llama3 Fp16 dense-natural-cache 경로에 대해 `tests/paged_scheduler_parity.rs`가 다루며, TF32가 비활성화되었을 때 logit RMS를 제한한 상태에서 greedy 토큰 동일성을 요구하는 CUDA qwen3 arm도 있습니다. 이 증거를 VLM front end, 모델 소유 cache, Turbo/quantized-KV 모드, 또는 모든 CUDA reduction 기하로 일반화하지 마십시오.

#### `content`만 되돌려 보내는 클라이언트의 reasoning 모델

일부 chat template은 이어지는 assistant 메시지에 `reasoning_content`가 있는지에 따라 이전 turn을 다르게 렌더링합니다. AI21 Jamba-Reasoning은 그 경우에만 이전 user turn에 thinking 지시를 유지하므로, `content`만 되돌려 보내는 클라이언트(OpenAI SDK 기본값)는 그 turn을 생성되었을 때보다 짧게 다시 렌더링하게 되고, 후속 요청은 prompt cache를 놓칩니다. prompt cache가 켜져 있으면 chat 라우트는 각 일반(tool-call이 아닌) 응답에 대해 반환한 정확한 reasoning을 기억하고, 클라이언트가 되돌려 보낸 것처럼 이후의 content만 있는 assistant turn에 다시 넣습니다. 대상은 `/v1/chat/completions`(`reasoning_content`, issue #2110), `/v1/responses`(`reasoning` 출력 항목), `/v1/messages`(`thinking` 블록. 확장 thinking이 활성화된 요청에서만 기록되며, 그렇지 않으면 클라이언트가 이를 받지 못하기 때문입니다)이며, 스트리밍과 비스트리밍 모두에 적용됩니다(issue #2118). 저장된 trace는 같은 모델, template 서명(template, kwargs, tools), 세션(`prompt_cache_key`, 그다음 `user`), 동일한 선행 메시지, 동일한 assistant `content`를 가진 요청에서만 찾아지며, 어떤 편집이든 miss가 되고 miss는 아무것도 바꾸지 않습니다. 클라이언트가 직접 되돌려 보낸 reasoning(`reasoning_content`, Responses의 `reasoning` 입력 항목, Anthropic의 `thinking` 블록)은 항상 우선합니다. 저장소는 32바이트 키와 reasoning 텍스트만 보관하며, 16 MiB와 4096개 항목으로 제한되고 least-recently-used 방식으로 축출합니다(`MLXCEL_REASONING_ECHO_MAX_BYTES`, `0`은 비활성화). 재시작하면 비어 있습니다. `cache_prompt: false`, `--reasoning-format none` 또는 `deepseek-legacy`(trace가 이미 `content`에 인라인으로 전달됨), `--skip-chat-parsing`에서는 참조되지 않으며, 프롬프트 검사 라우트는 다루지 않습니다. `/v1/chat/completions`의 다음 turn warm-up(#1144)은 저장소가 reasoning을 유지한 경우 항상 그 reasoning과 함께 응답을 렌더링하고, 요청은 렌더링된 그대로(채워진 turn 포함) 렌더링합니다. 따라서 warm-up된 prefix는, content만 있는 후속 요청이 자체 fill 이후에 렌더링하는 것과 일치합니다(issue #2118).

#### Fused decode 커널

v0.4.4부터 batched paged decode step은, 시퀀스마다 보이는 KV를 pool에서 복사해 낸 뒤 attention을 수행하던 시퀀스별 루프 대신 **batch 전체에 대한 하나의 fused attention launch**로 실행됩니다(issue #899). 커널은 CSR page table을 통해 pool의 흩어진 page를 제자리에서 읽고, KV 범위를 thread block에 나누므로, 병렬성이 포화되지 않고 context에 따라 커집니다. Apple M1 Ultra에서 이전 경로와 비교해 측정한 결과는, 1K~32K context에 걸친 batch 4에서 1.4x~3.1x, 16K와 32K의 단일 시퀀스에서 1.3x~1.5x입니다(`docs/benchmark_results/paged-decode-v2-m1ultra-2026-07-31.md`).

이 커널이 처리하지 *않는* 경우에는, 위에서 설명한 동일한 parity 한계 아래에서 launch마다 이전의 gather-then-SDPA 경로로 fallback합니다.

| 경우 | 이유 |
|---|---|
| 보이는 KV 토큰이 4096 미만인 단일 요청 | 그 구간에서 더 느리게 측정됨 (context 1K, batch 1에서 0.91x) |
| 요청당 보이는 토큰이 512 미만인 다중 요청 launch | 측정된 근거보다 낮은 구간이며, 그 크기에서는 gather가 매우 저렴함 |
| pool 행이 둘 이상의 slab에 걸쳐 있는 layer | 커널은 한쪽당 연속된 버퍼 하나를 읽음 |
| 다중 토큰 step: batched prefill, speculative / MTP verify | 커널은 시퀀스당 단일 query 토큰만 처리함 |
| logit-softcap 계열, 명시적 attention mask | 커널에 soft-cap이나 mask 항이 없음 |
| Int8 / Turbo KV 모드 | 해당 시퀀스는 dense cache를 유지하며 pool 기반이 아님 |

하한이 두 영역(two-regime)으로 나뉘는 이유는, 측정된 손실이 전체 작업량이 아니라 batch 1의 속성이기 때문입니다. 요청당 context 1024 토큰에서 커널은 batch 1에서는 0.91x이지만 batch 4에서는 1.41x, batch 8에서는 1.47x로 실행됩니다. batched launch는 같은 chunk 수를 더 많은 요청에 나누고 merge pass를 분할 상환하기 때문입니다.

#### dense 대 paged 선택이 이루어지는 곳

커널은 모델 forward가 아니라 cache가 고릅니다([ADR 0008](adr/0008-kv-attention-dispatch-in-the-cache.md), issue #2171; [ADR 0009](adr/0009-engine-step-api.md), issue #2172). `KVCache::attend`는 해당 step의 K/V를 추가하고, 그 cache 뒤의 저장소에 대해 attention을 실행합니다. `cache::attend_batched`는 batch의 한 layer에 대해 같은 일을 합니다. Qwen3, Llama 3, 그리고 attention을 재사용하는 계열(Qwen2, Qwen2.5, Helium, VLM text backbone)이 이것들을 호출하며, DeepSeek V2의 absorbed decode는 `MlaLatentCache::attend`를 거칩니다. latent MLA 계열은 압축된 latent와 위치 key cache 표현을 유지하고, 공유 projection과 MLP를 batch로 처리한 다음, 요청별 ragged cache 행을 통해 갱신하고 attend합니다. DeepSeek V3.2와 GLM MoE DSA도 batched index projection 이후 DSA indexer 선택을 행마다 독립적으로 적용합니다. 모델 소유 계열은 시퀀스별 상태를 자체 cache enum에 보관하고, 같은 규칙으로 attend합니다. Gemma 3의 `Cache::attend`는 global layer를 `KVCache::attend`로, sliding layer를 `RotatingKVCache::attend`로 라우팅하며(쓰기 경로는 `update_and_fetch`로 유지되고, 여기서 #2182 decode undo log가 모든 speculative 쓰기를 기록합니다), Llama 4의 `Llama4Cache::attend`는 chunked layer를 `ChunkedKVCache::attend`로 라우팅합니다. 이들 중 어떤 것이든 batch는 `cache::attend_batched_rows`를 실행하며, 행마다 그 행의 자체 cache를 통한 attention 하나씩을 수행합니다. 어떤 모델도 `is_paged_backed()`를 읽지 않으며, 스케줄러는 더 이상 step마다 `DecodeBatchContext`를 만들지 않습니다. `forward_batched_with_ids`는 id와 cache만 받으며, 저장소 정책은 시퀀스가 열릴 때 `CachePool`이 연결한 그대로입니다.

- pooled launch는 마스크 없는 단일 토큰 step에서 `caches[0].is_paged_backed()`를 기준으로 판단하므로, 모든 batched 호출자(동기 step, lookahead prime, `forward_batched`)가 같은 방식으로 fused launch에 도달합니다. pool 기반 행과 dense 행이 섞인 batch는 쓰기 전에 거절되고, 각 행을 자신의 저장소에서 `attend`로 실행합니다.
- 모델 소유 상태는 pool 기반이 아니므로, 서버가 어떤 `--decode-storage-backend`를 실행하든 Gemma 3과 Llama 4는 항상 표의 dense 행을 따릅니다. 이들이 paged 백엔드에서 선택하던 dense-pointer `paged_decode_attention_dense_compat` 및 `_rotating_compat` 경로는 행별 C++ 루프(block slice, concat, 행마다 SDPA 한 번)였습니다. 행별 `attend` 루프는 block concat이 없는 같은 attention이며, 이와 비트 단위로 동일하지는 않습니다. dense 백엔드에서 FP16 또는 Int8 Gemma 3 행은 이전과 정확히 같은 op를 실행합니다. dequant-first 또는 압축 decode 변형이 있는 어떤 Turbo 모드의 Gemma 3 global layer(Turbo4Asym, Turbo4Delegated, 그리고 환경 변수 gate 뒤의 sparse-V)는, `update_and_fetch`와 SDPA 대신 Qwen3와 마찬가지로 `KVCache::attend`가 선택하는 변형을 이제 사용합니다. Llama 4의 RoPE layer는 두 백엔드 모두에서 블록 전체를 batch로 실행합니다(projection, residual, post-attention norm, MoE feed-forward. #2172 이전에는 dense 백엔드가 모든 행을 단일 행 블록으로 실행했습니다). 행별 attention은 변하지 않았습니다. 실제 체크포인트 검사(`scheduler_real_batch_parity_tests`, `MLXCEL_SDPA_DETERMINISTIC=1`에서의 gemma-3-1b-it-4bit)에서는 두 백엔드 모두에서 동시 greedy 요청 두 개가 각 요청을 단독으로 실행했을 때와 같은 토큰 스트림을 얻었습니다.
- [2026-10-11 GB10 paired/null 저장소 sweep](benchmark_results/model-owned-storage-gb10-2026-10-11.md)은 두 모델 소유 계열 모두에 대해 `supports_paged_decode_backend()`를 true로 유지했습니다. 계열마다 프롬프트/동시성 셀 여섯 개 중 어느 것도, 어느 방향으로든 rotating paged-repeat null 분산을 넘지 못했습니다. 따라서 다중 슬롯 서버의 `auto`와 명시적인 `paged` 요청은 여전히 Gemma 3과 Llama 4에 paged 저장소를 선택합니다(pool 할당과 step별 mirror 포함). 그 attention은 위에서 설명한 dense 행을 통해 계속 실행됩니다.
- Turbo cache 위의 batched decode는 step마다 전체 dequant를 하는 대신, 단일 시퀀스 decode가 이미 사용하던 dequant-first 변형을 택합니다. attention은 연산 순서만 다른 같은 수학이므로, 이전 batched 경로와 비트 단위로 동일하지는 않습니다.
- non-FP16 paged 레이아웃의 FP16 layer(Turbo 모드가 FP16으로 유지하는 Boundary-V layer, `--kv-skip-last-layer` 아래의 마지막 layer)는 dense cache를 보관합니다. 이들은 이제 제거된 `paged_decode_attention_dense_compat` 경로 대신 행별 dense 갱신과 SDPA를 실행합니다. concat 복사가 없는 같은 attention이지만, 이와 비트 단위로 동일하지는 않습니다.

#### 어느 경로가 실행되었는지 확인하기

서버는 서로 다른 dispatch 결과마다 처음 발생한 것을 **info** 수준으로 알리며, `RUST_LOG`는 필요하지 않습니다.

```text
INFO ...: paged decode v2: fused v2 launch (batch 4, 3828 visible KV tokens, 16 chunks, merge on)
INFO ...: paged decode v2: gather: the layer spans 3 K / 3 V pool slabs and the fused kernels read one buffer per side (slab_blocks=32); raise --ctx-size or MLXCEL_PAGED_SLAB_BLOCKS so a layer's rows fit one slab
INFO ...: paged decode v2: gather: 1024 visible KV tokens across 1 request(s) is below the 4096-token dispatch floor
INFO ...: paged decode v2: gather: pinned by MLXCEL_PAGED_ATTENTION_NATIVE
INFO ...: paged decode v2: fused v2 cascade launch (batch 4, 4 of them sharing 256 pages / 8192 KV tokens read once; 8 shared-span chunks, 4 suffix chunks)
```

`fused v2 launch`를 한 번도 출력하지 않은 실행은, 처리량이 어떻게 보이든 fused 커널을 사용한 적이 없습니다. before/after 비교에서 결론을 내리기 전에 이 줄이 있는지 확인하십시오.

보통은 필요하지 않은 설정이 두 가지 있습니다.

- `MLXCEL_PAGED_ATTENTION_NATIVE=0`은 kill switch입니다. 커널 회귀가 의심될 때 bisect할 수 있도록 #899 이전의 gather 동작을 처음부터 끝까지 복원합니다. `=1`은 토큰 하한을 우회하여 서비스 가능한 모든 형태에 fused 경로를 강제합니다. 거절된 구석 사례를 벤치마크하는 지원되는 방법입니다. 하한 자체는 `MLXCEL_PAGED_V2_MIN_KV_TOKENS`(단일 요청, 기본값 4096)와 `MLXCEL_PAGED_V2_MIN_KV_TOKENS_PER_REQUEST`(다중 요청, 기본값 512)로 조정됩니다.
- `MLXCEL_CASCADE_ATTENTION=1`은 2단계 cascade decode를 활성화합니다(issue #903). batch에서 여러 시퀀스가 공유하는 전체 page 단위의 프롬프트 prefix는, 시퀀스마다 한 번씩 다시 읽는 대신 하위 그룹에 대해 한 번만 attend하고 각 구성원의 요청별 접미사 상태에 병합됩니다. 서빙 측정을 기다리는 동안 기본적으로 꺼져 있습니다. 자체 결과 줄이 실행 여부를 알려 주며, `/metrics`는 flat 및 gather 카운트와 함께 `mlxcel_paged_decode_launches_total{path="cascade"}`를 제공합니다. [Cascade attention](cascade-attention.md)을 참조하십시오.
- `MLXCEL_PAGED_SLAB_BLOCKS`는 pool의 slab 크기(block 단위)를 재정의합니다. 서버는 이를 `--ctx-size`와 `--parallel`로부터 유도하고 KV 예산으로 제한하며, 시작 시 확정된 값을 로그로 남깁니다(`Paged KV slab size: N blocks per layer`). layer의 첫 쓰기가 slab 전체를 할당하므로, 이는 시작 시 메모리 추정이 그 batch와 context에 대해 이미 보고하는 KV 바이트를 정확히 미리 확보합니다. **`--ctx-size`가 암시하는 것보다 긴 context를 서빙하면 layer가 slab을 넘어서고, 조용히 gather 경로로 되돌아갑니다.** 시작 시 줄이 너무 작아 보이면 `--ctx-size`를 올리거나 이 변수를 설정하십시오. `0`은 과거의 32 block 기본값으로 고정합니다.

Recurrent 및 hybrid SSM / linear-attention 계열은 임의의 KV block을 안전하게 재사용할 수 없으므로, hybrid-SSM/APC 제외 대상으로 남습니다. Baichuan-M1은 이 제외 대상에 추가로 포함되는 dense 계열입니다(#2567). K/V convolution 상태가 시퀀스별로 모델이 소유하며, 복원된 KV prefix는 이를 담지 못하므로, whole-prefix 저장, APC, KV handoff가 모두 이를 거절합니다. `supports_snapshot_reuse()`에 동의한 계열은 대신 별도의 정확한 prefix snapshot 버킷을 사용합니다. 정상 종료 시 스케줄러가 모델 소유 상태를 복사하고, 다음 turn에서는 저장된 토큰 벡터가 같은 세션의 들어오는 요청의 완전한 prefix일 때만 그 상태를 복원합니다. 검증된 batch 도중에 끝나는 speculative burst(batch의 마지막 토큰 이전에 발생한 stop 문자열, 생성 한계, context 한계, 반복 루프)는 donate하지 않습니다. 모델 소유 상태가 이미 커밋된 토큰보다 앞서 있기 때문입니다. 일치하지 않는 접미사는 여전히 정상적으로 prefill되며, recurrent 상태 truncation이나 세션 간 공유는 없습니다. 지원되는 snapshot 계열은 Mamba, Mamba2, Jamba, Nemotron-H, Qwen 3.5 / 3.6 text, MoE, VLM wrapper, 그리고 Gemma 4 text, VLM, Unified wrapper입니다.

멀티모달 서버에서는 `--enable-vlm-prefix-cache`가, 같은 이미지를 가진 같은 세션의 후속 turn에 걸쳐 이미지 요청이 prompt-prefix를 재사용하도록 옵트인합니다. VLM 요청의 기본값은 계속 꺼져 있으며, text 전용 prompt-cache 동작은 바뀌지 않습니다.

### Continuous batching에서의 speculative decoding

drafter 체크포인트를 제공하면(`mlxcel serve`에서는 `--draft-model`, `mlxcel-server`에서는 `--model-draft`이며, 두 표기는 서로 별칭입니다) 서버 측 speculative decoding이 켜집니다. 종류는 drafter의 `config.json`에서 자동 감지되고(`--draft-kind mtp|dflash`로 재정의), draft 블록 크기 K는 시작 시 결정됩니다(`--draft-block-size`와 두 환경 변수 표기가 이를 재정의). 우선순위는 다음과 같습니다. drafter 체크포인트가 스스로 선언한 폭이 가장 먼저이고, 그다음 실행 중인 compute capability와 타깃의 quantization 모드에 대해 측정된 기본값이 쓰입니다(issue #1797. GB10에서 affine-quantized 타깃에 대한 DFlash drafter는 4로 결정됨). 마지막으로 종류별 고정 상수(MTP는 4, DFlash는 16)가 쓰입니다. 결정된 폭과 그것이 위 중 어느 것에서 나왔는지는 시작 시 `speculative=` 줄에 기록됩니다.

싱글톤(B=1) speculative burst가 실제로 실행되는지는 플래그만으로 정해지지 않고, 적응형 정책(issues #333/#736)이 결정합니다. 각 (target, drafter, 하드웨어, K) 조합의 처음 4개 적격 요청은 speculation을 강제로 켠 채 프로파일링됩니다. 프로파일링되는 각 burst는 classic-step probe round 2개(실제 greedy 토큰을 내는, drafter 없는 단일 토큰 verify)도 함께 실행합니다. 그래서 정책은 측정된 speculative round 비용을 측정된 classic step 시간과 비교하고, Enable, Decline, 또는 추정이 모호할 때 하드웨어별 정적 기본값 중 하나로 정착할 수 있습니다. 판정은 `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/mtp-policy/` 아래에 저장되며 정착 시점에 기록됩니다(`adaptive MTP policy: settled verdict ... est_speedup=... [measured]`). 따라서 재시작하면 프로파일링을 건너뜁니다. K를 바꾸면 다시 프로파일링합니다. 결정이 측정에 기반하므로, 같은 바이너리가 speculative round가 비용을 회수하는 조합에서는 이를 활성화하고(Gemma 4 12B + assistant: 출력이 얼마나 예측 가능한지에 따라 M5 Max에서 1.90x~3.14x, multirow qmv verify를 쓰는 GB10에서 약 1.5x), 그렇지 않은 조합에서는 거부합니다(M1 Ultra에서의 같은 조합, 또는 `MLXCEL_QMV_MULTIROW=0`인 CUDA).

`GET /v1/internal/mtp-policy`는 버전이 지정된 지원 인터페이스를 통해 현재 조합의 상태를 보고합니다. 판정이 정착했는지와 그 방향, 프로파일링 윈도우가 아직 필요로 하는 적격 샘플 수, 정책이 실행되고 있지 않다면 그 이유가 포함됩니다. 호스트 애플리케이션은 비공개 디스크 형식인 hint 파일을 파싱하는 대신 여기서 읽어야 합니다. [Adaptive MTP policy API](mtp-policy-api.md)를 참조하십시오.

스케줄링: batch할 수 없는 타깃(12B Unified 계열)에서는 활성화된 B=1 burst가 그들의 고유한 단일 슬롯 직렬화와 같습니다. batch가 가능한 타깃에서는 B=1 MTP arm이 기본적으로 tick 협력(tick-cooperative) 방식으로 서비스됩니다(issue #734). 요청은 스케줄러 tick마다 speculative round를 하나씩 실행하고 classic decode/prefill 동작과 번갈아 진행하므로, 동시 classic-decode 행들은 요청 전체 동안 멈춰 있는 대신 round 사이에 전진합니다. 이는 speculative 요청이 동시 행들에 가하는 head-of-line stall을 약 한 round로 제한합니다. 스케줄러는 finalize 시점에 이를 단일 tick의 최대 `burst_wall_ms`로 기록합니다(검증 조합에서 이는 수 초 걸리던 burst 전체 정지를 정상 상태 점유의 약 한 round로 낮추었고, 동시 classic 행들의 첫 토큰까지 시간을 약 23배 빠르게 했습니다). 교차 실행은 공짜가 아닙니다. 검증 조합에서 엄격한 교대는 두 요청을 합한 총 decode 처리량의 약 27%를 소모했습니다. 따라서 동시 classic 트래픽 없이 speculative 요청을 서비스하고 순수한 speculative 처리량을 원하는 배포에서는 `MLXCEL_MTP_TICK_SLICE=0`이 기존의 run-to-completion burst를 복원합니다. DFlash B=1과 모든 B>1 batched burst는 플래그와 관계없이 여전히 하나의 tick 안에서 끝까지 실행됩니다. 프롬프트 길이, `max_tokens`, 샘플링 구성이 같은 동시 speculative 요청은 대신 하나의 B>1 batched burst로 실행될 수 있습니다. 다만 그 경로는 실험적이며, MTP의 경우 `MLXCEL_ENABLE_MTP_BATCH`(혼합 프롬프트 길이에는 `MLXCEL_ENABLE_MTP_BATCH_RAGGED` 추가) 뒤에 있습니다. 대기 중인 speculative 적격 요청은 기본 `--max-batch-prefill` 아래에서 burst에 도달합니다(issue #2442). classic 행이 decode 중이지 않은 동안에는, burst가 서비스할 선두(head) 요청이 classic batched prefill을 건너뜁니다. classic 행이 활성이면 대기 행은 batched prefill을 유지합니다. 그래야 tick당 run-to-completion burst가 그들의 decode를 막지 못합니다. 그래서 길이가 같은 대기 DFlash 요청들은 batched로 실행되는 타깃에서 하나의 batched DFlash window로 실행되며, burst가 거절만 하게 될 선두는 batched prefill을 유지합니다. MTP verify가 행 단위로 실행되는 Gemma 4 기하(어디서나 31B, CUDA에서는 12B)에서는, issue #2190 이후 batched burst가 decode-exact입니다. 각 행은 B=1 호출을 통해 prefill되고 verify되며, cache 쓰기만 batch됩니다. 이것은 CUDA에서 sliding cache가 뒤처진 행 아래에서 압축되지 않을 만큼 짧은, 최대 4행의 window에 대해 실행되며, 그 밖에는 classic decode로 거절됩니다. [GB10 기록](benchmark_results/gemma4-mtp-batched-gb10-2026-10-07.md)을 참조하십시오.

tick 협력 slice는 요청의 생성 전체에 걸치므로, 그렇지 않으면 단일 speculative 슬롯이 그 시간 내내 점유됩니다. issue #746 이후 슬롯은 첫 번째 요청만 서비스하는 대신 동시 speculative 요청들 사이에서 순환합니다. slice가 진행 중인 동안 적격한 tick-slice 요청 최대 2개가 grant 백로그에 보류되며(그 상한을 넘는 요청은 #746 이전의 동작인 classic decode로 fallback합니다), 활성 요청이 백로그가 비어 있지 않은 상태에서 `MLXCEL_MTP_SLICE_GRANT_ROUNDS`개의 slice(기본값 8, slice 0 포함)를 실행하면 다음 round 경계에서 보류되고 슬롯은 다음 요청에 grant됩니다. 우선순위 레인이 먼저이고 레인 안에서는 FIFO입니다. 또한 기아 방지 하한이 있습니다. grant 결정을 두 번 건너뛴 항목은 기한 초과(overdue)가 되어 레인과 관계없이 다음에 grant되어야 합니다. 따라서 높은 우선순위 도착이 지속되더라도(우선순위는 클라이언트가 `X-Priority` 헤더로 지정할 수 있음) 낮은 우선순위 요청의 지연은 유한한 횟수의 grant로만 늦춰지며, 스트림이 아예 멈추지는 않습니다. 백로그 상한은 admission 시점에 적용되고, 보류(park)는 다음 승격 전까지 일시적으로 상한 + 1개의 백로그 항목을 보유합니다. 따라서 살아 있는 speculative 세션 전체(활성 요청과 보류·대기 중인 요청들. 각각 시퀀스별 KV를 보유함)는 최대 3개까지 올라갑니다. 보류된 요청은 다음 grant에서 저장된 세션 상태로부터 재개되며, worker의 단일 drafter 핸들은 요청 종료 시 사용하는 것과 같은 return/take 경로로 넘겨집니다. 이것이 안전한 이유는 MTP assistant drafter의 reset이 no-op이고, 모든 round가 세션 자체에 저장된 verify 출력으로부터 drafter를 다시 준비하기 때문입니다. 따라서 토큰 스트림은 경합이 없는 실행과 바이트 단위로 동일합니다. 예산은 경합 상황에서만 적용됩니다. 단일 speculative 요청은 순환하지 않으며, 그 동작은 #734에서와 정확히 같습니다. `MLXCEL_MTP_SLICE_GRANT_ROUNDS=0`은 순환을 비활성화하고, 모든 동시 speculative 요청에 대한 classic fallback을 포함하여 생성 전체 점유를 복원합니다.

운영자 제어는 모두 환경 변수 수준입니다([environment-variables.md](environment-variables.md) 참조). `MLXCEL_ENABLE_MTP_B1`은 B=1 결정을 어느 방향으로든 고정하고 프로파일링을 억제합니다. `MLXCEL_MTP_ADAPTIVE=0`은 적응형 정책을 끄고 하드웨어별 정적 gate를 사용하게 합니다. `MLXCEL_MTP_SLICE_GRANT_ROUNDS`는 speculative 슬롯 grant 순환의 크기를 정하거나 비활성화하며, `MLXCEL_QMV_MULTIROW=0`은 verify 아래에서 기본 행별 CUDA quantized-matmul 커널을 복원합니다.

CUDA에서는 같은 draft/verify 형태의 다양성(batch 크기와 시퀀스 길이 버킷을 곱한 것)이 MLX 자체의 captured-graph-cache LRU를 가장 세게 밀어붙이기도 합니다. mlxcel은 시작 시 기본적으로 `MLX_CUDA_GRAPH_CACHE_SIZE`를 2000으로 올립니다. 그래야 오래 실행되는 서버가 MLX 자체 기본값 400에서 치명적인 `Cache thrashing` abort에 걸리지 않습니다(issue #818, [environment-variables.md](environment-variables.md)의 `MLX_CUDA_GRAPH_CACHE_SIZE`).

### Continuous batching에서의 prompt lookup

`--draft-kind prompt-lookup`(`mlxcel serve`와 `mlxcel-server`에서, drafter 체크포인트 없이 사용. `--draft-model`을 함께 주면 시작이 실패합니다)은 batch 스케줄러 안에서 요청별로 n-gram prompt-lookup speculation을 제공합니다(#2255). 프롬프트를 복사하는 응답(편집, 재포맷, 인용)에서 이득을 봅니다. prefill이 완료되면 적격 요청은 `PromptLookupDrafter`를 받습니다. `--draft-block-size`는 가장 큰 제안(기본값 7)을 정하며, 백엔드 draft 정책은 CUDA에서 `gated`, 그 밖에서는 `graded`입니다. 시작 요약에는 `max_draft`, `policy`, `max_batch`, `min_tokens_per_round`, `shadow_confirm_tokens`, `plain_rounds_before_pipeline`이 포함됩니다.

CUDA 서버는 손익분기 governor를 사용합니다(#2264). drafted round가 두 번 지난 뒤, verify round당 방출 토큰(수락된 제안과 bonus)의 EMA(가중치 0.5)가 4 미만이고 가장 최근 round도 그 하한 미만이면 행을 일시 중지합니다. 수락이 0인 round 하나는, 그 앞의 수락 토큰 EMA가 하한을 충족하면 면제됩니다. 이 유예는 첫 번째 연속 저수율 round를 넘어서 이어지지 않으며, 일시 중지에는 여전히 방출 토큰 EMA가 하한 미만이어야 합니다. 이는 강한 복사 증거 이후의 고립된 miss는 보존하고, 부분적으로만 수락되어 수익성이 없는 round는 그 유예 없이 일시 중지하게 합니다. 이것은 벤치마크 보고서의 작업 부하에서 측정된 휴리스틱이며, 임의의 산문에 대한 보장은 아닙니다. 이미 하한을 충족하는 round는 일시 중지 없이 시작 시 EMA 부채를 갚을 수 있습니다. 명시적으로 더 좁은 block에서는 하한이 `max_draft + 1`로 제한되므로, 전체 수락은 계속 drafting합니다. 일시 중지된 행은 즉시 pipelining을 허용하고 shadow lookup을 계속합니다. 연속된 제안 토큰 일곱 개가 맞아야 새로운 증거와 함께 verify가 재개됩니다. 제안이 없는 일반 요청은 CLI의 세 번(three asks)이 아니라 첫 번째 요청 뒤에 다시 prime됩니다. 각 행은 동시성 2에서도 자신의 증거를 유지하며, 제안하는 peer가 있으면 공유 tick은 여전히 동기식이 됩니다. CLI 기본값과 graded 정책은 바뀌지 않습니다. [GB10 손익분기 보고서](benchmark_results/server-prompt-lookup-break-even-gb10-2026-10-11.md)는 최종 통과한 매트릭스를 기록합니다. chat −0.275%, 짧은 복사 +149–159%, 긴 복사 1.52–2.82×입니다. 이 측정은 GB10의 Qwen3-1.7B 4-bit를 다루며, 경계 miss 휴리스틱은 다른 작업 부하에 대한 보장이 아닙니다.

각 decode tick은 적격 행들에게 제안을 요청합니다. 제안하는 행은 자체적으로 verify round 하나를 실행하고(`mlxcel generate --prompt-lookup`이 실행하는 round), 다른 모든 행은 일반 batched step을 택합니다. 그래서 동시 요청은 tick마다 계속 토큰을 받으며, run-to-completion burst는 없습니다. 아무도 제안하지 않는 tick은 lookahead 파이프라인에 머뭅니다. step이 진행 중일 때 발견된 제안은 다음 step을 제출하지 않고 그 step을 커밋하며, 이어지는 tick은 동기식으로 verify하므로 전환에 forward 비용이 들지 않습니다. 파이프라인은 drafter들이 제안 없는 round가 이어졌다고 보고하면 재개됩니다.

제안하는 각 행은 tick당 추가 forward 한 번을 비용으로 치르므로, verify round는 활성 decode batch의 행이 `--prompt-lookup-max-batch`개 이하(기본값 2. gate 8에서는 chat-template 작업 부하 하나가 GB10의 동시성 4에서 13%를 잃었음)일 때만 실행됩니다. 그보다 많으면 모든 행이 일반 decode를 하고, drafter는 miss를 기록하지 않으며, batch가 줄어들면 다시 재개됩니다. paged 저장소에서는 전체 context가 4608 토큰을 넘으면 시퀀스가 verify를 멈춥니다. 원래의 4096 타깃 벤치마크는 실제로 프롬프트 토큰 4099–4102개로 토큰화되며, 256 토큰 continuation과 함께 최대 4358개 위치가 필요합니다. 별도의 admission 진단에서는 기존 4096 컷오프가 그 작업 부하의 모든 제안을 억제했음이 드러났습니다. 4608 상한은 그 범위를 덮는 가장 작은 512 토큰 경계입니다. 8192 토큰 context는 여전히 제외하는데, 이 구간에서는 gathered verification이 이전에 성능을 잃었습니다(수락률 59% 프롬프트에서 0.62x). 이 경계의 근거는 새 구성의 성능을 입증하지 않으며, 동일 빌드의 전체 acceptance 실행이 필요합니다. [context 진단](benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/context-diagnostic.md)을 참조하십시오. dense 저장소에는 이런 제한이 없습니다.

손익분기 governor 이전에는 이 플래그가 프롬프트를 복사하는 출력에서는 이득을 보았습니다(GB10의 B=1에서 문단 반복 프롬프트에 대해 +53~+67%, 앞에 긴 문서가 있으면 최대 2.8x). 그러나 응답이 거의 복사하지 않는 chat(제안된 토큰의 25~45%만 수락)에서는 3.5%의 손해를 보았습니다. 각 제안이 한 tick 동안 lookahead도 끝내기 때문입니다. 입력을 인용하거나 편집하는 작업 부하에서 켜십시오. 측정: [`docs/benchmark_results/server-prompt-lookup-gb10-2026-10-11.md`](benchmark_results/server-prompt-lookup-gb10-2026-10-11.md). 서버 손익분기 governor는 이러한 저수율 round를 다룹니다(#2264). 통합 KV 예산(`--kv-unified`) 아래에서도 verify round는 꺼진 상태로 남습니다. 이미지, 오디오, 비디오가 있는 요청, 구조화된 출력, 토큰별 logprobs, mirostat, adaptive-p 또는 확장 sampler 체인(dynamic temperature, `min_keep`)이 있는 요청, 그리고 cache trim으로 상태를 되돌릴 수 없는 모델(recurrent, hybrid, 모델 소유 계열)은 일반 decode를 하며, `debug` 로그가 이유를 밝힙니다. 완료된 요청은 `timings` 블록에 `draft_kind: "prompt-lookup"`과 함께 verify round를 보고합니다([speculative-acceptance.md](speculative-acceptance.md) 참조). `--spec-type none`은 다른 drafter와 마찬가지로 prompt lookup을 끕니다.

첫 번째 적격 prefill(보통 서버의 시작 warmup 요청)은 모든 block 폭(2부터 `--draft-block-size` + 1까지)에서 verify forward를 한 번씩 실행하고 각각을 되돌립니다. `mlxcel generate --prompt-lookup`이 decode 전에 하는 것과 같으며, 따라서 첫 요청들이 decode 도중에 각 폭의 첫 사용 커널 비용을 치르지 않습니다.

### Continuous batching에서의 full model drafter

`--draft-model <일반 모델>`(자동 감지: DFlash 계열 표지도 MTP `model_type`도 없는 디렉터리. 또는 `--draft-kind full-model`이며, b10621의 `--spec-type draft-simple`이 여기에 대응합니다)은 프롬프트 lookup과 같은 행별 tick에서, 일반적인 더 작은 체크포인트를 token 전용 drafter로 제공합니다(#2757). worker는 시작 시 drafter를 target 옆에 로드하므로, 로드할 수 없는 체크포인트나 상태를 trim으로 되돌릴 수 없는 target 또는 drafter(#2495)는 시작에 실패하며, 일반 decode로 조용히 떨어지지 않습니다. chat 라우트는 worker를 사용할 수 없다고 보고하고, `/health`는 준비 완료 상태가 되지 않습니다. 시작 로그는 로드 후의 상주 메모리를 보고합니다. 적격 요청은 prefill이 완료되면 `DraftModelState`를 받습니다. 이 상태는 프롬프트를 기록하며 draft KV는 전혀 보유하지 않습니다. 게이트에 들어온 뒤에는 draft 모델이 프롬프트를 tick당 chunk 하나씩 따라잡으며, 그 이후의 모든 round는 `--draft-block-size`(기본값 4)번의 draft forward와 target에서의 verify forward 하나를 비용으로 치릅니다. 거절된 block은 둘 모두에서 trim됩니다. drafter는 자신의 argmax를 제안하고, 모든 위치의 결정은 target 자신의 draw가 내리므로, 샘플링된 스트림은 target의 분포를 유지합니다. temperature 0에서는 token 전용 루프의 기준에 따라 target의 greedy 선택을 따르지만, 이는 plain decode와 바이트 단위로 동일하다는 뜻은 아닙니다. [`docs/speculative-acceptance.md`](speculative-acceptance.md#오프라인-라우팅)를 참조하십시오.

**Batch 폭.** 각 행은 자신의 cache에서 drafting하고 롤백하므로 batch 단위 drafter는 필요하지 않지만, verification은 행 단위이며 직렬이고 batch로 처리되지 않습니다. `--prompt-lookup-max-batch`(기본값 2)는 동시에 verify할 행 수를 정합니다. 활성 행 수가 그보다 많은 tick에서는 모든 행이 일반 batched step을 택하며 speculation을 전혀 받지 못하고, batch가 줄어들면 재개됩니다. 따라서 기본값에서는 B = 4로 안정된 서버가 drafter를 로드하고 유휴 상태로 둔 채 일반 decode를 실행합니다. 이 플래그를 올리면 더 넓은 batch에서도 verify할 수 있지만, 그러면 행이 하나 추가될 때마다 모든 tick에 draft forward와 verify forward가 더해지며, 아래 측정에서 B = 4와 B = 8일 때 처리량의 35~68%를 잃었습니다. 게이트 위의 행은 처음 요청받을 때까지 draft KV를 보유하지 않습니다. verify round는 `--kv-unified`에서도 꺼져 있으며, 그곳에서도 drafter는 로드된 상태로 남습니다.

**Catch-up.** 안정된 꼬리(steady tail)보다 많은 양을 draft 모델에 빚진 게이트 진입 행(처음에는 프롬프트, 또는 게이트 밖에 있는 동안 놓친 토큰들)은, 따라잡는 동안 제안하지 않고 일반 decode를 하며, 따라잡은 뒤에 제안합니다. tick은 `--prompt-lookup-max-batch` 값과 무관하게, 빚진 행 중 batch에서 첫 번째 행에 최대 하나의 catch-up chunk만 공급합니다. 따라서 따라잡기는 tick마다 살아 있는 행들에 draft forward 한 chunk 분의 비용만 줍니다. 빚진 다른 행들은 차례를 기다리며 일반 decode를 합니다. chunk는 서버의 prefill chunk(`--prefill-chunk-size`, 기본적으로 단독 시 2048, 다른 행이 decode 중이면 512)입니다. target의 prefill에서 chunking을 끄는 `0`도 여기서는 이 기본값을 씁니다. draft에는 prompt cache가 없고, 꼬리 전체에 대한 forward는 중단할 수 없기 때문입니다. 행은 예산이 그 비용을 상환하지 못하면 따라잡지 않습니다. 남은 `max_tokens`가 context 한계로 제한된 상태에서 `ticks * (--draft-block-size + 2)`를 덮어야 합니다. 따라잡기 tick마다 일반 토큰 하나와 완전한 round 하나가 필요하기 때문입니다. 따라서 예산이 짧은 요청은 일반 decode를 하며 draft forward 비용을 전혀 내지 않습니다. 따라잡는 도중 취소된 행은 더 이상 아무것도 공급받지 않으며, 취소된 다른 행과 마찬가지로 떠납니다.

이 drafter가 제공하지 않는 경우와 그 이유는 다음과 같습니다. `--no-batch`(기존의 순차 worker에는 speculative dispatch가 없으며, 시작 시 경고합니다). KV handoff로 받아들인 disaggregated `--node-role` 행(drafter가 기록할 prefill이 없으므로 일반 decode를 합니다). 이미지, 오디오, 비디오, 구조화된 출력, logprobs 또는 확장 sampler 체인이 있는 요청(적격성은 prompt lookup과 같으며, `debug` 로그와 함께 일반 decode로 처리됩니다). target보다 큰 어휘, 또는 두 config 중 하나가 선언하지 않은 어휘는 시작 시 거절됩니다. target의 embedding 표를 넘는 제안은 검증할 수 없고, 선언되지 않은 쌍은 점검할 수 없기 때문입니다. context가 `--max-kv-size`를 넘는 행은 drafting을 멈추고 일반 decode를 하는데, draft cache에는 `--context-shift` 경로가 없기 때문입니다. router 모드(`--models-dir`)에서는 각 모델의 worker가 drafter를 자신의 사본으로 로드합니다. 완료된 요청은 `timings`에 `draft_kind: "full-model"`을 보고합니다.

**속도 향상은 측정되지 않았습니다.** 이 drafter는 엔진에 그 기능을 남겨 두는 것이며, 처리량을 올리리라 기대하지 마십시오. 유휴 상태의 M1 Ultra(Metal, 256 토큰, temperature 0)에서 Qwen3-8B 4-bit는 0.6B 4-bit drafter와 함께 B = 1에서 93.5에서 56.1 tok/s로 떨어졌고, 1B 4-bit drafter와 함께한 Llama 3.1 8B bf16은 40.3을 유지했습니다. 수락률은 0.58~0.80이었으므로, 높은 수락률이 draft forward 비용을 회수하지 못했습니다. B = 2에서 Llama의 34.1 대 25.8은 이득으로 계산하지 않습니다. 해당 체크포인트에서 plain B = 2가 plain B = 1보다 설명되지 않게 느리기 때문입니다. 전체 표와 catch-up 비용은 #2757 pull request에 있습니다.

## Disaggregated serving

Prefill은 compute-bound이고 decode는 memory-bound이므로, 배포는 이들을 router를 앞에 둔 별도의 프로세스(또는 머신)에서 실행할 수 있습니다. `--node-role`이 역할을 고릅니다.

- **`prefill`**: 프롬프트 prefill을 실행하고, 첫 토큰을 샘플링하며, 시퀀스의 KV cache를 TCP로 decode 노드에 넘깁니다.
- **`decode`**: KV handoff를 받아 autoregressive decode를 계속하고 continuation을 반환합니다.
- **`router`**: 모델이 없는 HTTP front입니다(모델 가중치가 아니라 tokenizer와 chat template만 로드합니다). 클라이언트 요청을 토큰화하고 prefill 노드로 라우팅하며, prefill 노드의 첫 토큰과 decode 노드의 continuation을 하나의 SSE 스트림으로 병합하여 클라이언트에 돌려줍니다.
- **없음 (hybrid)**: 단일 노드 서빙이며, 분산 플래그 없이 시작한 서버와 바이트 단위로 동일합니다.

요청 흐름(topology A): 클라이언트 HTTP 요청 -> router -> prefill 노드(TCP) -> decode 노드(TCP) -> router가 두 절반을 병합 -> 클라이언트 SSE.

네트워킹 플래그:

| 플래그 | 역할 | 용도 |
|------|------|---------|
| `--serving-bind <addr>` | prefill, decode, router | 이 노드 자신의 role-transport 리스너(`host:port`). |
| `--decode-peers <addr,...>` | prefill, router | router에서는 부하를 분산하는 decode pool이며, result 프레임을 받아들이는 유일한 decode 주소입니다. prefill 노드에서는 router가 요청에 대해 decode 대상을 고르지 않을 때만 쓰는 정적 fallback입니다. |
| `--prefill-peers <addr,...>` | decode, router | router에서는 부하를 분산하는 prefill pool이며, result 프레임을 받아들이는 유일한 prefill 주소입니다. decode 노드에서는 handoff를 받아들이는 prefill 노드입니다. |

router에서 각 peer 항목은 해당 노드가 연결해 오는 주소여야 합니다. 즉 그 노드의 `--serving-bind` 주소이며, 지정되지 않은 bind IP(`0.0.0.0`, `::`)는 router가 연결을 보는 IP로 대체됩니다. 그 밖의 인증된 주소에서 온 result 프레임은 `router: dropping result frame` 경고와 함께 버려지며, 요청은 handoff timeout으로 실패합니다(`docs/distributed.md` 참조).

### 다중 노드 라우팅, 부하 분산, failover

prefill 또는 decode 노드가 둘 이상이면 router가 두 pool을 모두 분산합니다(issue #201). 각 요청에 대해 prefill 노드 하나와 독립적으로 decode 노드 하나를 고르며, 둘 다 registry가 현재 online으로 간주하는 노드들에 대해 round-robin으로 고릅니다(router에는 노드별 실시간 부하 telemetry가 없으므로, 실제로 부하를 분산하는 전략은 round-robin입니다). 선택된 decode 노드는 요청 프레임의 `decode_target` 필드에 담겨 prefill 노드로 전달되므로, prefill 노드는 자체 `--decode-peers` 설정이 아니라 router가 분산한 decode 노드로 KV cache를 넘깁니다. 이 필드는 wire에서 선택 사항입니다. 이 필드가 없는 프레임(이전 router)은 prefill 노드를 설정 fallback에 두고, 이전 prefill 노드는 이를 무시하고 설정을 사용하므로, 버전이 섞인 pool도 계속 작동합니다.

옵트인 심층 방어(defense-in-depth)로서(issue #389), prefill 노드는 연결하기 전에 router가 선택한 `decode_target`을 allowlist에 대해 검증할 수 있습니다. allowlist의 출처는 전용 환경 변수 `MLXCEL_DECODE_ALLOWLIST`입니다. 이것은 router가 선택할 수 있는 decode 노드 전체 pool(공유 클러스터 설정)로 설정하는 숫자 `IP:port` 값(`SocketAddr`로 파싱됨)의 쉼표 구분 목록입니다. hostname:port 항목은 해석되지 않고 경고와 함께 건너뜁니다. 이는 정적 handoff fallback으로만 남는 `--decode-peers`와 독립적이므로, allowlist를 활성화해도 router의 부하 분산을 제약하지 않습니다. `MLXCEL_DECODE_ALLOWLIST`가 설정되지 않은 경우 prefill은 허용적 상태로 남아 거부하는 대신 경고를 기록하므로, 부하 분산이 조용히 깨지는 일은 없습니다. 보안 근거는 [`docs/distributed.md`](distributed.md)를 참조하십시오.

router는 노드 상태를 추적하며, 멈추지 않고 failover합니다.

- prefill 노드로 요청을 보낼 때의 전송 오류는 그 노드를 도달 불가(unreachable)로 표시하고, 진행 중인 요청을 다시 라우팅하며, 정상 노드에서 요청을 재시도합니다. 남은 prefill 노드가 없으면 요청은 멈추는 대신 오류와 함께 깔끔하게 실패합니다. 그 뒤로 종료된 노드에 대한 pool 연결도 그런 오류로 간주됩니다. 유휴 연결을 재사용하기 전에 transport는 바이트를 소비하지 않고 peer가 연결을 닫았는지 확인하며, 그렇다면 새 연결을 엽니다. 따라서 거부된 connect가 죽은 소켓에 쓰인 프레임이 아니라 failover를 주도합니다(issue #2695). router의 transport는 기본값인 100 ms부터의 5회 시도 대신, pool의 router 프로필(`PoolConfig::router()`: 50 ms 기본 지연에서 시작하는 재연결 2회, 2 s connect timeout)을 사용합니다. 따라서 거부된 connect는 3 s 뒤가 아니라 약 150 ms의 backoff 안에 failover합니다(issue #2712). 요청을 수락한 뒤에 죽은 노드는 여전히 그 요청에 전체 handoff timeout을 치르게 합니다.
- 백그라운드 health monitor는 모든 peer의 serving 주소를 TCP probe합니다. router가 시작할 때 한 번, 그 뒤로는 3 s마다 probe합니다(2 s connect timeout). 연결을 수락하지 않게 된 노드는 도달 불가로 표시되고(따라서 라우팅은 이를 건너뛰며, router가 직접 보내지 않는 죽은 decode 노드도 포함됩니다), 다시 수락하기 시작한 노드는 online으로 복원됩니다.
- Admission control은 모든 dispatch 전에 실행됩니다. prefill 대기열이 가득 찼거나 사용할 수 있는 prefill 노드가 없으면, router는 성공할 수 없는 dispatch를 시도하는 대신 HTTP 503을 반환합니다. 사용할 수 있는 prefill 노드가 없고 도달 불가로 표시된 prefill 노드가 하나 이상 있으면, router는 먼저 도달 불가로 표시된 노드들을 500 ms connect timeout으로 다시 probe하고, 수락하는 노드는 복원하며, 하나라도 수락하면 요청을 받아들입니다(issue #2763). 이 재-probe는 single-flight이며 router 전체에서 250 ms마다 최대 한 번 실행됩니다. 동시 요청은 실행 중인 probe를 기다리고 그 결과를 재사용하므로, 실제로 다운된 pool은 250 ms 윈도우마다 최대 500 ms의 probe timeout 한 번만 비용으로 치릅니다(거부된 connect는 즉시 반환하고, 패킷을 조용히 버리는 호스트는 전체 timeout을 치릅니다). 가득 찬 대기열은 probe 없이 거절됩니다.

`GET /router/stats`는 두 pool의 노드별 dispatch 횟수, 현재 health 상태가 포함된 등록 노드 목록, 그리고 라우팅 지표 스냅샷을 보고합니다. 운영자는 이를 통해 분산을 확인하고 어느 노드가 다운되었는지 볼 수 있습니다.

handoff는 메타데이터뿐 아니라 paged block 내용을 transport로 전송하므로, decode 노드는 prefill 노드가 만든 정확한 KV를 재구성합니다. `src/distributed/kv_cache_serde/`를 참조하십시오. disaggregated 출력은 단일 노드 실행과 바이트 단위로 동일하며, `tests/disaggregated_router_e2e.rs`(실제 프로세스 세 개)로 end-to-end 검증되고, `tests/paged_handoff_parity.rs`로 cache 수준에서 검증됩니다.

구현 진입점: `src/server/router_front.rs`(router front), `src/distributed/disaggregated/{coordinator,handoff_impl,serving_protocol}.rs`(역할 루프와 wire 프로토콜), `src/server/batch/`(스케줄러 serving-role 항목).

### 예시: localhost의 프로세스 세 개

모든 역할 프로세스는 공유 클러스터 시크릿이 필요합니다(issue #2655. `--cluster-secret-file`, `MLXCEL_CLUSTER_SECRET_FILE`, `MLXCEL_CLUSTER_SECRET`, 클러스터 TOML의 우선순위는 [cluster authentication](distributed.md#cluster-authentication)을 참조하십시오). 시크릿 없이 시작한 역할은 모든 출처를 나열하며 시작 시점에 종료됩니다.

```bash
# One secret, shared by all three processes (copy it to every host in a
# multi-host deployment).
openssl rand -hex 32 > cluster.secret && chmod 600 cluster.secret

# Decode node: receives KV handoffs on :8304.
mlxcel-server -m models/<ckpt> --port 8302 \
    --parallel 2 --max-batch-size 2 --decode-storage-backend paged \
    --node-role decode --serving-bind 127.0.0.1:8304 \
    --cluster-secret-file cluster.secret

# Prefill node: prefills, hands off to the decode node.
mlxcel-server -m models/<ckpt> --port 8301 \
    --parallel 2 --max-batch-size 2 --decode-storage-backend paged \
    --node-role prefill --serving-bind 127.0.0.1:8305 \
    --decode-peers 127.0.0.1:8304 \
    --cluster-secret-file cluster.secret

# Router: model-free HTTP front on :8300.
mlxcel-server -m models/<ckpt> --port 8300 \
    --node-role router --serving-bind 127.0.0.1:8306 \
    --prefill-peers 127.0.0.1:8305 --decode-peers 127.0.0.1:8304 \
    --cluster-secret-file cluster.secret

# Client talks only to the router.
curl http://127.0.0.1:8300/v1/chat/completions \
    -H 'content-type: application/json' \
    -d '{"model":"m","messages":[{"role":"user","content":"What is 2 + 2?"}],"stream":true,"temperature":0}'
```

### 범위와 제한 사항

- pool 기반 Fp16 계열만 지원합니다(qwen3, llama3와 같은 dense-natural 백엔드). 모델 소유 상태 계열, recurrent/hybrid SSM 모델, Baichuan-M1(KV slice 바깥의 conv 상태, #2567)은 paged handoff에서 제외됩니다. 위에서 설명한 exact-prefix snapshot cache는 단일 노드 prompt-cache 최적화이며, disaggregated prefill/decode 역할 사이에서 직렬화되지 않습니다.
- 텍스트 전용입니다. router는 `POST /v1/chat/completions`와 `POST /v1/completions`를 서비스하며, 멀티모달 요청은 거부됩니다.
- `/v1/completions`에서는 disaggregated wire 프로토콜이 필요한 데이터를 담지 못하므로, 세 가지 옵션 그룹이 HTTP 400으로 거부됩니다. `logprobs`, `response_format`(구조화된 출력), 명시적인 reasoning/thinking 예산입니다. 이 필드들이 없는 요청은 정상적으로 작동합니다.
- 강제된 `tool_choice`(`"required"` 또는 이름이 지정된 함수, #1319)는 `response_format`처럼 거부되지 않고 `/v1/chat/completions`에서 수용됩니다. router는 여전히 선언된 tool을 좁히고, 단일 노드 경로가 사용하는 것과 같은 프롬프트 지시를 주입합니다. 다만 disaggregated wire 프로토콜은 컴파일된 grammar 제약을 decode 노드에 전달할 수 없으므로, 이 호출은 프롬프트로만 강제되며 `warn` 수준으로 한 번 기록됩니다. 같은 요청에서 강제 `tool_choice`를 `response_format`과 함께 쓰면 단일 노드와 마찬가지로 여전히 400이 됩니다.
- `stop` 시퀀스는 router가 worker 노드로 전달하지 않습니다. 이는 chat 경로와 공유하는 기존의 제한입니다.
- usage 블록의 `completion_tokens`는 disaggregated wire 프로토콜로 전달되는 worker의 권위 있는 생성 토큰 수를 사용하므로(issue #387), byte-level BPE tokenizer(Qwen, Llama)와 byte-fallback tokenizer(Gemma `<0xXX>` 바이트 시퀀스) 모두에서 정확하며, `finish_reason`은 단일 노드와 일치합니다. wire 필드 이전 버전의 노드(버전이 섞인 클러스터)는 방출된 detokenize된 텍스트 조각 수를 세는 방식으로 fallback합니다. 이는 byte-fallback tokenizer에서 과소 계산될 수 있으며, `finish_reason`을 `"length"`와 `"stop"` 사이에서 뒤집을 수 있습니다.
- `/v1/chat/completions`는 `/v1/completions`와 같은 `usage` 형태(`prompt_tokens`, `completion_tokens`, `total_tokens`)를 보고합니다. 비스트리밍 응답은 항상 이를 포함하며, 스트리밍 응답은 요청이 `stream_options.include_usage`를 설정하면 `choices`가 빈 후행 chunk로 이를 전송합니다. 이는 단일 노드와 일치합니다(issue #398).
- 두 라우트 모두에서 `usage.prompt_tokens_details`는 요청을 처리한 worker에 적용된 단일 노드 규칙을 따릅니다. 즉 그 prefill 노드가 prompt cache를 켜 두었을 때(기본값) 존재하고, 꺼 두었을 때는 없습니다. prefill 노드는 첫 토큰 result 프레임에 자신의 설정을 보고합니다. 보고 기능이 없는 이전 prefill 노드는 그 결정을 router 자신의 prompt-cache 플래그에 맡깁니다(issues #2695, #2715). `cached_tokens`는 항상 `0`인데, 이는 정확합니다. handoff prefill은 prompt cache를 사용하지 않기 때문입니다.
- 비스트리밍 `/v1/chat/completions` 응답은 단일 노드의 tool 없는 응답과 같은 helper로 형태를 갖추므로, `content`와 `reasoning_content`가 단일 노드와 바이트 단위로 일치합니다(예: SentencePiece의 선행 공백은 양쪽 모두에서 제거됨). 스트리밍 delta는 두 경로 모두에서 형태를 갖추지 않습니다(issue #2715).
- decode 노드는 단일 노드와 마찬가지로, 모델 내장 EOS id뿐 아니라 체크포인트에 설정된 EOS(`config.json`)에서도 멈춥니다. 따라서 EOS를 `config.json`에만 선언한 체크포인트(llama 형식의 MiniCPM-2B)는 `max_tokens`까지 달리는 대신 `</s>`에서 멈춥니다(issue #2695).
- router 스트림 필터는 모델별 구조 마커(`<think>`, tool-call 구분자)를 억제하고, thinking 내용을 `reasoning_content`와, 기본값으로 동일한 `reasoning` 별칭으로 라우팅합니다. 이는 단일 노드 Chat Completions 경로와 일치합니다. `--reasoning-alias-field none`은 중복을 억제합니다. Tool-call 파싱은 router 경로에서 아직 지원되지 않으며, `content`와 설정된 reasoning 필드만 방출됩니다.
- 역할들을 한 머신에 함께 두면 prefill과 decode 하드웨어를 분리했을 때의 확장 이점 없이 transport hop만 추가되므로, 단일 머신 구성은 검증용입니다. disaggregation의 처리량 이점은 부하 상태에서 prefill과 decode pool을 분리하는 데 있습니다.

### 혼합 full/sliding cache 계열

Ministral 3, ExaOne 4, Mellum, Laguna, Step 3.5는 시퀀스마다 모델이 소유하는 별도의 cache를 유지합니다. 이들의 단일 토큰 decode는 행들에 걸쳐 projection, norm, feed-forward 블록과 LM head를 공유하며, attention은 각 행 자신의 KV 이력과 위치를 사용합니다. Step 3.7 VLM은 text decode와 embedding-prefill 시퀀스 소유권을 Step 3.5에 위임합니다. 이 계열들은 더 이상 `--parallel`을 1로 제한하지 않습니다. rotating cache가 pad trimming을 구현하지 않으므로 padded prefill은 비활성화합니다. Laguna의 독립적인 DFlash cache는 변경되지 않았습니다.
