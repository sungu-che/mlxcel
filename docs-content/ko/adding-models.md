# 모델 추가 가이드

이 가이드는 `mlxcel`에 새로운 텍스트 모델과 VLM을 추가하기 위한 실무 체크리스트입니다.
일관성을 유지해야 하는 구체적인 제어 지점(control surface)을 안내합니다. 이 저장소에 이후 유지관리자 워크플로 문서가 추가되면, 이 체크리스트를 해당 문서와 일관되게 유지하십시오.

## 목표

- 새 모델 추가를 예측 가능하게 유지합니다.
- 일회성 분기를 새로 추가하는 대신 기존 control-plane 헬퍼를 재사용합니다.
- 회귀가 가장 쉽게 발생하는 통합 지점 옆에 테스트를 추가합니다.
- `mlx-lm` / `mlx-vlm`은 유용한 참조일 뿐, 포팅의 유일하게 허용되는 소스로 취급하지 않습니다.

## 시작하기 전에

1. 포팅에 사용할 구현 소스를 식별합니다. MLX 참조 구현일 수도 있지만 반드시 그럴 필요는 없습니다.
   - 해당 패밀리가 이미 존재하는 경우 `mlx-lm` 텍스트 모델 구현.
   - 해당 패밀리가 이미 존재하는 경우 `mlx-vlm` 구현.
   - config 필드, 텐서 이름, 모듈 레이아웃, forward 의미론, 프로세서 동작에 대해 가장 명확한 기준(source of truth)인 경우 Hugging Face Transformers 또는 공식 PyTorch 구현.
   - 프로덕션 추론 동작이 유용한 참조인 경우 vLLM 또는 SGLang 구현. 특히 KV cache 레이아웃, paged-attention 가정, MoE 라우팅, rope 스케일링, speculative 경로, 멀티모달 요청 준비에 해당합니다.
   - 아키텍처 특이점에 대한 유일한 공개 출처인 경우 벤더 모델 저장소, 모델 카드, 변환 스크립트, 공개된 추론 예제.
   - 완전한 참조 구현이 없는 경우. 체크포인트, `config.json`, 토크나이저/프로세서 파일, 논문, 또는 부분적인 벤더 노트만 있는 경우입니다. 허용되지만, 더 엄격한 검증 체크포인트를 갖는 재구성(reconstruction) 작업으로 취급해야 합니다.
   - 로컬 `references/` 체크아웃이 없다면 관련 업스트림 저장소를 별도로 clone하거나 확인하십시오. 해당 저장소를 이 트리에 vendoring하지 마십시오.
2. 아키텍처가 다음 중 무엇인지 결정합니다.
   - 완전히 새로운 모델 패밀리
   - 기존 패밀리의 포맷 별칭(alias)
   - 기존 텍스트 모델을 감싸는 VLM 래퍼
3. 기존 로더 헬퍼가 새 모델과 이미 일치하는지 확인합니다.
   - `src/model_metadata.rs`
   - `src/loading/mod.rs`
   - `src/loading/vlm.rs`
   - `src/models/mod.rs`
   - `src/model_metadata.rs`의 통합된 등록 지점에서 시작하십시오. 표준 텍스트 모델은 먼저 해당 등록 테이블을 확장해야 합니다. `src/loading/config_backed.rs`가 이제 동일한 기준 소스를 사용하기 때문입니다.
4. 변경이 공유 실행 정책도 건드리는지 확인합니다.
   - 디바이스/환경 동작은 `src/execution/runtime.rs`
   - 사용자 대면 샘플링 기본값 및 greedy 대 sampled 조립은 `src/execution/sampling.rs`

## 참조 구현 선택과 활용

목표는 Python 파일 하나를 Rust로 기계적으로 번역하는 것이 아닙니다. 목표는 `mlxcel`이 구현해야 하는 모델 계약(contract)을 식별하는 것입니다. 즉 config 정규화, 텐서 이름, 그래프 토폴로지, 캐시 의미론, 프롬프트 또는 이미지 준비, 생성 동작입니다.

답하려는 질문에 가장 가까운 참조를 선호하십시오.

- **MLX 참조(`mlx-lm`, `mlx-vlm`)**는 이미 해당 모델을 지원하는 경우 보통 가장 빠른 경로입니다. 체크포인트 로딩, 양자화 규약, MLX 텐서 동작이 `mlxcel`과 거의 일치하는 경향이 있기 때문입니다.
- **Hugging Face Transformers 또는 공식 PyTorch 코드**는 흔히 아키텍처의 기준 소스입니다. 모듈 shape, config 필드 이름, 활성화 순서, 정규화 위치, rotary embedding 동작, tied embedding, 프로세서/토크나이저 규약을 확인하는 데 사용하십시오.
- **vLLM과 SGLang**은 유용한 프로덕션 추론 참조입니다. KV cache shape과 업데이트 정책, paged attention 가정, MoE 전문가(expert) 라우팅, prefix caching, speculative decode 동작, 멀티모달 배칭 제약 같은 서빙 시점 세부 사항을 이해하는 데 사용하십시오. 이러한 아이디어는 스케줄러나 CUDA 전용 구조를 직접 가져오지 말고 기존 `mlxcel` 캐시/런타임 추상화에 매핑하십시오.
- **기타 PyTorch 추론 엔진 또는 벤더 예제**는 모델 고유의 특이점에 대한 최선의 출처일 수 있으며, 특히 모델이 아직 MLX나 Transformers에 반영되지 않은 경우에 그렇습니다. 검증에 사용한 정확한 커밋 또는 릴리스를 PR 설명이나 벤치마크 노트에 기록하십시오.

참조들이 서로 다를 때는, 추가하는 체크포인트에 대해 어떤 동작이 권위 있는지 기록하십시오. 예를 들어 모델 카드는 chat template을 문서화하고, Transformers는 텐서/모듈 계약을 정의하며, vLLM은 서빙 시점 캐시 레이아웃을 보여 줄 수 있습니다. 포팅하는 동안 이러한 책임을 분리해서 유지하십시오.

### MoE 포트를 참조와 diff할 때는 `MLXCEL_FUSED_MOE=0`을 설정하십시오

융합된 단일 토큰 decode-MoE 커널(#268)은 기본적으로 켜져 있고 `l == 1`에서만 작동하므로, prefill 비교는 영향을 받지 않지만 greedy decode 비교는 영향을 받습니다. 이는 결함이 아닙니다. 모두 f32로 수행한 dequantize-and-matmul 기준값과 비교하면 Klear와 `qwen3-30b-a3b` 모두에서 `gather_qmm`보다 대략 6배 더 가깝습니다(#1045). 하지만 `gather_qmm`은 mlx-lm이 그대로 따르는 경로이므로, 커널을 켠 채 mlx-lm과 diff하면 비트 단위로 일치하도록 의도된 적이 없는 두 경로를 비교하게 됩니다.

차이가 greedy argmax를 뒤집는지는 체크포인트에 따라 다릅니다. `qwen3-30b-a3b`는 어느 쪽이든 바이트 단위로 동일하지만 Klear는 그렇지 않습니다. prefill에서는 정확해 보이지만 decode에서 벗어나는 포트는 포팅 버그가 아니라 이것에 해당할 가능성이 매우 높으므로, 커널을 끄고 다시 실행하여 먼저 이를 배제하십시오.

참조 코드의 경계를 맹목적으로 복사하지 마십시오.

- 라우트 선택은 `src/model_metadata.rs`와 `src/loading/`에 두십시오.
- 프롬프트, 미디어, 프로세서 정책은 기존 멀티모달 헬퍼에 두십시오.
- 서빙 동작은 공유 실행/서버 레이어에 두십시오.
- 참조가 다른 파일 레이아웃을 사용하더라도 `mlxcel`의 명명 및 테스트 규약을 유지하십시오.

## 완전한 참조가 없는 경우

일부 모델 추가는 동작하는 추론 구현이 아니라 체크포인트와 메타데이터에서 시작합니다. 이 경우 첫 PR은 폭넓은 패밀리 포트가 아니라 보수적인 로더/런타임 재구성으로 하십시오.

사용 가능한 모든 산출물을 부분적인 참조로 활용하십시오.

- `config.json`, `generation_config.json`, 토크나이저 파일, 프로세서 파일, chat template.
- SafeTensors 키 이름, 텐서 shape, 양자화 메타데이터, tied-weight 관계.
- 모델 카드 노트, 아키텍처 다이어그램, 논문 수식, 공개 예제, 변환 스크립트.
- `mlxcel`, `mlx-lm`, Transformers, vLLM, SGLang, 또는 다른 PyTorch 추론 엔진에서 가장 가까운 기존 패밀리.

권장 워크플로:

1. 먼저 체크포인트를 검사합니다. 모델 코드를 작성하기 전에 텐서 이름과 shape 목록을 만들고, 가장 가까운 기존 패밀리와 비교합니다.
2. 최소 실행 가능 경로를 식별합니다. VLM보다 텍스트 전용을, tensor/pipeline 병렬보다 단일 디바이스를, 고급 샘플링 동작보다 greedy decode를 먼저 합니다.
3. 추론한 모든 기본값에 대해 명시적인 config 정규화를 추가합니다. 추측한 기본값을 모델 생성자 안에 숨기지 마십시오.
4. 지원되지 않는 변형은 실제 체크포인트로 검증되기 전까지 감지(detection)에서 제외합니다.
5. 수치적 일치가 가능해지기 전이라도 shape/config 테스트를 추가합니다.
6. 실제 스모크 테스트를 실행하고 프롬프트, 생성된 토큰 수, 알려진 제한 사항을 PR 또는 벤치마크 노트에 기록합니다.

참조가 없으면 검증 기대치가 달라집니다. 처음에는 정확한 일치가 불가능할 수 있지만, 구현은 여전히 다음을 증명해야 합니다.

- 필요한 모든 텐서가 사용되거나 의도적으로 무시됨
- 텐서 shape이 재구성된 그래프와 일치함
- 캐시 업데이트가 prefill과 decode에 걸쳐 올바르게 진행됨
- 최소 하나의 실제 체크포인트에서 생성이 안정적임
- 지원되지 않는 config에 대해 잘못된 패밀리로 조용히 빠지지 않고 명시적으로 실패함

나중에 참조 구현이 나타나면, 해당 구현과의 후속 비교를 추가하고 그에 맞춰 테스트나 벤치마크 노트를 더 엄격하게 하십시오.

### 대형 VLM 베이스라인

대형 VLM 포트는 지원됨으로 노출되기 전에 명시적인 운영 문서가 필요합니다. 로더/런타임 테스트 외에도, 다음 사항으로 `docs/supported-models.md`, `mlxcel arch`, 그리고 런타임 기능을 요약하는 모든 최상위 도움말 텍스트를 업데이트하십시오.

- 체크포인트 식별 정보: 저장소, 고정된 리비전, 해당하는 경우 로컬 픽스처/체크포인트 경로, dtype, 대략적인 가중치 크기
- 메모리 기대치: 현실적으로 가장 작은 하드웨어 등급과 검증 중 다른 대형 작업을 직렬화해야 하는지 여부
- 컨텍스트 및 캐시 의미론: 실제 최대 컨텍스트와, 패밀리가 표준 growing KV, rotating/sliding KV, 또는 모델 소유의 혼합 캐시 중 무엇을 사용하는지
- 멀티모달 제한: 이미지/비디오/오디오 지원, 시각 토큰 상한, placeholder 확장 규칙, 프롬프트 토큰 계산
- 서빙 표면: CLI, OpenAI Chat Completions, Responses, Anthropic 호환성, 스트리밍, 배칭, tool-call 포맷
- 생성 기본값: `generation_config.json`에서 읽은 EOS id와 샘플링 기본값
- 지원되지 않는 경로: 양자화, speculative/DFlash, 어댑터, TP, PP, XLA, 분산/분리형 서빙, 그리고 의도적으로 거부되는 모든 모달리티

Muse Glimmer가 이 규칙의 현재 예시입니다. 첫 번째 베이스라인은 `meta-models/Muse-Glimmer-30B` 리비전 `97c77dff50b2797bcc558fa2d909761dbc575c59`, 약 59.55 GB의 dense BF16 가중치를 대상으로 하며, 양자화 계약은 `mlx-community/Muse-Glimmer-30B-4bit` 리비전 `3e7677d7a40d348a3daba263a2b1c0aa41910710`, dense 비전 타워를 가진 MLX affine-Q4 텍스트 및 fusion 가중치, 약 19.41 GB의 텐서를 대상으로 합니다. 둘 다 131072 컨텍스트, 2048 토큰 sliding 레이어와 growing full 레이어, 이미지당 4096 시각 토큰, ATEM tool call, 그리고 기본값이 `high`인 `reasoning_strength` `low`/`medium`/`high`/`xhigh`를 제공합니다. 하드웨어 게이트가 실제 체크포인트에 대한 로드, 메모리, 처리량, 텍스트/이미지/멀티 이미지, tool-call, 긴 컨텍스트, 스케줄러 증거를 기록하기 전까지는 대형 VLM을 실제 체크포인트 검증 완료로 표시하지 마십시오.

## 모델 이슈에 라벨 붙이기

모델 지원 이슈에는 일반적인 `type:` / `priority:` / `area:` 세트 외에 서로 직교하는 세 가지 라벨이 붙습니다. "우리가 무엇을 지원하고 여기서 실제로 무엇을 테스트할 수 있는가?"를 고고학적 탐색이 아니라 라벨 쿼리로 만들기 위한 것입니다.

**`modelsize:`**는 **공개된 가장 작은 체크포인트의 가장 낮은 공개 양자화**를 디스크 크기 기준으로 측정하며, MLX에서는 이것이 상주 메모리에 가깝습니다. 머신이 아니라 모델을 설명하므로 개발 하드웨어가 바뀌어도 다시 검토할 필요가 없습니다.

| 라벨 | 크기 | 의미 |
|-------|------|---------|
| `modelsize:small` | ≤ 10 GB | 빠른 반복 가능, 스모크 테스트에 사용하기 안전 |
| `modelsize:medium` | 10 ~ 50 GB | 128 GB 머신에서 여유 있음 |
| `modelsize:large` | 50 ~ 100 GB | 실행은 되지만 머신을 지배함, 다른 작업은 직렬화 |
| `modelsize:xlarge` | > 100 GB | 128 GB 머신을 초과함, 더 큰 하드웨어 필요 |

`modelsize:large`는 의도적으로 차단 요소가 아닙니다. DBRX는 70 GB이며 128 GB 개발 머신에서 토큰 단위로 정확하게 검증되었습니다. 모델을 사용 가능한 하드웨어에서 정말로 검증할 수 없다면 `status:blocked`로 이를 표시하고 이슈 본문에 이유를 기록하십시오. 두 가지를 분리해 두면, 모든 모델의 라벨을 다시 붙이지 않고도 하드웨어 업그레이드 시 `status:blocked`를 제거하는 것만으로 작업이 다시 열립니다.

**`modeltype:`**은 모달리티입니다: `text`, `vlm`, `audio`, `omni`.

**`arch:`**는 구조적 패밀리이며, 포팅 노력과 코드 재사용을 실제로 예측해 주는 것입니다.

| 라벨 | 해당 범위 |
|-------|--------|
| `arch:dense` | Dense transformer 디코더 |
| `arch:moe` | 희소 mixture-of-experts 디코더 |
| `arch:hybrid` | 혼합 어텐션 스택: linear/sliding/full 교차 배치, 또는 attention + SSM |
| `arch:ssm` | 상태 공간 또는 순환(recurrent) 모델(Mamba, RWKV) |

모델이 둘 다에 해당하면 `arch:hybrid`가 `arch:moe`보다 우선합니다. 하이브리드 캐시가 포팅에서 더 어려운 절반이기 때문입니다. 이는 단순한 `LanguageModel` 캐시 경로가 아니라 `ModelOwnedSequenceState<Cache>` 경로를 강제합니다. AFMoE와 MiMo v2 Flash는 정확히 이 이유로 `arch:hybrid` 라벨이 붙은 MoE 모델입니다.

## 텍스트 모델 체크리스트

1. `src/models/` 아래에 구현 파일을 추가합니다.
2. `src/models/mod.rs`에 모듈을 등록하고 re-export합니다.
3. `src/models/mod.rs`에 `ModelType` variant를 추가합니다.
4. `src/models/detection.rs`의 `get_model_type()`을 확장합니다.
   - 새 모델이 기존 패턴에 맞는다면 `detect_text_or_vlm()`과 `detect_hunyuan_model_type()` 같은 공유 헬퍼를 선호하십시오.
5. `src/loaded_model.rs`에 해당하는 `LoadedModel` variant를 추가합니다.
   - 반복되는 match 테이블을 새로 추가하는 대신 기존 디스패치 헬퍼를 확장하는 것을 선호하십시오:
     `src/loaded_model.rs`의 `delegate_language_model!`과 `src/loaded_model_capabilities.rs`의 `VlmRuntimeRef`
6. `src/loading/mod.rs`에 로딩을 연결합니다.
   - `load_pair_from_dir()`과 `load_owned_model_from_config!` 같은 기존 헬퍼를 선호하십시오.
   - 라우터를 건드리기 전에 `src/model_metadata.rs`를 업데이트하여 종류, 어댑터 지원, 라우트 선택, 표준 config 기반 등록이 중앙에 유지되도록 하십시오.
   - 모델이 표준 텍스트 모델 경로를 따른다면, `src/loading/config_backed.rs`에 병렬 항목 목록을 추가하는 대신 `src/model_metadata.rs`의 공유 등록 지점을 확장하십시오.
7. LoRA/어댑터가 지원된다면 `src/loading/mod.rs`의 `load_model_from_weights()`를 확인합니다.
   - 비표준 어댑터 경로는 `load_model_from_weights()`를 직접 키우는 대신 `src/loading/special.rs`를 확장해야 합니다.
8. 패밀리가 `switch_layers::SwitchLinear`를 사용하지 않고 **자체 양자화 MoE 전문가 타입**을 가진다면, 로더가 이를 저장하는 시점에서 선언된 양자화 쌍의 범위를 제한하십시오:
   `switch_layers::validate_expert_quantization_params(prefix, group_size, bits)?`.
   공유 로더는 이를 대신 수행해 주지만, 패밀리 로컬 `SwitchLinear` / `SwitchGLU` / `ExpertLinear`는 이를 상속하지 않습니다. [양자화 파라미터 범위](#quantization-parameter-bounds)를 참조하십시오.

## VLM 체크리스트

1. `src/vision/encoders/` 아래의 비전 인코더를 구현하거나 재사용합니다.
2. `src/vision/connectors/` 아래의 커넥터를 구현하거나 재사용합니다.
3. `src/vision/processors/` 아래의 프로세서를 구현하거나 재사용합니다.
4. `src/models/detection.rs`에 VLM `ModelType` 감지를 추가합니다.
   - 기본 텍스트 패밀리에 텍스트 전용 변형과 VLM 변형이 모두 있다면 `detect_text_or_vlm()`을 선호하십시오.
5. `src/loading/vlm.rs` 또는 `src/loading/` 아래의 해당 패밀리 모듈에 로더 항목을 추가합니다.
   - config 파싱, 토큰 기본값, 가중치 리매핑에는 공유 헬퍼를 선호하십시오.
   - 패밀리 고유의 조립은 가장 가까운 동료와 함께 묶어 두십시오:
     `src/loading/vlm_qwen.rs`, `src/loading/vlm_llava.rs`, `src/loading/vlm_gemma.rs`,
     `src/loading/vlm_pixtral.rs`, `src/loading/vlm_siglip.rs`, `src/loading/vlm_special.rs`.
   - 라우터가 해당 패밀리가 멀티모달임을 알고 어댑터 로딩 정책이 명시적으로 유지되도록 `src/model_metadata.rs`를 업데이트합니다.
   - `load_model()`이 얇은 디스패처로 유지되도록 `src/loading/mod.rs`의 `try_load_vlm_model_from_dir()`에 디렉터리 진입점을 등록합니다.
6. `src/loaded_model_capabilities.rs`에 `LoadedModel` 기능 헬퍼를 추가하거나 재사용합니다.
   - CLI/서버에서만 쓰이는 패밀리 고유 getter를 추가하는 대신 `VlmRuntimeRef`나 기존 멀티모달 헬퍼를 확장하는 것을 선호하십시오.
7. 가능하면 프롬프트 헬퍼를 재사용합니다.
   - Qwen-VL 토큰 삽입: `src/multimodal/qwen_vl.rs`
   - 일반 이미지 토큰 블록 확장: `src/multimodal/vlm_prompt.rs`
   - Phi3V 프롬프트 태그 처리: `src/multimodal/phi3v_prompt.rs`

## 새 패밀리 모듈을 만들어야 하는 경우

기본적으로 새 파일을 만들지 마십시오. 패밀리가 뚜렷한 control-plane 정체성을 가질 때 만드십시오.

다음의 경우 새 로더 패밀리 모듈을 만드십시오.

- config 정규화가 기존 패밀리의 작은 변형이 아닐 때
- 토큰 기본값이나 가중치 키 리매핑에 전용 테스트가 필요할 때
- VLM 래퍼가 다른 프롬프트/런타임 조립 경로를 사용할 때
- 로직을 인라인으로 추가하면 기존 패밀리 경계가 흐려질 때

다음의 경우 모델을 기존 모듈에 유지하십시오.

- 주로 별칭이거나 작은 config 차이일 때
- 동일한 로더 테스트가 이미 정책을 표현하고 있을 때
- 변경 후에도 패밀리를 여전히 알아볼 수 있을 때

확실하지 않다면 먼저 기존 패밀리 모듈을 확장하고, 테스트 파일이나 라우터가 명확한 경계를 잃기 시작할 때만 분리하십시오.

## 양자화 파라미터 범위

`config.json`은 신뢰할 수 없는 입력입니다. 다운로드한 HuggingFace 저장소와 함께 도착하기 때문입니다. 어떤 텐서 레이아웃으로도 기술할 수 없는 `group_size` 또는 `bits`는 Rust 측의 모든 검사를 통과한 뒤 문서화되지 않은 MLX 전제 조건을 위반합니다. MLX는 양자화된 행렬의 언패킹된 폭을 `w.shape(-1) * 32 / bits`로 재구성하므로, 선언된 `"bits": 0`은 0으로 나누기가 되고 32를 넘는 값은 몫을 붕괴시킵니다. `gather_qmm`, `quantized_matmul`, `quantized_embedding`, `dequantize`는 모두 `Result`가 아니라 `UniquePtr<MlxArray>`로 cxx 브리지를 건너기 때문에, 그 결과로 발생하는 C++ throw는 로드 오류가 아니라 **첫 번째 forward pass**에서의 포착할 수 없는 `std::terminate`이며 `catch_unwind`로도 막을 수 없습니다([`docs/adr/0003-release-panic-unwind-with-core-thread-abort.md`](adr/0003-release-panic-unwind-with-core-thread-abort.md) 참조). `mlxcel-server`에서는 이것이 모델을 로드하는 것만으로 유발되는 원격 서비스 거부(DoS)입니다.

따라서 규칙은 다음과 같습니다. **쌍(pair)이 저장되는 모든 곳에서, 그로부터 파생된 어떤 것이 유지되기 전에 범위를 제한하십시오.** 모델의 로드 경계에서가 아닙니다. 경계는 모든 경로를 지배하지 않습니다. 파이프라인 스테이지 실행기, `*StageModel::from_filtered_weights` 진입점, VLM 텍스트 래퍼, config 브리징 헬퍼는 모두 패밀리 자체의 모델 생성자를 한 번도 호출하지 않고 레이어를 만들며, 여러 패밀리는 다른 모듈에서 양자화 변형을 단순 struct 리터럴로 만들기 때문입니다.

이미 범위 제한을 갖추고 있어 공짜로 상속받는 것:

| 로더 | 포함 범위 |
|--------|--------|
| `reconcile_quantization_layout` | 모든 `UnifiedLinear` / `UnifiedEmbedding` |
| `SwitchLinear::from_stacked_parts` | 공유 `switch_layers` 로더를 통한 MoE 전문가 |
| `QuantizedMultiLinear::{new, from_weights}` | MLA `embed_q` / `unembed_out` |
| `FusedQKVLinear::from_weights_separate_with_mode` | 융합 QKV 프로젝션 |
| `infer_mla_quantization_params` | `sanitize_weights`에서의 MLA `kv_b_proj` 분해 |

그 목록에는 이제 직접 구성한 `QuantizedEmbedding` 생성자가 의도적으로 없습니다. Mamba / Mamba2를 위한 것이 하나 있었는데, 이들은 테이블을 두 가지 가능한 접두사 아래에서 해석하므로 단일 접두사 맵 로더로는 처리할 수 없는 것처럼 보였습니다. 이 생성자는 `mode: "affine"`을 하드코딩하고 `biases` 인자를 요구했으며, 바로 그 때문에 두 패밀리가 block-float 임베딩(`.scales`는 있고 `.biases`는 없음)을 비양자화로 취급하게 되었습니다(이슈 #976). 체크포인트가 임베딩 접두사를 둘 이상의 방식으로 표기한다면 `contains_key` 프로브로 접두사를 결정하여 `UnifiedEmbedding::from_weights`에 전달하십시오. 레이어를 직접 구성하지 마십시오.

`QuantizedMultiLinear::new`는 여전히 그 목록에 있으며, 두 생성자가 비슷해 보였기 때문에 그 차이를 정확히 짚어 둘 가치가 있습니다. 임베딩 생성자는 결함을 강제하는 시그니처 때문에 제거되었습니다. `biases` 인자를 요구하면 block-float 호출자가 자신의 체크포인트를 기술할 수 없었습니다. MLA 생성자는 `Option<biases>`를 받고 로더가 사용하는 것과 동일한 `infer_quantization_mode` 호출로 모드를 도출하므로(이슈 #1028), 로더가 기술할 수 있는 모든 plane 레이아웃을 기술할 수 있고 그것들과 모순되는 모드를 저장할 수 없습니다. 두 진입점은 또한 선언된 `group_size` / `bits` 쌍의 범위를 제한하는데, 이는 선언된 쌍을 텐서 shape에 대해 조정(reconcile)하지 않고 그대로 저장하는 유일한 공유 양자화 로더이므로 다른 곳보다 여기서 더 중요합니다.

직접 해야 하는 것:

- **패밀리 로컬 양자화 전문가 타입**: 양자화 변형을 만드는 분기에서 `switch_layers::validate_expert_quantization_params(prefix, group_size, bits)?`를 호출하십시오. 해당 분기에만 게이트를 두십시오. bf16 전문가 plane은 패킹이 없으며 선언된 어떤 쌍에서도 로드 가능해야 합니다.
- **접두사별 양자화 블록**(gpt-oss 스타일, 하나의 triple이 아니라 override의 맵): config 검증 중에도 모든 항목을 순회하십시오. 최상위 기본값에 대한 범위 제한은 개별 override에 대해 아무것도 말해 주지 않으며 그 반대도 마찬가지이기 때문입니다.
- 쌍을 저장하고 MLX 양자화 연산에 넘기는 모든 **새로운 수작업 생성자**: 호출자를 신뢰하는 대신 실패 가능하게(fallible) 만들고 범위를 제한하십시오.

악의적인 값을 `validate_quantization_params`를 직접 거치는 대신 실제 로더를 통해 구동하는 회귀 테스트를 추가하고, 모든 악의적 케이스에 긍정 대조군(positive control)을 짝지어 가드가 양자화된 모든 것을 거부함으로써 통과하는 일이 없도록 하십시오. `switch_layers::insert_stacked_quantized_expert_plane`과 `switch_layers::HOSTILE_QUANT_PARAMS`가 공유 테스트 픽스처입니다. 형태는 아무 `src/models/*_tests.rs` 가드 테스트를 보십시오.

## 회귀가 주로 발생하는 곳

- `src/models/mod.rs`
  - 모듈 export 또는 `ModelType` variant 누락
- `src/models/detection.rs`
  - `get_model_type()`의 별칭 누락
  - `vision_config`가 있을 때의 텍스트/VLM 오분류
- `src/loading/mod.rs`
  - `load_model()`과 `load_model_from_weights()` 사이의 불일치
  - 표준 config 기반 모델을 공유 로더 헬퍼 대신 일회성 특수 케이스로 추가
  - VLM을 `try_load_vlm_model_from_dir()` 대신 `load_model()`에 직접 추가
- `src/model_metadata.rs`
  - 로더를 연결하기 전에 텍스트/VLM 종류, 어댑터 지원, 라우트 정책, 또는
    공유 표준 텍스트 등록 항목 업데이트를 잊음
- `src/loading/config_backed.rs`
  - 공유 등록 지점을 우회하고 새로운 일회성 로더 로직을 추가
  - `Llama4`, `Gemma3`, `Ministral3` 같은 모델의 래퍼 생성자를 잊음
- `src/loading/nonstandard.rs`
  - 디렉터리 전용 로더 패밀리를 비표준 레지스트리 대신 `src/loading/mod.rs`에 남겨 둠
- `src/loading/special.rs`
  - 어댑터/owned-weight 특수 처리를 special-weight 레지스트리 대신 인라인으로 추가
  - 생성 전에 Qwen3.5 텍스트 config 정규화 또는 owned-weight sanitization을 잊음

`src/loading/mod.rs`는 라우트 선택에 집중하도록 유지하십시오. 새 모델 패밀리가 상당한 생성 로직을 추가한다면, `load_model()`이나 `load_model_from_weights()`를 인라인으로 키우는 대신 전용 형제 모듈을 두고 라우터에서 호출하는 것을 선호하십시오.

매우 큰 모델 패밀리의 경우, 코드가 메인 디코더 스택과 다른 이유로 변경된다면 내부 헬퍼 핫스팟을 집중된 형제 헬퍼 모듈로 추출하십시오. 현재 예시:

- `src/models/gemma3n_helpers.rs`
- `src/models/llama4_helpers.rs`
- `src/loading/vlm.rs`와 `src/loading/` 아래의 형제 패밀리 모듈
  - 잘못된 기본 토큰 ID
  - 최상위 양자화 상속 누락
  - 텍스트 타워와 비전 타워 사이의 잘못된 가중치 키 리매핑
- `src/loaded_model.rs` / `src/loaded_model_capabilities.rs`
  - 새 variant에 대한 디스패치 arm 누락
  - `VlmRuntimeRef`의 기능 연결 누락
  - 전체 모델 디스패치 매크로는 업데이트하고 멀티모달
    기능 스위치보드는 잊음

## 테스트 기대치

모델/control-plane 변경과 같은 슬라이스에서 테스트를 추가하십시오.

- 모델 감지 헬퍼의 경우:
  - `src/models/detection.rs` 근처에 단위 테스트를 추가합니다
- sanitization 헬퍼의 경우:
  - `src/models/sanitize.rs` 근처에 단위 테스트를 추가합니다
- 로더 정규화 또는 토큰 기본값 로직의 경우:
  - `src/loading/tests.rs` 또는 관련 `src/loading/vlm*_tests.rs` 근처에 테스트를 추가합니다
- 공유 비전 merge 계약의 경우:
  - `src/vision/merge_tests.rs` 근처에 테스트를 추가합니다
- 프롬프트/토큰 확장 로직의 경우:
  - `src/multimodal/qwen_vl_tests.rs`, `src/multimodal/vlm_prompt_tests.rs`, `src/multimodal/phi3v_prompt_tests.rs` 같은 전용 헬퍼 테스트 파일에 테스트를 추가합니다
- 런타임 검증의 경우:
  - `scripts/run_quality_gate.sh`를 실행합니다
  - 일치하는 모델이 있을 때는 최소 하나의 로컬 스모크 테스트를 추가합니다
  - 슬라이스가 MLX 비중이 큰 ignore 처리된 헬퍼 테스트를 건드렸다면
    `--ignored --test-threads=1`로 명시적으로 실행합니다

## 공유 함수 정책

여러 모델 패밀리가 사용하는 공유 함수를 건드릴 때는, 특히 다음 아래에 있는 공유 헬퍼 파일의 로컬 사용 주석을 업데이트하십시오.

- `src/lib/mlxcel-core/src/layers.rs`
- `src/lib/mlxcel-core/src/utils.rs`

이 주석들은 향후 변경 시 재테스트 목록으로 기능합니다.

## 공통 실행 헬퍼

동작이 CLI, 서버, 향후 프런트엔드 전반에서 동일해야 할 때는 진입점 정책을 공유 실행 레이어에 두십시오.

- `src/execution/runtime.rs`
  - 환경 변수 기반 디바이스 선택(`MLXCEL_DEVICE`)
  - GPU wired-memory 한도 설정
- `src/execution/sampling.rs`
  - 확정된 요청 기본값으로부터 중앙화된 `SamplingConfig` 조립
  - 공유 greedy 대 non-greedy 분기

새 프런트엔드나 요청 타입이 다른 기본값을 필요로 한다면, 그 기본값은 경계(edge)에서 결정하고 최종 변환은 `src/execution/`에 두십시오.

CLI 전용 프롬프트 포맷팅과 터미널 출력 동작은 공유 로딩이나 서버 모듈로 옮기지 말고 `src/commands/generate.rs`에 두십시오.

서버 전용 부팅 동작의 경우, 시작 정책은 `src/server/mod.rs`를 키우지 말고 `src/server/startup.rs`에 두십시오.

- API 키 / chat-template 결정 우선순위
- CLI 호환 플래그의 시작 시 정규화
- 워밍업 동작
- Unix 소켓 대 TCP 바인딩

공유 서버 타입도 집중된 모듈에 두십시오.

- 요청/기본 설정 구조체는 `src/server/config.rs`
- `AppState`와 메트릭 컨테이너는 `src/server/state.rs`
- 공개 요청/응답 채널 API는 `src/server/model_provider.rs`
- 장기 실행 워커 스레드, VLM 요청 준비, 디코드 상태는 `src/server/model_worker.rs`

둘 이상의 엔드포인트가 동일한 동작을 필요로 하면 서버 edge 어댑터를 라우트 파일 밖에 두십시오.

- OpenAI chat 메시지 평탄화(flattening)와 프롬프트 폴백 조립은 `src/server/chat_request.rs`
- 요청 기본값을 `ServerGenerateOptions`로 병합하는 것은 `src/server/request_options.rs`
- `data:` / `file://` 이미지 소스 파싱은 `src/server/media.rs`
- 공유 SSE 채널과 `[DONE]` 방출 헬퍼는 `src/server/streaming.rs`

## 드라이런 워크플로 검증

이 섹션은 새 모델 지원을 추가할 때 현재 아키텍처가 실제로 모호함을 줄여 주는지 검증합니다.

### 드라이런 A: 표준 텍스트 모델

기존 config 기반 로딩 경로를 따르는 새 텍스트 모델을 가정합니다.

현재 필요한 표면:

1. `src/models/<family>.rs`
2. `src/models/mod.rs`
3. `src/models/detection.rs`
4. 통합된 등록 지점과 `static_model_descriptor()` / `model_load_policy()`를 통한
   `src/model_metadata.rs`
5. 공유 config 기반 로딩 동작 자체를 바꿔야 하는 경우에만 `src/loading/config_backed.rs`
6. `src/loaded_model.rs`
7. 패밀리가 멀티모달 기능 노출을 변경하는 경우에만 `src/loaded_model_capabilities.rs`
8. `src/models/detection_tests.rs`, `src/models/sanitize_tests.rs`, `src/loading/tests.rs` 근처의 테스트

일어나서는 안 되는 일:

- `load_model()` 안에 새로운 일회성 생성 분기를 두지 않음
- 사용자에게 보이는 동작이 바뀌지 않는 한 CLI나 서버를 직접 변경하지 않음
- 기존 기능으로 충분하다면 `LoadedModel`에 패밀리 고유 getter를 추가하지 않음

이전 경로보다 나은 이유:

- 라우트 선택이 여러 로딩 match에 걸쳐 중복되는 대신 중앙화됨
- 어댑터 지원이 하나의 정책 표면에 선언됨
- 표준 텍스트 생성자 등록이 더 이상 별도의 병렬 테이블에 존재하지 않음
- 예상되는 편집 목록이 코딩을 시작하기 전에 검토할 수 있을 만큼 짧음

### 드라이런 B: 새로운 VLM 패밀리

새 VLM 패밀리가 자체 토큰 기본값과 가중치 키 리매핑을 필요로 한다고 가정합니다.

현재 필요한 표면:

1. `src/models/` 또는 `src/vision/` 아래의 텍스트 모델 및/또는 VLM 래퍼
2. `src/models/mod.rs`
3. `src/models/detection.rs`
4. `src/loading/vlm_<family>.rs`
5. `src/loading/vlm.rs`
6. `static_model_descriptor()` / `model_load_policy()`를 통한 `src/model_metadata.rs`
7. enum 연결을 통한 `src/loaded_model.rs`
8. `VlmRuntimeRef`를 통한 `src/loaded_model_capabilities.rs`
9. 프롬프트/런타임 준비가 정말 새로운 경우에만 `src/multimodal/`
10. `src/loading/vlm_<family>_tests.rs` 근처의 테스트와 새로운 멀티모달 헬퍼 테스트 파일

일어나서는 안 되는 일:

- CLI나 서버 요청 경로에 구체적인 모델 타입 검사를 추가하지 않음
- 패밀리 고유 로딩 로직을 `src/loading/mod.rs`에 직접 추가하지 않음
- CLI와 서버에 걸쳐 프롬프트 재작성 로직을 중복하지 않음

이전 경로보다 나은 이유:

- 패밀리 라우터가 `src/loading/vlm.rs`에 있음
- 패밀리 조립이 동료 VLM 로더 옆에 유지됨
- 멀티모달 프런트엔드가 패밀리 이름이 아니라 기능(capabilities)에 의존함

### 실제 추가를 위한 리뷰 체크리스트

새 모델 또는 VLM 패밀리에 대한 PR을 열기 전에 다음을 확인하십시오.

1. 로딩 정책이 `src/model_metadata.rs`를 통해 업데이트되었는가
2. `LoadedModel` 연결이 새로운 일회성 패밀리 getter를 만들지 않고
   `src/loaded_model.rs`와 `src/loaded_model_capabilities.rs` 안에 머물렀는가
3. CLI와 서버가 구체적인 모델 타입이 아니라 여전히 공유 헬퍼에 의존하는가
4. 단위 테스트가 새로운 정책 또는 정규화 로직을 다루는가
5. 로컬 모델을 사용할 수 있을 때 최소 하나의 스모크 테스트가 존재하는가

## 커밋 전략

하나의 control-plane 표면을 분리하는 작은 체크포인트를 선호하십시오.

- 모델 감지
- 로더 정규화
- 프롬프트 준비
- 런타임 초기화

이렇게 하면 회귀를 검색하기 쉬워지고 향후 모델 추가를 이전 슬라이스와 비교하기가 쉬워집니다.
