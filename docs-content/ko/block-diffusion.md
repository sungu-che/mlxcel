# Block-diffusion 생성 (DiffusionGemma)

DiffusionGemma(`google/diffusiongemma-26B-A4B-it`)는 왼쪽에서 오른쪽으로 진행하는
autoregressive decoding 대신 block-diffusion을 사용하는 텍스트 생성 모델입니다. 이
페이지에서는 생성 메커니즘, 이를 제어하는 CLI 플래그, 측정된 처리량, 현재의 제한 사항을
설명합니다.

## 동작 방식

Autoregressive 모델은 이전의 모든 token을 조건으로 한 번에 하나의 token을 예측합니다.
DiffusionGemma는 반복적인 denoising을 통해 token의 *캔버스(canvas)*를 한 번에 decode합니다.
캔버스는 균일한 무작위 token id로 시작하며, 모델은 token이 안정적이고 신뢰도가 높은
예측에 도달할 때까지 매 step마다 전체 블록을 다듬습니다. 그 후 step 루프는 조기에
종료되거나 step 예산을 모두 소진합니다.

생성 루프는 다음과 같이 동작합니다. 먼저 프롬프트를 dense FP16 KV cache에 저장되는
읽기 전용 접두사(prefix)로 인코딩합니다. 그런 다음 각 출력 블록마다 다음을 수행합니다.

1. 무작위 token id(재현성을 위해 `--seed`로 시드 지정)로 캔버스를 초기화하며, 길이는
   `min(max_canvas, max(remaining, min_canvas))`입니다.
2. 최대 `--max-denoising-steps`번의 denoising step을 실행합니다. 각 step은 학습된
   self-conditioning 신호와 함께 현재 캔버스를 임베딩하고, 캔버스 내부에서는 양방향으로,
   접두사에 대해서는 인과적(causal)으로 attention을 수행하며, sampler를 사용해 새로운
   token 예측을 수락합니다.
3. 각 step 이후 stable-and-confident 조기 종료는 캔버스 내용이 `stability_threshold`
   연속 step 동안 변하지 않았는지 확인합니다. 그렇다면 블록을 확정하고 루프가 다음으로
   넘어갑니다.
4. 확정된 캔버스를 접두사에 이어 붙이고, 출력이 남아 있으면 다음 캔버스 블록을 시작합니다.

### Sampler

`--diffusion-sampler`를 통해 두 가지 step별 수락 sampler를 사용할 수 있습니다.

- **`entropy-bound`**(기본값): 각 캔버스 위치에 대해 token별 엔트로피를 계산합니다.
  위치를 엔트로피 오름차순으로 정렬합니다. 엔트로피의 누적 합이 `entropy_bound`
  (체크포인트의 `generation_config`에서 읽음)를 초과할 때까지 그 순서대로 위치를
  수락합니다. 최소 한 개의 위치는 항상 수락됩니다. 이는 mlx-vlm 레퍼런스의
  `(cumsum - cummax) <= bound` 규칙을 재현합니다.

- **`confidence-threshold`**: top-1 token 확률이 `--diffusion-threshold`(기본값 0.9)를
  초과하는 아직 공개되지 않은 모든 위치를 수락합니다. 조건을 만족하는 위치가 없으면,
  루프가 항상 진행되도록 신뢰도가 가장 높은 공개되지 않은 위치를 무조건 수락합니다.

## CLI 플래그

모든 diffusion 플래그는 `mlxcel generate --help` 출력의 `Diffusion Options` 제목 아래에
나타납니다. Autoregressive 모델은 이를 완전히 무시합니다.

| 플래그 | 기본값 | 비고 |
|------|---------|-------|
| `--max-denoising-steps N` | 체크포인트 `generation_config` (일반적으로 48) | 캔버스 블록당 최대 denoising 반복 횟수. 값을 낮추면 품질을 희생하는 대신 속도를 얻습니다. |
| `--diffusion-sampler entropy-bound\|confidence-threshold` | `entropy-bound` | step별 token 수락 전략. 위를 참고하십시오. |
| `--diffusion-threshold FLOAT` | `0.9` | `confidence-threshold` sampler의 신뢰도 임계값. |
| `--diffusion-min-canvas-length N` | `64` | 생성 꼬리 부분의 가장 짧은 캔버스. 아주 작은 마지막 블록을 방지합니다. |
| `--diffusion-max-canvas-length N` | 모델 `canvas_length` (일반적으로 256) | 블록당 캔버스 길이의 상한. |
| `--diffusion-full-canvas` | off | 블록마다 항상 모델의 전체 `canvas_length`를 할당합니다. |
| `--seed N` | 설정 안 함 (비결정적) | MLX의 전역 RNG를 시드하며, 이는 캔버스 초기화도 제어합니다. 동일한 시드와 temperature 0으로 두 번 실행하면 동일한 출력이 생성됩니다. |

## 처리량

step당 비용이 처리량을 좌우합니다. 모델이 빠르게 수렴하면 stable-and-confident 조기
종료가 더 일찍 작동하므로, tokens/sec는 프롬프트와 출력의 난이도에 따라 달라집니다.

`diffusiongemma-26B-A4B-it-4bit`로 M1 Ultra에서 측정한 결과:

- 짧은 답변은 약 20 tok/s (64-token 캔버스, 48 step 중 약 12 step에서 수렴)
- 256-token 캔버스에서 48 step 예산의 대부분을 필요로 하는 더 긴 출력은 약 8 tok/s

동일한 하드웨어에서의 step당 시간: 64-token 캔버스는 254 ms/step(261 ms/step인 mlx-vlm
Python 레퍼런스의 103%), 256-token 캔버스는 651 ms/step(626 ms/step인 레퍼런스의 96%)입니다.

모델 로드: 13.5 GiB 상주(4-bit quantization 체크포인트, M1 Ultra).

## 재현성

캔버스 노이즈가 비결정성의 주된 원인입니다. 생성이 시작되기 전에 MLX의 전역 RNG를
시드하려면 `--seed N`을 전달하십시오. 동일한 시드, 동일한 프롬프트, 동일한
`--max-tokens`로 두 번 실행하면 temperature 0에서 동일한 커밋 스트림이 생성됩니다. 시드는
autoregressive 샘플링 경로에도 영향을 주므로 두 모델 패밀리 모두에 유용합니다.

환경 변수 `MLXCEL_DIFFUSION_DEBUG_CANVAS=1`은 모든 무작위 캔버스 초기화를 고정된 결정적
패턴으로 대체합니다. 이는 Python 레퍼런스와의 구현 간 동등성(parity) 테스트를 위한
것이며, 출력은 일반적인 생성에는 유용하지 않습니다. 자세한 내용은
[환경 변수](environment-variables.md)를 참고하십시오.

## 이미지 입력

프롬프트에 이미지를 포함하려면 하나 이상의 `--image <path>` 플래그를 전달하십시오. 이 플래그는 여러 이미지 프롬프트를 위해 반복해서 사용할 수 있습니다.

```
mlxcel generate -m google/diffusiongemma-26B-A4B-it \
  --image photo.jpg \
  -p "Describe what you see." \
  --max-tokens 256 \
  --seed 42
```

전처리는 Gemma 4 비전 파이프라인을 재사용합니다. 이미지는 768x768로 크기가 조정되고 패딩된 후 이미지당 256개의 soft token으로 투영됩니다. 각 이미지 블록은 prefill 중에 양방향 attention을 받으며, 이는 Gemma 4 Unified가 사용하는 것과 동일한 오버레이입니다. 이미지 token은 프롬프트 시퀀스에서 `boi`와 `eoi` 센티널 token 사이에 삽입됩니다.

비디오 및 오디오 입력은 지원되지 않으며 명확한 오류 메시지와 함께 거부됩니다.

## 서버 모드

`mlxcel-server --model <diffusion-model>`(또는 `mlxcel serve -m <diffusion-model>`)은 체크포인트를 로드하고 `/v1/chat/completions` 및 `/v1/completions`를 제공합니다. 스트리밍(SSE)과 비스트리밍 응답을 모두 지원합니다. 요청은 순차적으로 처리됩니다. DiffusionGemma는 batch 스케줄러에 참여할 수 없으므로, 서버는 요청을 도착 순서대로 큐에 넣고 한 번에 하나씩 처리합니다. 대기 중인 요청 큐는 여전히 `--max-queue-depth`를 따르며, 가득 차면 batch 서빙과 동일한 overloaded 응답을 반환합니다. 이는 일시적인 제한이 아니라 의도된 설계입니다.

Diffusion 모델에 영향을 주는 serve 수준 플래그:

| 플래그 | 기본값 | 비고 |
|------|---------|-------|
| `--diffusion-sampler entropy-bound\|confidence-threshold` | `entropy-bound` | step별 token 수락 전략. |
| `--diffusion-threshold FLOAT` | `0.9` | 신뢰도 임계값. 시작 시 `[0, 1]` 범위로 검증됩니다. |
| `--max-denoising-steps N` | 체크포인트 기본값 (일반적으로 48) | 캔버스 블록당 최대 denoising 반복 횟수. |

이미지 입력은 표준 OpenAI `image_url` content part 형식을 따릅니다. 비디오나 오디오를 포함하는 요청은 명확한 오류와 함께 거부되며, 연결은 열린 상태로 유지되고 다음 요청은 정상적으로 처리됩니다.

### 운영상 주의 사항

이 체크포인트는 thinking-channel 모델입니다. 어시스턴트는 보이는 응답을 작성하기 전에 `reasoning_content` 채널을 채웁니다. 몇 가지 실질적인 결과는 다음과 같습니다.

- **요청에 충분한 `max_tokens`를 지정하십시오.** 예산이 64 token이면 간단한 질문도 reasoning만 생성하여 `content`가 비어 있고 `finish_reason: "length"`가 될 수 있습니다. 256의 예산이면 대부분의 짧은 프롬프트에서 전체 답변이 생성됩니다.
- **Reasoning은 두 경로 모두에서 노출됩니다.** 스트리밍 응답은 이를 `delta.reasoning_content`와 동일한 `delta.reasoning` 별칭으로 전달하며, 비스트리밍 `/v1/chat/completions` 응답은 어시스턴트 메시지에 두 필드를 모두 포함합니다. 모델이 reasoning을 생성하지 않으면 생략되며, `--reasoning-alias-field none`은 `reasoning` 중복 필드만 제거합니다.
- **`/v1/completions`는 동작하지만 raw 프롬프트는 품질이 저하됩니다.** 이 엔드포인트는 템플릿이 적용되지 않은 텍스트를 기계적으로 받아들이지만, 모델이 채팅 구조에 의존하므로 출력 품질이 크게 저하됩니다. `/v1/chat/completions`를 사용하는 것을 권장합니다.
- **취소의 분해능.** 취소된 요청은 즉시가 아니라 최대 한 번의 denoising step 이내에 중단됩니다.

## 남은 제한 사항

- **Tensor parallelism**: TP 플래너에는 이 패밀리를 위한 자리 표시자(placeholder) 분기가 있습니다.

## 사용 예시

```
mlxcel generate -m google/diffusiongemma-26B-A4B-it \
  -p "Why is the sky blue?" \
  --max-tokens 256 \
  --seed 42
```

모델 스토어에 있는 4-bit quantization 스냅샷의 경우:

```
mlxcel generate -m diffusiongemma-26B-A4B-it-4bit \
  -p "Explain TCP vs UDP in one paragraph." \
  --max-tokens 512 \
  --diffusion-sampler entropy-bound \
  --max-denoising-steps 32
```
