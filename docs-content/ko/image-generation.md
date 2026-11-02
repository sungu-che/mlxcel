# 이미지 생성

mlxcel은 `mlxcel generate-image`와 OpenAI 호환 서버 라우트 `POST /v1/images/generations`, `POST /v1/images/edits`를 통해 이미지를 생성하고 편집합니다. 이 페이지는 `src/imagegen/`의 모델 독립 런타임, 즉 요청 타입, 체크포인트 probe, 스케줄러, latent packing, FLUX.2 VAE, Z-Image DiT 블록, PNG 출력, CLI, 서버 라우트, 그리고 모델 이슈가 패밀리를 등록하는 방법을 다룹니다.

**현재 상태.** 등록된 패밀리: `flux2`([FLUX.2 Klein](#flux2-klein-flux2)), `z_image`([Z-Image](#z-image)), `bonsai`([Prism Bonsai](#prism-bonsai-bonsai)), `ernie_image`([ERNIE-Image](#ernie-image-ernie_image)), `mage_flow`([Mage-Flow](#mage-flow-mage_flow)), `qwen_image`([Qwen-Image](#qwen-image-qwen_image)), `ideogram4`([Ideogram 4](#ideogram-4-ideogram4)), `ming_image`([Ming-Image](#ming-image-ming_image)), `llada_image`([LLaDA-Image](#llada-image-llada_image)). 이미지 생성 에픽(`lablup/mlxcel#2348`)의 각 모델은 자신의 transformer와 텍스트 conditioner, 그리고 패밀리 표에 한 행을 추가합니다. probe가 인식하지만 `FAMILIES`에 없는 체크포인트는 `image generation family <family> is not supported by this build`로 실패하며, `mlxcel-server --image-model`도 설계상 시작 시점에 실패합니다([서버](#서버) 참고).

| 패밀리 | 체크포인트 | Task | 기본 step / guidance |
|--------|-----------|------|---------------------|
| `flux2` | FLUX.2 Klein 4B, 9B, base-4B, base-9B, 9b-kv(Diffusers 트리)와 mlx-community 4-bit 변환본 | generate, edit(참조 이미지) | distilled 4 / 1.0, base 50 / 4.0 |
| `z_image` | Tongyi-MAI Z-Image-Turbo, Z-Image(원본 Diffusers 트리 또는 mlx-vlm 변환본) | generate, edit(img2img) | Turbo 8 / 0, base 50 / 4 |
| `bonsai` | Prism Bonsai ternary 2-bit FLUX.2 Klein 4B(`prism-ml/bonsai-image-ternary-4B-mlx-2bit`) | generate | 4 / 1.0 |
| `ernie_image` | Baidu ERNIE-Image-Turbo, ERNIE-Image(원본 Diffusers 트리) | generate, edit(SDEdit img2img) | Turbo 8 / 1.0(edit 8 / 3.0), base 50 / 4.0 |
| `mage_flow` | Mage-Flow, Mage-Flow-Base, Mage-Flow-Turbo, Mage-Flow-Edit, Mage-Flow-Edit-Base, Mage-Flow-Edit-Turbo(mlx-vlm 변환본) | 변형에 따라 generate 또는 edit(참조 이미지 하나 이상) | Turbo 4 / 1.0, Mage-Flow 20 / 5.0, Base와 Edit 30 / 5.0 |
| `qwen_image` | Qwen-Image-2.1(`Qwen/Qwen-Image-2.1`, Diffusers 트리) | generate, edit(참조 이미지 하나 이상, RGBA 유지) | generate 30 / 1.0, edit 40 / 1.0 |
| `ideogram4` | `ideogram-ai/ideogram-4-fp8`(라이선스 게이트, 행별 FP8) | generate | `V4_DEFAULT_20`: 20 / 7.0(마지막 두 step은 3.0) |
| `ming_image` | Ming-Image-0.1-Design(`inclusionAI/Ming-Image-0.1-Design` 트리, 또는 `nativ-community` MLX 8-bit 및 4-bit 변환본) | generate(RGBA 출력) | 12 / 1.0(guidance 고정) |
| `llada_image` | inclusionAI LLaDA-Image-Turbo, LLaDA-Image(Diffusers bf16 트리) | generate(텍스트 또는 VQ prior), edit(소스 이미지 하나) | Turbo 4 / 1.0, base 50 / 5.0 |

## Probe

`mlxcel::imagegen::detect_image_family(path)`는 작은 JSON 파일들로부터 체크포인트 디렉터리를 분류하며, 가중치는 읽지 않습니다. 이는 `Blaizzy/mlx-vlm`(`mlx_vlm/generate/image.py`)의 `_local_image_model_types`를 포팅한 것입니다. 순서대로 다음을 읽습니다.

1. `manifest.json`: `transformer-packed-mflux/diffusion_pytorch_model.safetensors`, `text_encoder-mlx-4bit/model.safetensors`, `tokenizer/tokenizer.json`을 나열한 manifest는 `bonsai`입니다. Bonsai는 `Flux2KleinPipeline` `model_index.json`을 제공하는데 클래스 이름 스캔은 이를 `flux2`로 판단하므로 이것을 가장 먼저 확인합니다.
2. `model_index.json`과 `config.json`: `model_type`, `_class_name`, 그리고 `transformer`, `vae`, `scheduler`, `text_encoder` 컴포넌트의 클래스 이름.
3. transformer `weight_map`(`transformer/model.safetensors.index.json`, 없으면 `transformer/diffusion_pytorch_model.safetensors.index.json`)을 마커 키 집합과 대조합니다. `z_image` transformer는 `mllm/config.json`이 있으면 `ming_image`입니다.
4. 각 `<dir>/config.json`.

`IMAGE_FAMILIES`의 패밀리를 지명하는 첫 번째 후보가 선택됩니다(`bonsai`, `flux2`, `z_image`, `ming_image`, `ernie_image`, `qwen_image`, `mage_flow`, `ideogram4`, `llada_image`). 후보는 위 순서로 나오므로, `model_index.json`의 `Ideogram4Pipeline`이 `vae/config.json`의 FLUX.2 VAE 클래스보다 우선합니다. 존재하지 않거나, 디렉터리가 아니거나, 이미지 메타데이터가 없는 경로는 `None`을 반환하며, 파싱할 수 없는 JSON은 건너뜁니다.

이미지 체크포인트에 대한 `mlxcel generate`와 `mlxcel serve -m`은 `model_type not found`나 단순한 `config.json` 누락 오류 대신 `image generation checkpoint (<family>): use mlxcel generate-image`로 실패합니다. 이 hook은 `src/models/detection.rs`에 있으며 디렉터리에 최상위 `model_type`이 없을 때만 동작하므로, 텍스트 체크포인트에는 영향이 없습니다.

## CLI

```bash
mlxcel generate-image -m <dir> -p "a watercolor fox"
mlxcel generate-image -m <dir> -p "a fox" --size 768x512 --seed 7 -n 2 -o fox.png
mlxcel generate-image -m <dir> --task edit --image fox.png -p "make it night"
```

| 플래그 | 의미 |
|------|------|
| `-m, --model PATH` | 체크포인트 디렉터리(Diffusers 레이아웃). |
| `-p, --prompt TEXT` | 프롬프트 또는 편집 지시. |
| `--negative-prompt TEXT` | negative 프롬프트. 생략하면 모델 기본값이 적용됩니다. |
| `--task generate\|edit` | 기본값 `generate`. |
| `--image PATH` | 참조 이미지. 여러 개면 반복합니다. `edit`에서는 필수이고 `generate`에서는 거부됩니다. |
| `--size WxH` | 생성의 기본값은 `512x512`. 편집의 기본값은 첫 번째 참조 이미지의 크기이거나 패밀리가 그로부터 도출한 값입니다(FLUX.2는 upstream처럼 최대 1 MP로 줄이고 16의 배수로 내림). 양쪽 변 모두 패밀리 grid의 배수여야 합니다(대부분의 패밀리에서 16). |
| `--steps N`, `--guidance F` | 생략하면 모델 기본값이 적용됩니다. |
| `--seed N` | 기본값은 무작위 32-bit seed이며, seed는 항상 출력됩니다. |
| `-n N` | 이미지 1개에서 10개. 이미지 `i`는 `seed + i`를 사용하며, FLUX.2 생성은 예외입니다([배치 seed](#배치-seed) 참고). |
| `-o, --output PATH` | 기본값 `outputs/image-<seed>.png`. 확장자는 `.png`로 강제되며, `-n`이 1보다 크면 이름은 `<stem>-01.png`, `<stem>-02.png` 등이 됩니다. |
| `--gen-kwargs JSON` | 모델별 옵션의 JSON 객체이며 `extra`로 모델에 전달됩니다. |

작업 순서: 모델 디렉터리, 인자와 참조 이미지, 출력 디렉터리 쓰기 가능 여부, 패밀리 분류, 런타임 초기화, 그리고 모델 로드. 따라서 잘못된 디렉터리, 포팅되지 않은 패밀리, 쓸 수 없는 출력 디렉터리는 가중치를 로드하기 전에 실패합니다.

모델이 프롬프트를 다시 쓴 경우(ERNIE-Image의 prompt enhancer), CLI는 저장된 각 이미지를 알리는 줄 다음에 `revised prompt: <text>`를 출력합니다.

`mlxcel generate-image`에는 `generate`의 `--output-modality image`에 해당하는 것이 없습니다. 그 명령의 sampling, KV-cache, speculative 플래그는 diffusion에 의미가 없고, `detect`가 이미 별도 서브커맨드라는 선례를 만들었기 때문입니다.

## 서버

```bash
mlxcel-server -m <chat model> --image-model <image checkpoint>
```

`--image-model`(`mlxcel-server`와 `mlxcel serve` 모두에서 `MLXCEL_IMAGE_MODEL`도 가능)은 `--embedding-model`처럼 보조 worker를 로드합니다. 전용 스레드가 모델을 소유하고, 제한된 큐가 요청을 받아들이며, 요청별 타임아웃이 호출자를 해방합니다. 로드 실패는 시작 시점에 치명적이므로, 이미지 라우트를 요청한 명령줄은 그 라우트에 대해 `501`을 서빙하는 일이 없습니다. diffusion 실행은 몇 초 동안 GPU를 점유하므로 요청은 한 번에 하나씩 실행되며, 큐가 가득 차면 `503`, 타임아웃은 `504`입니다. 큐 깊이(8)와 타임아웃(900초)은 `src/server/image_worker.rs`의 상수입니다.

| 메서드 | 경로 | 비고 |
|--------|------|------|
| POST | `/v1/images/generations` | 별칭 `/images/generations`. |
| POST | `/v1/images/edits` | 별칭 `/images/edits`. JSON 본문만 지원. |

`--image-model`이 없으면 두 라우트 모두 해당 플래그를 명시하며 `501`로 응답합니다. 패밀리가 실행 전에 거부하는 요청(`ImageGenerationModel::check_request`: FLUX.2의 크기 범위, 잘못된 `extra` 옵션)은 `400`이고, 모델이 실행 중에 발생시키는 `InvalidImageRequest`(Z-Image의 옵션, strength, 참조 개수. ERNIE-Image의 옵션, `image_strength`, 참조 개수, `pe/`가 없는 체크포인트에 대한 enhancer 요청)도 `400`입니다. 생성 중의 그 밖의 실패는 `500`입니다.

### 요청

`generations`: `prompt`, `model`, `n`(1에서 10), `size`(`WxH`) 또는 `width`와 `height`, `steps`, `seed`, `guidance`, `negative_prompt`, `response_format`, `output_format`. `edits`는 문자열 하나 또는 목록인 `image`를 추가합니다. 그 밖의 필드는 `extra`로 모델에 전달됩니다.

- `width`와 `height`는 함께 주거나 모두 생략해야 하며(그렇지 않으면 `400`), `size`를 덮어씁니다.
- `size`의 기본값은 생성에서는 `512x512`이고, 편집에서는 첫 번째 참조 이미지의 크기이거나 패밀리가 그로부터 도출한 값입니다(FLUX.2: 최대 1 MP, 16의 배수로 내림). 너비와 높이는 패밀리 grid의 배수여야 하며, 오류 메시지가 그 배수를 알려 줍니다.
- `response_format`은 `b64_json`만 받으며 `url`과 `path`는 `400`입니다. `output_format`은 `png`만 받습니다. `output_path`와 `output_dir`은 `400`입니다. 서버는 클라이언트를 대신해 파일을 쓰지 않습니다.
- `model`이 비어 있지 않으면 서빙 중인 모델 id와 같아야 합니다.
- 각 `image`는 data URL 또는 URL이며 chat 이미지와 같은 제한을 거쳐 읽습니다(`file://` URL은 `--media-path` 안에 머뭅니다). 원시 파일시스템 경로는 `400`입니다. OpenAI의 edit wire format인 multipart 업로드는 서빙하지 않습니다.

### 응답

```json
{
  "created": 1790000000,
  "data": [
    {"b64_json": "iVBORw0KGgo...", "mime_type": "image/png",
     "width": 512, "height": 512, "seed": 7, "revised_prompt": null}
  ],
  "output_format": "png",
  "size": "512x512"
}
```

RGB와 RGBA 이미지는 모두 그대로 인코딩되며, RGBA는 alpha 채널을 유지합니다.

`revised_prompt`는 항상 존재합니다. 모델이 프롬프트를 다시 쓰지 않았다면 `null`이며, 현재는 ERNIE-Image의 prompt enhancer와 Ideogram 4의 평문 caption 래핑이 이에 해당합니다.

## FLUX.2 Klein (`flux2`)

`src/imagegen/flux2/`에 있는 `mlx_vlm/models/flux2`(`Blaizzy/mlx-vlm`, git 73d366ec)의 포트입니다. 두 task 모두 지원됩니다.

| 체크포인트 | 변형 | Step | Guidance | 비고 |
|-----------|------|------|----------|------|
| `black-forest-labs/FLUX.2-klein-4B` | `flux2-klein-4b` | 4 | 1.0 | Diffusers 트리, bf16. Qwen3-4B 텍스트 encoder. |
| `black-forest-labs/FLUX.2-klein-9B` | `flux2-klein-9b` | 4 | 1.0 | Qwen3-8B 텍스트 encoder. |
| `black-forest-labs/FLUX.2-klein-base-4B`, `-base-9B` | `flux2-klein-base-*` | 50 | 4.0 | Undistilled. guidance가 1보다 크면 negative 프롬프트(`--negative-prompt`가 없으면 `" "`)로 두 번째 pass를 실행합니다. |
| `black-forest-labs/FLUX.2-klein-9b-kv` | `flux2-klein-9b-kv` | 4 | 1.0 | 편집은 첫 step에서 참조 token의 K/V를 캐시하며, 생성은 일반 9B 경로입니다. |
| `mlx-community/flux2-klein-4b-4bit`, `-9b-4bit` | dense 모델과 동일 | 4 | 1.0 | MLX affine 4-bit, group 64. 컴포넌트 `config.json`이 없으므로 텍스트 encoder config는 변형에서 합성하고, VAE의 양자화된 mid-block attention projection은 로드 시 dequantize합니다. |

변형은 upstream의 `variant_from_local_path`처럼 디렉터리 이름(snapshot 경로 포함)에서, 그다음 `model_index.json`의 `is_distilled`와 transformer 폭에서 결정됩니다. `transformer/`, `vae/`, `text_encoder/`, `tokenizer/tokenizer.json`이 없으면 해당 경로를 명시하며 로드 시 실패합니다.

- **크기.** 너비와 높이는 `256..=2048`, 16의 배수. 크기를 지정하지 않은 편집(CLI와 서버)은 upstream의 `_resolve_edit_dimensions`를 따릅니다. 첫 번째 참조 이미지를 최대 1024x1024 픽셀 면적으로 줄이고(절대 확대하지 않음) 각 변을 `256..=2048` 범위의 16의 배수로 내림하므로, 4000x3000 사진은 1168x880으로, 1000x667은 992x656으로 편집됩니다. 다른 패밀리는 첫 번째 참조 이미지 자신의 크기를 사용합니다.
- **프롬프트.** `enable_thinking=false`인 Qwen chat template을 사용하고, id 151643으로 오른쪽 padding하며, `max_sequence_length`로 truncate합니다. 생성은 맞는 가장 작은 bucket(32, 64, 128, 256, 512)으로 padding하고, 편집은 `max_sequence_length`까지 padding합니다. 임베딩은 Qwen3의 레이어 9, 18, 27을 쌓은 것입니다. 텍스트 encoder는 매 인코딩 후 evict되고 프롬프트는 캐시되며(32개 항목), transformer와 VAE는 첫 요청에서 로드되어 상주합니다.
- **Tiling.** 생성은 한 변이 256픽셀에 도달하면 upstream의 `auto` 모드처럼 128픽셀(16 latent cell) tile 단위로 디코딩하고 8 cell(64픽셀) 겹침 구간에서 raised-cosine ramp로 blend하므로, 이 패밀리가 받아들이는 모든 크기가 tiling됩니다. 편집은 tiling 없이 디코딩합니다. 같은 512x512 latent의 tiled와 untiled 디코딩은 upstream과 여기서 모두 25.05 dB PSNR만큼 차이가 납니다.
- **`--gen-kwargs` / `extra`.** `max_sequence_length`(1에서 512, 기본값 512)와 `tiled_vae`(`true` 또는 `false`, 기본값은 위와 같음).

### 배치 seed

FLUX.2 생성은 `Blaizzy/mlx-vlm#2355`를 따릅니다. `n`개의 이미지는 프롬프트 임베딩을 공유하고 `seed`에서 뽑은 배치 `n`의 단일 noise 텐서에서 나오므로, 모든 이미지가 `seed`를 보고하며 이미지 `i`는 `seed + i`를 단독으로 쓴 이미지와 다릅니다. 편집은 이미지마다 seed `seed + i`로 한 번씩 실행됩니다. 그다음 VAE는 배치의 각 이미지를 단일 이미지가 받는 tiling으로 따로 디코딩합니다. upstream은 배치 전체를 untiled로 디코딩하는데, 2048x2048에서는 이미지당 약 15 GiB가 필요합니다.

### 일치 여부

mlx 0.32.4, CUDA, NVIDIA GB10, `MLX_ENABLE_TF32=0`에서 upstream mlx-vlm 0.7.8(git 73d366ec)과 비교했습니다. `src/imagegen/flux2/parity_tests.rs`의 `#[ignore]` 테스트는 `tests/fixtures/flux2/`의 fixture를 읽으며, 이는 `tests/fixtures/flux2/generate_flux2_fixtures.py`가 재현합니다. `edit_reference.png`는 그 스크립트가 작성한 upstream FLUX.2-klein-4B 생성물(프롬프트 "A studio photo portrait of a smiling man with short brown hair, plain grey background", seed 7, 512x512, 4 step)이며 실제 인물의 사진이 아닙니다. 4B 체크포인트는 Apache-2.0입니다.

| 검사 | 결과 |
|------|------|
| 초기 noise(generate seed 9909, edit seed 123) | 비트 단위로 동일 |
| f32 transformer 대 f32 upstream, 같은 입력 | 비트 단위로 동일 |
| upstream의 최종 latent를 디코딩하는 VAE, tiled / untiled | 57.59 / 58.45 dB |
| 4B generate 512x512, 4 step, seed 9909 | 32.27 dB |
| 4B edit "Add black sunglasses", seed 123 | 40.54 dB |

bf16 최종 latent는 upstream과 상대 RMS로 1.03e-1(generate), 4.11e-2(edit) 차이가 납니다. 긴 reduction을 가진 bf16 matmul(7680 폭의 context embedder가 첫 번째)은 동일한 입력과 가중치에서 이 빌드의 MLX와 mlx 0.32.4에서 다르게 반올림되고, 네 번의 Euler step이 이를 증폭합니다. upstream 자신의 f32 실행도 같은 케이스에서 bf16 실행과 7.1e-2 차이가 납니다. generate는 30 dB 게이트보다 약 2 dB 위에 있으므로 MLX pin 변경으로 움직일 수 있습니다.

다른 체크포인트의 같은 요청에 대한 upstream 대비 PSNR: 9B edit 44.39 dB, 9b-kv edit 42.94 dB, 4B 4-bit generate 35.47 dB, base-4B generate 12 step 32.80 dB, 9B generate 25.78 dB, 9B 4-bit generate 20.70 dB(구도는 같고 세부에서 bf16 drift). 9b-kv 체크포인트는 자체 가중치를 가지므로 그 편집은 9B 편집이 아닙니다. upstream의 두 결과는 25.15 dB, 이 빌드의 두 결과는 25.09 dB 차이가 납니다.

**편집 fixture와 upstream의 encoder.** 편집 fixture는 VAE encoder의 `conv_in`을 전치한 상태로 생성됩니다. upstream이 그 텐서 하나만 PyTorch 순서로 남겨 두기 때문입니다([검증](#검증) 참고). 따라서 공개된 상태의 upstream은 이 빌드와 다르게 편집합니다.

## Prism Bonsai (`bonsai`)

`prism-ml/bonsai-image-ternary-4B-mlx-2bit`(Apache-2.0)를 위해 `src/imagegen/bonsai/`에 있는 `mlx_vlm/models/bonsai`(`Blaizzy/mlx-vlm`, git 73d366ec)의 포트입니다. FLUX.2 Klein 4B에 ternary transformer linear를 2-bit MLX affine 가중치로 packing한 것입니다. 생성 전용이며 편집 요청은 지원되지 않는 것으로 거부됩니다. 기본값: 4 step, guidance 1.0. `--gen-kwargs`는 FLUX.2의 `max_sequence_length`와 `tiled_vae`를 받습니다.

생성 루프는 FLUX.2의 것(`Flux2Pipeline`)이며, Bonsai가 컴포넌트를 제공합니다.

- **Transformer.** `config.json`이 있는 `transformer-packed-mflux/`. `quantization_config.json`은 `format: mlx-packed-affine`이어야 하고, 그 `bits`와 `group_size`는 mlxcel-core의 affine 경로가 구현한 폭이어야 합니다(공개 체크포인트에서는 2와 128이며, 1-bit 바이너리 변형은 그 값을 명시하며 거부됩니다). `quantized_modules`의 모든 module은 affine `UnifiedLinear`(`uint32` packed weight, 저장된 그대로의 bf16 scale과 bias)로 로드되고, 그 밖의 모든 텐서는 dense bf16입니다. 나열된 module에 `.scales`나 `.biases`가 없거나 나열되지 않은 module에 `.scales`가 있으면 해당 module을 명시하며 로드가 실패합니다. upstream의 `klein_fast` megakernel과 fused norm-RoPE Metal 커널은 같은 `quantized_matmul` 연산 위의 속도 작업이므로 포팅하지 않았습니다.
- **텍스트 encoder.** 공유 Qwen3 conditioner를 통한 `text_encoder-mlx-4bit/`(MLX 4-bit, group 64, tied embedding). tokenizer는 `tokenizer/`에서 옵니다. 매 인코딩 후 evict됩니다.
- **프롬프트.** FLUX.2 생성은 bucket화하는 반면, 여기서는 `max_sequence_length`까지 padding합니다(upstream `bucketed_seq_len=False`).
- **VAE.** 체크포인트의 `vae/`(전체 FLUX.2 VAE). upstream은 이를 무시하고 로드 시 `black-forest-labs/FLUX.2-small-decoder`(decoder 채널 96/192/384/384)를 다운로드하지만, 이 포트는 로드 시 두 번째 저장소를 가져오지 않습니다. 두 decoder는 일치 케이스에서 서로 34.96 dB이며, 이 빌드의 출력은 공개된 그대로의 upstream과 29.82 dB입니다.
- **배치.** `n`개의 이미지는 upstream의 서버 루프처럼 seed `seed + i`로 하나씩 실행되며, 각각 자신의 seed를 보고합니다.

### 일치 여부

mlx 0.32.4, CUDA, NVIDIA GB10, `MLX_ENABLE_TF32=0`에서 upstream mlx-vlm 0.7.8(git 73d366ec)과 "A tiny glass bonsai tree on a moonlit desk", 512x512, 4 step, seed 9909로 비교했습니다. `tests/fixtures/bonsai/generate_bonsai_fixtures.py`는 Metal 전용 fused 커널을 MLX 연산으로 된 동일 연산으로 바꾸고 체크포인트의 `vae/`를 decoder로 사용한 upstream 자체 Bonsai 파이프라인을 실행하며, `#[ignore]` 테스트는 `src/imagegen/bonsai/parity_tests.rs`에 있습니다.

| 검사 | 결과 |
|------|------|
| 초기 noise | 비트 단위로 동일 |
| Bonsai 가중치를 쓰는 f32 upstream FLUX.2 transformer 대 f32 transformer, 같은 입력 | 상대 RMS 1.19e-6 |
| upstream의 정확한 step-0 입력에 대한 bf16 transformer 한 step | 상대 RMS 8.1e-3 |
| upstream의 최종 latent를 디코딩하는 VAE | 57.61 dB |
| 엔드투엔드 PNG | 31.18 dB |

bf16 최종 latent는 upstream과 상대 RMS로 1.0e-1, 프롬프트 임베딩은 2.3e-2(padding 행에서는 1.7e-1) 차이가 납니다. 둘 다 포트 오류가 아닙니다. upstream의 bf16 4-bit conditioner는 f32 conditioner와 이미 1.95e-2 떨어져 있고(이 포트는 1.74e-2), padding을 따라 같은 방식으로 커지며, upstream은 같은 가중치에서 `klein_fast`를 FLUX.2 transformer로 바꾸면 최종 latent에서 자기 자신과 3.5e-2 불일치합니다. 에픽 전체의 기준에 따라 위의 상대 RMS 수치는 게이트하지 않고 non-blocking drift로 보고합니다. 테스트는 PSNR 30 dB 이상, 비트 단위로 동일한 noise, 그리고 잘못된 포트(약 1.4)가 통과할 수 없는 0.2 latent sanity 경계를 게이트로 사용합니다. megakernel 자체는 Metal 밖에서 실행할 수 없으므로 `quantized_matmul`과의 일치는 `klein_fast/blocks.py`를 읽어 확인했습니다. `QuantizedLinearKernel`은 `mx.quantized_matmul`을 호출하고, fused 커널은 f32 RMSNorm에 gain을 곱한 값과 interleaved-pair RoPE를 계산한 뒤 bf16으로 변환합니다.

## ERNIE-Image (`ernie_image`)

`baidu/ERNIE-Image-Turbo`와 `baidu/ERNIE-Image`를 위해 `src/imagegen/ernie_image/`에 있는 `mlx_vlm/models/ernie_image`(`Blaizzy/mlx-vlm`, git 73d366ec)의 포트입니다. 원본 Diffusers 트리는 바로 로드되고, 양자화 변환본(upstream `convert.py`)은 이름으로 거부됩니다.

- **변형.** 있으면 `mlx_ernie_image.json`의 `source` 또는 `variant`, 없으면 디렉터리 이름(`turbo`가 포함되면 Turbo. Hugging Face cache snapshot은 repo id를 사용). 두 체크포인트는 둘을 구분해 줄 모든 config 파일을 공유하므로 upstream도 이름으로 결정합니다.
- **Transformer.** 36층 single-stream DiT(hidden 4096, 32 head). 이미지 token이 먼저, 그다음 projection된 텍스트가 오며, 3축 RoPE와 텍스트 padding에 대한 key mask를 사용합니다.
- **텍스트 conditioner.** 트리 내 `Ministral3Model::forward_hidden_states`를 통한 `text_encoder/` Ministral 3 체크포인트. 마지막을 제외한 모든 decoder 레이어를 사용하고 최종 norm은 없으며, 프롬프트는 2048 token에서 truncate됩니다. 트리 내 Ministral 3은 이제 이전에 무시하던 체크포인트의 YaRN rope scaling을 적용하며, 이것이 없으면 conditioning이 upstream과 다릅니다.
- **Prompt enhancer.** `pe/`(Ministral 3 decoder)와 `pe_tokenizer/`가 upstream의 chat template, temperature 0.6, top-p 0.95, 최대 2048개의 새 token으로 이미지별 seed를 사용해 `{"prompt", "width", "height"}` JSON으로부터 프롬프트를 다시 씁니다. 렌더링된 enhancer 입력은 4096 token(conditioner의 2048의 두 배)으로 제한되며, 더 긴 프롬프트는 enhancer를 켜면 `400`이고 `use_prompt_enhancer: false`이면 2048 token으로 truncate됩니다. `use_prompt_enhancer`가 결정합니다. `true`이면 enhance하고(체크포인트에 `pe/`가 없으면 `400`), `false`이면 절대 하지 않으며, 생략하면 `pe/`가 있을 때 text-to-image는 enhance되고 편집은 enhancer가 소스 이미지를 볼 수 없으므로 enhance되지 않습니다. enhance된 텍스트가 각 이미지의 `revised_prompt`입니다. 공개된 체크포인트는 upstream처럼 중국어 프롬프트를 작성합니다.
- **Sampling.** bf16 noise `normal([1, 128, H/16, W/16], key(seed))`, upstream의 static-shift flow-matching 스케줄, f32 Euler step. CFG(negative가 먼저, batch 2)는 guidance가 1보다 클 때만 실행되므로, 기본 guidance의 Turbo는 negative 프롬프트를 인코딩하지 않습니다.
- **편집**(`--task edit`, 정확히 하나의 `--image`. 둘이면 `400`). SDEdit: 소스를 FLUX.2 VAE로 인코딩하고 batch-normalize한 뒤, `start = steps - max(1, min(steps, round(steps * image_strength)))`에서 `(1 - sigma) * source + sigma * noise`로 noise를 입힙니다. `--size`가 없으면 소스를 1 MP 쪽으로 줄이고(절대 확대하지 않음) 한 변을 최대 2048로, 16의 배수로 내림합니다.
- **옵션**(`--gen-kwargs` 또는 추가 JSON 필드): `image_strength`(별칭 `strength`, 편집 전용, 기본값 0.6, `(0, 1]` 범위), `negative_prompt`, `use_prompt_enhancer`, `prompt_enhancer_max_tokens`(1에서 2048). 잘못된 타입이나 범위는 각각 `400`입니다.
- **배치.** `n`개의 이미지는 enhancement를 포함해 upstream의 전체 파이프라인을 seed `seed + i`로 한 번씩 실행합니다.
- **메모리.** 텍스트 encoder, transformer, VAE가 상주합니다(bf16에서 약 24 GB). enhancer(약 7 GB)는 첫 사용 시 로드되어 유지됩니다. upstream은 프롬프트마다 encoder와 enhancer를 내릴 수 있지만(`evict_text_encoder`), 유지하면 서버가 다시 로드하지 않고 다음 요청에 응답할 수 있습니다.

```bash
mlxcel generate-image -m models/mlx/ernie-image-turbo -p "A red panda serving tea in a moonlit bamboo forest" --size 1024x1024
mlxcel generate-image -m models/mlx/ernie-image-turbo --task edit --image panda.png -p "Make it snow" \
    --gen-kwargs '{"image_strength": 0.7}'
```

### ERNIE-Image 정밀도

upstream은 모든 부동소수점 transformer 가중치를 bf16으로 변환하고 bf16 noise를 뽑으며, modulation, gate, 스케줄러 step은 f32로 실행합니다. 포트는 모든 백엔드에서 이를 유지합니다(`weights::TRANSFORMER_DTYPE`). f16 규칙에서는 transformer가 실패합니다. f16 가중치에서 Turbo 1024x1024 일치 케이스는 NaN latent와 9.54 dB 이미지를 만들므로 bf16을 유지합니다. 텍스트 encoder와 enhancer는 텍스트 모델 load policy(`load_text_weights`) 밖에서 `mlxcel_core::weights::load_weights_from_dir`로 체크포인트를 저장된 그대로(bf16) 로드합니다. 그 정책의 bf16에서 f16으로의 변환(`MLXCEL_CUDA_F16_NORMALIZE=1`, Apple Silicon) 아래에서는 conditioning이 상대 RMS 3.509e-3으로 drift했고 Turbo는 26.04 dB가 되었습니다. bf16에서는 이들이 만드는 conditioning이 일치 프롬프트에서 upstream과 정확히 일치합니다. batch-1 실행이 비트 단위로 정확하려면 두 가지 커널 세부 사항이 MLX의 `nn` 레이어를 따라야 했습니다. 입력 dtype으로 계산되는 GELU(`nn.gelu`. `mlxcel_core::gelu`는 f32로 계산함)와, 하나의 `addmm`으로 처리되는 bias가 있는 linear(`nn.Linear`. `Linear::forward`는 matmul 뒤에 bias를 더함)입니다.

### ERNIE-Image 일치 여부

1024x1024, seed 42, 프롬프트 "A red panda serving tea in a moonlit bamboo forest", enhancer 끔, 양쪽 모두 `MLX_ENABLE_TF32=0`으로 CUDA 백엔드(NVIDIA GB10)의 mlx 0.32.4에서 upstream mlx-vlm 0.7.8(git 73d366ec)과 비교한 seed 고정 실행입니다. `tests/fixtures/generate_ernie_image_fixtures.py`는 케이스별 oracle 파일과 커밋된 요약 `tests/fixtures/imagegen/ernie_image_parity.safetensors`(noise 체크섬과 앞부분, 8x8 pooling한 최종 latent와 이미지)를 만듭니다. 초기 noise는 모든 케이스에서 비트 단위로 동일하며, Turbo 케이스의 텍스트 conditioning도 그렇습니다.

| 케이스 | upstream 입력에 대한 첫 transformer forward | 최종 latent 상대 RMS | PNG PSNR | 게이트(latent / PSNR) |
|--------|---------------------------------------------|----------------------|----------|-----------------------|
| Turbo, 8 step, guidance 1 | 0 | 0 | 57.00 dB | 1e-2 / 30 dB |
| Turbo 이미지의 Turbo 편집, 8 step, guidance 3, `image_strength` 0.7 | 4.90e-2 | 1.35e-1 | 32.23 dB | 2.5e-1 / 30 dB |
| Base, 50 step, guidance 4, 빈 negative 프롬프트 | 5.02e-3 | 3.03e-1 | 24.24 dB(30 dB 미달) | 4.5e-1 / 20 dB |
| Base 대조, 8 step, guidance 4(oracle 디렉터리 전용) | 5.02e-3 | 1.64e-1 | 31.43 dB | 2.5e-1 / 30 dB |

batch 1에서는 모든 연산이 upstream을 비트 단위로 재현하므로 Turbo text-to-image 케이스는 정확합니다(57 dB는 동일한 latent의 PNG 왕복 결과입니다). 두 CFG 케이스는 transformer를 batch 2로 실행하는데, 여기서 작은 M의 bf16 GEMM 하나, 즉 텍스트 projection `[2, 13, 3072] x [3072, 4096]`이 mlxcel의 pinned MLX에서 mlx 0.32.4 wheel과 다른 커널을 선택해 마지막 비트에서 다르게 반올림되며, 36개 레이어와 8 또는 50 step이 이를 증폭합니다. upstream도 자기 자신에 대해 같은 편차를 보입니다. 그 projection만 배치 행 하나씩 평가하면 편집 케이스는 1.39e-1(32.14 dB), base 케이스는 2.89e-1(23.70 dB) 움직입니다. upstream도 batch에 불변이 아닙니다. batch 2의 행 1은 batch 1의 같은 행과 5.3e-2 차이가 납니다. **50 step base 케이스는 에픽의 30 dB PNG 기준을 충족하지 못합니다.** 8 step 대조 실행은 같은 체크포인트, 프롬프트, seed, CFG batch-2 forward를 실행하고 31.43 dB에 도달하므로 batch forward는 건전하며 미달은 50 step 루프가 반올림 차이를 증폭한 결과입니다. 50 step 이미지는 구도가 동일합니다(같은 판다, 찻잔, 주전자, 달이며 차이는 세부에 있음). 20 dB 하한은 심하게 망가진 파이프라인만 거르며 에픽 기준을 충족하지는 않습니다. 참고로, GELU와 `addmm` 수정 이전의 포트는 batch-1 Turbo 케이스에서 3.3e-1(24.8 dB)로 drift했으며 Turbo 게이트가 이를 거부합니다.

```bash
MLX_ENABLE_TF32=0 <lib test binary> --ignored --test-threads=1 imagegen::ernie_image::parity_tests::
```

`MLXCEL_ERNIE_IMAGE_TURBO`와 `MLXCEL_ERNIE_IMAGE_BASE`가 체크포인트를 가리킵니다(기본값 `models/mlx/ernie-image-turbo`, `models/mlx/ernie-image`). `MLXCEL_ERNIE_IMAGE_ORACLE_DIR`(생성기의 `--out`)는 전체 해상도 비교, 첫 forward와 conditioning 진단, 8 step 대조(`--control`로 생성)를 추가하고, `MLXCEL_ERNIE_IMAGE_OUT`은 PNG를 저장합니다.

## 공유 구성 요소

| 모듈 | 내용 | 사용처 |
|------|------|--------|
| `pipeline` | `ImageRequest`, `GeneratedImage`, `ImageSamplingDefaults`, `ImageGenerationModel` trait. | 모든 패밀리 |
| `probe` | `detect_image_family`. | 모든 패밀리 |
| `registry` | `FAMILIES`, `ImageFamily`, `resolve_family`, `load_image_model`. | 모든 패밀리 |
| `output` | `encode_png`, `to_b64_png`, `indexed_output_path`. | CLI, 서버 |
| `run` | `validate_request`, `run_request`, `generate_to_files`, `parse_size`, `random_seed`. | CLI, 서버 |
| `scheduler` | `FlowMatchEulerDiscrete::dynamic`(지수 shift, 선택적 `shift_terminal`)와 `::static_shift`. | QwenImage, MingImage. Z-Image, Mage-Flow |
| `latent` | `patchify_latents`, `unpatchify_latents`, `pack_latents`, `unpack_latents`, `prepare_packed_latents`(seed가 있는 noise를 grid id와 함께 packing), `prepare_grid_ids`. | FLUX.2, Bonsai, ErnieImage, LLaDAImage |
| `vae` | `Flux2Vae`, `Flux2VaeConfig`: decode, packed-latent decode, 선택적 encode(`quant_conv`는 선택적이며 `use_quant_conv: false`인 FLUX.1 `AutoencoderKL`에는 없음), `bn_normalize`와 `denormalize_packed_latents`. | FLUX.2, ErnieImage, LLaDAImage, Z-Image |
| `qwen3_text_encoder` | `Qwen3TextEncoder`: 모든 레이어의 hidden state(`num_layers + 1`, 인덱스 0은 최종 norm 이전의 임베딩)와 `stacked_prompt_embeds`(`[B, T, n * H]`, FLUX.2 순서). `Qwen3Model::forward_all_hidden` 위에 구축. | FLUX.2, Z-Image, Ideogram 4, Bonsai |
| `qwen_image::vae` | `QwenImageVae::new`, `::load`: 단일 프레임으로 실행하는 Wan 스타일 causal-3D VAE(residual 및 plain 레이아웃, Diffusers 또는 MLX conv 레이아웃), `decode_normalized`, `encode_normalized`. | QwenImage, MingImage |
| `dit_blocks` | `TimestepEmbedder`, `FeedForward`, `MRoPE`(`compute_freqs_on_device`는 upstream처럼 장치에서 inverse frequency를 만듦), `apply_rotary`, `Attention`, `ZImageTransformerBlock`, `FinalLayer`, `sanitize_transformer_weights`, `BiasAdd`(bias가 있는 linear를 위한 선택적 fused bias `addmm`). | ZImage, LLaDAImage, MingImage |

공유되지 않는 것: FLUX.2 transformer, 그 empirical-`mu` 스케줄러, tiled VAE 디코딩은 `src/imagegen/flux2/`에 있습니다. 아예 서빙하지 않는 것: multipart 업로드, 부분 이미지 스트리밍, 비디오 생성.

### 모델이 따라야 하는 규약

- **Seed.** `n`개 중 이미지 `i`는 `seed + i`를 사용합니다(FLUX.2 생성은 예외이며 [배치 seed](#배치-seed) 참고). mlxcel은 upstream과 같은 MLX C++ RNG를 링크하므로, 같은 seeding 호출(Qwen-Image와 LLaDA-Image는 `mx.random.seed(seed)`, ERNIE-Image와 Ming-Image는 `mx.random.key(seed)`)과 같은 `normal` shape 및 dtype이 같은 초기 noise를 만듭니다. 전체 실행 일치 테스트가 비교할 수 있도록 upstream의 추출 순서를 유지하십시오.
- **스레드.** MLX 작업은 스레드 친화적(thread-affine)입니다. 서버는 worker 스레드에서 모델을 만들며, `ImageGenerationModel`에는 `Send` 바운드가 없습니다.
- **Convolution 가중치.** Diffusers는 4차원 conv 가중치를 `(out, in, kH, kW)`로 저장하고 MLX는 `(out, kH, kW, in)`을 기대합니다. 레이아웃은 두 레이아웃의 shape이 다른 텐서로부터 체크포인트당 한 번 결정하고(`vae::model::detect_conv_layout`), 이미 변환된 체크포인트를 두 번 permutation하지 마십시오. 텐서별로 shape에서 결정하는 것은 입력 채널 수가 kernel 크기와 같은 모든 conv에서 틀립니다. [검증](#검증) 아래의 설명을 참고하십시오.
- **Half precision.** Norm은 f32로 실행하고 activation dtype으로 되돌립니다. `scaled_dot_product_attention`이 아니라 `fast_scaled_dot_product_attention`을 사용하십시오. 후자는 f32 스칼라를 곱해 half-precision score를 f32로 승격시킵니다.

### Qwen3 conditioner

`Qwen3TextEncoder::load(dir)`는 `config.json`(최상위 Qwen3 필드 또는 중첩된 `text_config`)과 safetensors를 읽고, `model.`, `model.language_model.`, `language_model.model.` 키 레이아웃을 받아들이며, `lm_head.*`와 텍스트가 아닌 모든 텐서를 버리고, MLX affine `quantization`을 존중합니다. `from_weights(config, weights)`는 메모리 내 `WeightMap`을 받으며(Ideogram 4는 먼저 FP8을 dequantize함) 아무것도 변환하지 않습니다. RoPE는 표준입니다. 텍스트 위치에서는 M-RoPE section이 아무 영향이 없으므로 `rope_scaling`은 무시됩니다.

mask는 causal에 key padding을 더한 것이며 f32로 더합니다. 차단된 위치는 upstream의 `-inf`가 아니라 `-30000`을 사용합니다. 보이는 key가 있는 모든 행에서는 동일하지만, 완전히 차단된 행(왼쪽 padding)은 NaN 대신 균일 평균이 되며, NaN은 다음 레이어를 통해 실제 위치로 전파될 것입니다.

작은 모델에서 upstream의 `Qwen3TextEncoder`와 대조했고, `mlx-community/Qwen3-0.6B-4bit`에서는 마지막 hidden state를 norm하고 projection한 결과가 생성 경로의 logit을 정확히 재현합니다(`#[ignore]` 테스트). `qwen3.rs`의 변경은 메서드 하나 추가입니다. 이전과 이후의 teacher-forced logit trace(`examples/logit_trace`, 컨텍스트 256 token 뒤의 8 token 8개 chunk, 96개 위치)는 바이트 단위로 동일합니다.

attention은 가중치의 dtype으로 실행됩니다. upstream의 encoder(`qwen/attention.py`)는 SDPA를 위해 q, k, v를 f32로 변환하고 출력을 되돌립니다. 이것이 일치 격차의 원인은 아닙니다. `MLX_ENABLE_TF32=0`으로 GB10의 CUDA oracle에 대해 측정한 결과(#2666), 이 encoder의 attention을 f32로 실행하면 FLUX.2 프롬프트 임베딩이 9.4e-4 움직이며 이는 upstream 자신의 bf16에서 f32 attention으로의 변화(1.0e-3)와 같은 크기이고, 이 포트의 upstream까지 거리는 어느 쪽이든 1.19e-2로 유지됩니다. 마찬가지로 Z-Image caption 특징은 9.3e-3, Ideogram 4 텍스트 특징은 6.4e-2로 유지됩니다. 남은 격차는 한 자릿수 더 큽니다. 일부는 cuBLASLt 버전([oracle과의 GEMM 차이](#oracle과의-gemm-차이) 참고)이며, 일부는 upstream이 세 개로 실행하는 q, k, v를 이 포트가 하나의 fused GEMM으로 실행하기 때문입니다. oracle의 cuBLASLt를 preload하면 FLUX.2 프롬프트 임베딩은 1.19e-2에서 7.8e-3으로, projection도 분리하면 6.9e-3으로 움직입니다(#2732). Z-Image caption 특징은 움직이지 않으므로 나머지는 다른 원인입니다. 엔드투엔드로는 f32 attention이 drift를 재배치할 뿐이며, Bonsai(29.2 dB)와 Ideogram 4(pooling한 latent 4.7e-2)에서는 일치 게이트를 넘겨 버리므로 conditioner는 activation dtype을 유지합니다.

## 정밀도

upstream은 DiT, VAE, conditioning을 bf16으로 실행합니다. 프로젝트 규칙은 양자화되지 않은 bf16을 새 로딩 경계마다 f16으로 변환하며, seed 고정 비교에서 f16 아래 inf/NaN이나 임계값 실패가 나타날 때만 컴포넌트가 bf16을 유지합니다.

| 컴포넌트 | 기본 dtype | 증거 |
|----------|-----------|------|
| FLUX.2 VAE | f16(`Flux2Vae::default_dtype`, `bf16_to_f16_at_load`에서) | 실제 ERNIE-Image-Turbo VAE, 고정 latent의 64x64 디코딩: 유한하며 upstream f32와의 상대 RMS 1.2e-3. upstream 자신의 bf16 실행은 upstream f32와 2.0e-2 차이가 나므로, f16이 bf16 upstream 기본값보다 f32에 더 가깝습니다. |
| Z-Image DiT 블록 | 체크포인트 dtype을 따름 | 블록은 dtype에 중립입니다. 회전, SDPA 입력, residual 덧셈은 upstream MLX의 promotion(bf16 또는 f16과 f32는 f32)으로 변환됩니다. mlxcel의 MLX CUDA 빌드는 bf16과 f32를 bf16으로 승격하기 때문입니다. |
| Z-Image transformer와 VAE | f16 가중치, f32 latent 스트림 | [Z-Image 정밀도](#z-image-정밀도)를 참고하십시오. |
| FLUX.2 transformer | 모든 백엔드에서 bf16 | f16 가중치(GB10의 `MLXCEL_CUDA_F16_NORMALIZE=1`)에서는 4B generate 케이스가 NaN latent(10.2 dB)로 디코딩됩니다. 편집은 upstream처럼 f32 참조 latent로부터의 promotion으로 transformer를 f32로 실행합니다. |
| FLUX.2 VAE(FLUX.2 내) | `Flux2Vae::default_dtype` | CUDA에서는 bf16. f16 정책에서 VAE 단독 일치는 tiled 54.76 dB, untiled 52.37 dB입니다. |
| FLUX.2 Qwen3 텍스트 encoder | Qwen3 로더의 정책 | f16이 유지됨: f16 정책과 bf16 transformer에서 4B generate는 34.47 dB, edit는 38.54 dB입니다. |
| Bonsai transformer | bf16 연산, 2-bit packed linear와 저장된 그대로의 bf16 scale 및 bias | upstream은 모든 linear 입력을 bf16으로 만듭니다. |
| ERNIE-Image transformer | 모든 백엔드에서 bf16, modulation, gate, 스케줄러 step은 f32 | f16 가중치에서는 Turbo 1024x1024 케이스가 NaN latent(9.54 dB)로 디코딩됩니다. [ERNIE-Image 정밀도](#ernie-image-정밀도)를 참고하십시오. |
| ERNIE-Image 텍스트 encoder와 prompt enhancer | 모든 백엔드와 모든 load policy에서 저장된 그대로의 bf16(bf16에서 f16 변환 없이 로드) | conditioning이 일치 프롬프트에서 upstream과 비트 단위로 일치합니다. f16 정책(`MLXCEL_CUDA_F16_NORMALIZE=1`)에서는 Ministral 3 encoder가 conditioning을 상대 RMS 3.509e-3으로 움직이고 Turbo가 비트 단위 일치에서 2.93e-1(26.04 dB)로 바뀌므로 bf16을 유지합니다. |
| ERNIE-Image VAE | `Flux2Vae::default_dtype` | FLUX.2 VAE 행과 같음. sampler의 bf16 latent는 디코딩 전에 VAE dtype으로 변환되므로 `MLXCEL_CUDA_F16_NORMALIZE=1`에서는 f16으로 디코딩합니다(이전에는 f32로 승격되어 `MLX_ENABLE_TF32=0`의 1024x1024에서 OOM이 났습니다). f16 VAE 디코딩은 그 정책에서 유일한 f16 컴포넌트이며, Turbo 하락의 원인이고 CFG 케이스는 거의 움직이지 않습니다. 1024x1024, `MLX_ENABLE_TF32=0`에서 upstream 대비 엔드투엔드 PNG PSNR, bf16 정책 대 f16 정책: Turbo 57.00 / 53.56 dB, Turbo edit 32.23 / 32.02 dB, Base 24.24 / 24.24 dB, Base 8 step 31.43 / 31.42 dB. |
| Qwen-Image transformer | `bf16_to_f16_at_load`: Metal에서는 f16, CUDA에서는 `MLXCEL_CUDA_F16_NORMALIZE=1`이 아니면 bf16 | f16이 유지됨: 1024x1024, 4 step, seed 7에서 f16 가중치(GB10의 `MLXCEL_CUDA_F16_NORMALIZE=1`)는 upstream bf16 PNG 대비 48.32 dB(generate)와 47.92 dB(edit)이고 전 구간 유한하며, bf16에서는 52.22 dB와 52.61 dB입니다. |
| Qwen-Image VAE | 저장된 그대로의 f32 | 체크포인트가 f32로 제공되며 upstream도 f32로 실행합니다. |
| Qwen-Image Qwen3-VL 텍스트 encoder | Qwen3-VL VLM 로더의 정책 | f16이 유지됨: transformer 행의 f16 수치는 conditioner도 같은 정책으로 실행했으며 전 구간 유한합니다. bf16 drift는 [Qwen-Image 일치 여부](#qwen-image-일치-여부)에 있습니다. |
| Bonsai VAE와 4-bit 텍스트 encoder | FLUX.2와 같음 | f16 정책(GB10의 `MLXCEL_CUDA_F16_NORMALIZE=1`)에서 일치 케이스는 bf16의 31.18 dB 대비 31.21 dB입니다. |
| Ideogram 4 transformer와 텍스트 tower | bf16, 로드 시 행별 FP8에서 dequantize | upstream 자신의 dequantization이 bf16으로 반올림합니다. [Ideogram 4 정밀도](#ideogram-4-정밀도)를 참고하십시오. |
| Ideogram 4 VAE | `Flux2Vae::default_dtype` | FLUX.2 VAE 행과 같음. 게이트된 512x512 실행은 엔드투엔드 37.1 dB입니다. |
| Ming-Image conditioner | bf16 backbone, 저장된 그대로이자 upstream과 같은 f32 connector와 bridge | bf16 state는 upstream의 promotion이 암묵적으로 하는 것처럼 `proj_in`과 `directvlm` norm 전에 f32로 확장됩니다. [Ming-Image 정밀도](#ming-image-정밀도)를 참고하십시오. |
| Ming-Image transformer | f32 latent 스트림을 읽는 텐서는 f16(변환이 정확하지 않으면 f32), bf16 activation을 읽는 텐서는 bf16 | CUDA에서 upstream의 bf16-f32 promotion을 재현합니다. 같은 입력에서 upstream과의 첫 step velocity는 4.2e-3과 2.6e-3. |
| Ming-Image VAE | 저장된 그대로의 bf16 | upstream은 bf16으로 실행합니다. upstream 최종 latent의 디코딩은 97.7 dB와 96.8 dB. |
| LLaDA-Image 텍스트 encoder, QueryFormer, projection, SigVQ, transformer | 저장된 그대로의 bf16, upstream의 반올림 지점 | [LLaDA-Image 정밀도](#llada-image-정밀도)를 참고하십시오. |
| LLaDA-Image VAE | `Flux2Vae::default_dtype` | FLUX.2 VAE 행과 같음. 편집 소스 latent는 upstream의 수정된 encoder와 8.8e-3 차이가 납니다. |

측정에는 CPU MLX 백엔드를 사용했습니다([검증](#검증) 참고).

### oracle과의 GEMM 차이

일치 여부 섹션들은 mlx 0.32.4 CUDA wheel의 upstream과 비교하며, 일부는 drift의 일부를 이 빌드와 그 wheel에서 다르게 반올림되는 bf16 GEMM 탓으로 돌립니다. `MLX_ENABLE_TF32=0`으로 GB10에서 측정한 결과(#2732), 원인은 MLX 소스가 아닙니다. 이 프로젝트의 MLX pin의 dense GEMM 경로(`cublas_gemm.cpp`, `Matmul`과 `AddMM`)는 v0.32.4와 일치합니다. 차이는 양쪽이 로드하는 cuBLASLt 라이브러리입니다. mlxcel은 CUDA 13.0 toolkit의 시스템 cuBLASLt(13.1.1.3)를 링크하는 반면, `mlx_cuda_13` wheel은 pip `nvidia-cublas` 13.8.1.7을 제공합니다. 작은 M의 bf16 GEMM에서는 두 버전이 서로 다른 커널과 split-K 계수를 선택하며, 이들은 다른 순서로 누산합니다. 다른 라이브러리를 preload(`libcublasLt`와 `libcublas`의 `LD_PRELOAD`)하면 측정한 모든 케이스에서 양쪽이 비트 단위로 동일해집니다. 어느 버전도 f64 참조에 일관되게 더 가깝지 않습니다. 512행에서는 어느 쪽도 split-K를 선택하지 않으며 출력은 이미 일치합니다. 따라서 이 페이지의 일치 수치에는 cuBLASLt 버전 항이 포함됩니다. 이를 제거하려면 시스템 cuBLASLt를 preload한 상태로 oracle을 실행하거나, oracle의 cuBLASLt 버전을 fixture 옆에 기록하십시오.

## 패밀리 등록

모델 이슈가 다음을 수행하며, 런타임 이슈는 이 중 어느 것도 제공하지 않습니다.

1. `src/imagegen/<family>/`에서 `ImageGenerationModel`을 구현합니다. `family()`(probe 이름), `supports(task)`, `defaults(task)`(step과 guidance. 가중치가 아니라 JSON 메타데이터에서 결정), `size_multiple()`, 그리고 seed `req.seed + i`로 `req.n`개의 이미지를 반환하는 `generate(&ImageRequest)`.
2. probe 이름이 `src/imagegen/probe.rs`의 `IMAGE_FAMILIES`에 없으면 이를 식별하는 마커 키나 클래스 이름과 함께 추가하고, `probe_tests.rs`에 fixture 기반 테스트를 추가합니다.
3. `src/imagegen/registry.rs`의 `FAMILIES`에 한 행을 추가합니다. `ImageFamily { name: "<family>", load: <constructor> }`. 생성자는 체크포인트 디렉터리를 받고 모델을 실행할 스레드에서 실행됩니다.
4. 패밀리를 `Runtime::ImageGenerate`, `OutputKind::Image`와 함께 `src/models/registry.rs`의 `STANDALONE_ARCHITECTURE_FAMILIES`에 추가하여 `mlxcel arch --json`이 이를 나열하도록 합니다. 두 테이블이 불일치하면 그곳의 테스트가 실패합니다.
5. 이 페이지와 `docs/supported-models.md`를 갱신합니다.

probe가 인식하지만 `FAMILIES`에 없는 패밀리는 `image generation family <family> is not supported by this build`로 보고됩니다.

## Z-Image

`src/imagegen/z_image/`는 `mlx_vlm/models/z_image`(mlx-vlm `73d366ec`)를 포팅한 것입니다. 원본 Diffusers 트리(`Tongyi-MAI/Z-Image-Turbo`, `Tongyi-MAI/Z-Image`)와, index 메타데이터에 `mlx_vlm_format: "z_image"`가 있는 mlx-vlm 변환 트리를 바로 로드합니다. 변환 트리의 양자화된 transformer와 VAE 레이어는 메타데이터의 모드로 로드 시 dequantize됩니다. `affine`은 2, 3, 4, 5, 6, 8 bit와 group size 32, 64, 128, `mxfp4`(4, 32), `mxfp8`(8, 32), `nvfp4`(4, 16)이며, 그 밖의 조합은 오류입니다. 각 packed 텐서는 MLX가 보기 전에 검사도 거칩니다(uint32 weight, `[..., in / group_size]` shape의 `.scales`, `affine`에만 `.biases`, `mx` 및 `nvfp4` 모드에는 uint8 scale). 형식이 잘못된 텐서는 텐서의 키와 함께 로드가 실패합니다. 텍스트 encoder의 양자화는 자체 `config.json`에서 옵니다.

- **변형.** `scheduler/scheduler_config.json`의 `shift <= 3`이면 Turbo(8 step, guidance 0. guidance는 `[0, 1]`이어야 함), 아니면 base(50 step, guidance 4)입니다. upstream의 Turbo 자체 생성 기본값은 9 step이지만 이 포트는 이슈의 8을 따르며, 이는 upstream의 img2img 기본값이기도 합니다.
- **프롬프트.** `enable_thinking=True`인 chat template을 `<|endoftext|>`로 512 token까지 padding하며, condition은 실제 token으로 자른 Qwen3 encoder의 끝에서 두 번째 hidden state입니다. negative 프롬프트(요청 필드, 없으면 `negative_prompt` 옵션, 없으면 빈 값)는 guidance가 1보다 클 때만 인코딩됩니다.
- **Sampling.** `normal([1, 16, 1, H/8, W/8], f32, key(seed))`, `FlowMatchEulerDiscrete::static_shift(steps - 1, shift)`, `t = 1 - sigma`를 받는 transformer, 그리고 부정(negate)한 예측에 대한 Euler step. CFG `pred + g * (pred - neg)`는 `t <= cfg_truncation`인 동안 실행됩니다.
- **img2img**(`--task edit`, 정확히 하나의 `--image`). 이미지는 크기가 `--size`와 다를 때만 Lanczos로 resize되고, posterior mean으로 인코딩되며, `static_shift(steps, shift)`의 `start = int(max(steps - min(steps * strength, steps), 0))`에서 `(1 - sigma) * latents + sigma * noise`로 noise가 입혀집니다. step이 남지 않는 strength는 거부됩니다.
- **옵션**(`--gen-kwargs` 또는 추가 JSON 필드): `cfg_truncation`(기본값 1.0, `[0, 1]` 범위), `strength`(편집 전용, 기본값 0.6, `(0, 1]` 범위), `negative_prompt`.
- **메모리.** 텍스트 encoder, transformer, VAE가 상주합니다(bf16 체크포인트에서 약 20 GB). upstream은 transformer를 로드하기 전에 텍스트 encoder를 내리지만, 유지하면 서버가 8 GB를 다시 로드하지 않고 다음 요청에 응답할 수 있습니다.

### Z-Image 정밀도

upstream은 transformer와 VAE 가중치를 bf16으로 유지하고 f32 noise와 timestep을 입력하므로 MLX가 latent 스트림을 f32로 승격합니다. mlxcel의 MLX CUDA 빌드는 bf16과 f32를 bf16으로 승격하므로 그 스트림이 bf16으로 실행되며(Turbo transformer forward 한 번에서 상대 RMS 2.1e-2, 최종 latent에서 1.1e-1), 포트는 이 가중치를 f16으로 저장하며 이는 모든 백엔드에서 f32와 함께 f32로 승격됩니다. bf16에서 f16으로의 변환은 범위상 안전하지만 이 체크포인트들에서는 비트 단위로 정확하지 않습니다. 가장 큰 크기가 47.75(f16의 한계는 65504)이고 f16의 정규 범위(6.1e-5) 미만의 bf16 값은 낮은 비트를 잃으므로, 텐서의 약 3분의 2에서 어떤 원소가 바뀌며 텐서별 최악 오차는 상대 RMS 2.0e-7입니다. 로더는 모든 텐서의 f16 왕복을 측정하여 상대 RMS 1e-5를 넘는 텐서(또는 변환이 overflow하는 텐서)를 f32로 유지하며, 이는 정확하고 같은 방식으로 승격됩니다. VAE에서는 그런 텐서가 하나라도 있으면 VAE 전체가 f32가 됩니다. 따라서 f32 체크포인트는 f32로 유지됩니다. 두 개의 섬은 체크포인트의 bf16을 유지합니다. caption 스트림이 처음 f32를 만나기 전에 upstream이 bf16으로 계산하기 때문이며, caption embedder(RMSNorm과 `addmm`)와 첫 번째 context-refiner 블록의 pre-attention 경로입니다. 텍스트 encoder는 upstream처럼 bf16으로 실행됩니다. transformer와 VAE mid-block의 self-attention은 score 행렬을 `MLXCEL_ATTENTION_CHUNK_BUDGET_MB` 예산(기본값 1024 MiB, `0`이면 비활성화. [환경 변수](environment-variables.md) 참고)으로 query chunk화하므로 큰 크기에서도 전체 `[heads, L, L]` score를 만들지 않습니다.

### Z-Image 일치 여부

512x512, seed 42, 양쪽 모두 `MLX_ENABLE_TF32=0`으로 CUDA 백엔드(GB10)의 mlx 0.32.4에서 upstream mlx-vlm `73d366ec`과 비교한 seed 고정 실행입니다. 초기 noise는 모든 케이스에서 비트 단위로 동일합니다.

| 케이스 | 최종 latent 상대 RMS | PNG PSNR |
|--------|---------------------|----------|
| Turbo, 8 step, guidance 0 | 8.6e-2 | 32.9 dB |
| Base, 50 step, guidance 4, negative 프롬프트 `blurry, low quality`, `cfg_truncation` 1.0 | 6.2e-2 | 34.7 dB |
| Turbo img2img, strength 0.6 | 1.7e-2 | 46.1 dB |

최종 latent의 drift는 upstream 자체가 bf16 caption 스트림의 반올림 방식에 민감한 정도와 같은 크기입니다. Turbo 케이스에서 upstream을 자기 자신과 비교하면, caption embedder의 linear만 f32로 실행할 때 2.7e-2(41.3 dB), 텍스트 encoder의 attention만 f32 대신 bf16으로 실행할 때 1.0e-1(32.1 dB), TF32를 켰을 때 9.5e-2(31.6 dB) 움직입니다. 공유 Qwen3 conditioner는 upstream encoder가 q, k, v를 f32로 변환하는 곳에서 attention을 bf16으로 실행하며, 그 caption 특징은 upstream과 9.3e-3 차이가 나고, upstream의 caption 특징을 이 포트의 denoiser에 넣으면 Turbo drift가 3.4e-2로 낮아집니다. attention dtype이 그 격차는 아닙니다. upstream의 f32 attention을 쓰더라도 caption 특징은 여전히 upstream과 9.3e-3 차이가 납니다([Qwen3 conditioner](#qwen3-conditioner) 참고). 커밋된 요약 `tests/fixtures/imagegen/z_image_parity.safetensors`(noise 체크섬, 4x4 pooling한 최종 latent와 이미지, `tests/fixtures/generate_z_image_fixtures.py`로 재생성)가 `#[ignore]` 테스트를 뒷받침합니다.

```bash
MLX_ENABLE_TF32=0 <lib test binary> --ignored --test-threads=1 imagegen::z_image::tests::
```

`MLXCEL_Z_IMAGE_TURBO`와 `MLXCEL_Z_IMAGE_BASE`가 체크포인트를 가리킵니다(기본값 `models/mlx/z-image-turbo-bf16`, `models/mlx/z-image-bf16`). `MLXCEL_Z_IMAGE_ORACLE_DIR`은 전체 해상도 비교와 img2img 케이스를 추가합니다.

## Mage-Flow (`mage_flow`)

`src/imagegen/mage_flow/`는 `mlx_vlm/models/mage_flow`(mlx-vlm `73d366ec`)를 포팅한 것으로, Qwen3-VL conditioner, 3축 RoPE를 사용하는 dual-stream transformer, Mage-VAE(DiCo convolutional encoder와 CoD + DiCo denoiser decoder, 16배 downsampling, 128 latent 채널)로 구성됩니다. mlx-vlm 변환 레이아웃(`transformer/`, `vae/`, `text_encoder/`, `_class_name: "MageFlowPipeline"`인 `model_index.json`)을 로드합니다. mlx-vlm이 양자화한 transformer projection(index 메타데이터에서 읽는 `affine`, `mxfp4`, `mxfp8`, `nvfp4`)은 양자화 matmul로 실행되며, 양자화된 VAE는 거부됩니다. 텍스트 encoder는 자체 `config.json` 양자화와 함께 Qwen3-VL VLM 로더를 통해 로드됩니다.

| 변형 | 체크포인트 | Task | Step / guidance |
|------|-----------|------|-----------------|
| `mage-flow-turbo` | `mage-flow-community/Mage-Flow-Turbo` | generate | 4 / 1.0 |
| `mage-flow` | `mage-flow-community/Mage-Flow` | generate | 20 / 5.0 |
| `mage-flow-base` | `mage-flow-community/Mage-Flow-Base` | generate | 30 / 5.0 |
| `mage-flow-edit-turbo` | `mage-flow-community/Mage-Flow-Edit-Turbo` | edit | 4 / 1.0 |
| `mage-flow-edit` | `mage-flow-community/Mage-Flow-Edit` | edit | 30 / 5.0 |
| `mage-flow-edit-base` | `mage-flow-community/Mage-Flow-Edit-Base` | edit | 30 / 5.0 |

- **변형.** upstream의 `variant_from_local_path`처럼 디렉터리 이름에서 결정됩니다. `turbo` 또는 `base`(`edit`와 함께)는 해당 변형을, `edit`만 있으면 Mage-Flow-Edit를, 그 외에는 Mage-Flow를 선택합니다. upstream은 전체 경로 문자열을 매칭하지만, 이 포트는 `mage`를 언급하는 가장 가까운 경로 구성 요소(없으면 leaf)만 읽으므로, `/data/database/` 같은 상위 디렉터리가 그 아래 모든 체크포인트를 `-base` 변형으로 만들지 않습니다. 각 체크포인트는 하나의 task를 실행하며, 다른 task는 서버에서 `400`이고 CLI 오류는 변형을 명시합니다.
- **크기.** 너비와 높이는 `512..=2048`, 16의 배수. 크기가 없는 편집은 첫 번째 참조 이미지의 크기를 16의 배수로 내림해 사용하거나, `max_size`가 있으면 긴 변이 `max_size`가 되도록(짝수로 반올림) 스케일한 뒤 내림합니다. `max_size`는 요청 크기가 그 기본값일 때만 적용되며, upstream의 명시적 `width`와 `height`처럼 명시적 크기가 우선합니다. 공유 16 메가픽셀 검사를 넘는 참조 이미지는 `max_size`가 출력을 줄이기 전에 여전히 거부됩니다(mlxcel의 fail-closed 순서).
- **편집 token 한도(mlxcel의 경계이며 upstream에는 없음).** 편집은 transformer를 target과 참조당 하나의 grid에 대해 모두 출력 크기로 실행합니다. forward당 `(1 + N) * (H / 16) * (W / 16)`개의 이미지 token(CFG에서는 step당 두 번)입니다. 합계는 2048x2048 grid 네 개인 65536으로 제한됩니다. 2048x2048 출력은 참조 최대 3개, 1248x1248 출력은 최대 9개를 받습니다. 더 큰 편집은 계산된 개수와 한도를 명시하는 `400`입니다. 어떤 크기든 참조 한두 개인 일반적인 편집은 한도 아래입니다.
- **프롬프트.** upstream의 generation 또는 edit system template. conditioner의 최종 norm hidden state에서 template 접두사를 버리며(generation은 34 위치, edit는 64 위치), 프롬프트 token은 최대 2048개입니다. 편집은 참조마다 `Image i: <|vision_start|><|image_pad|><|vision_end|>`를 접두사로 붙이고, 각 참조는 긴 변이 `vl_cond_long_edge`가 되도록(bicubic, 축소만) resize한 뒤 Qwen3-VL 규칙(factor 32)으로 크기가 정해집니다. 2048 token을 넘는 편집 지시는 truncate되는 대신 `400`입니다. 텍스트 encoder는 upstream처럼 매 인코딩 후 evict되며, transformer와 VAE는 첫 요청에서 로드되어 상주합니다.
- **Guidance.** CFG는 guidance 1 초과에서 실행됩니다. 생성은 negative 프롬프트(요청 필드, 없으면 `negative_prompt` 옵션, 없으면 `" "`)를 인코딩하며, 빈 negative 프롬프트는 CFG를 끕니다. 편집은 항상 `negative or " "`를 사용합니다. `renormalization`은 upstream처럼 각 token의 guided velocity를 조건부 velocity의 norm으로 다시 스케일합니다.
- **Sampling.** 초기 noise `normal([1, H/16 * W/16, 128], bf16, key(seed))`는 upstream과 비트 단위로 동일하며, shift 6의 `FlowMatchEulerDiscrete::static_shift`를 사용합니다. 편집은 각 참조(출력 크기로 resize, PIL 동등 bicubic)를 VAE posterior로 인코딩하는데 `sample_posterior`가 false가 아니면 `key(seed)`로 샘플링하며, 참조 token을 target token 뒤에 덧붙입니다.
- **옵션**(`--gen-kwargs` 또는 추가 JSON 필드): `negative_prompt`, `static_shift`(양수, 기본값 6), `renormalization`(기본값 false). 편집 전용: `max_size`, `vl_cond_long_edge`(기본값 384, 최대 2048. `null` 또는 `<= 0`이면 참조 크기를 유지하며 참조당 2048x2048 픽셀로 제한), `sample_posterior`(기본값 true). generate 요청의 편집 전용 옵션은 `400`입니다.

### Mage-Flow 정밀도

upstream은 포트가 유지하는 f32 섬을 제외하고 bf16으로 끝까지 실행합니다. transformer timestep 임베딩(bf16으로 반올림한 주파수), rotary 연산, VAE의 local-attention score와 softmax입니다. upstream과의 두 가지 차이는 의도적입니다.

- **LayerNorm.** MLX의 fused CUDA `layer_norm` 커널(`backend/cuda/layer_norm.cu`, mlx 0.32.4에도 있음)은 평균과 분산의 block reduction에 하나의 shared-memory 버퍼를 재사용하며 둘 사이에 barrier가 없습니다. 384 폭의 bf16 행, 즉 VAE encoder의 `norm_out`에서 호출마다 몇몇 행이 잘못 나오므로, 동일한 편집 실행 두 번이 upstream과 이 포트 모두에서 서로 달랐습니다(참조 latent가 약 1e-1, 최종 PNG는 서로 29.8에서 35.5 dB). 포트는 모든 Mage-Flow LayerNorm을 같은 f32 통계로 별도의 연산들로부터 계산하며 이는 결정적입니다. fixture 생성기는 upstream의 transformer와 VAE에 같은 형태를 적용합니다. 다른 패밀리는 여전히 fused 커널을 호출합니다.
- **공간 평균.** VAE의 channel attention은 공간 평균을 취합니다. oracle의 CUDA `mx.mean`은 bf16에서 bf16으로 누산하지만 포트는 f32로 누산한 뒤 한 번 변환합니다. 이는 마지막 denoiser 블록에서의 유일한 차이입니다(같은 입력에서 9.8e-3).

### Mage-Flow 일치 여부

512x512, seed 42, 4 step, 양쪽 모두 `MLX_ENABLE_TF32=0`으로 CUDA 백엔드(GB10)의 mlx 0.32.4에서 upstream mlx-vlm `73d366ec`과 비교한 seed 고정 실행입니다. 초기 noise는 비트 단위로 동일합니다.

| 케이스 | 최종 latent 상대 RMS | PNG PSNR | upstream의 conditioning에서 |
|--------|---------------------|----------|----------------------------|
| Turbo, generate | 6.6e-2 | 38.4 dB | 3.4e-2, 42.3 dB |
| Edit-Turbo, 참조 두 개 | 2.5e-1 | 21.8 dB | 5.1e-2, 39.9 dB |

편집의 격차는 denoiser나 VAE가 아니라 conditioner에 있습니다. upstream 자체의 conditioner 출력을 넣으면 편집 실행은 39.9 dB입니다. bf16 Qwen3-VL은 upstream의 bf16 hidden state를 가깝게 재현할 수 없습니다. 같은 입력의 decoder 레이어 하나가 이미 이 빌드의 MLX와 mlx 0.32.4 사이에서 5.7e-3 차이가 나고, upstream 자신의 bf16 hidden state도 같은 모델을 f32로 확장한 것과 1.8e-2(generate), 2.6e-1(edit, 24층 vision tower와 DeepStack 경유) 떨어져 있습니다. upstream은 conditioner만 f32로 실행하면(위의 결정적 LayerNorm을 양쪽 실행에 적용) 자기 자신의 edit PNG와 22.9 dB에 도달합니다. 따라서 conditioner 테스트는 입력(token id, 이미지 grid, M-RoPE 위치)을 정확히 검사하고, 포트의 hidden state가 f32 참조에서 upstream의 bf16 실행보다 더 멀지 않을 것을 요구합니다(측정 비율 0.96과 1.04, 경계 1.2). VAE는 upstream 자체 입력에서 단계별로 5e-3으로 게이트하고(CoD decoder, posterior head, pixel head는 정확. 마지막 denoiser 블록은 위의 공간 평균 누산 때문에 2e-2), 엔드투엔드로는 5e-2로 게이트합니다(측정 decode 1.0e-2, encode 평균 2.4e-2).

`tests/fixtures/imagegen/mage_flow_parity.safetensors`(token id, M-RoPE 위치, noise 체크섬, 4x4 pooling한 최종 latent와 이미지, bf16과 f32의 conditioner 출력의 64채널 pooling)와 `mage_flow_tiny_refs.safetensors`(RoPE 테이블, f32와 bf16의 작은 transformer, 실제 Mage-VAE 단계 참조)는 `tests/fixtures/generate_mage_flow_fixtures.py`로 재생성됩니다. 실제 체크포인트 테스트는 `#[ignore]`입니다.

```bash
MLX_ENABLE_TF32=0 <lib test binary> --include-ignored --test-threads=1 imagegen::mage_flow::
```

이 테스트는 `models/mlx/mage-flow-turbo`와 `models/mlx/mage-flow-edit-turbo`를 읽습니다. `MLXCEL_MAGE_FLOW_ORACLE_DIR`(생성기의 `--out` 디렉터리)은 전체 해상도 PNG 및 latent 비교, upstream의 conditioning에서의 실행, processor의 픽셀 값(PIL의 bicubic과 8-bit 두 단계 이내)을 추가합니다.

## Qwen-Image (`qwen_image`)

`src/imagegen/qwen_image/`는 `mlx_vlm/models/qwen_image`(mlx-vlm `73d366ec`)를 포팅했으며 `Qwen/Qwen-Image-2.1`의 원본 Diffusers 트리(`_class_name: "QwenImage21Pipeline"`인 `model_index.json`, `transformer/`, `vae/`, `text_encoder/`, `processor/`, `scheduler/`)를 로드합니다. 구성 요소: Qwen3-VL conditioner(mlxcel의 Qwen3-VL 로더와 `Qwen3VLModel::forward_hidden_pre_norm`의 최종 norm 이전 hidden state), 3축 RoPE(축 16, 56, 56)와 block-causal mask를 사용하는 32블록 single-stream transformer, Qwen-Image VAE(16배 공간 downsampling, 64 latent 채널). MLX affine으로 양자화된 transformer linear(`transformer/config.json`의 `quantization`)는 양자화 matmul로 실행됩니다.

- **크기.** 너비와 높이는 `256..=2048`, 16의 배수(명시적 편집 크기는 32). 자체 크기가 없는 편집은 마지막으로 resize된 참조의 크기입니다.
- **참조.** 각 참조는 RGBA로 열려 Pillow의 Lanczos(Pillow 13과 비트 단위로 일치)로, 자신의 종횡비에서 `output_resolution**2` 면적이 되도록 32의 배수로 반올림해 resize됩니다. conditioner는 흰색 위에 합성된 것을 보고, VAE는 4개 채널을 모두 인코딩하므로 편집 출력은 RGBA이며 참조의 투명도를 유지합니다. 생성 출력은 RGB입니다.
- **편집 한도(mlxcel의 경계이며 upstream에는 없음).** target과 모든 참조를 합쳐 이미지 token은 최대 32768개입니다(2048x2048 grid 두 개. 2048 편집은 2048 참조 하나, 기본 1024 편집은 최대 7개). 더 큰 편집은 개수를 명시하는 `400`입니다. 이 한도가 Mage-Flow의 절반인 이유는 block-causal mask가 `S x S`이고 편집 KV cache가 CFG 분기당 cache된 token마다 약 512 KiB를 보유하기 때문입니다(한도에서 약 17 GB). Qwen3-VL processor가 다시 resize할 resize된 참조(200:1 초과이거나 픽셀 범위 밖)와, vision 마커가 있는 편집 프롬프트 또는 CFG negative는 어떤 컴포넌트도 로드하기 전에 `400`입니다.
- **프롬프트.** upstream의 system template. hidden state는 14 token의 template 접두사를 버리고 최대 1024개의 프롬프트 token을 유지합니다(생성은 upstream처럼 truncate하고, 한도를 넘는 편집은 `400`). 편집 프롬프트는 참조마다 `<imageN><|vision_start|><|image_pad|><|vision_end|>`를 배치하며, vision 마커 자체를 포함하는 사용자 프롬프트는 `400`입니다. 텍스트 encoder는 upstream처럼 매 인코딩 후 evict되고(MLX cache를 비움) transformer와 VAE는 첫 요청에서 로드되어 상주합니다.
- **Conditioning.** target token은 실제 timestep의 modulation을 받고, 텍스트와 참조 token은 timestep 0 행(`causal_condition`)을 받으며, attention은 block-causal입니다. 텍스트와 참조는 target을 절대 attend하지 않습니다. `use_kv_cache`(기본값)가 켜지면 편집은 첫 step에서 참조의 K/V를 캐시하고 이후 step은 target만 실행하며, 결과는 캐시 없는 실행과 같습니다(upstream에서는 비트 단위, 이 포트에서는 4 step에서).
- **Guidance.** CFG(`neg + g * (pos - neg)`)는 guidance 1 초과에서 실행되며 negative 프롬프트(요청 필드, 없으면 `negative_prompt` 옵션, 없으면 `" "`)를 인코딩합니다. 기본값은 guidance 1을 사용하므로 step당 한 번의 pass입니다.
- **Sampling.** 초기 noise는 `mx.random.seed(seed)` 다음 `normal([1, 1, 64, H/16, W/16], f32)`를 bf16으로 변환한 것으로 upstream과 비트 단위로 동일합니다. 스케줄러의 shift 파라미터로 `FlowMatchEulerDiscrete::dynamic`을 사용하고 Euler step은 f32로 누산합니다.
- **옵션**(`--gen-kwargs` 또는 추가 JSON 필드): `negative_prompt`. 편집 전용: `output_resolution`(256에서 2048, 기본값 1024)과 `use_kv_cache`(기본값 true). generate 요청의 편집 전용 옵션은 `400`입니다.

### Qwen-Image 일치 여부

1024x1024, seed 7, 4 step, 양쪽 모두 `MLX_ENABLE_TF32=0`으로 CUDA 백엔드(GB10)의 mlx 0.32.4에서 bf16 transformer로 upstream mlx-vlm `73d366ec`과 비교한 seed 고정 실행입니다. 초기 noise는 비트 단위로 동일합니다. 두 최종 latent drift 모두 이슈가 정한 1e-2 미만이며, 게이트는 PSNR 30 dB 이상에 8x8 pooling한 latent에 대한 케이스별 경계를 더한 것입니다.

| 케이스 | 최종 latent 상대 RMS(drift) | PNG PSNR |
|--------|----------------------------|----------|
| Generate | 7.6e-3 | 52.22 dB |
| Edit, RGBA 참조, KV cache | 6.4e-3 | 52.61 dB(alpha 53.58 dB) |
| KV cache 없는 edit, cache된 실행과 비교 | 동일 | inf(CLI로 40 step에서도 비트 단위로 동일) |

conditioner의 텍스트 위치는 upstream과 상대 RMS 3.7e-2 떨어져 있고 위치 전반에 고르게 퍼져 있습니다. `<|image_pad|>` 행은 더 많이 다르지만(0.58) transformer가 이를 참조의 VAE token으로 교체합니다. 참조 latent는 1.7e-3이고, 실제 VAE는 고정 latent를 2.1e-4로 디코딩하고 5.7e-7로 인코딩합니다. latent 경계(2e-2)는 generate 케이스에서 측정한 망가진 경우들보다 훨씬 낮습니다. RoPE의 높이와 너비를 맞바꾸면 0.77(13.2 dB), 모든 token을 실제 timestep의 modulation 행에 두면 0.23(18.3 dB)입니다. LayerNorm은 Mage-Flow가 사용하는 조합된 형태를 사용합니다([Mage-Flow 정밀도](#mage-flow-정밀도) 참고). `qwen3_vl.rs`의 변경은 const-generic forward에 메서드 하나를 추가한 것이며, 이전과 이후의 teacher-forced logit trace(`examples/logit_trace`, 텍스트 및 이미지 arm)는 바이트 단위로 동일합니다.

`tests/fixtures/imagegen/qwen_image_parity.safetensors`(token id, `drop_idx`, noise 체크섬, 8x8 pooling한 latent와 32x32 pooling한 이미지)와 `qwen_image_tiny_refs.safetensors`(KV cache가 있는 경우와 없는 경우의 작은 transformer, RoPE 테이블, 작은 residual 및 plain VAE, 실제 VAE 참조)는 `tests/fixtures/generate_qwen_image_fixtures.py`로 재생성됩니다. 실제 체크포인트 테스트는 `#[ignore]`입니다.

```bash
MLX_ENABLE_TF32=0 <lib test binary> --include-ignored --test-threads=1 imagegen::qwen_image::
```

이 테스트는 `models/mlx/qwen-image-2.1`을 읽습니다. `MLXCEL_QWEN_IMAGE_ORACLE_DIR`(생성기의 `--out` 디렉터리)은 전체 해상도 PNG 및 latent 비교와 편집 참조 이미지를 추가하며, `MLXCEL_QWEN_IMAGE_OUT`은 PNG와 conditioner 출력을 저장합니다.

## Ideogram 4 (`ideogram4`)

`src/imagegen/ideogram4/`는 `ideogram-ai/ideogram-4-fp8` snapshot을 위해 `mlx_vlm/models/ideogram4`(mlx-vlm `73d366ec`)를 포팅한 것입니다. 체크포인트는 라이선스가 게이트되어 있습니다(Ideogram 4 Non-Commercial License). Hugging Face에서 약관에 동의하고 직접 다운로드하십시오. mlxcel은 이를 절대 가져오지 않습니다. snapshot 레이아웃은 upstream의 것입니다(`_class_name: "Ideogram4Pipeline"`인 `model_index.json`, `tokenizer/`, `transformer/`, `unconditional_transformer/`, `text_encoder/`, `vae/`). 없는 파일은 로드 시 명시됩니다. probe는 snapshot이 함께 가진 FLUX.2 VAE 클래스보다 앞서 파이프라인 클래스 이름에서 `ideogram4`를 선택합니다.

모델은 34층 single-stream DiT(256 크기의 18 head, 이미지 위치가 65536만큼 offset된 interleaved 3축 M-RoPE)이며, 공유 [Qwen3 conditioner](#qwen3-conditioner)를 통해 Qwen3-VL 8B 텍스트 tower의 13개 hidden state(레이어 0, 3, ..., 33, 35)를 hidden 우선으로 쌓아 token당 53248개 특징으로 conditioning합니다. 두 번째 unconditional transformer가 classifier-free guidance 분기를 제공합니다. latent는 2x2로 patchify한 128채널 FLUX.2 VAE latent이며 고정된 채널별 shift와 scale로 정규화됩니다. 생성 전용입니다. `/v1/images/edits`는 task를 명시하는 `400`이고 CLI는 참조 이미지를 거부합니다.

- **가중치.** Linear 가중치는 f32 행별 `weight_scale`을 가진 F8_E4M3 바이트로 제공됩니다. upstream처럼 로드 시 bf16으로 dequantize됩니다(E4M3을 256개 항목 테이블로 디코딩하고 bf16으로 반올림한 뒤 bf16으로 반올림한 scale을 곱함). 따라서 두 transformer와 텍스트 tower는 upstream처럼 bf16으로 실행됩니다. VAE(디스크에서 bf16)는 `Flux2Vae::default_dtype`으로 `Flux2Vae`를 통해 로드됩니다. `unconditional_transformer`는 텍스트를 읽지 않으므로 그 `llm_cond_*` 가중치는 읽지도 dequantize하지도 않으며, conditional transformer는 이를 가져야 합니다. transformer config는 범위를 검사하고(`num_layers`는 1에서 4096, overflow를 검사한 head와 M-RoPE 크기) 모든 가중치의 shape과 dtype을 로드 시 검사하므로, shape이 잘못된 텐서나 아직 원시 FP8 바이트인 텐서는 이름이 명시된 로드 오류가 됩니다.
- **메모리.** 텍스트 encoder(bf16에서 약 16 GB)는 요청마다 로드되어 transformer가 실행되기 전에 MLX cache를 비우며 내려집니다. 두 transformer(bf16에서 각각 약 18.6 GB)와 VAE는 첫 요청에서 로드되어 상주하므로, 가중치 크기로 볼 때 서빙 중인 모델은 요청 사이에 약 38 GB를, 이후 요청이 프롬프트를 인코딩하는 동안 약 54 GB를 보유합니다(실측 피크가 아니라 텐서 크기에서 계산).
- **크기.** 너비와 높이는 `256..=2048`, 16의 배수, 종횡비는 최대 6:1(upstream의 `validate_dimensions` 메시지). 래핑 후 프롬프트는 최대 2048 token입니다.
- **프롬프트.** `{`로 시작하는 프롬프트는 JSON caption이며 그대로 전달되고, 잘못된 JSON은 `400`입니다. 평문은 upstream의 최소 caption(`high_level_description`, background placeholder 하나, element 하나)으로 래핑되며 래핑된 caption이 `revised_prompt`로 반환됩니다. `auto_json_caption: false`는 텍스트를 그대로 보내고 `revised_prompt`는 `null`로 유지됩니다. tokenizer는 upstream처럼 snapshot의 chat template을 하나의 user 메시지에 적용합니다. 1 MiB(`MAX_PROMPT_BYTES`)를 넘는 프롬프트는 토큰화 전에 `400`이고, 2048 token을 넘으면 upstream의 메시지와 함께 `400`입니다. Caption 스키마 경고는 최대 64개까지 유지되고 최대 8개까지 로그에 남으며(로그 줄이 나머지를 셈), 한도를 넘어 버려진 경고도 caption을 구조화되지 않은 것으로 표시합니다.
- **Sampling.** 초기 noise `mx.random.seed(seed); normal([1, H/16 * W/16, 128], f32)`는 `key(seed)`가 아니라 `split(key(seed))`의 행 1이며 upstream과 비트 단위로 동일합니다. 스케줄은 해상도 shift가 있는 log-SNR 위의 logit-normal이며, step 루프, guidance blend, Euler 갱신은 upstream의 `generate_array`를 따릅니다.
- **프리셋**(`sampler_preset`): `V4_DEFAULT_20`(기본값: 20 step, guidance 7이며 마지막 두 denoising step은 3, `mu` 0, `std` 1.75), `V4_QUALITY_48`(48 step, `std` 1.5), `V4_TURBO_12`(12 step, `mu` 0.5, `std` 1.75). 명시적 `guidance`(7과 같더라도)나 프리셋의 step 수와 다른 `steps`는 upstream처럼 프리셋의 guidance 스케줄을 상수 guidance(주어지지 않으면 7)로 대체하며, `guidance_schedule`은 대신 step별 스케줄을 지정합니다.
- **옵션**(`--gen-kwargs` 또는 추가 JSON 필드): `sampler_preset`, `auto_json_caption`(기본값 true), `guidance_schedule`(step당 숫자 하나, 최대 step 한도 개수. upstream처럼 항목 `i`는 루프 인덱스 `i`에 적용되고 루프가 거꾸로 세므로 목록은 끝에서부터 소비됨), `mu`, `std`(양수). `prompt_expansion_model`(프롬프트를 다시 쓰는 upstream의 선택적 두 번째 모델)은 mlxcel이 실행하지 않으므로 `400`입니다. 알 수 없는 프리셋은 유효한 이름을 나열합니다. 비어 있지 않은 `negative_prompt`는 `400`입니다. upstream의 unconditional 분기는 텍스트를 받지 않아 그렇지 않으면 조용히 버려지기 때문입니다. `n`개의 이미지는 seed `seed + i`를 사용합니다(공유 규약). upstream의 `num_images`는 대신 `seed`에서 하나의 배치 noise 텐서를 뽑습니다.

### Ideogram 4 정밀도

upstream은 f32 섬이 있는 bf16으로 실행하며 포트는 각각을 유지합니다. 행별 FP8을 bf16으로 dequantize, RMSNorm 평균을 f32로 취한 뒤 `rsqrt` 전에 bf16으로 반올림(fused 커널이 아닌 별도 연산으로 구성), 파라미터 없는 LayerNorm은 f32, M-RoPE 테이블은 f32로 계산해 bf16으로 변환, attention score와 softmax는 f32, timestep은 f32 sinusoid 전에 bf16으로 반올림, f32 출력 head, f32 latent 스트림과 스케줄러 연산입니다. bias가 있는 projection은 MLX `nn.Linear`처럼 하나의 fused `addmm`을 사용합니다. f16 정책은 transformer나 텍스트 tower에 적용되지 않습니다. 가중치가 bf16이 아니라 FP8로 들어오며 upstream의 bf16 dequantization이 참조의 일부이기 때문입니다.

### Ideogram 4 일치 여부

양쪽 모두 `MLX_ENABLE_TF32=0`으로 CUDA 백엔드(NVIDIA GB10)의 mlx 0.32.4에서 upstream mlx-vlm `73d366ec`과 비교한 seed 고정 실행입니다. 초기 noise는 비트 단위로 동일합니다(f32 바이트에 대한 FNV-1a).

| 케이스 | 최종 latent 상대 RMS(drift) | PNG PSNR | upstream의 텍스트 특징에서 |
|--------|----------------------------|----------|---------------------------|
| `V4_TURBO_12`, 512x512, seed 0, 빨간 사각형 JSON caption(게이트됨) | 6.2e-2 | 37.1 dB | 2.9e-2, 46.0 dB |
| `V4_DEFAULT_20`, 1024x1024, seed 0, 같은 caption(CLI 실행) | 기록하지 않음 | 47.9 dB | |

같은 noise에서 첫 step의 velocity는 6.2e-3(conditional)과 3.2e-3(unconditional, 텍스트 미관여) 차이가 나며 텍스트 특징은 6.4e-2 차이가 납니다. 엔드투엔드 drift의 대부분은 공유 conditioner가 비트 단위로 재현하지 못하는 bf16 텍스트 tower에서 옵니다. tower의 attention을 upstream처럼 f32로 실행해도 텍스트 특징은 6.4e-2로 유지됩니다([Qwen3 conditioner](#qwen3-conditioner) 참고). 테스트는 pooling한 최종 latent(상대 RMS 4.5e-2 이하, 측정 3.2e-2), 8x8 pooling한 이미지(38 dB 이상, 측정 43.7 dB), 그리고 oracle 디렉터리가 있으면 전체 PNG(30 dB 이상)를 게이트합니다. 두 개의 upstream 대조가 경계가 거르는 것을 보여 줍니다. 잘못된 noise(seed 1)는 pooling한 latent 4.2e-1과 PNG 12.3 dB를 기록하고, 프리셋의 guidance 스케줄을 빼면(마지막 step에서 3 대신 guidance 7) pooling한 latent 5.2e-2이지만 PNG 48.1 dB이므로 latent 경계만 이를 잡아냅니다. 무작위 bf16 가중치를 가진 작은 2층 transformer는 upstream과 비트 단위로 일치합니다(상대 RMS 0).

일부 프롬프트와 seed에서 이 체크포인트는 "Image blocked by safety filter"라고 적힌 회색 또는 체크무늬 placeholder를 렌더링합니다(1024x1024 빨간 사각형 실행과 seed 1의 래핑된 평문 "red fox" 프롬프트). upstream도 같은 이미지를 만들므로(47.9 dB와 51.3 dB) 이는 mlxcel 필터가 아니라 체크포인트의 출력입니다.

`tests/fixtures/imagegen/ideogram4_parity.safetensors`(token id, noise 체크섬과 앞부분, 4x4 token으로 pooling한 최종 latent, 8x8 pooling한 이미지)와 `ideogram4_tiny_refs.safetensors`(작은 transformer의 가중치, 입력, M-RoPE 테이블, 출력)는 `tests/fixtures/generate_ideogram4_fixtures.py`로 재생성됩니다(두 번째는 `--tiny PATH`). 실제 체크포인트 테스트는 `#[ignore]`이며 `models/mlx/ideogram-4-fp8`이 없으면 건너뜁니다.

```bash
MLX_ENABLE_TF32=0 <lib test binary> --ignored --test-threads=1 imagegen::ideogram4::parity_tests
```

`MLXCEL_IDEOGRAM4_ORACLE_DIR`(생성기의 `--out` 디렉터리)은 전체 해상도 PNG 및 latent 비교, 텍스트 특징과 첫 step 진단, upstream의 텍스트 특징에서의 실행을 추가하며, `MLXCEL_IDEOGRAM4_OUT`은 PNG를 저장합니다.

## Ming-Image (`ming_image`)

`src/imagegen/ming_image/`는 출력이 RGBA인 text-to-image 모델 Ming-Image-0.1-Design(inclusionAI)을 위해 `mlx_vlm/models/ming_image`(mlx-vlm `73d366ec`)를 포팅한 것입니다. 원본 트리(`inclusionAI/Ming-Image-0.1-Design`: `mllm/`, `connector/`, `mlp/`, `transformer/`, `vae/`)와 MLX 변환본(`nativ-community/Ming-Image-0.1-Design-MLX-*`. conditioner를 `text_encoder/`에 병합하고 양자화된 각 컴포넌트의 `config.json`에 affine `quantization` 블록이 있음)을 로드합니다.

- **Conditioner.** Ming-flash-omni의 BailingMoeV2 backbone(20층, causal)이 upstream의 chat template 뒤에 `<image_start>`, 256개의 학습된 query token, `<image_end>`가 이어지는 입력에 대해 실행됩니다. query token은 token별 `image_gate` router(`src/models/bailing_moe_image_router.rs`, 가산적: 텍스트 router와 다른 모든 `bailing_moe` 경로는 변경 없음)를 통해 라우팅됩니다. 최종 normed query state는 `proj_in`, f32의 28층 양방향 Qwen2 connector, `proj_out`을 거쳐 `cap_feats`가 되고, 레이어 5, 12, 20의 텍스트 token을 이어 붙여 RMS-norm한 것은 `directvlm` projection을 거쳐 `cap_feats_2`가 됩니다. `mllm/`의 vision tower, audio gate, 출력 head는 읽지 않습니다.
- **Denoiser와 VAE.** [`dit_blocks`](#공유-구성-요소)의 Z-Image NextDiT 블록에 두 번째 caption 스트림을 이어 붙이고 zero padding을 attention에서 mask합니다. 그리고 4개 채널을 디코딩하는 plain 레이아웃의 공유 [Qwen-Image VAE](#qwen-image-qwen_image)를 사용합니다(원본 트리의 5차원 causal kernel은 마지막 temporal slice로 줄임). 모든 이미지는 RGBA이며 PNG는 CLI와 서버 모두에서 alpha 채널을 유지합니다.
- **메모리.** bf16 backbone은 약 33 GB입니다. 텍스트 encoder는 요청마다 로드되어 transformer와 VAE가 실행되기 전에 내려지며, 그 둘은 첫 요청에서 로드되어 상주합니다. 체크포인트는 디스크에서 50 GB(bf16), 24 GB(8-bit), 13 GB(4-bit)입니다.
- **요청.** 생성 전용입니다. 편집은 task를 명시하는 `400`입니다. 너비와 높이는 `256..=2048`, 16의 배수(Qwen-Image의 `validate_dimensions`). 기본값은 guidance 1.0에서 12 step이며, 모델은 CFG를 끄고 학습되었으므로 다른 guidance는 `400`입니다(`Ming-Image-0.1-Design is trained for guidance 1.0 (CFG off), got 3`). `n`개 중 이미지 `i`는 seed `seed + i`를 사용합니다.
- **Sampling.** 초기 noise `normal([1, 16, 1, H/8, W/8], f32, key(seed))`는 upstream과 비트 단위로 동일하며, 체크포인트의 `scheduler_shift`로 `FlowMatchEulerDiscrete::dynamic`을 사용합니다. latent는 디코딩 전에 VAE의 scaling factor(8.0064)와 shift로 unscale됩니다.

### Ming-Image 정밀도

upstream은 backbone, DiT, VAE를 bf16으로, connector와 bridge는 저장된 f32로 실행합니다. 포트는 이 dtype을 유지합니다. 이 빌드에서는 두 곳에 명시적 변환이 필요합니다. mlxcel의 MLX CUDA 빌드는 upstream이 f32로 승격하는 곳에서 bf16과 f32를 bf16으로 승격하므로, bf16 backbone state는 `proj_in`과 `directvlm` norm 전에 f32로 확장되며, f32 latent 스트림을 읽는 transformer 텐서는 f16(변환이 정확하지 않으면 f32)로 저장되고 bf16 activation을 읽는 텐서는 bf16으로 유지됩니다. 양자화된 transformer는 로드 시 dequantize되고(scale이 bf16이므로 dense 가중치는 bf16), 양자화된 conditioner는 양자화 matmul로 실행됩니다.

### Ming-Image 일치 여부

1024x1024, 12 step, seed 0, 양쪽 모두 `MLX_ENABLE_TF32=0`으로 CUDA 백엔드(NVIDIA GB10)의 mlx 0.32.4에서 upstream mlx-vlm `73d366ec`과 비교한 seed 고정 실행입니다. 초기 noise는 비트 단위로 동일하고(f32 바이트에 대한 FNV-1a) template이 적용된 id도 동일합니다.

| 케이스 | pooling한 최종 latent 상대 RMS | PNG PSNR | Alpha PSNR | upstream의 conditioning에서 |
|--------|-------------------------------|----------|------------|----------------------------|
| `gen`: 포스터 프롬프트, bf16 | 3.37e-1 | 24.98 dB(기록) | 65.81 dB | 3.90e-2, 32.45 dB |
| `rgba`: 투명 배경 사과 아이콘, bf16 | 3.03e-2 | 30.67 dB | 72.16 dB | 1.38e-3, 52.69 dB, alpha 82.02 dB |
| `q4`: 포스터 프롬프트, 같은 변환본에 대한 upstream과 비교한 4-bit 변환본(기록) | 4.44e-2 | 41.91 dB | 75.97 dB | |

upstream 자체 입력에서의 단계별 결과: 첫 step의 velocity는 4.2e-3(`gen`)과 2.6e-3(`rgba`), 4-bit 변환본은 2.3e-3이고, upstream의 최종 latent를 디코딩하는 VAE는 97.7 dB와 96.8 dB(4-bit에서는 100.0 dB)입니다. 같은 id에서의 conditioner: `cap_feats` 2.05e-1, `cap_feats_2` 5.5e-2와 7.0e-2, 텍스트 state 1.2e-2와 6.2e-3.

**`gen` 예외.** 포스터 케이스는 문서화된 conditioner 정밀도 예외이며, Mage-Flow의 edit 케이스와 마찬가지로 메인테이너가 수용했습니다. upstream 자체 입력에서 backbone 레이어 하나하나는 upstream과 1e-4에서 7.5e-4(반올림 수준) 차이가 나며 최종 norm은 단독으로 비트 단위로 정확하지만, 20개 MoE 레이어를 거치며 위의 0.2 query-span 차이로 누적됩니다. 이 프롬프트는 그 수준에서 혼돈적입니다. 같은 drift(query state를 1e-3 offset)로 upstream을 자기 자신에 대해 교란하면 포스터 프롬프트에서는 자기 PNG와 16.37 dB, RGBA 프롬프트에서는 29.98 dB에 도달합니다. upstream의 conditioning을 넣으면 포트는 32.45 dB이므로 denoiser와 VAE가 격차의 원인은 아닙니다. 따라서 `gen` 게이트는 upstream의 conditioning에서의 실행(30 dB 이상, pooling한 latent 8e-2 이내. `cap_feats_2`를 빼면 7.73e-1로 측정됨)에 conditioner 경계(`cap_feats` 최대 0.3, `cap_feats_2` 0.12, 텍스트 state 3e-2)를 더한 것이며, query token에 텍스트 router를 쓰는 conditioner는 이를 넘습니다(`cap_feats` 4.59e-1). 엔드투엔드 PNG는 기록하고, pooling한 latent에는 느슨한 sanity 경계(0.6)만 둡니다.

**Alpha.** `rgba` 케이스는 엔드투엔드로 게이트됩니다. PNG 30 dB 이상, alpha 채널 30 dB 이상, pooling한 latent 6e-2 이내. 시도한 두 RGBA 프롬프트(사과 아이콘과 RGBA 포스터) 모두에서 upstream 자체의 출력은 사실상 불투명하며 bf16과 4-bit 모두 마찬가지입니다. alpha가 16 미만인 픽셀이 없고 최소 alpha는 227에서 252입니다. 따라서 게이트는 배경이 투명한지가 아니라, 포트가 실제 alpha 채널을 기록하고 그것이 upstream과 일치하는지입니다.

`tests/fixtures/imagegen/ming_image_parity.safetensors`(template이 적용된 id, noise 체크섬과 앞부분, 8x8 cell로 pooling한 최종 latent, 32x32 픽셀로 pooling한 RGBA 이미지)와 `ming_image_tiny_refs.safetensors`(`cap_feats_2`가 있을 때와 없을 때, 작은 transformer의 출력과 upstream의 출력)는 `tests/fixtures/generate_ming_image_fixtures.py`로 재생성됩니다. 실제 체크포인트 테스트는 `#[ignore]`입니다.

```bash
MLX_ENABLE_TF32=0 <lib test binary> --ignored --test-threads=1 imagegen::ming_image::parity_tests
```

이 테스트는 `models/mlx/ming-image-0.1-design`과 `models/mlx/ming-image-0.1-design-mlx-4bit`를 읽습니다. `MLXCEL_MING_IMAGE_ORACLE_DIR`(생성기의 `--out` 디렉터리)은 전체 해상도 PNG, alpha, latent 비교, conditioner 및 레이어별 진단, 잘못된 router arm, upstream의 conditioning에서의 실행을 추가하며, `MLXCEL_MING_IMAGE_OUT`은 PNG를 저장합니다.

## LLaDA-Image (`llada_image`)

`src/imagegen/llada_image/`는 bf16 Diffusers 레이아웃(`_class_name: "LLaDAImagePipeline"`인 `model_index.json`, `text_encoder/`, `queryformer/`, `text_projection/`, `sigvq/`, `transformer/`, `vae/`, `scheduler/`, `tokenizer/`)의 `inclusionAI/LLaDA-Image-Turbo`와 `inclusionAI/LLaDA-Image`를 위해 `mlx_vlm/models/llada_image`(mlx-vlm `73d366ec`)를 포팅한 것입니다. probe는 파이프라인 클래스 이름에서 `llada_image`를 선택합니다. 없는 컴포넌트는 로드 시 명시되며, 모든 가중치의 shape과 dtype은 해당 config에 대해 검사됩니다.

프롬프트는 단어 임베딩 뒤에 QueryFormer query가 이어지는 입력에 대해 트리 내 LLaDA-2 MoE decoder(`Llada2MoeModel`, 20층, expert 256개, token당 8개)가 인코딩하며, 최종 hidden state의 QueryFormer 위치는 text projection을 거쳐 2560개의 caption 특징이 됩니다. denoiser는 공유 Z-Image 블록(`dit_blocks`)으로 만든 30층, 폭 3840의 single-stream DiT이며, 3축 RoPE와 2층의 noise, context, SigVQ refiner를 가집니다. VQ와 편집 요청은 caption에 폭 4096의 SigVQ 의미 특징을 추가합니다. latent는 FLUX.2 VAE latent(128채널, 16배 downsampling)이며 공유 `Flux2Vae`가 디코딩하고, 편집에서는 인코딩합니다.

| 체크포인트 | Step / guidance | 스케줄 |
|-----------|-----------------|--------|
| `llada-image-turbo` | 4 / 1.0 | f32의 `linspace(1, 0, steps + 1)`, shift 3, stochastic: 모든 step이 예측된 clean latent로부터 다시 noise를 입힘 |
| `llada-image` | 50 / 5.0 | f64로 만들어 f32로 반올림한 `linspace(0.001, 1, steps + 1)[:-1]` 위의 비선형 `1 - (1 - (1 - s^1.17)^0.8)^1.1`, shift 1, Euler step |

- **모드.** `generation_mode`가 `text`(기본값)이면 caption에만 conditioning합니다. `vq`는 먼저 LLaDA-2 decoder를 이산 이미지 prior로 실행합니다. 32 token 블록에 대한 block diffusion, 블록당 8번의 greedy step, CFG 2, confidence 0.95 초과 또는 스케줄의 상위 개수에 따라 공개합니다. SigVQ는 생성된 id를 의미 특징(512x512에서 1024 token)으로 임베딩합니다. 편집(`--task edit`, `/v1/images/edits`, 정확히 하나의 소스 이미지)은 소스를 VAE latent로 인코딩하고, 절반 해상도에서 SigVQ의 40블록 ViT를 거쳐 의미 특징으로 인코딩합니다. 여기서 `generation_mode`는 `editing`이거나 없어야 합니다.
- **크기.** 너비와 높이는 양의 16의 배수이며 편집에서는 32의 배수입니다. 크기가 없는 편집은 소스를 1024x1024 픽셀 면적 쪽으로 줄이고(절대 확대하지 않음) 각 변을 32로 내림하며, 한 변이 주어지면 다른 변은 종횡비를 유지합니다. 편집은 denoiser를 소스와 target grid에 대해 실행하므로 forward당 `2 * (H / 16) * (W / 16)`개의 이미지 token이며, 이 합계는 65536(`MAX_EDIT_IMAGE_TOKENS`, Mage-Flow의 값. mlxcel의 경계이며 upstream에는 없음), 즉 약 2896x2896 픽셀 면적으로 제한되고, 더 큰 편집은 아무것도 로드하기 전에 계산된 개수와 한도를 명시하는 `400`입니다. 프롬프트는 최대 2048 token입니다. 텍스트와 편집 프롬프트는 upstream처럼 거기서 truncate되고, prior 입력(system template, 프롬프트, 이미지 header)이 2048 token을 넘는 VQ 프롬프트는 아무것도 로드하기 전에 `400`입니다(mlxcel의 경계이며 upstream에는 없음. prior는 KV cache 없이 그 윈도 전체를 다시 실행함). 소스 이미지는 디코딩된 그대로 사용됩니다. upstream은 먼저 `ImageOps.exif_transpose`를 적용하지만 mlxcel의 이미지 로딩은 어떤 패밀리에서도 이를 하지 않습니다.
- **Guidance.** CFG는 guidance 1 초과에서 실행됩니다. negative 프롬프트는 요청 필드, 없으면 `negative_prompt` 옵션, 없으면 빈 값입니다.
- **옵션**(`--gen-kwargs` 또는 추가 JSON 필드): `generation_mode`, `stochastic_sampling`(스케줄러의 설정을 덮어씀), `negative_prompt`. 잘못된 옵션, 크기, 모드는 서버에서 `400`이고, 모델 내부의 평가 실패는 `500`입니다.
- **Noise.** `mx.random.seed(seed)` 다음 upstream의 추출을 순서대로(초기 latent, 그다음 stochastic 재-noise마다 한 번) 수행하므로 모든 추출이 upstream과 비트 단위로 동일합니다.
- **메모리.** 컴포넌트는 upstream 파이프라인처럼 로드되고 내려집니다. 텍스트 encoder(bf16에서 31 GB)는 프롬프트가 인코딩된 뒤 transformer(13 GB)가 로드되기 전에 내려지고, transformer는 VAE가 디코딩하기 전에 내려집니다. 서빙 중인 모델은 요청 사이에 projection된 프롬프트를 최대 두 개 유지하므로, 반복된 프롬프트는 텍스트 encoder를 건너뜁니다.

### LLaDA-Image 정밀도

upstream의 텍스트 encoder는 LLaDA-2 MoE decoder를 상속하며 수치를 바꾸고, 이미지 prior의 token은 1 ulp 차이에서 뒤집히는 MoE 라우팅을 따릅니다. 포트는 각 변경을 선택적인 `Llada2Numerics::LladaImage` 경로 뒤에서 재현하며, 기본 텍스트 생성 경로는 변하지 않습니다(`llada2.0-mini-preview-4bit`에서 teacher-forced logit trace, 256 token x 4 chunk: 1024개 위치 중 0개 불일치, 바이트 단위로 동일한 trace, perplexity는 이전과 이후 모두 7.6092). 이미지 경로는 RMSNorm gain 전에 정규화된 activation을 bf16으로 반올림하고, RoPE 테이블을 activation dtype으로 반올림하며, 라우팅된 expert를 점수 내림차순으로 정렬하고, down projection 전에 activation에 f32 라우팅 점수를 곱하고, expert 출력을 f32로 합산합니다.

denoiser에서 upstream의 bias가 있는 `nn.Linear`는 하나의 fused `addmm`이며, 별도의 matmul과 bias 덧셈은 첫 예측을 0.18로 움직였습니다. `dit_blocks`는 이를 위해 선택적인 `BiasAdd::Fused`를 받습니다(LLaDA-Image 전용. Z-Image와 Mage-Flow는 별도 덧셈을 유지하며, fused 형태는 런타임 LoRA를 우회함). 회전된 q와 k는 upstream처럼 입력 dtype으로 반올림되며, f32로 유지하면 첫 예측이 8.4e-2 움직입니다. RoPE inverse frequency는 upstream의 `MRoPE`처럼 장치에서 계산됩니다(`MRoPE::compute_freqs_on_device`, LLaDA-Image 전용). 호스트의 `powf`는 40개 주파수 중 하나를 다르게 반올림하여 이미지 테이블 항목의 2.8%와 첫 예측을 6.0e-2 움직였습니다. 두 가지를 모두 적용하면 upstream이 입력한 값을 받는 모든 denoiser 단계가 비트 단위로 동일합니다(`transformer_stage_tests.rs`는 `tests/fixtures/generate_llada_image_denoiser_stages.py`의 단계별 텐서를 읽음). 단, VQ 512의 최종 projection은 예외입니다. `[1024, 3840] x [3840, 128]` bf16 `addmm`이 mlx 0.32.4와 다른 GEMM 커널을 선택합니다(입력은 비트 단위로 동일하고 상대 RMS 3.0e-3, 출력의 38.6%).

같은 seed의 Turbo CLI 실행 두 번은 바이트 단위로 동일한 PNG를 기록합니다.

### LLaDA-Image 일치 여부

oracle은 양쪽 모두 `MLX_ENABLE_TF32=0`으로 CUDA 백엔드(NVIDIA GB10)의 mlx 0.32.4에서 실행한 upstream mlx-vlm `73d366ec`이며, oracle에서 upstream의 결함 두 개를 수정했습니다(`tests/fixtures/generate_llada_image_oracle.py`, 플래그 `--unsorted-gather`와 `--fix-conv-in`).

- **`sorted_indices=True`인 `gather_mm`은 mlx 0.32.4 CUDA에서 잘못된 행을 반환합니다**(정렬하지 않은 호출이나 행별 matmul과 비교해 상대 RMS 0.8, 무작위 가중치에서도 동일. CPU f32는 정확). upstream의 `TextMoE`는 정렬된 형태를 사용하므로 텍스트 encoder는 첫 MoE 레이어부터 틀리고 VQ prior는 다른 token을 생성합니다. mlxcel의 MLX 빌드에는 이 버그가 없으므로(정렬 경로가 f32 CPU 참조와 6e-3으로 일치함), oracle은 `gather_mm`을 정렬하지 않고 호출합니다.
- **VAE encoder의 `conv_in`은 PyTorch 레이아웃을 유지합니다**([검증](#검증) 아래에 설명된 `_match_conv_layout` 결함). 그러면 upstream의 VAE는 편집 소스를 18.0 dB로 왕복하고, 가중치를 전치하면 33.8 dB이며, mlxcel은 이를 올바르게 로드합니다. 수정하지 않은 upstream의 base 편집은 소스 collage를 색만 바꾼 사본으로 렌더링하지만, 수정한 upstream과 mlxcel은 같은 프롬프트에 대해 둘 다 물의 caustic 질감을 렌더링하며 이는 체크포인트의 출력입니다.

프롬프트는 붉은 판다 장면이며 seed 42입니다. 편집 소스는 체크포인트의 `assets/edit_demo.jpg`(1440x704, "make it blue")입니다. Turbo는 guidance 1에서 4 step, base 체크포인트는 guidance 5에서 50 step을 실행합니다. "Upstream features"는 upstream 자체의 caption과 의미 특징을 denoiser에 넣는 경우입니다.

| 케이스 | upstream 입력에 대한 첫 예측 | 최종 latent 상대 RMS / PNG PSNR | upstream 특징에서 | 경계(upstream 특징에서) |
|--------|-----------------------------|-------------------------------|-------------------|------------------------|
| Turbo text, 1024x1024 | 0 | 0.882 / 16.10 dB | 0 / 57.97 dB | 5e-3 / 50.0 dB |
| Turbo VQ, 512x512 | 3.00e-3 | 1.087 / 14.34 dB | 0.0454 / 44.79 dB | 0.08 / 40.0 dB |
| Turbo edit | 0 | 0.154 / 26.71 dB | 0 / 56.21 dB | 5e-3 / 50.0 dB |
| Base text, 1024x1024 | 0 | 0.623 / 18.02 dB | 0 / 57.94 dB | 5e-3 / 50.0 dB |
| Base edit | 0 | 0.547 / 22.24 dB | 0 / 58.45 dB | 5e-3 / 50.0 dB |

upstream의 특징에서는 text와 edit 케이스가 upstream의 최종 latent를 정확히 재현하며, 그 PSNR은 VAE 디코딩의 잔차입니다. mlxcel 자체 특징에서는 30 dB 기준을 충족하지 못하며, 그 격차는 아래의 텍스트 encoder에서 옵니다.

모든 초기 및 재-noise 추출은 비트 단위로 동일하고, token id는 정확히 일치하며, upstream의 hidden state에서 text projection은 비트 단위로 정확하고(그 GELU와 QueryFormer의 GELU는 MLX가 `nn.gelu_approx`와 `nn.gelu`를 컴파일하는 것과 같은 컴파일된 그래프를 실행하며, eager 연산 체인은 1.9e-3으로 측정되었습니다), caption 특징은 1.1e-2에서 1.4e-2입니다. 편집 소스 latent는 8.8e-3이고 SigVQ codebook id 990개 중 973개가 일치합니다(나머지는 단위 구에서 거의 동점인 경우). base VQ 케이스에는 oracle이 없습니다. upstream이 mlx 0.32.4의 CUDA graph 코드에서 실패하기 때문입니다(`cudaGraphAddKernelNode ... invalid argument`). mlxcel 자체의 base VQ 실행은 완료됩니다.

mlxcel 자체 특징에서의 엔드투엔드 격차는 denoiser나 VAE가 아니라 텍스트 encoder를 통해 들어옵니다. upstream의 특징에서는 text와 edit가 정확하고 VQ는 4.5e-2입니다. 이는 포트의 결함이 아닙니다. teacher-forced(각 레이어에 upstream 자체의 해당 레이어 입력을 입력)로 보면, attention과 dense 레이어 0은 모든 레이어에서 비트 단위로 정확하고 모든 MoE 레이어는 모든 token을 같은 expert로 같은 순서로 라우팅합니다. MoE 출력의 0.6%만 한 번의 반올림(레이어당 상대 RMS 2e-4)만큼 다른데, mlxcel의 expert 정렬 `gather_mm`이 oracle의 정렬하지 않은 호출과 다른 순서로 누산하기 때문입니다. 정렬하지 않은 호출을 쓰면 20개 레이어 모두 upstream의 입력에서 비트 단위로 정확하지만 프롬프트는 여전히 drift합니다. QueryFormer의 28행 k 및 v projection(`addmm`)이 이 빌드의 MLX에서 mlx 0.32.4와 다른 GEMM 커널을 선택하고(57344개 값 중 40개가 다름), 라우팅이 이를 마지막 레이어에서 3e-2로 증폭합니다. 정렬하지 않은 호출(과 컴파일된 GELU)로 측정한 결과: Turbo text 18.5 dB, Turbo edit 36.0 dB, VQ id는 처음 48개 token이 같고(1024개 중 377개), VQ prior 시간은 두 배(206초 대 101초)입니다. 포트는 upstream이 직접 호출하는 정렬된 호출을 유지합니다. 파이프라인은 작은 차이를 증폭하며 upstream도 자신의 것을 같은 방식으로 증폭합니다. upstream의 image-embedder bias 덧셈을 따로 반올림하면 첫 예측이 8.6e-2 움직이고 PNG는 26.5 dB가 되며, QueryFormer query를 2^-9만큼 교란하면 caption이 3.1e-3 움직이고 PNG는 17.9 dB가 되며, 텍스트 backbone 레이어 2에서 bf16 1 ulp가 최종 hidden state를 0.167 움직입니다(mlxcel 대 upstream: 0.172). MoE 라우팅이 뒤집히기 때문입니다. 따라서 VQ prior의 id는 upstream과 정확히 일치하지 않습니다. 1024개 중 254개가 같고 처음 6개가 동일합니다. 엄격한 게이트는 타이트한 단계(token id, noise, 1e-4 미만의 projection, 2e-5 미만으로 teacher-forced한 텍스트 backbone의 처음 두 레이어(이는 5.1e-5인 FP32 RoPE 테이블을 거부함), upstream id의 SigVQ 임베딩 1e-2 미만, VAE 인코딩, text와 edit에서 1e-3 미만, VQ에서 1e-2 미만의 첫 예측)에 있고, 엔드투엔드 경계는 측정된 포트와 잘못된 포트 대조군 사이에 있습니다.

| 대조군(Turbo, upstream 특징) | Text | VQ | Edit |
|------------------------------|------|----|------|
| Seed 43 | 1.256 / 12.78 dB | 0.978 / 17.71 dB | 0.505 / 20.46 dB |
| shift 3 대신 1 | 0.791 / 18.02 dB | 0.669 / 22.20 dB | 0.230 / 25.37 dB |
| RoPE 후 FP32 q와 k(첫 예측) | 0.250 / 28.62 dB (8.43e-2) | 0.117 / 35.28 dB (6.69e-2) | 0.0152 / 48.85 dB (1.28e-2) |

모든 대조군은 해당 케이스의 엔드투엔드 경계를 벗어나며, FP32 RoPE는 모든 케이스에서 첫 예측 경계도 벗어납니다.

```bash
MLX_ENABLE_TF32=0 MLXCEL_LLADA_IMAGE_ORACLE_DIR=<oracle --out> \
    <lib test binary> --ignored --test-threads=1 imagegen::llada_image::parity_tests
```

테스트는 `models/mlx/llada-image-turbo`와 `models/mlx/llada-image`를 읽습니다(`MLXCEL_LLADA_IMAGE_TURBO`, `MLXCEL_LLADA_IMAGE_BASE`). `MLXCEL_LLADA_IMAGE_OUT`은 PNG를 저장합니다. fixture는 커밋되지 않았습니다(oracle이 케이스별로 전체 해상도 텐서를 저장함). `MLXCEL_LLADA_IMAGE_ORACLE_DIR`이 없으면 테스트는 모든 케이스를 건너뜁니다.

## 검증

```bash
cargo test --workspace --profile test-fast --features metal,accelerate imagegen
```

테스트는 자기 자신이 아니라 upstream Python(`Blaizzy/mlx-vlm`)과 비교합니다. `tests/fixtures/generate_imagegen_fixtures.py`는 커밋된 fixture가 어떻게 만들어졌는지를 문서화하며 이를 정확히 재현합니다.

- **스케줄러**: 이미지 시퀀스 길이 1024와 4096에서의 1, 4, 12, 30 step에 대한 sigma 및 timestep 배열(`shift_terminal` 0.02와 비활성화), 그리고 static-shift 스케줄.
- **VAE, 작은 것**: upstream의 decode, packed decode, encode 출력을 가진 무작위 가중치의 FLUX.2 VAE(Diffusers 키 이름, PyTorch conv 레이아웃). 같은 가중치를 MLX 레이아웃으로 재배치해도 같은 결과여야 합니다.
- **VAE, 실제 체크포인트**(`#[ignore]`): `baidu/ERNIE-Image-Turbo`의 `vae/`를 upstream f32 출력과 비교합니다. `models/ERNIE-Image-Turbo/vae/`가 없으면 건너뜁니다.

  ```bash
  huggingface-cli download baidu/ERNIE-Image-Turbo --include "vae/*" "model_index.json" \
      --local-dir models/ERNIE-Image-Turbo
  cargo test --profile test-fast --features metal,accelerate -p mlxcel --lib -- \
      --ignored ernie_image_turbo
  ```

  upstream f32와 비교한 측정값(상대 RMS): decode 3.4e-6, packed decode 1.2e-6, encode 3.0e-6.
- **Qwen3 conditioner**: 작은 모델의 hidden state와 쌓은 프롬프트 임베딩을 upstream과 비교, padding 및 key 레이아웃 불변식, 그리고 위의 실제 체크포인트 일관성 테스트.
- **DiT 블록**: timestep embedder, feed-forward, MRoPE 테이블, `apply_rotary`, modulated 및 plain 블록(attention mask 유무 모두), final layer에 대한 upstream 출력. 가중치와 입력은 양쪽이 구현하는 닫힌 형태의 생성기에서 도출합니다.

**encoder fixture가 우회하는 upstream 결함.** upstream의 `_match_conv_layout`은 가중치의 shape이 이미 target과 같으면 가중치를 그대로 반환합니다. encoder의 RGB `conv_in`은 두 레이아웃 모두 `(128, 3, 3, 3)`이므로, upstream은 거기서 PyTorch 데이터 순서를 유지하고 encoder가 틀립니다. 매끄러운 이미지를 encode한 뒤 decode하면 상대 RMS 0.96으로 재구성되며, 텐서를 전치하면 0.096입니다. fixture는 그 텐서 하나를 전치한 상태로 생성되며 실제 체크포인트 테스트도 encode-decode 재구성을 단언합니다. 레이아웃을 체크포인트마다 결정하므로 이 빌드는 영향을 받지 않습니다.

**실행한 것과 실행하지 않은 것.** 위의 수치는 CPU MLX 백엔드(`MLXCEL_BUILD_METAL=OFF`)에서 나온 것입니다. 이를 만든 머신에 Metal toolchain이 설치되어 있지 않았기 때문입니다. 이 테스트들의 Metal 백엔드 실행은 여기에 기록되어 있지 않습니다.

### 리소스 및 큐 제한

공유 요청 validator는 각 출력 이미지를 16,777,216 픽셀로, 명시적 step 수를 1–1,000으로 제한합니다. 이 검사는 모델 실행 전에 CLI와 HTTP 라우트 모두에 적용됩니다. 이미지 worker는 큐 대기 시간을 요청 타임아웃에 포함하고, GPU 작업을 시작하기 전에 만료된 큐 요청을 버립니다. 이미 실행 중인 생성은 현재 모델 trait을 통해 중단할 수 없으며, 소유 worker 스레드에서 끝까지 실행됩니다.

CLI는 현재 디렉터리의 파일 이름만 지정한 출력의 경우를 포함해, 배타적으로 생성한 임시 파일로 출력 디렉터리의 쓰기 가능 여부를 검사합니다. 기존 파일과 심볼릭 링크는 쓰기 probe로 사용되지 않습니다.
