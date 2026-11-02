# 비전 모델 범위

이 페이지는 `lablup/mlxcel#524`가 VLM 레퍼런스에서 모은 작업 특화 비전 및 생성형 미디어 패밀리, 즉 객체 detection, 인스턴스 segmentation, 3D body 추정, grounding, diffusion 이미지·비디오 생성의 범위 기록입니다. 각 패밀리에는 근거가 되는 증거와 함께 하나의 처리 결과(disposition)가 있으므로, 커버리지에 대한 질문에는 기록된 답이 있습니다. 생성형 chat 또는 completion VLM인 패밀리는 [model-triage.md](model-triage.md)(`lablup/mlxcel#548`)와 개별 포트 이슈 `lablup/mlxcel#533`부터 `lablup/mlxcel#547`까지에서 별도로 추적합니다.

마지막 확인 기준: `main`의 `68d7cc36`.

## 이 패밀리들이 범위 밖이었던 이유

mlxcel의 생성 런타임은 token 스트림을 내보내는 decoder LLM과 VLM을 대상으로 합니다. 아래 패밀리는 대신 box, mask, mesh, 픽셀을 내보내거나, 원래 런타임에 없던 encoder-decoder 루프가 필요합니다. detection 항목이 없는 패밀리의 체크포인트는 `src/models/detection.rs::get_model_type`에서 두 가지 방식 중 하나로 실패합니다.

- `model_type`에 맞는 match arm이 없는 `config.json`은 `Unsupported model type: <type>`으로 실패합니다.
- 다중 컴포넌트 diffusion 디렉터리에는 최상위 `model_type`이 없습니다. 이미지 probe가 이를 인식하지 못하면 `model_type not found`로 실패하며, 최상위 `config.json`이 아예 없으면 단순한 파일 읽기 오류로 실패합니다(Diffusers 체크포인트는 `model_index.json`을 키로 사용합니다). 인식된 이미지 체크포인트는 아래 이미지 생성 항목에 설명한 대로 대신 리디렉션됩니다.

mlxcel에는 이미 비-chat 경로가 있으며, 이들이 범위를 재설정할 때의 선례입니다.

- `mlxcel detect`(`src/commands/detect.rs`, `src/vision/detection/`): bounding-box를 출력하는 객체 detection.
- Whisper(`src/models/whisper/`): `/v1/audio/transcriptions`를 통한 encoder-decoder 음성 인식.
- Kokoro(`src/models/kokoro/`): `/v1/audio/speech`를 통해 오디오를 출력하는 text-to-speech.
- 이미지 생성(`src/imagegen/`, `src/commands/generate_image.rs`): `lablup/mlxcel#2341`(PR `lablup/mlxcel#2497`)로 추가된, `mlxcel generate-image`와 `/v1/images/generations` / `/v1/images/edits` 뒤의 공유 런타임. [image-generation.md](image-generation.md)를 참고하십시오.

## 이미 지원됨

| 패밀리 | 체크포인트 / 이슈 | 트리 내 증거 |
|---|---|---|
| `rt_detr_v2` | Docling layout 체크포인트(`docling-layout-heron`)를 포함한 RT-DETRv2 체크포인트 (PR `lablup/mlxcel#80`) | `src/vision/detection/rt_detr_v2/`(ResNet-vd backbone, AIFI와 FPN/PAN 하이브리드 encoder, iterative box refinement를 갖춘 6층 deformable decoder, `id2label`을 통한 label), `src/vision/detection/rt_detr_v2/sanitize.rs`(레이아웃을 확인하는 conv permutation), `src/commands/detect.rs`(`RtDetrV2Predictor::from_pretrained`). generate 루프 밖의 독립 런타임 패밀리이며 `mlxcel arch --json`에 나열됩니다. [supported-models.md](supported-models.md)를 참고하십시오. |
| `pp_doclayout_v3` | PaddlePaddle/PP-DocLayoutV3 및 변환된 `PP-DocLayoutV3-mlx` 체크포인트 (`lablup/mlxcel#2412`) | `src/vision/detection/pp_doclayout_v3/`가 HGNetV2-L, 하이브리드 encoder, mask-enhanced query 초기화, 6층 deformable decoder, GlobalPointer 읽기 순서 head를 구현합니다. 로더는 출처가 다른 Paddle 레이아웃과 변환된 레이아웃을 모두 받아들이며, `mlxcel detect`는 Falcon-OCR layout 입력에 바로 쓸 수 있도록 1부터 시작하는 dense 모델 읽기 순서를 내보냅니다. |
| `florence2` | `mlx-community/Florence-2-base-ft-bf16`과 그 4-bit, 8-bit 변환본 (에픽 `lablup/mlxcel#850`, PR `lablup/mlxcel#1060`, `lablup/mlxcel#1063`, `lablup/mlxcel#1064`, `lablup/mlxcel#1069`, `lablup/mlxcel#1071`, `lablup/mlxcel#1082`, `lablup/mlxcel#1083`) | `src/models/detection.rs`(`"florence2" => Florence2VLM`), `src/models/florence2/`(DaViT tower, BART encoder-decoder, task 프롬프트, location-token 후처리), `src/loading/vlm_florence2.rs`, 그리고 PR `lablup/mlxcel#1083`의 seq2seq 서버 worker. 이 기록이 걸림돌로 지목했던 seq2seq 루프가 이제 존재합니다. `-large-ft` MLX 변환본은 로드되지만 쓸 만한 텍스트를 생성하지 못하므로(upstream `Blaizzy/mlx-vlm#1840`), [supported-models.md](supported-models.md)는 `-base-ft`를 권장합니다. |
| `locateanything` | `mlx-community/LocateAnything-3B-4bit` (`lablup/mlxcel#847`, PR `lablup/mlxcel#1070`) | `src/models/detection.rs`(`"locateanything" => LocateAnythingVLM`), `src/vision/locateanything.rs`, `src/loading/vlm_locateanything.rs`. 표준 autoregressive decode 경로는 지원됩니다. 병렬 box 디코딩 head(`pbd`, `n_future_tokens = 6`)와 coordinate-token-to-box 후처리는 구현되지 않았으며, [supported-models.md](supported-models.md)가 이를 후속 작업으로 기록하고 있고 이를 추적하는 열린 이슈는 없습니다. |
| `nemotron_parse` | `nvidia/NVIDIA-Nemotron-Parse-2.0`과 그 `mlx-community/Nemotron-Parse-2.0-{4bit,8bit}` 변환본 (`lablup/mlxcel#1369`, PR `lablup/mlxcel#2034`) | `src/models/detection.rs`(`"nemotron_parse" => NemotronParseVLM`), `src/models/nemotron_parse/`(C-RADIO ViT-H/16 tower, compression neck, 10층 pre-norm mBART decoder), `src/loading/vlm_nemotron_parse.rs`, `src/commands/generate_nemotron_parse.rs`(CLI driver), `src/server/nemotron_parse_worker.rs`(단일 스트림 서버 worker). box와 class token이 포함된 markdown을 내보내는 문서 OCR 및 layout 모델이므로 `lablup/mlxcel#524`가 모은 패밀리는 아니지만, `florence2` 행이 설명하는 encoder-decoder 구조를 공유합니다. 참조 구현과의 greedy-token 일치 여부는 로컬 체크포인트가 있을 때만 실행되도록 `#[ignore]`로 게이트된 `tests/nemotron_parse_real_model.rs`가 확인합니다. [supported-models.md](supported-models.md)를 참고하십시오. |

## 에픽과 포트 이슈로 재분류됨

메인테이너가 이미지 생성, depth, pose, 3D 지원을 요청했으므로, 이 패밀리들은 이 기록에서 전용 작업으로 옮겨졌습니다(`lablup/mlxcel#524`의 2026-10-16 코멘트). 이제 각 패밀리에는 에픽 아래에 포트 이슈가 있습니다.

### 이미지 생성, 에픽 `lablup/mlxcel#2348`

`lablup/mlxcel#2341`(PR `lablup/mlxcel#2497`)의 공유 런타임이 병합되었습니다. `src/imagegen/registry.rs::FAMILIES`는 `z_image`(`lablup/mlxcel#2425`), `ernie_image`(`lablup/mlxcel#2345`), `mage_flow`(`lablup/mlxcel#2426`), `qwen_image`(`lablup/mlxcel#2342`), `ideogram4`(`lablup/mlxcel#2427`), `ming_image`(`lablup/mlxcel#2347`)를 등록합니다. `src/imagegen/probe.rs::IMAGE_FAMILIES`는 `bonsai`, `flux2`, `z_image`, `ming_image`, `ernie_image`, `qwen_image`, `mage_flow`, `ideogram4`를 인식합니다. 따라서 인식되었지만 등록되지 않은 패밀리의 체크포인트는 명령에 따라 두 가지 방식 중 하나로 실패합니다.

- `mlxcel generate`, `mlxcel serve` 및 다른 텍스트 모델 로더는 `src/models/detection.rs`에서 `image generation checkpoint (<name>): use mlxcel generate-image`로 중단됩니다(`src/imagegen/probe.rs::image_checkpoint_redirect_error`).
- `mlxcel generate-image`는 포트 이슈가 해당 패밀리를 등록하기 전까지 `image generation family <name> is not supported by this build`로 실패합니다(`src/imagegen/registry.rs::resolve_family`).

probe가 인식하지 못하는 체크포인트는 위에서 설명한 `model_type not found` 또는 파일 읽기 실패를 그대로 유지합니다.

| 패밀리 | 포트 이슈 | 설명 | 현재 probe 상태 |
|---|---|---|---|
| `flux2` | `lablup/mlxcel#2424` | FLUX.2 Klein(4B, 9B): dual-stream과 single-stream MMDiT flow-matching text-to-image 및 편집, 내장 텍스트 encoder, VAE. | 인식됨, 등록되지 않음. |
| `z_image` | `lablup/mlxcel#2425` | Z-Image / Z-Image-Turbo: Qwen3 스타일 텍스트 encoder, diffusion transformer, VAE, flow-matching 스케줄러. | 등록됨: `mlxcel generate-image`와 `mlxcel-server --image-model`이 text-to-image와 img2img를 실행합니다. |
| `mage_flow` | `lablup/mlxcel#2426` | Mage-Flow와 Mage-Flow-Edit(4B, 변형 6개): Qwen3-VL conditioner, NR-MMDiT transformer, Mage-VAE, flow-matching sampler. | 등록됨: `mlxcel generate-image`와 `mlxcel-server --image-model`이 변형에 따라 정해지는 text-to-image 또는 편집을 실행합니다. |
| `ideogram4` | `lablup/mlxcel#2427` | Ideogram 4: 외부 LLM 텍스트 encoder를 사용하는, VAE latent 위의 34층 DiT. FP8이며 라이선스가 게이트된 체크포인트로 `model_index.json`을 키로 사용합니다. | 등록됨: `mlxcel generate-image`와 `mlxcel-server --image-model`이 text-to-image를 실행합니다(생성 전용). |
| `qwen_image` | `lablup/mlxcel#2342` | Qwen-Image-2.1: Qwen3-VL conditioner(pre-norm hidden state), 3축 RoPE를 사용하는 32블록 single-stream DiT, Qwen-Image VAE(Ming-Image와 공유). | 등록됨: `mlxcel generate-image`와 `mlxcel-server --image-model`이 text-to-image와 하나 이상의 참조 이미지를 사용하는 편집을 실행합니다(RGBA 유지). |
| `ming_image` | `lablup/mlxcel#2347` | Ming-Image-0.1-Design: 256개 query token에 대한 token별 이미지 router를 가진 BailingMoeV2 conditioner, 양방향 Qwen2 connector, 두 번째 caption 스트림을 가진 Z-Image NextDiT, 4채널을 디코딩하는 Qwen-Image VAE. | 등록됨: `mlxcel generate-image`와 `mlxcel-server --image-model`이 RGBA 출력의 text-to-image를 실행합니다(생성 전용, guidance 1.0). |
| `bonsai` | `lablup/mlxcel#2428` | Prism Bonsai: ternary 2-bit FLUX.2 text-to-image. [model-triage.md](model-triage.md)에도 기록되어 있습니다. | 인식됨, 등록되지 않음. |
| `ernie_image` | `lablup/mlxcel#2345` | ERNIE-Image / ERNIE-Image-Turbo: single-stream DiT, Ministral 3 텍스트 encoder 및 프롬프트 enhancer, FLUX.2 VAE. | 등록됨: `mlxcel generate-image`와 `mlxcel-server --image-model`이 text-to-image와 SDEdit img2img를 실행합니다. |

### Dense prediction, segmentation, 3D, 에픽 `lablup/mlxcel#2382`

트리에는 `mlxcel predict` 출력 표면(`lablup/mlxcel#2349`), 그 dense 패밀리 레지스트리, 그리고 공유 `mlxcel detect` 표면이 포함되어 있습니다. 아래의 모든 에픽 패밀리는 이제 이 네이티브 런타임 중 하나로 구현되어 있습니다.

| 패밀리 | `model_type` | 포트 이슈 | 출력 경로 |
|---|---|---|---|
| `dinov2` | `dinov2` 또는 `dinov2_with_registers` | `lablup/mlxcel#2350` | 네이티브 `mlxcel predict` 특징 텐서와 메타데이터. |
| `sapiens2` | `sapiens2` | `lablup/mlxcel#2371` | 네이티브 pose, segmentation, normal, pointmap, matting, backbone 예측. |
| `video_depth_anything` | `video_depth_anything` | `lablup/mlxcel#2352` | 네이티브 정지 이미지 및 시간축 depth 예측. |
| `moge3` | `moge3` | `lablup/mlxcel#2355` | 네이티브 단안(monocular) geometry, depth, normal, validity, intrinsics 예측. |
| `sam3` | `sam3_video` 또는 `sam3` | `lablup/mlxcel#2429` | 텍스트 또는 픽셀 box 프롬프트를 사용하는 네이티브 `mlxcel predict` detector, 인스턴스 mask, memory tracker. |
| `sam3_1` | `sam3.1_video` 또는 `sam3.1` | `lablup/mlxcel#2430` | `sam3`와 같되, 한 번의 pass에서 최대 16개 객체를 추적하는 Object Multiplex tracking이 추가됩니다. 비디오 체크포인트에는 `sam3.1` detector 하위 config가 있으며 대부분의 컴포넌트를 `sam3`와 공유합니다. |
| `sam3d_objects` | `sam3d_objects` | `lablup/mlxcel#2381` | 이미지와 mask로부터 RGB OBJ 또는 Gaussian PLY로 객체를 재구성하는 네이티브 처리. |
| `sam3d_body` | `dinov3` vision 하위 config를 가진 `sam3d_body` | `lablup/mlxcel#2432` | 네이티브 사람 mesh, 3D keypoint, joint 좌표, 카메라 예측. |
| `rf-detr` | `rf-detr`(별칭 `rfdetr`) | `lablup/mlxcel#2431` | 공개된 windowed DINOv2 backbone, C2f projector, deformable decoder, COCO box, 그리고 `--masks-dir`를 통한 선택적 원본 해상도 segmentation mask를 갖춘 네이티브 `mlxcel detect` 구현. |

## 재분류되지 않음

두 에픽 모두 이 패밀리들을 지명하지 않으므로, 2026-10-16의 범위 재설정은 이들에 영향을 주지 않습니다.

| 패밀리 | 처리 결과 | 증거 |
|---|---|---|
| `minimax_h3` | 범위 밖. | MiniMax-H3는 비디오와 오디오를 함께 생성하는 모델(T2VA, FL2VA, Ref2VA)로, 50블록 rectified-flow transformer, causal 3D-conv 비디오 VAE, DAC/BigVGAN 스타일 오디오 VAE, Qwen3-VL-32B conditioner로 구성되며 워크플로당 bf16 기준 약 130 GB입니다. 이미지 런타임은 비디오 생성을 제외하며([image-generation.md](image-generation.md)), 포트 이슈가 없고, 일치하는 detection 항목도 없습니다. |
| `nemotron_labs_diffusion` | 범위 안, 포트 이슈 대기 중. | 이름과 달리 비전 모델이 아닙니다. autoregressive, block-diffusion(block size 32), self-speculative decode 모드를 가진 8B 텍스트 전용 diffusion 언어 모델(GQA, SwiGLU, YaRN RoPE, untied diffusion head, mask token)입니다. DiffusionGemma(`src/models/diffusion_gemma/`, `src/models/detection.rs`의 `"diffusion_gemma" \| "diffusion_gemma_text"`, [block-diffusion.md](block-diffusion.md))가 이 런타임 형태에 대한 트리 내 선례입니다. `src/models/detection.rs`에 이에 해당하는 arm이 없으므로 로드는 `Unsupported model type`으로 실패합니다. 포트 이슈는 없습니다. 패밀리 이름으로 저장소 이슈를 검색하면 이 기록, 확인할 패밀리 중 하나로 나열한 f32 attention-guard 수정 `lablup/mlxcel#1710`, 그리고 이를 포팅되지 않은 것으로 기록한 감사 `lablup/mlxcel#1829`(`docs/f32-attention-widen-audit.md`)만 일치합니다. |

## 패밀리 범위 재설정

여기에 나열된 패밀리는 다음 조건이 모두 충족되면 전용 포트 이슈로 옮겨집니다.

1. 구체적인 사용자 또는 제품 요구가 해당 패밀리를 지목하고, 누군가 서빙하려는 체크포인트가 있습니다.
2. 출력 경로가 존재하거나 구축 중입니다. box는 `mlxcel detect`, 이미지는 `src/imagegen/`, mask, depth map, keypoint, mesh는 `lablup/mlxcel#2349`가 반영된 이후 `mlxcel predict`를 거칩니다. 그 밖의 출력 경로는 먼저 그 자체로 인프라 이슈가 됩니다.
3. 검증용으로 구체적인 공개 체크포인트가 지정되어 있고, 그 `config.json` 또는 컴포넌트 레이아웃이 확인되었습니다.
4. 가중치 레이아웃이 고려되었습니다. 이 패밀리들은 `(out_channels, in_channels, kernel_h, kernel_w)`로 직렬화된 convolution 스택(CNN backbone, ViT patch embedding, VAE)을 가지는 반면, MLX는 `(out_channels, kernel_h, kernel_w, in_channels)`를 기대하는 channels-last convolution을 실행합니다. 포트는 로드 시 이를 permutation해야 하며, 이미 MLX 레이아웃으로 저장된 체크포인트를 두 번 전치해서는 안 됩니다. `src/vision/detection/rt_detr_v2/sanitize.rs`가 선례입니다.

포트 이슈는 `lablup/mlxcel#533`부터 `lablup/mlxcel#547`까지의 형식(`model_type` 문자열을 포함한 Summary, What it is, In-tree reuse, Scope, Effort)으로 작성하고, `lablup/mlxcel#524` 또는 해당 에픽에 연결한 뒤 이 페이지를 갱신하십시오.

## 비목표

- 이 기록에서 이 패밀리들을 구현하는 것. 이 문서는 결정 기록일 뿐입니다.
- 여기서 범용 이미지 생성, segmentation mask, 3D geometry 출력 파이프라인을 구축하는 것. 각각은 해당 에픽에서 범위를 정합니다.
- 해당 에픽 밖에서 OpenAI 호환 API에 이미지, mask, mesh 응답 타입을 확장하는 것.
- 생성형 VLM 포트 후보를 추적하는 것. 그 후보는 [model-triage.md](model-triage.md)와 포트 이슈 `lablup/mlxcel#533`부터 `lablup/mlxcel#547`까지에 있습니다.
