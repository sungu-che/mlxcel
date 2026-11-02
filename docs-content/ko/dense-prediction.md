# Dense prediction

`mlxcel predict`는 depth, normal, segmentation, point, keypoint, 인스턴스별 mask, mesh, Gaussian 결과를 반환하는 체크포인트를 위한 오프라인 파일 인터페이스입니다. 이런 출력은 `generate`의 token 스트림이나 `detect`의 box 행에 맞지 않는 shape과 수명을 가집니다.

같은 인터페이스는 독립형 이미지 encoder 특징도 노출합니다. DINOv2는 레지스트리에서 처음으로 활성화된 패밀리이며, 이후의 dense 모델 포트는 이 encoder와 공유 파일 계약을 재사용합니다.

오프라인 비전 표면에는 `predict` 패밀리 여덟 개와 `detect`의 RF-DETR이 포함됩니다. task는 체크포인트나 입력 종류에서 선택됩니다.

```bash
mlxcel predict \
  --model models/dinov2-small \
  --image input.png \
  --output outputs/features \
  --task features

mlxcel predict \
  --model models/sapiens2 \
  --image person.png \
  --output outputs/pose \
  --opts '{"boxes":[[140,90,920,700]],"flip_test":true}'

mlxcel predict \
  --model models/video-depth-anything-small \
  --video clip.mp4 \
  --max-frames 64 \
  --output outputs/depth

mlxcel predict \
  --model models/sam3-bf16 \
  --image cats.jpg \
  --prompt "a cat" \
  --boxes "0,50,350,480" \
  --output outputs/cats
```

| 패밀리 | 런타임과 체크포인트 표기 | Task | 실제 출력 파일 |
|---|---|---|---|
| DINOv2 | `predict`; `dinov2`, `dinov2_with_registers` | `features` | `result.safetensors`, `meta.json` |
| Sapiens2 | `predict`; `sapiens2` | `pose`, `seg`, `normal`, `pointmap`, `matting`, `backbone` | 모든 task가 `result.safetensors`와 `meta.json`을 기록함. 비어 있지 않은 pose는 `keypoints.json`도 기록함. `seg`는 `segmentation.png`, `normal`은 `normal.png`를 기록함 |
| SAM 3 | `predict`; `sam3_video`, `sam3` | 이미지: `detect`, `segment`. tracker 체크포인트: `detect`, `segment`, `track` | `result.safetensors`, `meta.json`, `instances.json`. 이미지 mask는 `masks/000.png` 같은 형태, 비디오 mask는 `masks/f000012_o003.png` 같은 형태. `detect`는 mask PNG를 기록하지 않음 |
| Video Depth Anything | `predict`; `video_depth_anything` | `depth` | `result.safetensors`, `meta.json`. 정지 이미지는 `depth.png`, 비디오 프레임은 `depth/000000.png`, `depth/000001.png`, ... |
| MoGe3 | `predict`; `moge3` | `geometry` | `result.safetensors`, `meta.json`, `depth.png`, `normal.png`, `mask.png` |
| SAM 3.1 | `predict`; `sam3.1_video`, `sam3.1` | `detect`, `segment`, `track` | `result.safetensors`, `meta.json`, `instances.json`. 이미지 mask는 `masks/000.png` 같은 형태, 비디오 mask는 `masks/f000012_o003.png` 같은 형태. `detect`는 mask PNG를 기록하지 않음 |
| SAM 3D Objects | `predict`; `sam3d_objects` | `mesh` | 항상 `result.safetensors`, `meta.json`. 선택한 형식에 따라 `mesh.obj` 또는 `gaussian.ply`, 선택적으로 `gaussian_4.ply` |
| SAM 3D Body | `predict`; `sam3d_body` | `body_mesh` | `mesh.obj`, `result.safetensors`, `meta.json`, 빈 `instances.json`. 텐서 파일에는 `vertices`, `keypoints_3d`, `joint_coords`, `camera`가 들어 있으며 `keypoints.json`은 없음 |
| RF-DETR Base / Seg Small | `detect`; `rf-detr`(`rfdetr` 별칭) | Detection. segmentation 체크포인트는 선택적으로 mask를 추가 | stdout의 JSON. segmentation 체크포인트와 `--masks-dir DIR`를 쓰면 출력된 detection 순서대로 원본 해상도 8-bit `DIR/mask-000.png`, `DIR/mask-001.png`, ...를 기록. detection 전용 실행은 mask 파일을 기록하지 않음 |

모든 `predict` 패밀리는 `instances.json` 파일을 포함해 아래에 설명하는 공유 writer를 사용하며, 검출된 인스턴스가 없는 출력에서는 이 파일이 비어 있습니다. 표는 인스턴스와 mask 경로가 해당 패밀리 task의 일부일 때 이를 명시했습니다.

DINOv2는 표준 Hugging Face 체크포인트와 register token이 있는 체크포인트를 모두 받아들입니다. predictor는 체크포인트의 `preprocessor_config.json`을 따릅니다. RGB로 변환하고, Pillow 호환 bicubic uint8 반올림으로 짧은 변을 resize하고, center crop하고, rescale하고, 설정된 채널 통계로 정규화합니다. 이 파일이 없으면 공개 체크포인트 기본값인 짧은 변 256 픽셀, 224x224 center crop, 255로 나누기, ImageNet 정규화를 사용합니다. `last_hidden_state`는 class token과 모든 register token을 유지하고, `pooler_output`은 정규화된 class token이며, `hidden_patch_tokens`는 공간 patch만 담습니다. 비디오 입력은 거부됩니다.

`--image`와 `--video` 중 정확히 하나가 필요합니다. 선택한 패밀리가 해당 입력에 대한 기본값을 선언하면 `--task`를 생략할 수 있습니다. `--prompt`는 반복할 수 있습니다. `--boxes`도 반복할 수 있으며 세미콜론으로 구분한 픽셀 `x1,y1,x2,y2` box를 받습니다. prompt를 지원하지 않는 패밀리는 두 플래그를 모두 거부합니다. `--opts`는 JSON 객체여야 하며, 명령은 predictor를 호출하기 전에 선택한 task를 `task` 아래에, prompt 문자열을 `prompts` 아래에, 파싱된 숫자 box를 `boxes` 아래에 추가합니다. 비디오 샘플링은 기본적으로 최대 32프레임을 유지하며 `--max-frames`로 이 상한을 바꿉니다. `--target-fps`가 없으면 Video Depth Anything과 SAM tracking은 상한까지의 연속된 앞부분을 유지하고, `--target-fps`를 주면 시간 기반 균등 샘플링을 사용합니다. 명령은 디코딩된 RGBA 최악 크기가 512 MiB를 넘는 샘플 집합을 ffmpeg가 디코딩하기 전에 거부합니다. 출력 디렉터리는 없으면 생성됩니다. 표준 이름의 기존 파일은 교체되고 관련 없는 파일은 유지됩니다.

Sapiens2는 정지 이미지를 받습니다. task 체크포인트는 정확히 자신의 task 하나만 노출하므로 `--task`는 선택 사항입니다. Pose는 픽셀 공간 `xyxy` box 배열로 최대 16개의 `boxes`와 boolean `flip_test`를 받습니다. 활성화 메모리 피크를 제한하기 위해 box를 한 번에 하나씩 처리합니다. `boxes`를 생략하면 pose는 전체 이미지에 대해 실행됩니다. 빈 배열은 유효한 빈 pose 텐서를 만듭니다. mlxcel은 box를 이미지 범위로 clip하고, 형식이 잘못되었거나 유한하지 않거나 퇴화한 box는 거부합니다. pose 경로는 upstream의 종횡비 crop, affine warp, flip-pair 재매핑, DARK refinement, unbiased data processing(UDP) 좌표 변환을 그대로 유지합니다. 사람 detector는 실행하지 않습니다.

공개된 Sapiens2 체크포인트는 고정된 1024x768(높이 x 너비) 입력을 사용합니다. Segmentation과 matting은 이미지를 그 캔버스에 맞게 늘립니다. Normal과 pointmap 추정은 대칭 padding으로 종횡비를 유지한 뒤, 결과를 원본 해상도로 resize하기 전에 그 padding을 제거합니다. 로더는 공개된 두 config 레이아웃, 즉 이전의 중첩된 backbone/deconvolution 스키마와 현재의 flat Transformers 스키마를 모두 받아들입니다. backbone 차원은 표준 0.4B, 0.8B, 1B, 5B 아키텍처에 대해 config로 결정됩니다.

Video Depth Anything은 공개된 Small, Base, Large의 relative 및 metric 체크포인트를 지원합니다. 원본 종횡비를 유지하고, OpenCV 호환 cubic 보간과 14픽셀 배수로 하한을 518로 resize한 뒤 ImageNet 정규화를 적용합니다. 추론은 upstream의 32프레임 윈도, 10프레임 overlap, keyframe conditioning, relative scale/shift 정렬, 8프레임 선형 blend를 사용합니다. Metric 체크포인트는 절대 스케일을 유지하며 `units: "metres"`를 보고하고, relative 체크포인트는 `units: "relative_disparity"`를 보고합니다. Depth는 원본 해상도로 반환됩니다. 정지 이미지는 프레임을 반복하여 같은 시간축 경로를 따릅니다. `--target-fps`가 없을 때 `--max-frames`는 디코딩된 연속 프레임을 유지합니다.

DINOv2 backbone은 token 예산을 지키는 프레임 chunk 단위로 실행되며, 각 시간축 head는 여전히 32프레임 윈도 전체를 봅니다. 공개된 체크포인트는 F32 텐서를 저장합니다. BF16 체크포인트는 load policy가 지원하는 ROCm 및 다른 백엔드에서 BF16을 유지합니다.

Sapiens2는 모든 백엔드에서 공개된 BF16 가중치 dtype을 유지합니다. 공개 pose 체크포인트를 F16으로 변환하면 일부 디코딩된 keypoint가 upstream 일치 기준인 1.5픽셀을 넘게 바뀌므로, 이 패밀리는 일반적인 Apple Silicon 로드 시 BF16-to-F16 최적화의 명시적 예외입니다.

성공한 실행은 항상 다음을 기록합니다.

| 경로 | 내용 |
|---|---|
| `result.safetensors` | u8로 유지되는 `masks`를 제외한 모든 출력 텐서를 연속된 f32로 변환한 것. 인스턴스 출력에는 `boxes` `(N, 4)`와 `scores` `(N,)`가 포함됨. 인스턴스에 mask가 없을 수 있으므로 `masks`는 원본 해상도의 `(M, H, W)`. 비디오 맵은 축 0으로 쌓임. |
| `meta.json` | 패밀리, 선택한 task, 원본 크기, MLX 평가 이후의 추론 시간, 모든 텐서의 유한한 min/max, 모델 메타데이터, 그리고 `intrinsics` 텐서가 있을 때 평탄화한 intrinsics. |
| `keypoints.json` | 사람이 검출되었을 때 존재. 각 항목은 `bbox`, 원본 픽셀 단위의 `(x, y, score)` 점, 선택적인 점 이름을 포함함. |
| `instances.json` | 원본 픽셀 단위 `bbox`, `label`, `mask_index`, 선택적인 비디오 `object_id`와 `frame`, 그리고 있을 경우 상대 경로의 `mask` PNG를 가진 점수가 매겨진 인스턴스. |
| `masks/000.png` | 이미지 인스턴스 mask. 0과 255 값을 가진 8-bit 단일 채널. 번호는 `masks` 텐서의 행. |
| `masks/f000012_o003.png` | 프레임 12와 안정적인 객체 3에 대한 비디오 mask. |
| `depth.png` | 16-bit grayscale. 유한한 맵의 min/max가 0과 65535가 되고, NaN과 무한대는 0이 됨. |
| `normal.png` | `(normal + 1) / 2`를 `[0, 1]`로 clamp한 뒤의 8-bit RGB. |
| `mask.png` | 8-bit grayscale. 0.5 이하는 0, 0.5 초과는 255. |
| `segmentation.png` | 8-bit class ID. |

Sapiens2는 다음 텐서 키와 shape을 사용합니다.

| 체크포인트 task | 텐서 키 | Shape |
|---|---|---|
| Pose | `keypoints`, `scores`, `boxes` | 원본 이미지 픽셀 단위의 `(N, K, 2)`, `(N, K)`, `(N, 4)` |
| Segmentation | `segmentation` | `(H, W)` 정수 class ID |
| Normal | `normals` | `(H, W, 3)` normal 벡터 |
| Pointmap | `pointmaps`, `scales` | `(H, W, 3)`, `(1)` |
| Matting | `alphas`, `foregrounds` | `(H, W)`, `(H, W, 3)` |
| Backbone | `last_hidden_state`, `pooler_output` | token 및 pooled 특징 텐서 |

Pose는 box당 하나의 `keypoints.json` 항목도 기록합니다. 점 이름은 체크포인트에 `id2label` 맵이 있으면 거기서 옵니다.

## MoGe3 geometry

MoGe3는 하나의 RGB 이미지에서 카메라 공간 point map, depth, validity mask, 표면 normal, 정규화된 카메라 intrinsics를 추정합니다. 공개된 `moge-3-vitl-mlx-fp32`와 `moge-3-vitg-mlx-fp32` 체크포인트는 각각 DINOv2-L/14와 DINOv2-g/14를 사용합니다. 가중치는 F32로 유지됩니다. 설정된 네 개의 중간 DINOv2 grid가 projection되어 합산되고, convolution pyramid로 디코딩된 뒤, sparse 3D U-Net으로 refine됩니다. sparse 경로는 점유된 `(batch, row, column, depth-bin)` voxel만 저장하며 dense 3D volume은 절대 할당하지 않습니다.

```bash
MLX_ENABLE_TF32=0 mlxcel predict \
  --model models/moge-3-vitl-mlx-fp32 \
  --image room.png --task geometry \
  --opts '{"resolution_level":9,"refine_steps":3}' \
  --output outputs/room
```

`resolution_level`은 0부터 9까지의 정수이며 체크포인트의 최소 1200 token에서 최대 3600 token까지 선형으로 매핑됩니다. `num_tokens`는 그 범위의 정확한 값을 선택할 수 있으며 두 필드를 모두 주면 우선합니다. `refine_steps`의 기본값은 3이고 0부터 16까지 받습니다. `force_projection`의 기본값은 true이며 복원된 depth와 intrinsics로 point를 재구성합니다. `apply_mask`의 기본값은 true이며, 유효하지 않은 depth와 point는 양의 무한대가 되고 유효하지 않은 normal은 0이 됩니다. `fov_x`는 수평 시야각을 도 단위로 고정하며 0보다 크고 180보다 작아야 합니다. 생략하면 모델이 초점 거리와 depth shift를 복원합니다. `return_per_step`은 CLI 직렬화에 `points_stepN`, `depth_stepN`, `intrinsics_stepN` 텐서를 추가합니다. Rust 라이브러리는 이 값을 `Moge3Output`의 step별 벡터로 유지합니다.

MoGe3는 큰 decoder, sparse-refiner, 유지되는 출력 그래프를 구성하기 전에 입력 grid와 검증된 설정으로부터 확인된 live-work 추정치를 도출합니다. refiner는 선택적 이웃 맵을 해제하고 필요할 때 다시 계산합니다. 지원되는 필수 단계에는 명시적인 8 GiB 상한이 있고, 선택적 캐시 맵은 512 MiB를 목표로 합니다. 공개된 resolution level 9 입력은 계속 지원됩니다. 너무 큰 입력, 지원되지 않는 설정, 산술 overflow는 resource-budget 오류를 반환합니다. 이 할당 보호 장치는 계산 전체를 보존하며 요청한 해상도나 refinement step을 줄이지 않습니다.

정지 이미지의 표준 텐서 shape은 다음과 같습니다.

| 키 | Shape | 의미 |
|---|---|---|
| `points` | `(H, W, 3)` | 카메라 공간 XYZ. masking이 활성화되면 유효 mask 밖은 양의 무한대 |
| `depth` | `(H, W)` | 같은 무효 값 규칙을 쓰는 metric 스케일의 카메라 depth |
| `mask` | `(H, W)` | 양의 depth와 결합된 boolean 예측 validity |
| `normal` | `(H, W, 3)` | 유효 픽셀에서는 단위 표면 normal이고, masking이 활성화되면 그 외에는 0 |
| `intrinsics` | `(3, 3)` | 이미지 너비와 높이로 정규화된 OpenCV pinhole 행렬 |

입력 정수는 원본 dtype의 전체 범위를 사용해 변환됩니다. 라이브러리에 직접 전달하는 부동소수점 입력은 `[0, 1]` 범위의 유한한 RGB 값이어야 하며, 다른 범위는 거부됩니다. 배치 CLI 추론, 비디오, 스트리밍, 서버 라우트는 이 패밀리의 범위 밖입니다. 지원되는 GPU 백엔드에서 수치 비교를 위해 전체 F32 matmul 누산이 필요하면 프로세스를 시작하기 전에 `MLX_ENABLE_TF32=0`을 설정하십시오.

비디오의 경우 사용 가능한 각 프리뷰는 `<tensor-key>/000000.png`, `<tensor-key>/000001.png` 등으로 기록됩니다. 프리뷰 정규화는 `meta.json`에 기록된 것과 같은 텐서 전체 범위를 사용합니다.

인스턴스 writer는 모든 mask 행이 인스턴스의 `mask_index`로 정확히 한 번씩 참조될 것을 요구합니다. predictor가 `boxes` 또는 `scores` 텐서를 제공하면 그 값은 인스턴스 레코드와 일치해야 하고, 제공하지 않으면 writer가 그 레코드로부터 텐서를 합성합니다.

## Promptable segmentation과 tracking

SAM 3는 텍스트와 box prompt를 받습니다. 이미지 입력의 기본값은 `segment`이며, mask PNG 없이 box와 점수만 기록하려면 `--task detect`를 사용하십시오. `segment`에는 텍스트 prompt가 최소 하나 필요하며, box는 prompt된 객체를 다듬거나 추가할 수 있습니다. 하나의 캐시된 이미지 backbone에 대해 여러 label을 실행하려면 `--prompt`를 반복하십시오.

```bash
mlxcel predict -m models/sam3-bf16 --image street.jpg \
  --prompt "a car" --prompt "a bicycle" \
  --boxes "20,40,300,420;340,80,620,430" -o outputs/street
```

비디오에서는 `--task track`이 frame-zero 텍스트 detection과 box prompt로 객체를 초기화한 뒤 안정적인 `object_id` 값으로 전파합니다.

```bash
mlxcel predict -m models/sam3-bf16 --video clip.mp4 --task track \
  --prompt "a car" -o outputs/tracked-car
```

`--opts '{"threshold":0.4,"nms":0.6}'`은 이미지 점수 임계값과 NMS IoU를 덮어씁니다. SAM 3는 이미지에서 임계값 0.3과 NMS IoU 0.5를 기본값으로 사용합니다. 임계값을 넘는 이미지 인스턴스가 없으면 `instances.json`은 빈 배열이고 mask PNG는 기록되지 않습니다. 비디오의 기본 임계값은 0.15이며, 가장 점수가 높은 frame-zero query로부터 각 텍스트 prompt 객체를 초기화할 때 적용됩니다. 임계값 미만의 prompt는 track을 만들지 않으며, 이미 만들어진 track은 이후 tracker 점수가 낮아져도 유지됩니다.

frame zero 이후의 tracking 점수는 확률이 아니라 upstream tracker의 원시 값이므로 음수일 수 있으며 clamp 없이 기록됩니다. 고정된 upstream 모델은 이후의 mask가 비어 있어도 객체 ID를 유지합니다. 3프레임 DAVIS car 수락 클립에서 고정된 참조 구현은 모든 프레임에서 ID 1을 유지하지만 프레임 1과 2에서는 빈 mask를 만듭니다. fixture는 빈 mask 두 개를 의미 있는 IoU 결과로 취급하지 않고 이 한계를 명시적으로 기록합니다.

SAM 3.1은 model type `sam3.1`과 `sam3.1_video`로 같은 이미지 detector와 출력 파일을 사용합니다. Object Multiplex tracker는 모든 frame-zero 텍스트 detection을 하나의 session에서 초기화하고, 16개를 넘는 객체가 여러 내부 bucket을 필요로 할 때도 이후 프레임마다 tracker를 한 번씩 호출합니다. 비디오 box는 텍스트 detection에 대한 영역 필터이므로 여전히 `--prompt`가 최소 하나 필요합니다.

```bash
mlxcel predict -m models/sam3.1-bf16 --video traffic.mp4 --task track \
  --prompt "a car" --boxes "0,0,640,360" -o outputs/sam31-cars
```

frame-zero 점수는 detector confidence를 사용합니다. 이후 SAM 3.1 점수는 multiplex tracker의 객체 logit에 f32 sigmoid를 취한 값이며, mask는 nearest 샘플링으로 resize되고 0 초과로 임계 처리된 뒤 그 범위로부터 원본 픽셀 단위 box가 도출됩니다. 고정된 공개 MLX 체크포인트의 이름은 `sam3.1-bf16`이지만 저장되고 로드되는 텐서는 모두 f32이므로, mlxcel은 공유 로드 시 dtype 정책에 따라 f32로 유지합니다.

## 모델 통합

모델은 `src/vision/dense/predictor.rs`의 object-safe 인터페이스를 구현합니다.

```rust,ignore
pub trait DensePredictor {
    fn family(&self) -> &'static str;
    fn tasks(&self) -> &[&'static str];
    fn default_task(&self, input: &DenseInput) -> Option<&'static str>;
    fn predict(
        &self,
        input: &DenseInput,
        opts: &serde_json::Map<String, serde_json::Value>,
    ) -> anyhow::Result<DenseOutput>;
}
```

`MlxArray`는 불투명한 C++ bridge 타입이므로 `DenseOutput::tensors`는 `BTreeMap<String, UniquePtr<MlxArray>>`입니다. `depth`, `points`, `normal`, `mask`, `intrinsics`, `segmentation`, `normals`, `pointmaps`, `alphas` 같은 upstream 출력 이름을 유지하십시오. `DenseOutput::meta`는 표준 writer가 보존하는 패밀리별 JSON 값을 담습니다.

패밀리 predictor는 내부 전처리 실패에 `.context("preprocess")`로 단계 컨텍스트를 붙여야 합니다. 명령은 바깥쪽 `predict <task>` 컨텍스트를 제공하고 모델 로딩과 출력 기록은 별도로 명명하므로, 결과 오류 체인이 실패한 단계를 식별합니다.

다음 시그니처의 생성자를 노출하고 `src/vision/dense/registry.rs`에 그 행을 추가하십시오.

```rust,ignore
fn load(path: &std::path::Path) -> anyhow::Result<Box<dyn DensePredictor>>;

DenseFamily {
    model_type: "video_depth_anything",
    supports_prompts: false,
    load: crate::vision::dense::video_depth_anything::load,
}
```

레지스트리 키는 `config.json`의 최상위 `model_type`을 소문자로 한 것과 같아야 합니다. 패밀리가 dense predictor인 경우에만 같은 model type을 `KNOWN_MODEL_TYPES`에 추가하십시오. 이 목록은 실수로 `mlxcel generate`를 쓴 운영자를 `mlxcel predict`로 안내하며, 해당 패밀리가 아직 반영되지 않은 빌드에서도 마찬가지입니다.

공유 전처리 API는 `src/vision/dense`에 있습니다.

```rust,ignore
resize_bilinear_nhwc(x, (height, width), align_corners, antialias)
resize_bicubic_nhwc(x, Some((height, width)), None, antialias)
resize_bicubic_nhwc(x, None, Some((height_scale, width_scale)), antialias)
resize_nearest_nhwc(x, (height, width))
image_to_nhwc_f32(image)
normalize_imagenet(x)
resize_to_patch_multiple(x, DINOV2_PATCH_SIZE)
conv_weight_to_channels_last(weight, expected_in_channels)
```

보간 함수는 rank-4 NHWC 배열을 받으며, 호출 지점에서 기본적으로 `align_corners=false`인 `torch.nn.functional.interpolate`와 일치합니다. Bilinear와 bicubic 보간 연산은 f32를 반환하고, upstream 호환 동일 크기 fast path는 입력 dtype의 복사본을 반환하며, nearest는 dtype을 유지합니다. bicubic `scale_factor`의 경우, upstream과 같이 antialias를 쓰지 않는 경로는 전달된 factor를 유지하고 antialias 경로는 floor된 출력 크기로부터 scale을 도출합니다. convolution 헬퍼는 OIHW에서 OHWI로의 permutation을 적용하기 전에 이미 channel-last인 마지막 입력 채널 축을 인식하여 두 번째 변환을 방지합니다. 모델 로더는 채널 축과 kernel 축이 모호한 shape에 대해 체크포인트 키 레이아웃 감지도 사용해야 합니다.

스트리밍, 웹캠 입력, 서버 라우트는 클라이언트용 dense 출력 wire format이 정의될 때까지 이 파일 계약의 범위 밖입니다.

## SAM 3D Objects geometry

RGB 이미지와 같은 크기의 mask, 또는 alpha가 mask를 제공하는 RGBA 이미지로부터 객체를 재구성합니다.

```bash
mlxcel predict -m models/sam-3d-objects-bf16 --image chair.png --mask chair_mask.png -o out/
mlxcel predict -m models/sam-3d-objects-bf16 --image chair-rgba.png --opts '{"estimate_depth":false}' -o out-no-depth/
```

기본값은 seed 42, structure step 25, latent step 25, 자동 내장 MoGe-3 depth, `formats: ["gaussian", "mesh"]`입니다. `seed`, `ss_steps`, `slat_steps`, `formats`, `estimate_depth`는 `--opts`로 설정하며, `gaussian_4`는 4-Gaussian head를 선택합니다. `--pointmap points.safetensors`는 추정된 depth 대신 HWC 부동소수점 `points` 텐서를 제공합니다. Point map은 +X가 왼쪽, +Y가 위, +Z가 앞쪽이며 NaN은 유효하지 않은 점을 나타냅니다. point map 없이 depth를 비활성화하면 학습된 point-token dropout 경로를 사용합니다.

`mesh.obj`에는 RGB vertex color와 1부터 시작하는 삼각형 face가 들어 있습니다. `gaussian.ply`와 선택적인 `gaussian_4.ply`에는 로그 scale과 logit opacity를 가진 바이너리 little-endian Gaussian 레코드가 들어 있습니다. geometry는 객체 로컬 좌표를 유지합니다. `result.safetensors`에는 `pose.rotation`(WXYZ), `pose.rotation_matrix`, `pose.translation`, `pose.scale`, 정수 `coords`, `latents`, boolean `pointmap_conditioned`가 기록되며, `meta.json`에는 seed와 실제 step 수가 기록됩니다. geometry를 장면에 배치할 때는 반환된 pose를 별도로 적용하십시오.

mask는 최소 두 행과 두 열에 걸쳐 있어야 합니다. 전처리는 큰 할당 전에 원본 변환, crop padding, 정사각형 view, 보간 중간 결과, point-map 정규화를 보수적인 1 GiB live-buffer 보호 장치에 대해 검사합니다. 극단적인 종횡비는 원본 픽셀 수가 입력 제한 이내여도 이 보호 장치를 넘을 수 있으며, 거부된 입력은 암묵적으로 resize되거나 잘리지 않습니다. 다른 prediction 패밀리는 `--mask`와 `--pointmap`을 거부합니다. 공개된 메인 체크포인트는 BF16을 유지하고 내장 depth 모델은 F32를 유지합니다. 공유 DINO trunk는 파라미터를 정확히 비교한 뒤에만 재사용됩니다. 비디오, 배치 JSONL, 스트리밍, 장면 배치, 서빙은 이 패밀리에서 지원되지 않습니다.

## SAM 3D Body human mesh

```bash
mlxcel predict -m models/sam-3d-body-dinov3-bf16 --image person.png \
  --opts '{"bbox":[100,50,400,500]}' -o outputs/body
```

body 전용 predictor는 DINOv3, 카메라 조건이 있는 6개의 decoder 레이어, MHR body 모델을 실행합니다. 공개된 설정은 512x512 affine crop을 명시적으로 지정하며 이 런타임은 그것을 요구합니다. upstream dataclass 기본값인 512x384는 공개 체크포인트와 별개이며 crop 크기가 없을 때 선택되지 않습니다. dense 체크포인트 설정 읽기는 JSON 파싱 전에 8 MiB로 제한됩니다. 저장된 backbone은 BF16을 유지하고, decoder와 MHR 텐서는 F32를 유지하며, 정수 topology는 정수로 유지됩니다. F32 입력 활성화는 upstream 그래프의 자연스러운 promotion을 따릅니다.

`bbox`는 원본 픽셀 단위 `[x1,y1,x2,y2]` box이며 기본값은 전체 이미지입니다. 일부가 밖으로 나간 box는 clip 없이 검은색 padding을 받고, 유한하지 않거나 퇴화했거나 완전히 밖에 있는 box는 실패합니다. `cam_int`는 아홉 개의 row-major 숫자 또는 3x3 행을 받으며, 값은 유한하고 초점 거리는 양수이며 행렬은 가역이어야 합니다. 기본값은 이미지 대각선을 초점 거리로, 이미지 중심을 principal point로 사용합니다. Grayscale과 RGBA 입력은 RGB로 변환됩니다. 사람 box는 호출자가 제공하며, 사람 detection과 별도의 손 refinement pass는 이 런타임의 범위 밖입니다.

색이 없는 `mesh.obj`에는 소수점 여섯 자리의 vertex 18,439개와 1부터 시작하는 삼각형 face 36,874개가 들어 있습니다. `result.safetensors`에는 F32 `vertices [18439,3]`, `keypoints_3d [70,3]`, `joint_coords [127,3]`, `camera [3]`이 들어 있습니다. Vertex와 keypoint는 MHR의 Y/Z 축을 부정한 body 로컬 미터 단위이며, camera는 원시 weak-perspective `[scale,tx,ty]` 출력 그대로입니다. `meta.json`에는 변경되지 않은 box, intrinsics, mesh 경계가 기록됩니다. 공유 writer는 빈 `instances.json`도 내보내며 2D `keypoints.json`은 만들어지지 않습니다. 색이 있는 SAM 3D Objects OBJ 형식은 변경되지 않습니다.
