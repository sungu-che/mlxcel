# 모델 커버리지 분류(triage)

`lablup/mlxcel#548`에서 추적하는 모델 패밀리의 처리 결과(disposition)를 저장소 트리와 공개 Hugging Face 체크포인트 헤더에 대조해 확인한 기록입니다. 각 항목에는 그 판단의 근거가 된 구체적인 증거를 함께 적었습니다. 비생성형 비전 패밀리와 생성형 미디어 패밀리(`lablup/mlxcel#524`)는 [vision-model-scope.md](vision-model-scope.md)에 기록되어 있습니다.

마지막 확인 기준: `main`의 `cf1354f5`.

## 이미 지원됨

| 패밀리 | 체크포인트 / 이슈 | 트리 내 증거 |
|---|---|---|
| `falcon_ocr` | `tiiuae/Falcon-OCR` (`lablup/mlxcel#848`, PR `lablup/mlxcel#1075`) | `src/models/detection.rs` (`"falcon_ocr" => FalconOcrVL`), `src/models/falcon_ocr.rs`, `src/models/falcon_ocr_rope.rs`, `src/vision/falcon_ocr.rs`, `src/vision/processors/falcon_ocr.rs`, `src/loading/vlm_falcon_ocr.rs`, `src/commands/generate_falcon_ocr.rs`, [supported-models.md](supported-models.md) |
| `jina_vlm` | `jinaai/jina-vlm` (`lablup/mlxcel#849`, PR `lablup/mlxcel#1076`) | `src/models/detection.rs` (`"jvlm" \| "jina_vlm" => JinaVLM`), `src/models/jina_vlm.rs`, `src/vision/jina_vlm.rs`, `src/vision/encoders/jina_vlm.rs`, `src/vision/processors/jina_vlm.rs`, `src/loading/vlm_jina_vlm.rs`, [supported-models.md](supported-models.md) |
| `laguna` | Poolside Laguna XS/S (`lablup/mlxcel#1347`, PR `lablup/mlxcel#1739`; DFlash drafter `lablup/mlxcel#1351`, PR `lablup/mlxcel#1771`) | `src/models/detection.rs` (`"laguna" => Laguna`), `src/models/laguna.rs`, `src/models/laguna_layers.rs`, `src/models/laguna_sanitize.rs`, `src/models/laguna_speculative.rs`, [supported-models.md](supported-models.md) |

## 전용 에픽과 포트 이슈로 재분류됨

| 패밀리 | 체크포인트 | 추적 |
|---|---|---|
| `bonsai` | `prism-ml/bonsai-image-ternary-4B-mlx-2bit` (3.9 GB, Apache-2.0) | `lablup/mlxcel#548`에서는 텍스트 생성 런타임의 범위 밖으로 표시되었고, 이미지 생성 에픽 `lablup/mlxcel#2348`(공유 런타임은 `lablup/mlxcel#2341` / PR `lablup/mlxcel#2497`로 반영됨) 아래의 전용 포트 이슈 `lablup/mlxcel#2428`로 재분류되었습니다. 이 포트는 `lablup/mlxcel#2424`의 FLUX.2 Klein 4B transformer를 재사용하고, `transformer-packed-mflux/`의 2-bit `mlx-packed-affine` group-128 가중치를 `UnifiedLinear`로 로드합니다. |

## `lablup/mlxcel#548`에 남은 후보

`falcon_perception`, `zaya1_vl`, `hrm_text`, `multi_modality` 중 현재 `src/models/detection.rs`에 `model_type` arm이 있는 것은 없습니다. 이들의 공개 체크포인트, `config.json` 스키마, safetensors 헤더는 아래에 기록했습니다.

---

## `falcon_perception`

공개 체크포인트는 두 개이며, 모두 Apache-2.0이고 `model_type: "falcon_perception"`, fp32 가중치입니다.

| | `tiiuae/Falcon-Perception` | `tiiuae/Falcon-Perception-300M` |
|---|---|---|
| 크기 | 2.53 GB, 텐서 231개 | 1.27 GB, 텐서 123개 |
| Decoder | `dim` 1024, `n_layers` 28, `n_heads` 16, `n_kv_heads` 8, `head_dim` 128, `ffn_dim` 3072, `vocab_size` 65536 | `dim` 768, `n_layers` 22, `n_heads` 16, `n_kv_heads` 8, `head_dim` 64, `ffn_dim` 2304, `vocab_size` 65536 |
| `freqs_cis_golden` | `[16, 32, 2]` | `[16, 16, 2]` |
| `do_segmentation` | `true` (`segm_out_dim` 256, `num_segm_layers` 3) | `false` (detection 전용) |

### Decoder와 트리 내 `falcon_ocr`과의 호환성

- `modeling_falcon_perception.py`의 `Attention`, `FeedForward`, `TransformerBlock`을 `modeling_falcon_ocr.py`와 diff했습니다. 주석과 import를 제외하면 동일합니다.
- `tiiuae/Falcon-Perception-300M`의 `modeling_falcon_perception.py`를 `tiiuae/Falcon-Perception`과 diff했습니다. 모든 모델 및 레이어 클래스(`Attention`, `FeedForward`, `TransformerBlock`, `FourierEncoder`, `MLPDecoder`, `FalconPerceptionModel`)가 동일합니다. 300M 파일은 `generate()` / `_postprocess_aux()`에 `task: str | None = None` 인자(`"detection"` 대 `"segmentation"`)를 추가하고, `upsample_img_features`와 `proj_segm`을 `self.config.do_segmentation` 뒤에서 보호하는 것만 다릅니다.
- 두 체크포인트 모두 `falcon_ocr`과 같은 5개의 레이어별 텐서 이름 `layers.{i}.attention.{wqkv.weight, wo.weight, sinks}`, `layers.{i}.feed_forward.{w13.weight, w2.weight}`를 사용하며, 그 외에 `tok_embeddings.weight`, `img_projector.weight`, `freqs_cis_golden`, `norm.weight`, `output.weight`가 있습니다.
- `src/models/falcon_ocr.rs`, `src/models/falcon_ocr_rope.rs`, `src/vision/falcon_ocr.rs`, `src/loading/vlm_falcon_ocr.rs`를 두 config에 대해 감사했습니다. `FalconOcrConfig`는 `head_dim`(`self.head_dim.unwrap_or(self.dim / self.n_heads)`), `n_kv_heads`, `patch_dim`, `temporal_inv_freq(cfg.head_dim() / 2, cfg.rope_theta)`를 `config.json`에서 도출하고, `golden_cos_sin`은 `freqs_cis_golden`의 `[n_heads, head_dim / 4, 2]` shape을 런타임에 읽습니다. `falcon_ocr` decoder에는 `dim`이나 `head_dim`을 하드코딩한 곳이 없습니다.

### `falcon_ocr` 위에 포트가 더하는 것

- Detection head(두 체크포인트 모두에 있으며 텐서 8개): `coord_encoder.{embed,transform}.weight`(`[256, 2]`, `[dim, 512]`), `size_encoder.{embed,transform}.weight`(`[256, 2]`, `[dim, 512]`), `coord_decoder.{w1,w2}.weight`(`[8192, dim]`, `[2048, 8192]`), `size_decoder.{w1,w2}.weight`(`[8192, dim]`, `[2048, 8192]`). Encoder는 연속값 `(x, y)` / `(h, w)` 예측을 `coord_token_id`(240)와 `size_token_id`(241) 위치의 임베딩으로 되돌려 매핑하며, decoder는 2048개의 좌표/크기 bin에 대한 2층 squared-ReLU MLP입니다.
- Box 디코딩 루프: 좌표와 크기는 `coord_token_id` / `size_token_id`에서 bin argmax로 디코딩되며, 이전 box와 1% 이내인 box 예측은 거부되고 다시 샘플링됩니다(최대 100회 시도).
- 큰 체크포인트에만 있음(300M safetensors 파일에는 없는 추가 텐서 78개): `seg_token_id`(262)에서의 segmentation. `proj_segm.{layers.0, layers.1, pixel_layer}`, `conv_segm.{weight,bias}`(`conv_segm.weight`는 `[256, 1024, 3, 3]`로 저장되어 있어 로드 시 표준 `(0, 2, 3, 1)` channels-last permutation이 필요), `itok_upsampler.*`(`anyup.py`의 479줄짜리 AnyUp cross-attention 특징 업샘플러로, `image_encoder`, `query_encoder`, `cross_decode`, `aggregation`, `rope.freqs`를 가지며 4차원 `1x1` 및 `3x3` conv 가중치도 channels-last permutation이 필요)를 사용합니다.
- 프롬프트 계약: `<|image|>Segment these expressions in the image:<|start_of_query|>{query}<|REF_SEG|>`.

### 제안하는 분할과 미결 사항

1. `tiiuae/Falcon-Perception-300M`(detection 전용)을 먼저 포팅합니다. `FalconOcrTextModel` / `FalconOcrProcessor`를 재사용하고, 8개의 Fourier/MLP head 텐서와 box 디코딩 루프를 추가하며, box는 `mlxcel detect`(`src/commands/detect.rs`)를 통해 노출합니다.
2. `tiiuae/Falcon-Perception`의 segmentation 경로(`conv_segm`, `proj_segm`, `itok_upsampler`, RLE mask)는 에픽 `lablup/mlxcel#2382`(`lablup/mlxcel#2429`)의 인스턴스 마스크 출력 계약이 마련된 뒤에 별도로 범위를 정합니다.

---

## `zaya1_vl`

- **체크포인트**: `Zyphra/ZAYA1-VL-8B` (bf16 safetensors shard 4개에 걸쳐 19.4 GB, 텐서 5,833개).
- **확인된 `config.json` 키**: 최상위 `model_type: "zaya1_vl"`, `architectures: ["Zaya1VLForConditionalGeneration"]`, `hidden_size: 2048`, `num_hidden_layers: 40`, `num_attention_heads: 8`, `num_key_value_heads: 2`, `num_query_groups: 2`, `head_dim: 128`, `ffn_hidden_size: 4096`, `num_experts: 16`, `moe_router_topk: 1`, `zaya_mlp_expansion: 256`, `zaya_use_eda: true`, `zaya_use_mod: true`, `cca: true`, `rope_pct: 0.5`, `rotary_base: 1000000`, `scale_residual_merge: true`, `tie_word_embeddings: true`, `vocab_size: 262272`, `image_token_id: 262147`, `vision_start_token_id: 255999`, `vision_end_token_id: 256000`, `vision_lora: true`, `vision_lora_rank_attn: 8`, `vision_lora_rank_mlp: 32`, 그리고 `model_type: "qwen2_5_vl"`인 중첩 `vision_config`(`hidden_size: 1280`, `out_hidden_size: 2048`, `spatial_patch_size: 14`, `temporal_patch_size: 1`).
- **확인된 가중치 레이아웃**:
  - Vision tower(`vision_tower.patch_embed.proj.weight` `[1280, 3, 1, 14, 14]`, `vision_tower.blocks.{0..31}.*`, `vision_tower.merger.{ln_q, mlp.0, mlp.2}`)는 `src/vision/encoders/qwen2_5_vl.rs`와 일치합니다.
  - 텍스트 decoder(`model.layers.{0..39}.*`, 레이어당 텐서 133개):
    - Convolution이 추가된 attention(`attn.self_attn.qkv.*`): `linear_q.weight` `[1024, 2048]`, `linear_k.weight` `[256, 2048]`, causal 1차원 convolution `conv_qk.0.{weight,bias}`(`[1280, 1, 2]`)와 `conv_qk.1.{weight,bias}`(`[1280, 128, 2]`, `(out, in/groups, k)`에서 `(out, k, in/groups)`로의 permutation 필요), 시간 이동된 hidden state를 위한 이중 value projection `val_proj1.weight`와 `val_proj2.weight`(각각 `[128, 2048]`), 학습되는 KV head별 temperature `qkv.temp` `[2]`, 출력 projection `o_proj.weight` `[2048, 1024]`, 그리고 rank-8 vision LoRA adapter(`lora_linear_{q,k,o}.{0,1}.weight`, `lora_val_proj{1,2}.{0,1}.weight`).
    - MoE + Mixture-of-Depths 블록(`mlp.zaya_block.*`): router(`router.down_proj.{weight,bias}` `[256, 2048]`, `router.rmsnorm_eda.weight` `[256]`, `256 -> 256 -> 256 -> 17`로 projection하는 `router.router_mlp.{0,2,4}`, fp32의 `router.balancing_biases` `[17]`)에서 17번째 슬롯은 MoD skip expert이며, 여기에 각각 rank-32 vision LoRA adapter(`lora_fc1.{0,1}.weight`, `lora_fc2.{0,1}.weight`)를 가진 16개의 SwiGLU expert `experts.local_experts.{0..15}.{linear_fc1, linear_fc2}.weight`(`[4096, 2048]`, `[2048, 2048]`)가 있습니다.
    - 서브레이어별 residual scaling(`attn.res_scale.{hidden_states_scale, hidden_states_bias}`와 `mlp.res_scale.{hidden_states_scale, hidden_states_bias, residual_scale, residual_bias}`), 그리고 최상위 `model.res_scale.*`.
- **포트 변경 범위(delta)**: `lablup/mlxcel#548` 목록에서 가장 큽니다. Vision tower는 `qwen2_5_vl.rs`를 재사용하지만, 텍스트 backbone에는 사용자 정의 레이어별 하이브리드 cache(KV cache + 2단계 causal conv/token-shift 상태), 레이어 간 router 상태 전달(`zaya_use_eda`), MoD skip-expert 라우팅(`zaya_use_mod`), 그리고 attention과 expert 양쪽의 image-token 마스크 LoRA 분기가 필요합니다.

---

## `hrm_text`

- **체크포인트**: `sapientinc/HRM-Text-1B`(bf16 텐서 131개)와 `Xiaoye08/HRM-Text-0.6B`(bf16 텐서 99개, `configuration_hrm_text.py`와 `modeling_hrm_text.py` 포함). `sapientinc/HRM-Text-1B-v3.2`는 gated입니다.
- **확인된 `config.json` 키**:
  - `sapientinc/HRM-Text-1B`: `model_type: "hrm_text"`, `architectures: ["HrmTextForCausalLM"]`, `vocab_size: 65536`(`lablup/mlxcel#548`에서 처음 적은 151808이 아님), `hidden_size: 1536`, `intermediate_size: 4096`, `num_hidden_layers: 16`, `num_attention_heads: 12`, `num_key_value_heads: 12`, `head_dim: 128`, `H_cycles: 2`, `L_cycles: 3`, `L_bp_cycles: [0, 3]`, `embedding_scale: 39.191835884530846`, `prefix_lm: true`, `rms_norm_eps: 1e-6`, `rope_theta: 10000.0`, `tie_word_embeddings: false`.
  - `Xiaoye08/HRM-Text-0.6B`: 같은 스키마에 `hidden_size: 1280`, `intermediate_size: 3584`, `num_hidden_layers: 12`, `num_attention_heads: 10`, `num_key_value_heads: 10`, `head_dim: 128`이며, `L_bp_cycles`의 키 별칭으로 `L_bp_steps: [0, 3]`을 사용합니다.
- **확인된 가중치 레이아웃**: `model.embed_tokens.weight`(`[65536, hidden_size]`), `model.z_L_init`(`[hidden_size]`), `lm_head.weight`(`[65536, hidden_size]`), 그리고 `num_hidden_layers` 깊이의 별도 스택 두 개 `model.H_module.layers.{0..N-1}.*`와 `model.L_module.layers.{0..N-1}.*`. 각 레이어에는 가중치 텐서 4개가 있으며(norm은 파라미터가 없음) `attn.gqkv_proj.weight`(`[4 * hidden_size, hidden_size]`, gate + Q + K + V를 fuse한 것), `attn.o_proj.weight`(`[hidden_size, hidden_size]`), `mlp.gate_up_proj.weight`(`[2 * intermediate_size, hidden_size]`), `mlp.down_proj.weight`(`[hidden_size, intermediate_size]`)입니다.
- **포트 변경 범위(delta)**: 개별 레이어 연산은 간결하지만(sigmoid attention 출력 gate가 있는 fused `gqkv_proj`, fused `gate_up_proj` SwiGLU, 가중치가 없는 RMSNorm), 모든 token step이 `H_cycles * (L_cycles + 1) = 8`번의 module 평가(1B에서 유효 레이어 평가 `128`번)를 재생하고 순환 상태 덧셈(`z_L`, `z_H`)과 prefix-LM 양방향 프롬프트 마스킹(`prefix_lm: true`)이 필요하므로, 전용 순환 cache 관리가 필요합니다.

---

## `multi_modality` (DeepSeek-VL v1)

- **체크포인트**: `deepseek-ai/deepseek-vl-1.3b-chat`(fp16 텐서 529개; `mlx-community/deepseek-vl-1.3b-chat-4bit`)와 `deepseek-ai/deepseek-vl-7b-chat`(shard 3개에 걸쳐 fp16 텐서 775개, 14.7 GB; `mlx-community/deepseek-vl-7b-chat-4bit`).
- **확인된 `config.json` 및 가중치 레이아웃**:
  - 두 체크포인트 모두 최상위 `model_type: "multi_modality"`, `architectures: ["MultiModalityCausalLM"]`, 그리고 `model_type: "llama"`, `vocab_size: 102400`인 `language_config`를 가집니다(1.3B는 2048 x 24 레이어, 7B는 4096 x 30 레이어). 텍스트 가중치는 `language_model.model.*`와 `language_model.lm_head.weight` 아래에 있으며 `src/models/llama3.rs`에 그대로 매핑됩니다.
  - **1.3B (`CLIPVisionTower` + `mlp_gelu`)**: `vision_model.vision_tower.{patch_embed.proj, pos_embed, blocks.0..23, norm, attn_pool}` 아래의 단일 384px SigLIP-L/16 tower(`vision_config.params.model_name: "siglip_large_patch16_384"`; `patch_embed.proj.weight` `[1024, 3, 16, 16]`, `pos_embed` `[1, 576, 1024]`, fused `attn.qkv`)와 2층 GELU MLP projector `aligner.layers.{0,2}.{weight,bias}`(`1024 -> 2048 -> 2048`)로 구성됩니다. 이 vision tower는 DeepSeek-VL2(`lablup/mlxcel#533`)용으로 `src/vision/encoders/deepseek_vl2.rs`에 이미 구현된 것과 키와 블록 구조가 정확히 같습니다.
  - **7B (`HybridVisionTower` + `low_high_hybrid_split_mlp_gelu`)**: 384px SigLIP tower(`vision_model.vision_tower_low.vision_tower.*`)와 1024px SAM-B 고해상도 tower(`vision_model.vision_tower_high.vision_tower.{patch_embed, pos_embed, blocks.0..11, neck, neck_hd, downsamples, hd_alpha_downsamples}`, windowed relative-position attention `rel_pos_h` / `rel_pos_w`와 conv downsampler 사용), 그리고 분리된 projector(`vision_model.{low,high}_layer_norm.{weight,bias}`, `aligner.{low,high}_up_proj.{weight,bias}`, `aligner.layers.1.{weight,bias}`)를 결합합니다.
- **분류 맥락**: DeepSeek-VL2(`model_type: "deepseek_vl_v2"`, `lablup/mlxcel#533`)는 이미 `src/vision/deepseek_vl2.rs`와 `src/loading/vlm_deepseek_vl2.rs`에서 지원되며 v1을 대체합니다. 나중에 v1 지원이 필요해지더라도 1.3B 변형은 `src/models/llama3.rs`와 `src/vision/encoders/deepseek_vl2.rs` 위의 작은 wrapper이면 되지만, 7B 변형은 `sam_b_downsample` encoder와 `low_high_hybrid_split_mlp_gelu` aligner를 포팅해야 합니다.
