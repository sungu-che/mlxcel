# 임베딩 및 리랭킹 API와 CLI

mlxcel은 OpenAI 호환 `POST /v1/embeddings` 엔드포인트와 오프라인 `mlxcel embed` 명령으로 임베딩 체크포인트를 서비스합니다. 이 페이지는 모든 임베딩 계열이 공통으로 사용하는 기반, 즉 감지, 풀링, 정규화, 길이 제한, 요청 및 응답 스키마, 서버 플래그, CLI를 다룹니다. 계열별 프롬프트 접두사, instruction 형식, 이미지 지원은 아래에 문서화되어 있으며, 관리되는 계열 표는 [supported-models.md](supported-models.md#embedding-models)에 있습니다.

관련성 점수 계산은 같은 페이지의 [리랭킹](#reranking-v1rerank-and-mlxcel-rerank)에서 다룹니다. `POST /v1/rerank`와 `mlxcel rerank`는 이 서브시스템의 토크나이저, 길이 제한, worker 구성을 공유하며, 세 가지 reranker 종류 중 두 가지는 임베딩 계열이 이미 보유한 인코더 위에서 동작합니다.

**현재 상태.** 두 엔드포인트, 전용 worker 스레드, 배칭, 오프라인 명령, 그리고 임베딩 및 reranker 지원 표에 있는 모든 forward pass가 구현되어 있습니다. [임베딩](supported-models.md#embedding-models) 표와 [reranker](supported-models.md#reranker-models) 표가 서비스되는 계열과 체크포인트별 제약 사항에 대한 공식 기준입니다.

## 구현된 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|-------------|
| POST | `/v1/embeddings` | 문자열, 문자열 목록, token id 배열, token id 배열의 목록, 또는 타입이 지정된 part 목록을 임베딩합니다. |
| POST | `/embeddings` | `/v1` 접두사가 없는 별칭입니다. |
| POST | `/v1/rerank` | 쿼리를 문서 목록과 비교하여 점수를 매깁니다. Cohere 및 Jina와 호환되며, [리랭킹](#reranking-v1rerank-and-mlxcel-rerank)을 참고하십시오. |
| POST | `/rerank` | `/v1` 접두사가 없는 별칭입니다. |
| POST | `/v1/reranking` | `/v1/rerank`에 대한 llama-server 별칭입니다. |
| POST | `/reranking` | `/rerank`에 대한 llama-server 별칭입니다. |
| GET | `/v1/models` | `--embedding-model` / `--reranker-model`이 두 번째 체크포인트를 로드한 경우, 채팅 모델과 함께 임베딩 모델과 reranker를 나열합니다. |

## 구현 소스 맵

| 모듈 | 역할 |
|--------|----------------|
| `src/embeddings/model.rs` | `EmbeddingModel` trait, `EmbeddingBatch`, `EmbeddingOutput`, `ImageInput`. |
| `src/embeddings/pooling.rs` | `PoolingMode`, `1_Pooling/config.json` 리더, `pool`, `normalize_l2`, `truncate_dimensions`, `MLXCEL_EMBEDDING_POOLING` 재정의. |
| `src/embeddings/limits.rs` | `max_length` 도출, pad token 및 어휘 크기. |
| `src/embeddings/tokenize.rs` | 오른쪽 패딩 배치 토크나이즈, 끝 특수 토큰 보존 truncation, 쌍(pair) 인코딩. |
| `src/embeddings/loader.rs` | `load_embedding_model`: 계열 dispatch, 하위 폴더 가중치, bf16 규칙. |
| `src/embeddings/engine.rs` | 길이 순 정렬 micro-batching, 정규화, `dimensions`, readback. |
| `src/embeddings/maxsim.rs` | `maxsim` / `maxsim_mlx`: multi-vector 계열의 순위를 매기는 late-interaction 점수. |
| `src/lib/mlxcel-core/src/utils.rs` | `create_bidirectional_padding_mask`, `create_causal_padding_mask`, `create_bidirectional_window_mask`. |
| `src/lib/mlxcel-core/src/weights.rs` | `load_weights_from_dir_with_subfolders` (`2_Dense/...` 텐서에 `2_Dense.` 접두사가 붙음). |
| `src/models/detection.rs` | `is_embedding_checkpoint`: 아래의 감지 규칙. |
| `src/models/bert.rs` | BERT / XLM-RoBERTa 인코더 trunk: 가중치 sanitize, position id, 블록. |
| `src/models/bert_config.rs` | BERT / XLM-RoBERTa config 해석, `bert`를 통해 재노출됩니다. |
| `src/models/bert_heads.rs` | 해당 trunk 위의 `BertEmbeddingModel`과 `BertSequenceClassifier`. |
| `src/models/modernbert.rs`, `src/models/modernbert_heads.rs` | ModernBERT 인코더와 그 임베딩 / 시퀀스 분류 head. |
| `src/models/siglip_text.rs` | SigLIP 텍스트 tower, 고정 폭 토크나이즈 규약, projection 및 풀링. |
| `src/server/embedding_model.rs` | `EmbeddingModelProvider` trait과 `EmbeddingError`. |
| `src/server/embedding_worker.rs` | `EmbeddingWorker`: MLX를 소유하는 전용 스레드, 유한 큐, 타임아웃, panic 경계. |
| `src/server/routes/embeddings.rs` | HTTP 핸들러, 검증 순서, 오류 매핑. |
| `src/server/types/embeddings.rs` | 요청 및 응답 타입, base64 인코딩. |
| `src/commands/embed.rs` | `mlxcel embed`. |
| `src/models/embedding_sanitize.rs` | decoder-backbone 계열이 공유하는 가중치 키 정규화: `{N}_Dense.linear.*` 병합, head 제거, `model.` 접두사 부여. |
| `src/models/gemma3_embedding.rs` | EmbeddingGemma: 양방향 Gemma 3, mean 풀링, 두 개의 `Dense` projection. |
| `src/models/embedding_gemma2.rs` | EmbeddingGemma 2: 멀티모달 양방향 Gemma 4, projection 전용 PLE, 512 -> 768 `embedding_projection`, mean 풀링. |
| `src/models/qwen3_embedding.rs` | Qwen3-Embedding: causal Qwen3, last-token 풀링. |
| `src/models/qwen3_vl_embedding.rs` | Qwen3-VL-Embedding: 멀티모달 채팅 포맷팅, vision 주입 및 last-token 풀링. |
| `src/models/llama_nemotron_vl_embedding.rs` | Llama-Nemotron-VL-Embed: SigLIP 타일링, projection 및 양방향 Llama 풀링. |
| `src/models/col_late_interaction.rs` | 두 late-interaction 계열이 공유: `embedding_dim`, `1_Dense` projection 재정의, LoRA 전용 거부, token별 projection 및 정규화, 쿼리 형식. |
| `src/models/colidefics3.rs` | ColIdefics3: head가 없는 SmolVLM / Idefics3에 128차원 projection을 더한 것. |
| `src/models/colqwen2_5.rs` | ColQwen2.5: head가 없는 Qwen2.5-VL에 128차원 projection을 더한 것. |
| `src/models/llama_bidirec.rs` | 양방향 Llama (LLM2Vec): padding 전용 마스크 아래의 Llama 3 레이어, mean 풀링. |
| `src/models/ministral3_embedding.rs` | Nemotron-3-Embed: 양방향 Ministral 3, offset 0에서의 Llama 4 attention scaling, mean 풀링. |
| `src/models/lfm2_embedding.rs` | LFM2.5-Embedding: 양방향 LFM2 (비인과 short conv), CLS 풀링. |
| `src/rerank/mod.rs` | `Reranker` trait, `RerankerKind`, `RerankItem`, `detect_reranker_kind`. |
| `src/rerank/loader.rs` | `load_reranker`: 종류별 dispatch, batch 크기 및 길이 재정의. |
| `src/rerank/sequence_classifier.rs` | Cross-encoder 경로: 쌍 토크나이즈, longest-first truncation, `sigmoid(logit)`. |
| `src/rerank/qwen3_generative.rs` | Qwen3 yes/no 경로: 프롬프트 레시피, 왼쪽 패딩, `sigmoid(logit(yes) - logit(no))`. |
| `src/rerank/qwen3_vl_generative.rs` | Qwen3-VL yes/no 경로: `reranker.jinja`, 이미지 병합, `1_LogitScore` token id. |
| `src/server/rerank_model.rs` | `RerankModelProvider` trait과 `RerankError`. |
| `src/server/rerank_worker.rs` | `RerankWorker`: MLX를 소유하는 전용 스레드, 유한 큐, 타임아웃, panic 경계. |
| `src/server/routes/rerank.rs` | HTTP 핸들러, 검증 순서, 정렬, `top_n`, `return_documents`. |
| `src/server/types/rerank.rs` | 요청 및 응답 타입, 정렬 후 truncation 규칙. |
| `src/commands/rerank.rs` | `mlxcel rerank`. |

## 감지

다음 중 하나라도 성립하면, 일반적인 `model_type` dispatch가 실행되기 전에 체크포인트가 임베딩 모델로 서비스됩니다.

- `config.json`의 `model_type`이 인코더 전용 계열입니다: `bert`, `xlm-roberta`, `modernbert`, `siglip`, `embedding_gemma2`, `embedding_gemma2_text`. 이들은 텍스트를 생성하지 않으며, `BertForMaskedLM` 또는 `ModernBertForMaskedLM` export는 MLM head를 제거한 임베더로 로드됩니다.
- `config.json`의 `architectures[0]`이 임베딩 아키텍처입니다: `BertModel`, `XLMRobertaModel`, `ModernBertModel`, `SiglipModel`, `SiglipTextModel`, `EmbeddingGemma2Model`, `use_bidirectional_attention: true`인 `Gemma3TextModel`, `LlamaBidirectionalModel`, `LlamaNemotronVLModel`, `Lfm2BidirectionalModel`, `is_causal: false`인 `Ministral3Model`, `ColIdefics3`, `ColQwen2_5`, `ColQwen2ForRetrieval`.
- `modules.json`이 존재하고 `type`이 `.Pooling`으로 끝나는 모듈을 나열합니다(sentence-transformers 레이아웃). 추가 모듈이 `1_LogitScore`뿐인 `modules.json`(Qwen3-VL-Reranker)은 해당되지 않습니다.
- `1_Pooling/config.json`이 존재합니다.

`architectures[0]`이 `ForSequenceClassification`으로 끝나는 체크포인트는 나머지 레이아웃이 어떻든 임베더가 아니라 reranker입니다. `bert`, `xlm-roberta`, `modernbert`에서는 `ModelType::SequenceClassifier`로 감지되어 `/v1/rerank`에서 서비스되고, 다른 계열에서는 일반적인 `model_type` dispatch로 넘어가 거기서 오류가 보고됩니다.

결정된 계열은 `model_type`을 키로 합니다.

| `model_type` | `ModelType` |
|--------------|-------------|
| `bert` | `Bert` |
| `xlm-roberta` | `XlmRoberta` |
| `modernbert` | `ModernBert` |
| `siglip` | `SiglipText` |
| `gemma3_text`, `gemma3` | `Gemma3Embedding` |
| `embedding_gemma2`, `embedding_gemma2_text` | `EmbeddingGemma2` |
| `qwen3` | `Qwen3Embedding` |
| `qwen3_vl` | `Qwen3VLEmbedding` |
| `lfm2` | `Lfm2Embedding` |
| `ministral3` | `Ministral3Embedding` |
| `llama`, `llama_bidirec` | `LlamaBidirec` |
| `llama_nemotron_vl` | `LlamaNemotronVLEmbedding` |
| `idefics3` | `ColIdefics3` |
| `qwen2_5_vl`, `colqwen2` | `ColQwen25` |

그 외의 `model_type`에 풀링 레이아웃이 있으면 causal 생성기로 잘못 라우팅되지 않고, `model_type`을 명시한 오류로 보고됩니다. 풀링 레이아웃이 없는 `Qwen3ForCausalLM`은 여전히 `Qwen3` 생성기로 감지됩니다.

`mlxcel arch`는 임베딩 변형을 `Embedding` 계열 아래에, `SequenceClassifier`를 `Reranker` 아래에 나열합니다. 생성 로더(`mlxcel generate`, `mlxcel run`, 채팅 worker가 사용하는 `load_model`)는 모든 임베딩 변형을 `/v1/embeddings`와 `mlxcel embed`를 안내하는 메시지와 함께 거부합니다. 따라서 `mlxcel-server -m <embedding checkpoint>`는 임베딩을 서비스하고 채팅은 로드하지 않은 상태로 둡니다(Whisper 체크포인트가 채팅을 로드하지 않은 상태로 두는 것과 같은 방식). 이때 `/v1/chat/completions`는 기존의 "model is not loaded" 오류를 반환합니다.

## 풀링

hidden state `[B, L, D]`(f32로 캐스팅)와 attention mask `[B, L]`(int32, `1` = 실제 token, `0` = 패딩)가 주어졌을 때:

| 모드 | 정의 |
|------|------------|
| `cls` | `first_real_index = argmax(mask, axis=1)`일 때 `hidden[b, first_real_index[b], :]`. 오른쪽 패딩이면 인덱스 0, 왼쪽 패딩이면 첫 번째 `1`입니다. |
| `mean` | `sum(hidden * mask[..., None], axis=1) / max(sum(mask, axis=1), 1e-9)`. |
| `max` | `max(where(mask == 0, -inf, hidden), axis=1)`. |
| `lasttoken` | `hidden[b, last_real_index[b], :]`. 모두 패딩인 행은 `L - 1`을 사용합니다. 왼쪽 및 오른쪽 패딩 모두에서 올바릅니다. |

지원되지 않는 모드(`weightedmean`, `mean_sqrt_len_tokens`, `include_prompt: false`, 둘 이상의 legacy 플래그 설정)는 조용한 fallback이 아니라 모드를 명시한 로드 오류입니다.

해석 순서:

1. `<model_dir>/1_Pooling/config.json`이 있으면 이를 사용합니다. 두 가지 키 스타일을 모두 허용합니다. 새 스타일 `"pooling_mode": "cls" | "mean" | "max" | "lasttoken"`과 legacy 불리언 `pooling_mode_cls_token`, `pooling_mode_mean_tokens`, `pooling_mode_max_tokens`, `pooling_mode_lasttoken`, `pooling_mode_weightedmean_tokens`, `pooling_mode_mean_sqrt_len_tokens`입니다. legacy 플래그는 정확히 하나만 true일 수 있으며, 모두 false이면 `mean`입니다.
2. 없으면 계열 기본값(`EmbeddingModel::default_pooling()`)을 사용합니다.
3. `MLXCEL_EMBEDDING_POOLING=cls|mean|max|lasttoken`은 디버깅용으로 위 둘을 모두 재정의하며 시작 시 로그에 기록됩니다.

정규화는 벡터별 L2, 즉 `v / max(||v||_2, 1e-9)`이며 `EmbeddingModel::normalize()`가 true일 때 적용됩니다(기본값. 이를 선언하는 계열에서는 `config.json`의 `normalize: false`로 끌 수 있습니다). 요청 필드 `dimensions`는 처음 `dimensions`개의 성분을 유지하며 정규화가 켜져 있으면 다시 정규화합니다. `1 <= dimensions <= D`여야 하며 그 외의 값은 `400`입니다.

## 길이 제한과 패딩

`max_length`는 다음 중 가장 작은 값입니다. `sentence_bert_config.json`의 `max_seq_length`(있는 경우), `0 < value < 1_000_000`일 때 `tokenizer_config.json`의 `model_max_length`, 절대 위치 인코더(BERT, XLM-RoBERTa, SigLIP)의 경우 `config.json`의 `max_position_embeddings`, 하드 상한 `8192`, 그리고 설정된 경우 `--embedding-max-length`입니다. `max_length`보다 긴 입력은 특수 토큰이 추가된 뒤 오른쪽에서 잘리며, 토크나이저가 끝 특수 토큰을 덧붙이는 경우 그 토큰은 유지됩니다. 따라서 BERT 입력은 `[SEP]`를, Qwen3-Embedding 입력은 `<|endoftext|>`를 유지합니다. token id 입력은 그대로 오른쪽에서 잘립니다.

`sentence-transformers/all-MiniLM-L6-v2`에서는 `256`으로 결정되며(`sentence_bert_config.json`에서), `Qwen/Qwen3-Embedding-0.6B`에서는 상한인 `8192`로 결정됩니다(`sentence_bert_config.json`이 없고, `model_max_length`는 131072이며, rotary 위치를 사용하므로 `max_position_embeddings`는 참조되지 않습니다).

micro-batch는 가장 긴 항목에 맞춰(또는 SigLIP 텍스트처럼 계열이 고정 폭을 요구하면 그 폭에 맞춰) 오른쪽에 pad token id로 패딩됩니다. pad token id는 `tokenizer_config.json`의 `pad_token`이고, 없으면 `eos_token`, 그다음 `0`입니다. `tokenizer.json`에 박혀 있는 패딩이나 truncation(MiniLM은 128로 고정 패딩)은 로드 시 제거됩니다. 엔진이 micro-batch마다 패딩하고 체크포인트 제한에 따라 truncation하기 때문입니다.

## 계열별 참고 사항

프롬프트 접두사는 어디서나 호출자 측 책임입니다. 서버 측에서 주입되는 것은 없으므로 입력은 보낸 그대로 임베딩됩니다. `instruction` 요청 필드(그리고 `mlxcel embed --instruction`)는 `EmbeddingModel::format_text`에 전달되며, 아래에 형식을 문서화한 계열만 이를 사용합니다.

### 풀링 및 정규화 재정의

llama-server b10621 플래그 두 개는 체크포인트가 요구하는 값을 재정의하며, 서버와 `mlxcel embed` 모두에서 동일하게 동작합니다(issue #1452).

| 플래그 | 값 | 미설정 시 |
|---|---|---|
| `--pooling` | `mean`, `cls`, `last` (b10621에는 `none`과 `rank`도 있습니다. 아래 참고) | `1_Pooling/config.json`, 그다음 계열 기본값 |
| `--embd-normalize` | `-1` 없음, `0` max absolute int16, `1` taxicab, `2` euclidean, 2 초과는 해당 p-norm | 체크포인트 `config.json`의 `normalize` 플래그. 달리 지정하지 않으면 euclidean입니다 |

`--pooling`은 `1_Pooling/config.json`과 `MLXCEL_EMBEDDING_POOLING` 변수보다 우선하며, `GET /props`는 실제로 결정된 모드를 보고하므로 체크포인트가 재정의한 값을 가정하지 않고 확인할 수 있습니다. mlxcel 고유의 `max` 풀링은 b10621에 대응하는 표기가 없으므로 `MLXCEL_EMBEDDING_POOLING`과 체크포인트 config를 통해서만 사용할 수 있습니다.

b10621의 나머지 두 `--pooling` 값은 풀링 커널이 아닙니다. `rank`는 모델을 reranking 경로에 올리는 방법이므로 mlxcel은 이를 `--reranking`의 동의어로 받아들입니다. `none`은 token마다 하나의 벡터를 요구하는데, 모든 mlxcel 임베딩 계열은 엔진이 출력을 보기 전에 자체 forward pass 안에서 풀링하므로 어떤 계열도 이를 반환할 수 없습니다. 따라서 그 이유와 함께 시작 시 거부됩니다.

`--embd-normalize`는 `/embedding`, `/embeddings`, `/v1/embeddings`에서 요청별로 `embd_normalize`로도 지정할 수 있습니다. 영벡터는 upstream과 동일하게 어떤 모드에서도 NaN이 아니라 0으로 정규화됩니다.


### BERT 및 XLM-RoBERTa

하나의 인코더가 둘 모두를 처리하며(`src/models/bert.rs`), `BertVariant` 스위치로 선택되는 절대 위치 임베딩 위의 post-LayerNorm 블록 스택입니다. `model_type: bert`는 position id를 `0..L`로 만들고 실제 `token_type_ids` 축을 가집니다. `model_type: xlm-roberta`는 `cumsum(input_ids != pad_token_id) * mask + pad_token_id`로 만들기 때문에 첫 번째 실제 token이 `pad_token_id + 1`에 놓이고 패딩은 `pad_token_id`에 머뭅니다. `intfloat/multilingual-e5-small`은 XLM-RoBERTa 토크나이저를 포함해 배포되지만 `BertModel`이므로 BERT 규칙을 따릅니다.

| 체크포인트 | 풀링 | 체크포인트가 기대하는 접두사 |
|------------|---------|-------------------------------|
| `sentence-transformers/all-MiniLM-L6-v2` | mean | 없음 |
| `intfloat/multilingual-e5-small` (및 다른 `multilingual-e5-*` 크기) | mean | 검색 쿼리에는 `query: `, 색인된 문서에는 `passage: `. 대칭 작업을 포함한 모든 입력에 적용 |
| `BAAI/bge-m3` | cls | 없음 |

이를 그대로 보내십시오. 예: `{"input": ["query: how much protein should a female eat", "passage: As a general guideline, the average daily protein intake is 46 grams."]}`. e5 접두사를 생략하면 실패하지 않고 검색 품질이 조용히 저하됩니다.

길이는 절대 위치 테이블에 의해 제한됩니다. BERT는 0부터 인덱싱하므로 `max_position_embeddings`가 곧바로 token 상한입니다. XLM-RoBERTa는 `pad_token_id + 1`에서 시작하므로 `pad_token_id + 1`개 적은 token을 담을 수 있습니다. `bge-m3`의 8194개 행은 실제 token 8192개로 제한되며, 계열이 이를 `EmbeddingModel::max_sequence_length()`로 보고하고 로더가 도출된 `max_length`에 반영합니다. 이보다 긴 배치는 로드 시점에 truncation되며, 범위를 벗어난 gather가 되는 일은 없습니다.

양자화되지 않은 f32 체크포인트는 f32로 실행됩니다. `all-MiniLM-L6-v2`는 f16에서 underflow되는 `layer_norm_eps: 1e-12`와 함께 f32로 배포되므로, 공유 텍스트 bf16-to-f16 규칙은 bf16 export에만 적용됩니다. 양자화된 체크포인트(`config.quantization`)는 모든 projection과 word-embedding 테이블을 `UnifiedLinear` / `UnifiedEmbedding`을 통해 로드합니다.

`BertForSequenceClassification` 및 `XLMRobertaForSequenceClassification` 체크포인트(`BAAI/bge-reranker-v2-m3`, `cross-encoder/ms-marco-MiniLM-L6-v2`)는 reranker입니다. 감지는 이를 임베딩 변형이 아니라 `ModelType::SequenceClassifier`로 라우팅하며, `src/models/bert_heads.rs`의 `BertSequenceClassifier::load(dir)`가 경로로 하나를 로드합니다. 같은 trunk를 재사용하고, 임베더 경로에서 버려지는 `pooler.` 텐서를 유지하며, `tanh(dense(h[:, 0, :]))`에 이어 label projection(BERT는 `pooler.dense`와 `classifier`, XLM-RoBERTa는 `classifier.dense`와 `classifier.out_proj`)을 적용한 `[B, num_labels]` logit을 반환합니다. `num_labels`는 config의 주장이 아니라 해당 projection 텐서의 행 수입니다. [리랭킹](#reranking-v1rerank-and-mlxcel-rerank)이 이들을 서비스합니다.

### ModernBERT (`modernbert`)

ModernBERT는 절대 위치 테이블 대신 RoPE를 쓰는 8192 컨텍스트 양방향 인코더이므로 `max_position_embeddings`는 입력 길이를 제한하지 않으며, `max_length`는 `sentence_bert_config.json`과 `tokenizer_config.json`에서 오고 공개된 두 체크포인트 모두 `8192`로 결정됩니다. 세 레이어 중 두 레이어는 양쪽으로 `local_attention / 2`개의 key(공개된 `local_attention: 128`에서는 64)를 보는 sliding window를 사용하며 `local_rope_theta`(10000)로 회전합니다. 나머지 레이어는 시퀀스 전체를 보고 `global_rope_theta`(160000)로 회전합니다. 레이어 0에는 정당하게 `attn_norm`이 없으며(upstream은 이를 `nn.Identity()`로 만듭니다), 다른 레이어에서 `attn_norm`이 없으면 로드 오류입니다. `Wqkv`와 `Wi`는 fused 텐서이며 가중치를 슬라이스하는 것이 아니라 projection 이후에 분할되므로, 양자화된 export도 그대로 로드됩니다. `ModernBertModel`과 `ModernBertForMaskedLM`은 모두 임베더로 로드되며 MLM `head.*` / `decoder.*` 텐서는 버려집니다.

`nomic-ai/modernbert-embed-base`는 비대칭이며 텍스트에 명시적인 작업 접두사를 기대합니다. 쿼리 앞에는 `search_query: `, 문서 앞에는 `search_document: `를 붙입니다. 접두사는 서버 측 래퍼가 아니라 입력 문자열의 일부이므로 `input`(또는 `mlxcel embed -p`)에 직접 지정하십시오. 생략하면 검색 품질이 떨어집니다. Matryoshka truncation도 지원하므로 `dimensions`를 256까지 낮춰도 의미가 유지됩니다(엔진이 truncation 후 다시 정규화합니다).

```sh
mlxcel embed -m nomic-ai/modernbert-embed-base \
  -p "search_query: What is TSNE?" \
  -p "search_document: t-SNE is a dimensionality reduction technique used for visualizing high-dimensional data." \
  -p "search_document: The Eiffel Tower is a wrought-iron lattice tower in Paris."

mlxcel-server -m nomic-ai/modernbert-embed-base --port 8080
curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["search_query: What is TSNE?", "search_document: t-SNE is a dimensionality reduction technique used for visualizing high-dimensional data."]}'
```

`Alibaba-NLP/gte-reranker-modernbert-base`는 `ModernBertForSequenceClassification`을 선언하며, 감지는 이를 어떤 임베딩 변형으로도 라우팅하지 않도록 의도적으로 거부합니다(reranker는 임베더가 아닙니다). 대신 `ModelType::SequenceClassifier`로 감지되므로 해당 디렉터리에 `-m`을 지정하면 `/v1/rerank`가 서비스됩니다. 그 head는 `crate::models::modernbert_heads::ModernBertSequenceClassifier`이며, 디렉터리로 로드하고 `classifier(norm(gelu(dense(pooled))))`의 `[B, num_labels]` logit을 반환하고 풀링은 `classifier_pooling`(`cls` 또는 `mean`)이 결정합니다.

### SigLIP 텍스트 (`siglip`, `SiglipModel` / `SiglipTextModel`)

`src/models/siglip_text.rs`는 SigLIP 체크포인트의 텍스트 tower를 서비스합니다. vision tower는 아직 `/v1/embeddings`에서 서비스되지 않으므로 `image_url` 항목은 거부됩니다. tower는 token 및 학습된 위치 임베딩, VLM vision tower가 사용하는 것과 같은 pre-norm 인코더 블록(`src/vision/encoders/siglip.rs`), 최종 LayerNorm, 선형 projection `head`로 구성됩니다.

- 토크나이저는 분할하기 전에 정규화합니다. 소문자로 바꾸고, ASCII 구두점을 제거하며, 연속된 공백을 하나로 합치므로 `"A photo of a Cat!"`과 `"a photo of a cat"`은 같은 벡터를 만듭니다. 이는 mlxcel의 선택이 아니라 체크포인트 자체의 `tokenizer.json`입니다.
- 입력은 64 token으로 제한됩니다(`tokenizer_config.json`의 `model_max_length`와 tower의 `max_position_embeddings`). 63 token에 post-processor가 덧붙이는 끝의 `</s>`가 더해진 것입니다. 이후 모든 행은 정확히 64로 오른쪽 패딩되며, 계열은 이를 `EmbeddingModel::pad_to_max_length`를 통해 요청합니다.
- `pad_token`은 `</s>`(id 1)이며 토크나이저가 덧붙이는 것과 같은 id이므로, 위치 63에는 항상 `</s>`가 있습니다. 참조 구현이 풀링하는 "sticky EOS"입니다.
- attention mask는 적용되지 않으며 모든 위치가 64개 전체를 봅니다. 이는 학습 시점의 레시피이고 체크포인트도 이에 부합합니다. `tokenizer_config.json`이 `model_input_names: ["input_ids"]`를 선언하므로 참조 프로세서도 마스크를 내보내지 않습니다.
- 풀링은 위치 63으로 고정되며 이어서 projection `head`와 L2 정규화가 적용됩니다. `1_Pooling/config.json`은 참조되지 않고 `MLXCEL_EMBEDDING_POOLING`도 적용되지 않으며, `EmbeddingModel::default_pooling`은 시작 로그를 위해서만 `lasttoken`을 보고합니다.
- `text_config` 키는 모두 선택 사항이며 참조 `SiglipTextConfig` 기본값(`vocab_size` 32000, `hidden_size` 768, `intermediate_size` 3072, 12 head, 12 레이어, `max_position_embeddings` 64, `layer_norm_eps` 1e-6, `projection_size` = `hidden_size`, `hidden_act` `gelu_pytorch_tanh`)으로 fallback됩니다. `google/siglip-base-patch16-224`는 이 중 다섯 개만 선언합니다.
- `vision_model.*`, `logit_scale`, `logit_bias`, 그리고 모든 `position_ids` 버퍼는 로드 시 버려지며 나머지 키는 그대로 사용됩니다.

이 벡터를 텍스트 간 검색에 사용하기 전에 알아 둘 특성이 하나 있습니다. SigLIP의 텍스트 tower는 이미지에 대해 contrastive하게 학습되었을 뿐 다른 텍스트에 대해서는 학습되지 않았기 때문에, 서로 무관한 두 캡션을 멀어지게 하는 학습 신호가 없고 텍스트끼리의 cosine 유사도는 높은 비등방성 바닥에 놓입니다. `google/siglip-base-patch16-224`에서 동물, 기계, 금융, 음식, 물리를 아우르는 문장 6개로 측정한 결과, 무관한 14개 쌍은 0.519에서 0.725 사이(평균 0.653)였고 관련된 한 쌍(`a photo of a cat`과 `a photo of a kitten`)은 0.966에 이르렀습니다. 절대 점수에 임계값을 두지 말고 margin으로 후보의 순위를 매기십시오. 이는 sentence-transformers 인코더가 유도하는 방식과 반대입니다.

### EmbeddingGemma (`Gemma3Embedding`)

Gemma 3 텍스트 backbone을 양방향으로 실행합니다. full-attention 레이어에는 padding 전용 마스크가 적용되고, sliding 레이어에는 같은 padding 마스크와 대칭 `|q - k| < sliding_window` 밴드의 교집합이 적용됩니다. `sliding_window`(공개된 체크포인트에서는 512) token까지는 두 마스크가 동일하므로 window는 더 긴 입력에서만 영향을 미칩니다. 최종 norm 출력은 mean 풀링된 뒤, 엔진의 L2 정규화 이전에 bias가 없는 두 `Dense` 모듈(`768 -> 3072 -> 768`, 활성화 없음)을 통해 projection됩니다.

공개된 두 레이아웃이 모두 로드됩니다. mlx 변환본은 projection을 `dense.0.*`과 `dense.1.*`로 메인 shard에 병합하고 `model.` 접두사를 유지합니다. sentence-transformers 원본은 이를 `2_Dense/`와 `3_Dense/` 모듈 폴더에 저장하며, 이들은 `2_Dense.linear.weight`와 `3_Dense.linear.weight`로 들어와 `dense.0`과 `dense.1`로 순위가 매겨집니다. `Dense` 폭은 backbone hidden size에서 이어지는지 검사되므로, 순서가 바뀐 쌍은 조용히 잘못된 벡터가 되지 않고 로드 오류가 됩니다.

교대 주기는 `config.json`에 `layer_types`가 있으면 거기서 가져옵니다. transformers 4.57은 스칼라 값의 이름을 `_sliding_window_pattern`으로 바꾸었는데 Gemma 3 args 타입은 이를 파싱하지 못하며, 주기가 하나 어긋나도 로드와 실행은 여전히 됩니다.

호출자가 적용하는 프롬프트 접두사(`config_sentence_transformers.json`에서):

| 작업 | 접두사 |
|------|--------|
| 검색 쿼리, reranking, bitext mining | `task: search result \| query: ` |
| 검색 문서 | `title: none \| text: ` |
| 의미 유사도(STS) | `task: sentence similarity \| query: ` |
| 분류 | `task: classification \| query: ` |
| 클러스터링 | `task: clustering \| query: ` |
| 코드 검색 쿼리 | `task: code retrieval \| query: ` |
| 요약 | `task: summarization \| query: ` |

`max_length`는 2048입니다(`sentence_bert_config.json`). 학습된 Matryoshka 폭은 768, 512, 256, 128이므로 `dimensions`가 이 값이면 순위가 유지되며, `1..=768`의 다른 값도 허용되고 다시 정규화되지만 학습된 폭은 아닙니다.

```sh
mlxcel embed -m mlx-community/embeddinggemma-300m-4bit \
  -p "task: search result | query: Which planet is known as the Red Planet?" \
  -p "title: none | text: Mars, known for its reddish appearance, is often referred to as the Red Planet."
```

### Qwen3-Embedding (`Qwen3Embedding`)

causal-plus-padding 마스크와 last-token 풀링을 사용하는 causal Qwen3 backbone입니다. 문장 임베딩은 토크나이저가 덧붙인 `<|endoftext|>` 위치의 hidden state입니다. 배치는 오른쪽 패딩되므로 실제 token에 대한 causal 마스크는 패딩되지 않은 단일 행의 경우와 동일합니다. tied `lm_head`는 로드 시 버려지고 backbone은 최종 norm에서 끝나므로 `[B, L, vocab_size]` logit 텐서는 만들어지지 않습니다.

쿼리는 `Instruct: {task}\nQuery: {query}` 형식을 사용하고 문서는 원문 텍스트입니다. 작업을 `instruction`(또는 `--instruction`)으로 전달하면 계열이 텍스트를 감싸며, 완전히 포맷된 문자열을 보내고 `instruction`을 설정하지 않으면 직접 처리할 수 있습니다. `instruction`은 요청의 모든 입력에 적용되므로, 쿼리와 문서가 섞인 배치는 두 개의 요청으로 나누거나 쿼리를 텍스트 안에서 직접 포맷해야 합니다. `max_length`는 하드 상한인 8192입니다. 체크포인트는 32부터 1024까지의 `dimensions`를 지원합니다.

```sh
mlxcel embed -m Qwen/Qwen3-Embedding-0.6B \
  --instruction "Given a web search query, retrieve relevant passages that answer the query" \
  -p "What is the capital of China?"
```

### Qwen3-VL-Embedding (`Qwen3VLEmbedding`)

head 대신 풀링을 사용하는 생성용 Qwen3-VL 스택입니다. 동일한 vision tower, 선택된 decoder 레이어로의 동일한 DeepStack 주입, 동일한 interleaved M-RoPE를 사용하며 같은 VLM 로더로 로드됩니다. `Qwen3VLModel::forward_hidden`은 최종 norm에서 멈추므로 tied `lm_head`는 적용되지 않고 `[B, L, vocab_size]` 텐서도 만들어지지 않습니다. 풀링은 causal-plus-padding 마스크 위에서의 last token이며, 그래서 오른쪽 패딩된 배치가 단일 입력 벡터를 정확히 재현합니다.

다른 모든 계열과 달리 입력은 그대로 임베딩되지 않습니다. 체크포인트의 `chat_template.jinja`가 입력을 instruction system 메시지로 감싸며, 풀링되는 위치는 assistant 헤더의 마지막 개행입니다.

```text
<|im_start|>system
{instruction}<|im_end|>
<|im_start|>user
[<|vision_start|><|image_pad|><|vision_end|>]{text}<|im_end|>
<|im_start|>assistant
```

instruction의 기본값은 체크포인트의 `config_sentence_transformers.json` 프롬프트입니다(공개된 2B에서는 `Represent the user's input.`). 요청의 `instruction` 또는 CLI의 `--instruction`으로 재정의할 수 있으며, 문장 중간에서 끝나는 instruction에는 참조 래퍼와 동일하게 끝에 `.`이 붙습니다. `max_length`는 하드 상한인 8192이고 벡터는 2048차원입니다.

이미지 항목을 사용할 수 있습니다. 엔진은 계열이 이미지를 보기 전에 토크나이즈하므로, 템플릿은 `<|image_pad|>`를 정확히 하나만 내보내고 `EmbeddingModel::expand_image_tokens`가 행이 패딩되기 전에 이를 Qwen2-VL 프로세서가 해당 이미지에 대해 계산하는 patch 수로 늘립니다. 이후 forward pass가 그 위치에 vision feature를 병합합니다. 픽셀 한도는 생성용 기본값이 아니라 체크포인트의 `preprocessor_config.json`에서 오며, 그래서 공개된 2B에서 이미지 하나는 1280개의 visual token으로 유지됩니다. DeepStack 주입과 M-RoPE delta가 모두 시퀀스별이므로 이미지는 forward pass당 하나씩 실행됩니다.

```sh
mlxcel embed -m Qwen/Qwen3-VL-Embedding-2B \
  -p "A woman playing with her dog on a beach" \
  -p "Quarterly revenue report for a software company" \
  --image beach.jpg

curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}, {"type": "text", "text": "A woman playing with her dog on a beach"}]}'
```

### Llama-Nemotron-VL-Embed (`LlamaNemotronVLEmbedding`)

mlxcel이 이미 실행하는 세 가지 구성 요소를 조합한 것입니다. `vision_model.vision_model`의 SigLIP-400M tower(`select_layer: -1`이 선택하는 `post_layernorm` 출력), InternVL의 `pixel_shuffle(0.5)`와 `mlp1` connector(`LayerNorm(4608) -> Linear(4608 -> 2048) -> GELU -> Linear(2048 -> 2048)`), 그리고 양방향 padding 마스크로 구동되고 최종 norm 이후 mean 풀링되는 Llama 3.2 1B decoder입니다. 체크포인트의 `llm_config`는 모든 attention 모듈에서 `is_causal = False`인 Llama decoder인 `LlamaBidirectionalModel`(`model_type: llama_bidirec`)을 선언합니다. SigLIP attention-pooling `head.*`는 임베더가 읽지 않으므로 로드 시 버려집니다.

프롬프트 접두사는 텍스트 전용 형제 모델인 `nvidia/llama-nemotron-embed-1b-v2`와 마찬가지로 호출자 측 책임입니다. 질문 앞에는 `query: `, 문서 앞에는 `passage: `를 붙입니다. 이 계열은 `instruction`을 사용하지 않습니다.

이미지 항목에는 호출자 텍스트가 없으므로, 계열이 문서 형식을 직접 내보냅니다: `passage: <img>{<IMG_CONTEXT> repeated 256 * tiles}</img> `. 타일 수는 체크포인트 자체의 면적 인식 `find_closest_aspect_ratio`를 사용한 512 px의 InternVL 스타일 동적 타일링에서 오며, `max_input_tiles`(6)로 제한되고 이미지가 실제로 분할된 경우 전체 이미지 썸네일이 하나 추가되므로, 가로 방향 페이지는 `7 * 256`개의 visual token에 이릅니다. 정규화는 ImageNet이 아니라 SigLIP의 것(평균과 표준편차 `0.5`)입니다.

이 체크포인트에 대해 알아 둘 점이 두 가지 있습니다. `modules.json`에는 `2_Normalize` 모듈이 없지만 mlxcel은 어쨌든 L2 정규화합니다(`config.json`에 `normalize: false`가 없습니다). 체크포인트 자체의 `similarity_fn_name`이 cosine이므로 순위는 영향받지 않습니다. 또한 `max_length`는 참조 구현의 문서용 `p_max_length` 4096이나 쿼리용 `q_max_length` 512가 아니라 공유 상한인 8192로 결정되는데, 이 키들이 공유 길이 도출 로직이 읽지 않는 `processor_config.json`에 있기 때문입니다. 참조 구현의 truncation을 재현하려면 `--embedding-max-length 4096`(쿼리 전용 서버라면 `--embedding-max-length 512`)을 지정하십시오.

```sh
mlxcel embed -m nvidia/llama-nemotron-embed-vl-1b-v2 \
  -p "query: what does the chart say about 2023 revenue" \
  -p "passage: Revenue grew 12% in 2023 driven by subscriptions." \
  --image chart.png
```

두 멀티모달 계열 모두 `EmbeddingModel::expand_image_tokens`를 구현하므로 placeholder가 행이 패딩되기 전에 실제 연속 구간으로 바뀌며, `usage.prompt_tokens`는 forward pass가 실제로 소비하는 시퀀스를 설명합니다.

### ColIdefics3 (`idefics3`, `ColIdefics3`)

SmolVLM / Idefics3 스택 위의 late-interaction 시각 문서 검색기입니다. SigLIP tower, `pixel_shuffle(scale_factor)` connector, SmolLM2 decoder는 `mlxcel generate`가 이미 실행하는 것들입니다. 검색은 양 끝만 바꿉니다. decoder는 최종 norm에서 멈추고(체크포인트에는 `lm_head`가 아예 없습니다), 하나의 `Linear`(`linear.{weight,bias}`, `[128, 576]`)가 모든 token의 hidden state를 128차원으로 projection하며, 각 token 벡터는 L2 정규화되고 패딩 행은 0으로 채워집니다.

두 항목 종류가 서로 다른 형식을 사용하기 때문에 프롬프트는 호출자가 아니라 계열이 만듭니다. 이미지 문서는 `<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:`로 렌더링되고, 이후 엔진이 단일 `<image>`를 프로세서의 프레임이 있는 타일 구간(`<fake_token_around_image><row_r_col_c>`에 타일당 이미지 token 64개, 한 줄에 한 행, 이어서 전역 썸네일 블록)으로 교체합니다. 쿼리는 `Query: {text}` 뒤에 열 개의 `<end_of_utterance>` 증강 token을 붙여 렌더링되며, 이것이 짧은 쿼리에 MaxSim이 구별하기에 충분한 벡터 수를 제공합니다. 이 계열은 `instruction` 필드를 사용하지 않습니다.

`vidore/colSmol-256M`은 `vidore/ColSmolVLM-Instruct-256M-base`에 대한 LoRA adapter이며 학습된 projection을 `1_Dense/`에 담고 있습니다. mlxcel은 adapter를 병합하지 않으므로, base shard 없이 `adapter_model.safetensors`만 있는 디렉터리는 병합된 체크포인트를 요청하는 메시지와 함께 거부됩니다. 체크포인트 자체의 도구로 병합하고 `1_Dense/model.safetensors`를 `config.json` 옆에 두십시오. 해당 폴더가 있으면 그 `linear.{weight,bias}`가 메인 shard의 것을 대체하며, 이는 sentence-transformers가 하는 방식입니다. base 저장소는 단독으로 로드되어 유한하고 올바른 shape의 출력을 만들며 이는 가중치 레이아웃을 검증하지만, projection이 학습되지 않았으므로 순위는 의미가 없습니다. `mask_non_image_embeddings: true`는 조용히 무시되지 않고 로드 시 거부됩니다.

```sh
mlxcel embed -m models/colSmol-256M-merged \
  -p "What was the total revenue in 2023?" \
  --image revenue-table.png --image unrelated-page.png
```

### ColQwen2.5 (`qwen2_5_vl` / `colqwen2`, `ColQwen25`)

Qwen2.5-VL 스택 위의 같은 레시피입니다. windowed vision tower와 patch merger가 M-RoPE를 쓰는 Qwen2 decoder에 입력을 공급하고, `Qwen2VLModel::forward_hidden`은 최종 norm에서 멈추며, `custom_text_proj`(`[128, 2048]`와 bias)가 모든 token을 projection합니다. position id는 입력을 따릅니다. 텍스트 전용 micro-batch는 backbone이 텍스트 prefill용으로 만드는 순차적인 `[3, B, L]` 위치를 사용하고, 이미지 입력은 실제 M-RoPE grid를 유지하며, 이전 이미지 요청의 공간 위치가 새어 들어오지 않도록 텍스트 배치 전에 요청별 M-RoPE 슬롯이 비워집니다.

이미지 문서는 `<|image_pad|>` 구간이 병합된 patch당 token 하나로 확장된 `<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|><|endoftext|>`로 렌더링되고, 쿼리는 `Query: {text}`에 열 개의 `<|endoftext|>` token을 붙인 형태로 렌더링됩니다. 문서 턴은 assistant 헤더가 아니라 `<|endoftext|>`로 닫히며, 이는 참조 프로세서와 체크포인트 자체의 `additional_chat_templates/sentence_transformers.jinja`가 모두 내보내는 방식입니다. 페이지 뒤에 생성되는 것은 없습니다. `preprocessor_config.json`은 이미지를 `768 * 28 * 28` 픽셀로 제한하며, 이는 한 페이지를 768개의 visual token으로 제한합니다.

세 가지 키 레이아웃이 로드됩니다. `vidore/colqwen2.5-base`와 그 병합 파생본은 `model.*`, `visual.*`, `custom_text_proj.*`를 저장합니다. 네이티브 `transformers` retrieval export는 모든 것을 `vlm.` 아래에 중첩시키고 projection을 `embedding_proj_layer.*`로 명명합니다. mlx 변환본은 이미 `model.*`와 `vision_tower.*`를 저장합니다. sanitizer는 먼저 `vlm.` 래퍼를 제거한 다음 `embedding_proj_layer.`를 `custom_text_proj.`로, `model.language_model.`과 `language_model.`을 `model.`로, `visual.` / `model.visual.`을 `vision_tower.`로 매핑합니다. tied `lm_head`는 로드 시 버려집니다. 원본 HuggingFace export는 vision patch 필터도 `Conv3d`의 네이티브 `[out, in, kT, kH, kW]` 레이아웃으로 저장하는 반면 인코더는 channels-last mlx 변환을 기대하므로, 해당 텐서는 로드 시 permute됩니다. 이를 하지 않으면 tower가 뒤섞인 필터를 읽어 모든 페이지가 거의 같은 벡터로 임베딩됩니다. ColIdefics3와 마찬가지로 `vidore/colqwen2.5-v0.2`는 LoRA adapter이므로 먼저 base에 병합해야 하며, base 저장소는 학습되지 않은 projection으로 단독 로드되어 레이아웃은 검증하지만 순위는 검증하지 못합니다.

```sh
mlxcel embed -m models/colqwen2.5-v0.2-merged \
  -p "What was the total revenue in 2023?" \
  --image revenue-table.png --image unrelated-page.png
```

## Multi-vector (late-interaction) 출력

`EmbeddingModel::multi_vector()`가 `true`인 계열은 입력당 하나가 아니라 token당 하나의 벡터를 반환합니다. `EmbeddingModel::embed`는 패딩 token의 행이 0으로 채워진 `[B, L, D]`를 만들고, 엔진은 정규화하고 `dimensions`를 적용한 뒤 각 항목을 자신의 실제 token 수에 맞게 잘라내므로 응답에는 `[num_real_tokens, D]`가 담깁니다. 이미지 항목의 실제 token 수는 이미지 placeholder가 확장된 뒤의 수이며, 이는 배치가 패딩되기 전에 `EmbeddingModel::expand_image_tokens`가 계산하는 값이므로 `usage.prompt_tokens`와 반환되는 행 수는 항상 같은 시퀀스를 설명합니다.

후보는 cosine이 아니라 MaxSim으로 순위가 매겨집니다.

```
maxsim(q, d) = sum over query rows i of ( max over document rows j of dot(q_i, d_j) )
```

`crate::embeddings::maxsim`은 readback된 행에 대해, `maxsim_mlx`는 device 배열에 대해 이를 구현합니다. 모든 행이 L2 정규화되어 있으므로 각 내적은 cosine이고 점수는 쿼리의 행 수로 제한됩니다. 하나의 쿼리에 대해 두 문서를 비교하는 것이 이 숫자의 용도이므로 평균을 내지 않고 raw 값으로 보고합니다. 점수는 비대칭입니다. 바깥쪽 합은 항상 첫 번째 인자에 대해 실행됩니다.

`mlxcel embed`는 로드된 모델이 multi-vector일 때마다 cosine 행렬 대신 `MaxSim similarity`라는 라벨의 이 행렬을 출력합니다.

### 양방향 Llama, LLM2Vec (`LlamaBidirec`)

LLM2Vec 레시피입니다. 일반적인 Llama 3.2 1B decoder에서 causal 마스크를 제거하여 인코더로 변환한 것입니다. `nvidia/llama-nemotron-embed-1b-v2`는 이를 `model_type: llama_bidirec`, `architectures: ["LlamaBidirectionalModel"]`, `use_bidirectional_attention: true`, 그리고 mean 풀링을 선언하는 `1_Pooling` 모듈로 export합니다. 레이어에 바뀌는 것은 없습니다. factor 32의 동일한 `rope_scaling` `llama3` 스케줄, 동일한 norm, `head_dim` 64의 동일한 GQA 32/8입니다. 마스크와 head의 부재만 다르므로, 이 포트는 `Llama3Model`을 거치지 않고 가중치 맵에서 바로 `llama3::TransformerBlock` 레이어를 구성합니다. `Llama3Model`의 `lm_head` 필드는 선택 사항이 아니어서 이 경로가 적용하지 않는 tied `128256 x 2048` projection을 불필요하게 만들어 내기 때문입니다.

공개된 export는 내부의 `LlamaBidirectionalModel`을 저장하므로 루트가 접두사 없이 들어오며(`embed_tokens.weight`, `layers.0.…`, `norm.weight`) 로드 시 `model.` 접두사가 붙습니다. `language_model.` 래퍼 접두사는 먼저 제거되고, 파생된 `rotary_emb.inv_freq`와 `position_ids` 버퍼는 이 트리가 `rope_theta`와 `rope_scaling`에서 주파수 테이블을 다시 만들기 때문에 버려집니다.

PEFT adapter로 배포되는 LLM2Vec 체크포인트는 로드할 수 없습니다. `adapter_model.safetensors`만 있고 전체 shard가 없는 디렉터리는, adapter를 base 모델에 병합하고 완전한 `LlamaBidirectionalModel` 체크포인트를 먼저 export하라는 메시지와 함께 거부됩니다. mlxcel은 adapter를 병합하지 않습니다.

`max_length`는 8192입니다(`sentence_bert_config.json`이며 우연히 하드 상한과 같습니다). `config_sentence_transformers.json`의 프롬프트 접두사는 호출자가 적용합니다. 쿼리 앞에는 `query: `, 문서 앞에는 `passage: `입니다.

```sh
mlxcel embed -m nvidia/llama-nemotron-embed-1b-v2 \
  -p "query: how do solar panels generate electricity" \
  -p "passage: Photovoltaic cells convert sunlight into electricity through the photovoltaic effect."
```

### Nemotron-3-Embed (`Ministral3Embedding`)

Ministral 3 backbone을 양방향으로 실행합니다. `nvidia/Nemotron-3-Embed-1B-BF16`과 `mlx-community`의 8-bit 변환본은 `model_type: ministral3`, `architectures: ["Ministral3Model"]`, `is_causal: false`를 선언하며, 감지는 이 플래그를 기준으로 합니다. 풀링 모듈은 mean이라고 명시합니다.

포트를 좌우하는 세부 사항이 두 가지 있습니다. 위치별 Llama 4 attention scale(`1 + beta * ln(1 + floor(pos / original_max_position_embeddings))`, `beta` 0.1, window 16384)은 offset 0에서 계산되는데, 임베더는 prefill을 한 번 실행하고 decode하지 않으며, 이는 같은 길이에 대해 생성기의 새 캐시가 보고하는 값과 같기 때문입니다. 그리고 두 체크포인트 모두 `tie_word_embeddings`가 true이므로 `lm_head`가 없고 forward pass는 최종 norm에서 멈춥니다. 공개된 두 체크포인트는 모두 `sliding_window: null`이고 `layer_types`가 없으므로 모든 레이어가 full attention입니다. `sliding_attention` 레이어를 선언하는 체크포인트라면 그 레이어에는 `create_bidirectional_window_mask`가 사용됩니다.

`max_length`는 하드 상한인 8192이며 이는 체크포인트가 선언한 32768보다 작습니다. 호출자가 적용하는 프롬프트 접두사는 `query: `와 `passage: `입니다.

```sh
mlxcel embed -m nvidia/Nemotron-3-Embed-1B-BF16 \
  -p "query: how do solar panels generate electricity" \
  -p "passage: Photovoltaic cells convert sunlight into electricity through the photovoltaic effect."
```

### LFM2.5-Embedding (`Lfm2Embedding`)

두 mixer가 모두 양방향이 된 하이브리드 LFM2 backbone입니다. `LiquidAI/LFM2.5-Embedding-350M`은 gated short convolution 10개와 full-attention 레이어 6개가 교대하는 16개 레이어(`conv_L_cache` 3, head별 Q/K RMSNorm이 있는 GQA 16/8, 최종 norm으로서의 `embedding_norm`)이며, CLS를 선언하는 `1_Pooling` 모듈과 함께 `Lfm2BidirectionalModel`로 export됩니다.

attention 레이어는 일반적인 padding 전용 마스크를 사용합니다. short conv에는 두 가지 변경이 필요하며, 이것이 생성기의 파일에 손을 대는 유일한 변경입니다.

- `ModelArgs`에 기본값이 `true`인 `conv_causal` 플래그가 추가되며 임베딩 로더가 이를 `false`로 설정합니다. 그러면 mixer는 `L_cache - 1`개의 zero padding을 전부 앞에 붙이는 대신 양쪽으로 나누므로 위치 `t`가 `t - 1`, `t`, `t + 1`을 섞고 출력 길이는 `L`로 유지됩니다. 또한 일회성 양방향 pass에서는 쓸모가 없는 conv state를 더 이상 기록하지 않습니다. 생성은 기본값을 유지하며 바이트 단위로 동일합니다.
- conv 입력은 `[B, L, 1]` 승수를 통해 패딩 위치에서 0으로 만들어지며, 이는 참조 구현의 `apply_mask_to_padding_states`가 하는 것과 같습니다. convolution에는 attention 마스크가 작용할 key 축이 없으므로, 이것이 없으면 pad-token 임베딩이 경계에 인접한 실제 위치에 섞여 들어가고 위쪽 attention 레이어가 이를 행 전체로 퍼뜨립니다. 수정 전에 공개된 체크포인트에서 측정했을 때, 패딩된 행의 마스크된 꼬리 부분만 바꿔도 풀링된 벡터가 cosine 0.94만큼 움직였습니다.

풀링은 CLS입니다. 토크나이저의 post-processor가 `<|startoftext|>`를 앞에 붙이고 문장 벡터는 그 위치의 hidden state입니다. 오른쪽 패딩에서는 모든 행에서 인덱스 0에 놓이지만, 풀링은 이를 가정하지 않고 첫 번째 실제 token의 argmax로 찾습니다.

LFM2와 LFM2.5 ColBERT late-interaction 체크포인트는 이 `model_type`과 아키텍처를 공유하지만 모든 token을 `1_Dense` 모듈로 projection하고 token당 하나의 벡터를 내보냅니다. 이들은 풀링된 단일 벡터로 서비스되지 않고 로드 시 거부됩니다.

`max_length`는 512이며(`sentence_bert_config.json`) 하드 상한보다 한참 작습니다. 호출자가 적용하는 프롬프트 접두사는 `query: `와 `document: `입니다.

```sh
mlxcel embed -m LiquidAI/LFM2.5-Embedding-350M \
  -p "query: how do solar panels generate electricity" \
  -p "document: Photovoltaic cells convert sunlight into electricity through the photovoltaic effect."
```

### 배치 geometry와 bf16

실행 간에 벡터를 비교하기 전에 알아 둘 특성이 하나 있습니다. bf16 체크포인트에서는 attention 커널이 배치 geometry에서 누산 shape를 고르므로, 같은 텍스트를 단독으로 임베딩한 경우와 더 큰 배치의 한 행으로 임베딩한 경우는 정확히 일치하지 않고 대략 cosine 0.9999 수준으로 일치합니다. 이는 패딩이 아니라 산술의 문제입니다. 하나의 텍스트를 패딩되지 않은 여러 사본으로 구성한 단일 배치로 임베딩해도 같은 편차가 나타나며, 여기 있는 모든 bf16 decoder-backbone 계열에서 확인됩니다. 패딩 자체는 아무것도 바꾸지 않으며, 각 계열은 배치 shape를 고정하고 마스크 아래의 token id만 바꾸는 방식으로 이를 허용 오차 0으로 정확히 검증합니다.

## Attention 마스크

계열은 `[B, L]` attention mask로부터, `mlxcel_core::utils`의 `create_causal_mask` 옆에 있는 세 개의 빌더를 사용하여 가산 f32 마스크(`0.0` = attend, `-inf` = 차단)를 만듭니다.

- `create_bidirectional_padding_mask(mask) -> [B, 1, 1, L]`: `mask[b, k] == 0`일 때에 한해 key 열 `k`가 차단됩니다. 패딩 query 행도 모든 실제 key를 볼 수 있으므로 완전히 차단되는 행은 없습니다.
- `create_causal_padding_mask(mask, offset) -> [B, 1, L, L + offset]`: `k <= q + offset`이고 `mask[b, k] == 1`입니다. 완전히 패딩된 query 행은 대각선 열을 유지하며, 이는 `create_causal_mask_with_left_padding`이 적용하는 것과 같은 구제 방식입니다.
- `create_bidirectional_window_mask(mask, window) -> [B, 1, L, L]`: `mask[b, k] == 0`이거나 `|q - k| >= window`일 때에 한해 차단됩니다.

마스크는 activation dtype에서 유한한 `C`를 사용한 `(1 - m) * C`로 만들어지지 않습니다. f16에서는 `-1e9`가 `-inf`로 overflow되고 `0 * -inf`는 NaN이기 때문입니다. 빌더는 f32로 `0 / -inf`를 만들며, 이는 `fast_scaled_dot_product_attention`이 f16 및 bf16 activation에 적용하는 캐스트에서 유지됩니다.

## 요청과 응답

```json
{
  "model": "optional; must equal the served embedding model id when given",
  "input": "string | [string] | [int] | [[int]] | [{\"type\": \"text\", \"text\": ...} | {\"type\": \"image_url\", \"image_url\": {\"url\": ...}}]",
  "encoding_format": "float | base64",
  "dimensions": 256,
  "instruction": "optional; forwarded to EmbeddingModel::format_text",
  "user": "ignored"
}
```

```json
{
  "object": "list",
  "data": [{"object": "embedding", "index": 0, "embedding": [0.01, 0.02]}],
  "model": "all-MiniLM-L6-v2",
  "usage": {"prompt_tokens": 17, "total_tokens": 17}
}
```

- token id 입력은 그대로 사용되며(특수 토큰은 추가되지 않음) 모든 id는 `< vocab_size`여야 합니다.
- `encoding_format: base64`는 각 벡터를 little-endian f32 바이트로 인코딩하며, 패딩이 있는 표준 base64를 사용합니다. multi-vector(late-interaction) 모델에서 `embedding`은 float 모드에서는 벡터의 목록 `[num_real_tokens][D]`이고, base64 모드에서는 row-major 바이트와 형제 필드 `"shape": [num_real_tokens, D]`입니다.
- `usage.prompt_tokens`는 특수 토큰을 포함하여 모든 입력의 실제(패딩이 아닌) token 수를 셉니다. `total_tokens`는 이와 같습니다.
- `image_url` 항목은 `data:image/...;base64,` URI, `file://` 경로, `http(s)://` URL 또는 로컬 경로를 받아들이며 공유 이미지 제한의 적용을 받습니다. 이미지는 한 번에 하나씩 임베딩되고, 텍스트 항목은 token 길이순으로 정렬되어 `--embedding-batch-size` 크기의 micro-batch로 잘린 뒤 요청 순서대로 다시 기록됩니다.

### 오류

| 상태 | 타입 | 발생 조건 |
|--------|------|------|
| 400 | `invalid_request_error` | 잘못된 형식의 본문, 비어 있는 `input`, 빈 문자열 항목, 빈 token 목록, `>= vocab_size`인 token id, `supports_images() == false`인 모델에 대한 이미지 항목, 범위를 벗어난 `dimensions`, 지원되지 않는 `encoding_format`, 또는 서비스 중인 id와 일치하지 않는 `model`. |
| 501 | `not_implemented` | 로드된 임베딩 모델이 없음: `No embedding model loaded; start with -m <embedding checkpoint> or --embedding-model <path>`. |
| 503 | `server_busy` | 유한 worker 큐가 가득 참(`--embedding-queue-depth`). |
| 504 | `server_timeout` | worker가 `--embedding-request-timeout-secs` 이내에 응답하지 않음. |
| 500 | `server_error` | forward pass 실패. |

## 서버 플래그

| 플래그 | Env | 기본값 | 의미 |
|------|-----|---------|---------|
| `-m <embedding checkpoint>` | `LLAMA_ARG_MODEL` | | 체크포인트를 `/v1/embeddings`에서 서비스하며, 채팅은 로드되지 않은 상태로 유지됩니다. |
| `--embedding-model <path or repo-id>` | `LLAMA_ARG_EMBEDDING_MODEL`, `MLXCEL_EMBEDDING_MODEL` | unset | `-m`의 채팅 모델과 함께 `/v1/embeddings`에서 서비스되는 두 번째 체크포인트입니다. `-m`과 같은 방식으로 해석되고 자동 다운로드됩니다. `-m`의 임베딩 체크포인트와 함께 지정하면 시작 오류("two embedding models")입니다. 명시한 `--embedding-model`의 로드 실패는 시작 오류이며, `-m` 임베딩 체크포인트의 로드 실패는 로그에 기록되고 라우트는 `501`로 응답합니다. |
| `--embedding-batch-size N` | `MLXCEL_EMBEDDING_BATCH_SIZE` | `16` | forward pass당 텍스트 수. |
| `--embedding-max-length N` | `MLXCEL_EMBEDDING_MAX_LENGTH` | derived | 도출된 `max_length`를 낮춥니다. |
| `--embedding-queue-depth N` | `MLXCEL_EMBEDDING_QUEUE_DEPTH` | `8` | 각 임베딩/reranking worker 명령 큐의 상한이며, 큐가 가득 차면 `503`을 반환합니다. |
| `--embedding-request-timeout-secs N` | `MLXCEL_EMBEDDING_REQUEST_TIMEOUT_SECS` | `120` | 요청별 임베딩/reranking 응답 타임아웃이며, `0`이면 기본값으로 fallback됩니다. |

`mlxcel serve`와 `mlxcel-server`는 모든 플래그를 받아들입니다. 임베딩 모델은 자체 전용 MLX 스레드에서 실행되므로(모델, 토크나이저, 모든 배열이 그곳에 존재합니다) 채팅과 임베딩은 stream을 공유하지 않습니다.

```sh
# Embeddings only
mlxcel-server -m sentence-transformers/all-MiniLM-L6-v2 --port 8080

# Chat plus embeddings
mlxcel-server -m mlx-community/Qwen3-4B-4bit --embedding-model Qwen/Qwen3-Embedding-0.6B

curl -s localhost:8080/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"input": ["The weather is lovely today.", "It is so sunny outside!", "He drove to the stadium."]}'
```

## `mlxcel embed`

```text
mlxcel embed -m <path or repo-id> -p "text" [-p "text2" ...] [--image <file> ...]
             [--instruction "..."] [--dimensions N] [--max-length N] [--batch-size N]
             [--models-dir PATH] [--json]
```

입력마다 하나의 벡터를 출력하고(`[v1, v2, ...]`, 한 줄에 하나. multi-vector 모델에서는 행의 목록), 입력이 둘 이상이면 유사도 행렬을 함께 출력합니다. single-vector 모델은 cosine, multi-vector 모델은 MaxSim이며 헤더에 어느 쪽인지 표시됩니다. `--json`은 대신 `embeddings`, `shapes`, `prompt_tokens`, `similarity`를 담은 하나의 객체를 출력합니다. 이것은 모든 계열을 위한 오프라인 검증 도구로, 서버와 동일한 로더, 풀링, 정규화, 배칭을 listener 없이 사용합니다.

## 리랭킹 (`/v1/rerank`)과 `mlxcel rerank`

reranker는 문서가 쿼리와 얼마나 관련 있는지 점수를 매기고 `[0, 1]` 범위의 확률을 반환합니다. 두 개의 독립된 벡터가 아니라 쌍에 대한 cross-encoder이므로 색인할 수는 없지만, 임베딩에 대한 cosine보다 훨씬 잘 구분합니다. 일반적인 흐름은 `/v1/embeddings`로 검색한 뒤 `/v1/rerank`로 상위 후보를 재정렬하는 것입니다.

세 가지 체크포인트 형태가 서비스되며, 세 가지 모두 확률로 끝납니다.

| 종류 | 선택 조건 | 점수 | 체크포인트 |
|------|---------------|-------|-------------|
| `sequence_classifier` | `architectures[0]`이 `ForSequenceClassification`으로 끝나고 `model_type`이 `bert`, `xlm-roberta` 또는 `modernbert` | 단일 label head의 `sigmoid(logit)` | `BAAI/bge-reranker-v2-m3`, `cross-encoder/ms-marco-MiniLM-L6-v2`, `Alibaba-NLP/gte-reranker-modernbert-base` |
| `generative_text` | `--reranker-model`이 `model_type: qwen3` causal 체크포인트를 가리킴 | 마지막 프롬프트 위치에서의 `sigmoid(logit("yes") - logit("no"))` | `Qwen/Qwen3-Reranker-0.6B`와 그 MLX 양자화본, 예: `mlx-community/Qwen3-Reranker-0.6B-4bit` |
| `generative_vl` | `--reranker-model`이 `model_type: qwen3_vl` 체크포인트를 가리킴 | 쿼리와 문서가 이미지를 담을 수 있는 프롬프트 위에서의 동일한 yes/no 읽기 | `Qwen/Qwen3-VL-Reranker-2B` |

정확히 하나의 출력 label을 노출하지 않는 classifier는 로드 시 `reranker sequence classifiers must expose exactly one output label`로 거부됩니다. 개수는 `config.json`이 아니라 projection 텐서의 행 수에서 가져옵니다. 그 외의 `model_type`은 `Unsupported reranker model type`으로 거부됩니다.

체크포인트만으로 감지할 수 있는 것은 cross-encoder 종류뿐이므로, `-m`을 통해 서비스할 수 있는 것도 이것뿐입니다. generative reranker의 `config.json`은 일반적인 채팅 export(`Qwen3ForCausalLM`, `Qwen3VLForConditionalGeneration`)이며 `--reranker-model`로만 접근할 수 있습니다. `Qwen/Qwen3-VL-Reranker-2B`는 `modules.json`을 포함해 배포되지만 추가 모듈이 `Pooling`이 아니라 `1_LogitScore`뿐이며, 감지가 이를 임베딩 export로 오인하지 않는 이유가 바로 이것입니다.

### Cross-encoder 점수 계산

쌍은 체크포인트 자체의 pair 템플릿(BERT는 `[CLS] query [SEP] document [SEP]`, XLM-RoBERTa는 `<s> query </s></s> document </s>`)을 통해 하나의 시퀀스로 토크나이즈되며, `token_type_ids`는 BERT 방식에서만 사용됩니다. truncation은 `max_length`에서의 토크나이저 자체 `longest_first`입니다. 현재 더 긴 쪽에서 잘라내며 무엇이든 자르기 전에 템플릿의 특수 토큰을 먼저 확보하므로, 긴 문서 때문에 쿼리가 첫 token을 잃는 일은 없습니다. `max_length`는 `sentence_bert_config.json`의 `max_seq_length`, `tokenizer_config.json`의 `model_max_length`, 절대 위치 BERT trunk의 `config.json` `max_position_embeddings`, 그리고 공유 상한 8192 중 가장 작은 값입니다.

### Qwen3 generative 점수 계산

프롬프트는 모델 카드가 공개한 레시피와 바이트 단위로 동일합니다.

```text
PREFIX  = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
CONTENT = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
SUFFIX  = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
```

기본 `instruction`은 `Given a web search query, retrieve relevant passages that answer the query`입니다. 세 부분은 따로 토크나이즈된 뒤 이어 붙여지므로 `CONTENT`만 truncation되며, prefix와 점수를 읽는 assistant 헤더는 항상 남습니다. `yes`와 `no`는 특수 토큰 없이 인코딩하여 로드 시 결정되며(Qwen3 토크나이저에서 9693과 2152), 둘 중 하나라도 단일 token이 아니면 로드가 실패합니다.

행은 체크포인트의 pad token으로 **왼쪽** 패딩되며, 이것이 모든 행의 마지막 실제 token을 열 `L - 1`에 놓는 요소입니다. 마스크는 `create_causal_padding_mask`이므로 pad prefix는 어디서나 key로서 차단됩니다. 절대 위치는 참조 `Qwen3ForCausalLM.forward(input_ids, attention_mask)`가 cache position에서 `position_ids`를 도출할 때와 똑같이 패딩된 행 전체에 걸쳐 여전히 `0..L`로 진행됩니다. 알아 둘 결과가 하나 있습니다. 왼쪽 패딩된 행의 실제 token은 이동된 위치에 놓이므로 문서의 점수는 함께 점수가 매겨진 배치에 약간 의존합니다. 이는 이 포트가 아니라 참조 레시피의 특성입니다.

### Qwen3-VL generative 점수 계산

프롬프트는 체크포인트 자체의 `additional_chat_templates/reranker.jinja`에서 오며 `add_generation_prompt: true`로 렌더링됩니다. 이 템플릿은 `user`가 아니라 `role: query`와 `role: document` 메시지를 읽고, `system` 메시지가 없으면 자체 기본 instruction(`Given a search query, retrieve relevant candidates that answer the query.`)을 제공하며, 각 이미지 콘텐츠 항목을 `<|vision_start|><|image_pad|><|vision_end|>`로 렌더링합니다. 렌더링은 이 템플릿에 그대로 위임되므로 프롬프트가 다른 재export 체크포인트도 올바르게 유지됩니다. 두 정답 token id는 `1_LogitScore/config.json`(`true_token_id` 9693, `false_token_id` 2152)에서 가져옵니다.

이미지 placeholder는 패딩 전에 Qwen2-VL 프로세서가 각 이미지에 대해 계산하는 visual-token 수로 확장되고, feature는 `Qwen3VLModel::get_input_embeddings`를 통해 병합됩니다. 이미지를 가진 행은 단독으로 점수가 매겨집니다. Qwen3-VL의 M-RoPE 인덱스와 DeepStack 주입은 시퀀스별로 계산되며, 이는 Qwen3-VL-Embedding 계열이 문서화한 것과 같은 제약입니다. 같은 요청의 텍스트 전용 행은 여전히 배치되고 왼쪽 패딩되므로, 혼합 요청은 이미지 문서에 대해서만 행당 비용을 지불합니다. 쿼리와 문서 텍스트는 템플릿이 렌더링하기 전에 longest-first로 truncation되며, scaffold와 visual token을 먼저 확보합니다.

비디오 문서는 범위 밖입니다.

### 요청과 응답

```json
POST /v1/rerank
{
  "model": "optional; must match the served reranker id",
  "query": "string, or {\"text\": ..., \"image\": url}",
  "documents": ["string, or {\"text\": ..., \"image\": url}", "..."],
  "top_n": 3,
  "return_documents": false,
  "instruction": "optional; generative rerankers only"
}
```

항목은 단순 문자열이거나 `text`, `image`, 그리고/또는 `image_url`을 가진 객체입니다. `image`와 `image_url`은 `/v1/embeddings`와 같은 URL 형식(`data:` URI, `file://`, `http(s)://`, 또는 로컬 경로)과, 단순 문자열 또는 `{"url": ...}` 표기를 모두 받아들이며, 클라이언트가 둘 다 보내면 `image`가 우선합니다. 이미지만 있는 객체도 유효한 이미지 문서입니다.

```json
{
  "model": "bge-reranker-v2-m3",
  "results": [
    {"index": 0, "relevance_score": 0.9948, "document": "..."},
    {"index": 1, "relevance_score": 0.0003}
  ],
  "usage": {"prompt_tokens": 312, "total_tokens": 312}
}
```

`results`는 `relevance_score` 내림차순으로 정렬되고 동점은 `index` 오름차순으로 해소된 뒤 `top_n`으로 잘립니다. `document`는 `return_documents`가 true일 때만 존재하며 요청 항목을 그대로 되돌려 줍니다. `prompt_tokens`는 점수가 매겨진 모든 쌍의 실제 token 수를 셉니다.

| 상태 | 타입 | 발생 조건 |
|--------|------|------|
| 400 | `invalid_request_error` | 잘못된 형식의 본문, 비어 있는 `documents`, 텍스트도 이미지도 없는 항목, 1 미만의 `top_n`, `supports_images() == false`인 reranker에 대한 이미지 항목, sequence-classifier reranker에 대한 `instruction`, 또는 서비스 중인 id와 일치하지 않는 `model`. |
| 501 | `not_implemented` | 로드된 reranker가 없음: `No reranker loaded; start with -m <sequence-classifier checkpoint> or --reranker-model <path>`. |
| 503 | `server_busy` | 유한 worker 큐가 가득 참(`--embedding-queue-depth`, 임베딩 worker와 공유). |
| 504 | `server_timeout` | worker가 `--embedding-request-timeout-secs` 이내에 응답하지 않음. |
| 500 | `server_error` | forward pass 실패. |

### Reranker 플래그

| 플래그 | Env | 기본값 | 의미 |
|------|-----|---------|---------|
| `-m <cross-encoder checkpoint>` | `LLAMA_ARG_MODEL` | | 단일 label `ForSequenceClassification` 체크포인트를 `/v1/rerank`에서 서비스하며, 채팅은 로드되지 않은 상태로 유지됩니다. |
| `--reranker-model <path or repo-id>` | `LLAMA_ARG_RERANKER_MODEL`, `MLXCEL_RERANKER_MODEL` | unset | `-m`의 채팅 모델과 함께 `/v1/rerank`에서 서비스되는 체크포인트입니다. `-m`과 같은 방식으로 해석되고 자동 다운로드됩니다. generative reranker에 접근하는 유일한 방법입니다. |
| `--rerank-batch-size N` | `MLXCEL_RERANK_BATCH_SIZE` | `0` (종류별 기본값: 텍스트는 8, 멀티모달은 2) | forward pass당 쿼리/문서 쌍 수. |

큐 깊이와 응답 타임아웃은 임베딩 worker와 공유됩니다(`--embedding-queue-depth`, `--embedding-request-timeout-secs`). 두 단일 스레드 worker가 동일한 방식으로 부하를 흘려보내기 때문입니다. 명시한 `--reranker-model`의 로드 실패는 시작 오류이며, `-m` reranker 체크포인트의 로드 실패는 로그에 기록되고 라우트는 `501`로 응답합니다.

`-m`에 reranker 체크포인트를 지정하고 `--reranker-model`에 *다른* 체크포인트를 지정하면 시작 오류("two rerankers")입니다. 두 곳에 **같은** 디렉터리를 지정하는 것은 rerank 전용 구성입니다. 서버는 `-m`을 요구하고 generative reranker는 `--reranker-model` 없이는 접근할 수 없으므로, 이 조합은 하나의 체크포인트를 `/v1/rerank`에서 서비스하며 채팅 worker는 같은 가중치의 두 번째 사본을 의도적으로 로드하지 않습니다. 이때 채팅은 텍스트 생성기가 아닌 `-m`의 경우와 동일하게 동작합니다. `/v1/chat/completions`는 오류를 반환하고 `/health`는 `loading model`을 보고합니다.

```sh
# A cross-encoder alone: -m is enough.
mlxcel-server -m BAAI/bge-reranker-v2-m3 --port 8080

# Chat plus reranking in one process.
mlxcel-server -m mlx-community/Qwen3-4B-4bit --reranker-model mlx-community/Qwen3-Reranker-0.6B-4bit --port 8080

# A generative reranker alone.
mlxcel-server -m mlx-community/Qwen3-Reranker-0.6B-4bit --reranker-model mlx-community/Qwen3-Reranker-0.6B-4bit --port 8080

curl -s localhost:8080/v1/rerank -H 'Content-Type: application/json' -d '{
  "query": "What is the capital of China?",
  "documents": ["The capital of China is Beijing.", "Gravity is a force that attracts two bodies towards each other.", "Berlin is in Germany."],
  "top_n": 2, "return_documents": true}'
```

### `mlxcel rerank`

```text
mlxcel rerank -m <path or repo-id> -q "query" -d "doc" [-d "doc2" ...] [--image <file> ...]
              [--query-image <file>] [--instruction "..."] [--top-n N]
              [--max-length N] [--batch-size N] [--models-dir PATH] [--json]
```

요청 순서대로 문서마다 하나의 관련성 점수를 출력한 뒤 순위를 출력합니다. 이미지 문서는 결과 순서에서 텍스트 문서 뒤에 옵니다. `--json`은 대신 `scores`, `ranking`, `kind`, `prompt_tokens`를 담은 하나의 객체를 출력합니다. 이것은 모든 reranker 종류를 위한 오프라인 검증 도구로, 서버와 동일한 로더, 프롬프트 조립, 배칭을 listener 없이 사용합니다.

```sh
mlxcel rerank -m BAAI/bge-reranker-v2-m3 -q "what is panda?" \
  -d "hi" \
  -d "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China."

mlxcel rerank -m Qwen/Qwen3-VL-Reranker-2B -q "a chart of quarterly revenue" \
  --image revenue-chart.png --image cat.jpg
```

## 계열 추가하기

1. `src/models/` 아래에 `EmbeddingModel` 구현을 가진 계열 모듈을 추가합니다. 가중치는 `crate::embeddings::loader::load_embedding_weights`(모듈 하위 폴더 포함, 텍스트 bf16 규칙 적용)로 읽고, 풀링 모드는 `crate::embeddings::resolve_pooling_mode(model_dir, family_default)`로 결정합니다. attention 마스크는 위의 빌더를 사용해 `EmbeddingBatch::attention_mask`로부터 만듭니다. 풀링된 `[B, D]` 벡터를 반환합니다(multi-vector 계열은 패딩 행이 0인 `[B, L, D]`). 정규화와 truncation은 엔진이 수행합니다. 프롬프트에 이미지 placeholder가 있는 vision-language 계열은 `EmbeddingModel::expand_image_tokens`도 구현해야 하며, 엔진은 패딩 전에 이를 호출하여 보고되는 token 수가 forward pass가 보는 시퀀스와 일치하도록 합니다.
2. `build_family_model`(`src/embeddings/loader.rs`)에서 해당 계열의 `not yet supported` arm을 생성자로 교체합니다.
3. 양자화된 체크포인트(`config.quantization = {group_size, bits}`)는 `.scales`가 있든 없든 텐서를 받아들이는 `UnifiedLinear::from_weights` / `UnifiedEmbedding::from_weights`를 거치며, `quantization_params(config)`가 해당 블록을 읽습니다.
4. 실제 체크포인트에 대해 `mlxcel embed`로 검증하고 [supported-models.md](supported-models.md#embedding-models)에 해당 계열 행을 추가합니다.

감지(`src/models/detection.rs`), `ModelType` 변형, `mlxcel arch` 항목, 그리고 `src/model_metadata.rs`의 `ModelKind::Embedding` 등록은 위 표의 모든 계열에 대해 이미 존재합니다.
