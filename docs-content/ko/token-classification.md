# Token classification과 span extraction

mlxcel은 token별 출력을 내는 두 종류의 모델을 오프라인으로 실행하며, 각각 별도의 서브커맨드를 가집니다.

| 형태 | 서브커맨드 | Label | 출력 |
|---|---|---|---|
| Token classification | `mlxcel tag` | 체크포인트가 고정(`id2label`, BIO 또는 BIOES) | 태그가 붙은 문자 span과 텍스트의 redaction 사본 |
| Span extraction | `mlxcel extract` | 요청마다 선택(`--labels`, `--labels-json`, `--classify`) | label을 키로 하는 span 목록, 또는 task당 하나의 class |

두 형태는 입력이 다르기 때문에 별도의 trait(`src/token_classification/`의 `TokenClassifierModel`과 `SpanExtractorModel`)입니다. label이 고정된 모델에 요청 label을 넘기면 그것을 조용히 무시해야 하기 때문입니다. 체크포인트를 다른 형태의 서브커맨드에 넘기면, 올바른 서브커맨드를 알려 주는 메시지와 함께 실패합니다.

mlxcel은 의도적인 결정에 따라 어느 형태도 HTTP로 서빙하지 않습니다. 참조 구현인 mlx-vlm은 라우트를 정의하지 않으므로 맞춰야 할 upstream 요청/응답 스키마가 없고, OpenAI 호환 클라이언트에는 token별 태그나 요청 시점 span을 위한 엔드포인트가 없습니다. 클라이언트가 필요로 하면 라우트는 별도의 이슈에서 설계하며, 그때까지는 CLI가 유일한 표면입니다.

## 지원 패밀리

| 패밀리 | 감지 | Kind | 디코딩 | 검증된 체크포인트 |
|---|---|---|---|---|
| BERT / XLM-RoBERTa token classifier | `model_type`이 `bert`, `xlm-roberta`, `xlm_roberta`이고 `architectures[0]`이 `ForTokenClassification`으로 끝나며 비어 있지 않은 `id2label`이 있음 | `bert_token_classifier` | token별 argmax | [`nativ-community/rampart-mlx-fp16`](https://huggingface.co/nativ-community/rampart-mlx-fp16), [`nativ-community/rampart-mlx-4bit`](https://huggingface.co/nativ-community/rampart-mlx-4bit) |
| OpenAI Privacy Filter | `model_type`이 `openai_privacy_filter`이고 비어 있지 않은 `id2label`이 있음 | `openai_privacy_filter` | 제약된 BIOES Viterbi(기본값), `--argmax`를 쓰면 token별 argmax | [`mlx-community/openai-privacy-filter-bf16`](https://huggingface.co/mlx-community/openai-privacy-filter-bf16), [`mlx-community/openai-privacy-filter-8bit`](https://huggingface.co/mlx-community/openai-privacy-filter-8bit) |
| GLiNER2.5 span extractor | `model_type`이 `gliner2` 또는 `gliner2_5`이거나, `architectures`가 `Gliner2ForSchemaExtraction` 또는 `BoundaryExtractor`를 지명함. `architecture`는 `boundary` | `gliner2_5` | 공유 후보 pool에 대한 sigmoid span 점수, label별로 겹침을 해소 | [`fastino/gliner2.5-small-v1`](https://huggingface.co/fastino/gliner2.5-small-v1) |

GLiNER2.5는 첫 번째 span-extraction 패밀리입니다. `mlxcel extract` 아래의 [GLiNER2.5](#gliner25)를 참고하십시오.

이 명령들의 감지는 `config.json`만 읽으며, 변경되지 않은 generation, `/v1/embeddings`, `/v1/rerank` dispatch와는 별개입니다. 특히 `mlxcel embed`는 이 기능이 생기기 전과 똑같이 `BertForTokenClassification` 체크포인트를 BERT 임베더로 로드합니다(`model_type`이 encoder 전용이기 때문입니다).

## `mlxcel tag`

```sh
mlxcel tag -m nativ-community/rampart-mlx-fp16 --keep CITY,STATE,ZIP_CODE \
  "I'm Sarah Connor, 1984 Cyberdyne Ave, Los Angeles CA 90012, sarah@sky.net"
```

```json
{
  "text": "I'm Sarah Connor, 1984 Cyberdyne Ave, Los Angeles CA 90012, sarah@sky.net",
  "spans": [
    {"label": "GIVEN_NAME", "start": 4, "end": 9, "text": "Sarah", "placeholder": "<GIVEN_NAME>"},
    ...
  ],
  "redacted_text": "I'm <GIVEN_NAME> <SURNAME>, <BUILDING_NUMBER> <STREET_NAME>, Los Angeles CA 90012, <EMAIL>"
}
```

플래그는 `python -m mlx_vlm.token_classification`을 그대로 따릅니다.

| 플래그 | 의미 |
|---|---|
| `-m, --model` | 체크포인트 디렉터리 또는 HuggingFace repo-id(`mlxcel generate -m`처럼 자동 다운로드됨) |
| `TEXT` | 태그를 붙일 텍스트. `-`이면 stdin에서 읽음 |
| `--argmax` | 기본 decoder가 Viterbi인 모델에서 token별 디코딩을 강제 |
| `--replacement STR` | redaction된 모든 span을 placeholder 대신 이 문자열로 치환 |
| `--keep L1,L2` | `spans`에는 보고하지만 `redacted_text`에는 그대로 두는 entity label. 알 수 없는 label은 무시됨 |
| `--operating-point NAME` | Viterbi로 디코딩하는 모델(OpenAI Privacy Filter)의 보정 operating point(기본값 `default`). argmax만 쓰는 모델에서는 가중치를 로드하기 전에 거부됨 |
| `--models-dir` | repo-id 해석을 위한 model-store 루트 |

JSON은 stdout으로 출력되고, 한 줄 요약(span 수, kind, `model_type`)은 stderr로 출력됩니다.

redaction되는 것은 `redacted_text`뿐입니다. `text`와 모든 `spans[].text`에는 여전히 원래 값이 들어 있으므로, 출력이 신뢰 경계 밖으로 나갈 때는 `redacted_text`만 전달하십시오.

### 파이프라인

1. special token 없이 토크나이즈합니다. 토크나이저는 문자 offset을 보고해야 하며, 모든 `tokenizer.json` 토크나이저가 그렇습니다. SentencePiece나 tiktoken fallback은 추론 전에 `Token classification requires a tokenizer with offsets`로 거부됩니다. `tokenizer.json`에 내장된 padding과 truncation은 제거되므로 긴 텍스트가 잘리지 않습니다.
2. id를 `context_size - prefix - suffix` token 크기의 윈도로 나누고 각 윈도를 special token으로 감쌉니다(둘 다 있으면 토크나이저의 `[CLS]` / `[SEP]`, XLM-RoBERTa는 `<s>` / `</s>`). BERT의 context는 `max_position_embeddings`이고, XLM-RoBERTa는 테이블이 실제로 지정하는 위치 수(`max_position_embeddings - pad_token_id - 1`)입니다.
3. 각 윈도를 점수화하고, f32로 log-softmax를 취한 뒤, special 위치를 제거하고 윈도들을 이어 붙입니다.
4. label 경로를 디코딩합니다(Privacy Filter의 Viterbi처럼 모델이 제약된 decoder를 기본값으로 쓰지 않는 한 argmax).
5. 모델이 요구하면(BERT가 그렇습니다) sub-token을 단어로 묶습니다. 단어는 `O`가 아닌 첫 piece의 태그를 가지고 나머지는 그 entity의 `I` 상태가 되므로, 윈도 경계에 걸쳐 쪼개진 단어도 하나의 span을 이룹니다.
6. BIO / BIOES 구간으로부터 span을 만듭니다. 다른 entity 뒤의 `I`는 새 span을 시작하고, 짝이 되는 열린 span이 없는 `E`는 한 token짜리 span이 됩니다. 문자 범위는 양끝의 공백을 잘라낸 offset에서 얻으며, 빈 span은 버립니다. span은 `(start, -length, label)` 순으로 정렬되고, 이전에 유지된 span의 끝보다 앞에서 시작하는 span은 버려집니다. 또한(단어 묶기가 있으면) 이전 span이 끝난 바로 그 지점에서 시작하는 같은 label의 span은 그 span에 병합되며, 이 덕분에 `617-555-0142`가 하나의 전화번호로 유지됩니다.
7. `--replacement` 또는 placeholder로 오른쪽에서 왼쪽으로 redaction하며, `--keep` label은 건너뜁니다. placeholder는 `<` + 모든 영숫자가 아닌 연속 문자를 `_`로 바꾸고 양끝의 `_`를 제거한 대문자 label(아무것도 남지 않으면 `REDACTED`) + `>`입니다.

offset은 `text`에 대한 Unicode scalar 인덱스이며 이는 Python 문자열 인덱싱이 쓰는 방식이므로, span 앞에 멀티바이트 문자가 있어도 출력이 mlx-vlm과 필드 단위로 일치합니다. 빈 텍스트는 span 없이 텍스트를 그대로 반환합니다.

### 로드 시점 오류

| 조건 | 메시지 |
|---|---|
| `O`도 아니고 `B-`, `I-`, `E-`, `S-` 뒤에 이름이 오는 형태도 아닌 label | `Invalid token label '<l>'; expected BIO or BIOES labels` |
| `O` label 없음 | `Token labels must include the background label 'O'` |
| `id2label`이 없거나 비어 있음 | `Token-classification checkpoints must define id2label` |
| `id2label` 크기가 `classifier` 텐서와 다름 | `id2label has <n> entries but the classifier has <m> outputs` |
| span-extraction 체크포인트 | `mlxcel extract`를 지명 |
| `B`, `I`, `E`, `S`가 모두 갖춰지지 않은 Privacy Filter entity | `Privacy Filter labels for '<entity>' are missing ["E", "S"]` |
| `viterbi_calibration.json`에 없는 `--operating-point` | `Unknown Privacy Filter operating point '<name>'` |
| 여섯 개의 bias 중 하나가 빠진 operating point | `Viterbi calibration is missing biases: [...]` |
| argmax 전용 모델에 대한 `--operating-point` | `--operating-point '<name>' applies only to Viterbi-decoded checkpoints (openai_privacy_filter)` |

### OpenAI Privacy Filter

[`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter)는 BIOES label로 PII에 태그를 붙이는 14억 파라미터의 양방향 sparse-MoE encoder(8 레이어, expert 128개, top-4)입니다. label은 `O`와, `account_number`, `private_address`, `private_date`, `private_email`, `private_person`, `private_phone`, `private_url`, `secret`에 대한 `B-`, `I-`, `E-`, `S-`입니다. mlxcel은 upstream `PrivacyFilter` 설정(`default_n_ctx`(128000)개의 raw id 윈도, special token 없음, 단어 묶기 없음)으로 같은 `mlxcel tag` 파이프라인을 통해 이를 실행합니다.

```sh
mlxcel tag -m mlx-community/openai-privacy-filter-bf16 "Alice Smith can be reached at alice@example.com."
```

```json
{
  "text": "Alice Smith can be reached at alice@example.com.",
  "spans": [
    {"label": "private_person", "start": 0, "end": 11, "text": "Alice Smith", "placeholder": "<PRIVATE_PERSON>"},
    {"label": "private_email", "start": 30, "end": 47, "text": "alice@example.com", "placeholder": "<PRIVATE_EMAIL>"}
  ],
  "redacted_text": "<PRIVATE_PERSON> can be reached at <PRIVATE_EMAIL>."
}
```

encoder는 GPT-OSS 구성 요소(`src/models/gpt_oss_common.rs`: YaRN 주파수, clipped SwiGLU, bias가 있는 expert projection)로 만든 `src/models/openai_privacy_filter.rs`입니다. 대칭 128 token band 위에서 head별 sink를 가진 bias GQA attention, `1 + 0.1 ln(factor)` attention factor를 쓰는 interleaved YaRN RoPE(`factor` 32, `truncate: false`), q와 k 양쪽의 `head_dim^-0.25`, 그리고 f32의 router와 expert로 구성됩니다(라우팅된 행은 chunk에 64개가 쌓이면 dense와 양자화 plane 모두에서 gather 전에 expert별로 정렬됩니다. dense plane을 정렬하지 않은 채로 두게 했던 "할당 이력에 의존하는" drift는 CUDA overlay가 크기 1인 축의 stride를 정렬된 `gather_mm`의 leading dimension으로 읽던 문제였고 #2462에서 수정되었습니다). attention은 band 안의 key에 대해 `attention_chunk_size`(4096) 크기의 query chunk로 실행되므로 `T x T` mask가 만들어지지 않으며, 라우팅은 `moe_chunk_size`(2048) 크기의 chunk로 실행됩니다. bf16과 8-bit 빌드 외에 다른 `mlx-community/openai-privacy-filter-*` 양자화(`-4bit`, `-5bit`, `-6bit`, `-mxfp4`, `-mxfp8`, `-nvfp4`)도 같은 경로로 로드되지만 검증되지는 않았습니다. `openai/privacy-filter`의 fused Hugging Face expert 레이아웃(`gate_up_proj`, `down_proj`)은 upstream `sanitize`가 하는 것처럼 로드 시 분리됩니다.

**디코딩.** 기본값은 `mlx_vlm.privacy_filter.ViterbiDecoder`를 포팅한 `src/token_classification/viterbi.rs`의 제약된 Viterbi decoder입니다. 경로는 `O`, 임의의 `B`, 또는 임의의 `S`로 시작할 수 있고 `O`, 임의의 `E`, 또는 임의의 `S`로 끝나야 합니다. `O`는 `O` 또는 닫힌 `E` / `S` 뒤에 오고, `B`와 `S`는 `O` 또는 닫힌 상태 뒤에 오며, entity의 `I`와 `E`는 그 entity의 `B` 또는 `I` 뒤에 옵니다. 동점이면 나열된 첫 번째 옵션을, 닫힌 상태나 종료 상태 중에서는 label 출력 순서가 아닌 upstream 순서의 첫 후보(모든 `E`가 모든 `S`보다 앞, 종료 상태 중 `O`가 가장 앞)를 선택합니다. `--argmax`는 독립적인 token별 디코딩으로 전환하며, 이 경우 span builder가 나중에 바로잡는 경로(`O` 뒤의 `I`는 새 span이 됨)가 나올 수 있습니다.

**Operating point.** 체크포인트의 `viterbi_calibration.json`은 `operating_points.<name>.biases`, 즉 lattice 전이에 더해지는 여섯 개의 log 영역 bias(`transition_bias_background_stay`, `transition_bias_background_to_start`, `transition_bias_inside_to_continue`, `transition_bias_inside_to_end`, `transition_bias_end_to_background`, `transition_bias_end_to_start`)를 가집니다. `--operating-point NAME`이 하나를 선택하며(기본값 `default`), 이 파일이 없는 체크포인트는 모두 0으로 디코딩하고, 알 수 없는 이름이나 빠진 키는 가중치를 읽기 전에 실패하며, Viterbi decoder가 없는 모델에서는 이 플래그가 거부됩니다. 공개된 `openai/privacy-filter`와 mlx-community 체크포인트는 모든 bias가 0인 `default` operating point 하나만 제공합니다.

mlx-vlm 0.7.8과의 일치 여부(`generate_privacy_filter_fixture.py`가 생성한 `tests/fixtures/token_classification/privacy_filter_mlx_vlm.json`): bf16 빌드에서 한 문장의 token별 label 확률은 2e-2 이내이고(-18 nat보다 낮은 꼬리 log 확률은 같은 가중치의 bf16과 f32 forward 사이에서 몇 nat씩 움직이므로 log 영역은 -10 nat보다 높은 label에서만 경계가 의미가 있습니다), 여섯 문장(CJK와 움라우트 텍스트 포함)과, Viterbi와 argmax 양쪽에서의 `--replacement` 및 `--keep`의 span이 동일합니다. 8-bit 빌드도 같은 span을 출력합니다. 4096 / 2048 token을 넘는 chunked attention과 라우팅은 같은 가중치의 chunk 없는 forward와 비교해 검증합니다. 긴 반복 텍스트에서는 모델 자체가 많은 token에서 50/50에 놓여 oracle span 비교가 의미가 없기 때문입니다. `mlxcel generate -m <privacy-filter>`는 `openai_privacy_filter is a token classifier; use \`mlxcel tag\``로 실패하며, `mlxcel embed`는 이 체크포인트를 가져가지 않습니다.

## `mlxcel extract`

```sh
mlxcel extract -m fastino/gliner2.5-small-v1 --labels company,person,location --include-confidence --include-spans "Apple hired Sam in New York."
mlxcel extract -m fastino/gliner2.5-small-v1 --labels-json '{"person": "a human name", "company": null}' "Ada Lovelace worked with Charles Babbage."
mlxcel extract -m fastino/gliner2.5-small-v1 --classify '{"sentiment": ["positive", "neutral", "negative"], "topics": {"labels": ["tech", "sports"], "multi_label": true, "threshold": 0.3}}' "The launch was excellent."
```

```json
{
  "entities": {
    "company": [{"text": "Apple", "confidence": 0.9993, "start": 0, "end": 5}],
    "person": [{"text": "Sam", "confidence": 0.9905, "start": 12, "end": 15}],
    "location": [{"text": "New York", "confidence": 0.9995, "start": 19, "end": 27}]
  }
}
```

`--labels`, `--labels-json`, `--classify` 중 정확히 하나가 필요합니다. `--threshold`(기본값 0.5, 0과 1 사이)는 최소 entity 점수이자 기본 multi-label 임계값입니다. `--include-confidence`와 `--include-spans`는 entity 추출에만 적용됩니다. `--max-len <N>`과 `--word-splitter whitespace|char`는 GLiNER2.5 전용입니다. 첫 번째는 체크포인트의 `max_len`을 덮어쓰며(1..=32768 범위의 정수, 기본값은 체크포인트의 값으로 `fastino/gliner2.5-small-v1`에서는 4096), 두 번째는 기본값인 공백 구간 대신 공백이 아닌 문자 하나마다 단어로 자릅니다(`char`, CJK처럼 띄어쓰기가 없는 문자 체계용).

entity 출력은 mlx-vlm의 GLiNER2.5 `extract_entities`를 따릅니다. 요청한 label마다 하나의 키를 요청 순서대로 가진 `{"entities": {"<label>": [...]}}`입니다. 각 항목은 매칭된 문자열이며, `--include-confidence` 또는 `--include-spans`가 설정되면 `{"text", "confidence"?, "start"?, "end"?}`입니다. `--classify`는 `{"<task>": [labels] | {"labels", "prompt"?, "label_descriptions"?, "multi_label"?, "threshold"?}}`를 받아 `{"<task>": "<label>"}`을 출력하며, multi-label task에서는 label 목록을 출력합니다.

token-classification 체크포인트를 `mlxcel extract`에 넘기면 가중치를 로드하기 전에 `mlxcel tag`를 지명하며 실패합니다.

### GLiNER2.5

[GLiNER2.5](https://huggingface.co/fastino/gliner2.5-small-v1)(Fastino)는 *boundary head*를 가진 DeBERTa-v3 encoder입니다. 요청의 label들이 텍스트와 함께 인코딩되고, span은 생성 없이 시작/끝 boundary, 내용, label query로 점수가 매겨집니다. `src/token_classification/gliner2_5.rs`가 mlx-vlm의 `mlx_vlm/models/gliner2_5`(`gliner2_5.py`, `boundary.py`)를 포팅한 것이며, encoder는 재사용 가능한 `src/models/deberta_v2.rs`입니다.

**감지.** 실제 `fastino/gliner2.5-small-v1`의 `config.json`은 `model_type: gliner2`와 `architectures: ["Gliner2ForSchemaExtraction"]`를 선언하는데, mlx-vlm의 로더는 이를 매핑하지 않습니다(`BoundaryExtractor`를 키로 사용). mlxcel은 `model_type`이 `gliner2` 또는 `gliner2_5`이고 둘 중 어느 architecture든 받아들입니다. `architecture`는 `boundary`여야 하며(`BoundaryExtractor` export에서만 없을 수 있음), 다른 head는 `GLiNER2 architecture '<a>' is not supported; only the boundary extractor is`로 실패합니다. encoder config는 `encoder_config`에서 인라인으로 읽고, 없으면 `encoder_config/config.json`에서 읽으며, 그것도 없으면 그 파일을 지명하며 로드가 실패합니다. upstream과 마찬가지로 `shared`가 아닌 `candidate_pool`, 0보다 큰 `candidate_attention_layers` 또는 `query_attention_layers`는 로드 시 거부됩니다. 1024를 넘는 `boundary_head.pool_boundary_top_k`(후보 격자가 그 제곱이 됨), 32768을 넘는 `max_len`, `share_att_key: true`가 없는 encoder config(Hugging Face의 기본값은 `false`이며, 이 포트가 구현하지 않는 별도 projection 레이아웃)도 거부됩니다. `boundary_proposer`, `pair_scorer`, `candidate_encoder`, `record_decoder`, `relation_scorer` 텐서(relation 및 record 추출용)는 버려집니다. 가중치는 체크포인트 자체의 dtype으로 로드됩니다(`gliner2.5-small-v1`은 float32 텐서를 제공하며 mlx-vlm에서처럼 float32로 실행됩니다).

**입력.** 시퀀스는 `( [P] <prompt> ( [E] label ... ) ) [SEP_TEXT]` 뒤에 각 단어의 sub-token이 이어지는 형태입니다. 여기서 prompt는 추출의 경우 `entities`(classification은 task 이름이며, task가 prompt를 주면 `"<task>: <prompt>"`)이고, 설명이 있는 모든 label에는 ` [DESCRIPTION] <label>: <description>`이 덧붙습니다. classification은 `[L]` 마커를 사용합니다. 토크나이저는 `[SEP_TEXT]`, `[P]`, `[E]`, `[L]`, `[DESCRIPTION]`을 이미 정의하고 있어야 하며, 그렇지 않으면 로드가 `checkpoint tokenizer is missing GLiNER2.5 special tokens`로 실패합니다. 단어는 upstream의 정규식(URL, 이메일 주소, `@handles`는 통째로 유지되고, `\w+` 구간은 `-` 또는 `_`로 이어질 수 있으며, 그 밖의 공백이 아닌 문자는 각각 하나의 단어)에서 나오고 소문자로 변환됩니다. sub-token이 없는 단어는 `[UNK]`가 됩니다. 단어는 다음 단어가 `max_len`(기본 4096)을 넘기기 직전까지 추가됩니다. 스키마만으로 `max_len`에 도달하면 `schema alone exceeds max_len`으로 실패하고, 텍스트의 어떤 단어도 들어가지 않으면 `no text tokens fit within max_len`으로 실패합니다. `.`, `!`, `?`로 끝나지 않는 텍스트에는 offset을 보고하기 전에 `.`가 덧붙습니다. 호출 한 번은 시퀀스 하나를 인코딩합니다. Python 모델의 `char` 단어 분리기(ASCII 문자, 숫자, `@._-+`의 연속은 단어이고, 그 밖의 공백이 아닌 문자는 각각 하나의 단어. 띄어쓰기가 없는 문자 체계용)는 `--word-splitter char`로 사용할 수 있으며, `max_len`은 config에서 오고 `--max-len`으로 덮어쓸 수 있습니다(둘 다 GLiNER2.5 전용, 1..=32768 범위의 정수). 단어 문자는 Python의 `\w`(문자, 숫자, `_`이며 결합 부호나 joiner는 제외)를 따르고, 단어 정규식의 이메일 분기는 RFC 5321 길이로 제한되어 긴 점 구간이 단어마다 다시 스캔되지 않습니다. 어차피 유효하지 않은 주소만 upstream과 다르게 분리됩니다. `max_len`을 넘는 단어는 `tracing` 경고와 함께 버려집니다(`RUST_LOG=warn`으로 확인 가능). mlx-vlm이 변환한 체크포인트(`encoder.layers.`, `.self_attn.`, `.layer_norm.` 이름, 양자화 여부 무관)는 Hugging Face 레이아웃과 같은 경로로 로드됩니다.

**디코딩.** span logit은 sigmoid를 거치며, sigmoid 이후 abstention logit이 0.5를 넘는 label은 아무것도 반환하지 않습니다. `--threshold` 이상이고 유효한 단어 범위를 가진 후보는 upstream의 가중 구간 동적 계획법(`_resolve_flat_overlaps`)으로 label별로 해소되어, 총점이 가장 큰 겹치지 않는 집합을 유지하며 점수 내림차순, 다음으로 start, 다음으로 end 순으로 나열됩니다. 문자 offset은 (`.`로 끝나도록 보정되었을 수 있는) 텍스트에 대한 Unicode scalar 인덱스입니다. `--classify`는 task당 하나의 시퀀스를 실행하고, label 마커에 대한 classifier의 sigmoid를 취해 가장 좋은 label을 반환하며, `multi_label`이 설정되면 task 임계값 이상인 모든 label을 반환합니다.

**일치 여부.** `tests/fixtures/token_classification/gliner2_5_mlx_vlm.json`(git `73d366ec`의 mlx-vlm 0.7.8로 `generate_gliner2_5_fixture.py`가 작성)에는 label 설명이 있는 경우와 없는 경우의 README 추출 예제, 끝 구두점이 없는 이메일 / URL 텍스트, `char` 분리기를 쓰는 CJK 텍스트, 두 개의 classification task가 들어 있습니다. mlxcel은 입력 id, 단어 및 마커 위치, span, offset을 정확히 재현하고 confidence는 1e-3 이내로 재현합니다. 모든 fixture 케이스가 128 token 미만인 이유는, mlx-vlm의 DeBERTa가 `max_relative_positions`가 -1일 때 상대 위치를 bucket화하지 않고 clip하는 반면 Hugging Face는 log-bucket화하기 때문입니다. mlxcel은 Hugging Face를 따르며, `src/models/deberta_v2_tests.rs`가 306 token 입력을 `transformers`의 `DebertaV2Model`과 비교합니다(fixture `deberta_v2_hf_long.json`). 여기서 mlx-vlm의 hidden state는 `transformers`와 최대 7e-2까지 차이가 납니다. 같은 테스트는 encode와 그 replay가 비트 단위로 일치함을 단언하며, `real_encoder_first_run_equals_replay`는 이를 새 프로세스에서 확인합니다. CUDA에서는 GB10의 새 프로세스 중 약 40%에서 첫 실행이 잘못된 상태를 반환했는데, 이는 cuBLASLt GEMM이 같은 CUDA graph의 무관한 커널 노드와 겹쳐 실행되기 때문이었고, CUDA device overlay가 캡처된 child graph를 barrier로 만들어 이를 방지합니다(#2462, `MLXCEL_CUDA_GRAPH_CAPTURE_BARRIER`). `mlxcel generate -m <gliner>`는 `GLiNER2.5 is a span extractor; use \`mlxcel extract\``로 실패합니다.

## 패밀리 추가하기

1. `src/token_classification/loader.rs`에 `as_str`, `description`, `output_shape` arm을 갖춘 `TokenModelKind` variant를 추가합니다.
2. 같은 파일의 `DETECTORS`에 감지 함수를 추가합니다(config만 사용하며, 자신이 담당하지 않는 config에는 `Ok(None)`을 반환).
3. `TokenClassifierModel`(고정 label. `log_probs`는 감싸진 윈도 하나를 받아 id당 한 행을 반환하며, `decode_modes()[0]`이 기본값)이나 `SpanExtractorModel`을 구현하고, `load_token_classifier` 또는 `load_span_extractor`에 로더 arm을 추가합니다.
4. 참조 구현이 만든 fixture에 대한 실제 체크포인트 테스트를 추가합니다. `src/token_classification/real_checkpoint_tests.rs`가 `tests/fixtures/token_classification/generate_rampart_spans.py`로 Rampart에 대해, `src/token_classification/privacy_filter_checkpoint_tests.rs`가 `generate_privacy_filter_fixture.py`로 Privacy Filter에 대해, `src/token_classification/gliner2_5_checkpoint_tests.rs`가 `generate_gliner2_5_fixture.py`로 GLiNER2.5에 대해 하는 방식과 같습니다.
