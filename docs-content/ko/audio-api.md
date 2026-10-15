# OpenAI 오디오 API (`/v1/audio`)

`mlxcel serve`와 `mlxcel-server`는 OpenAI 호환 오디오 엔드포인트 세 가지를 제공합니다. 텍스트 음성 변환(`/v1/audio/speech`), 전사(`/v1/audio/transcriptions`), 번역(`/v1/audio/translations`)입니다.

**현재 상태.** 세 라우트 모두 마운트되어 동작합니다. `/v1/audio/transcriptions`는 llama-server도 마운트하는 유일한 라우트이므로 llama-server의 의미론을 따르며, 로드된 채팅 모델이 오디오를 받을 수 있으면 그 모델이 처리하고 그렇지 않으면 Whisper 프로바이더로 폴백합니다(이슈 #1446). `/v1/audio/translations`는 mlxcel 고유의 라우트이며 Whisper 프로바이더가 처리합니다. `-m`으로 Whisper 체크포인트를 전달하면 STT 슬롯이 자동으로 채워집니다. 텍스트 음성 변환(`/v1/audio/speech`)은 Kokoro-82M 프로바이더가 처리합니다. `-m`으로 Kokoro 체크포인트를 전달하면 TTS 슬롯이 자동으로 채워집니다. 모델 종류가 로드되지 않은 라우트는 요청이 완전히 파싱된 후 501을 반환합니다.

## 구현된 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|-------------|
| POST | `/v1/audio/speech` | 텍스트 음성 변환: JSON 본문을 입력받아 바이너리 WAV를 출력합니다. |
| POST | `/v1/audio/transcriptions` | 음성 텍스트 변환: `multipart/form-data`를 입력받아 JSON 텍스트를 출력합니다. |
| POST | `/v1/audio/translations` | 영어 출력의 음성 텍스트 변환: 전사와 동일한 multipart 형식입니다. |

`/v1` 접두사가 없는 별칭 경로도 마운트됩니다: `/audio/speech`, `/audio/transcriptions`, `/audio/translations`.

## 구현 소스 맵

| 모듈 | 책임 |
|--------|----------------|
| `src/server/audio_model.rs` | `AudioModelProvider` trait, 입출력 타입, `AudioModelKind`, `AudioModelError`. |
| `src/server/audio_worker.rs` | `AudioWorker`와 `AudioEngine`: 오디오 모델을 로드하고 실행하는, MLX를 소유한 전용 스레드. |
| `src/server/whisper_stt.rs` | `WhisperSttProvider`: WAV 리더와 Whisper 프런트엔드를 `AudioModelProvider` 접점에 연결합니다. |
| `src/server/kokoro_tts.rs` | `KokoroTtsProvider`: g2p 프런트엔드와 Kokoro 음향 모델을 `AudioModelProvider` 접점에 연결합니다. |
| `src/server/routes/audio.rs` | HTTP 핸들러, multipart 파서, 포맷 결정, 바이너리/JSON 응답 빌더. |
| `src/server/types/request.rs` | `AudioSpeechRequest`, `AudioTranscriptionRequest`(스키마 참조). |
| `src/server/types/response.rs` | `AudioTranscriptionResponse`, `ErrorResponse`(`not_implemented` 포함). |
| `src/audio/wav_writer.rs` | `encode_wav_pcm16`: `f32` PCM 샘플을 RIFF WAV 바이트로 변환합니다. |
| `src/audio/whisper_mel.rs` | Log-mel 프런트엔드: STFT, Slaney mel 필터뱅크, 정규화, 16 kHz 리샘플러. |
| `src/server/app.rs` | 라우트 등록과 `AUDIO_MAX_UPLOAD_BYTES`(25 MiB) 본문 제한 레이어. |

## Whisper STT 설정

`-m`에 Whisper 체크포인트 디렉터리를 전달하십시오. 서버는 `config.json`의 `model_type: "whisper"`를 감지하고, 모든 MLX 그래프 평가를 위한 전용 워커 스레드를 소유하는 `WhisperSttProvider`를 통해 STT 슬롯을 채웁니다.

```sh
mlxcel-server -m models/whisper-base
```

네이티브 MLX 키 레이아웃과 HuggingFace `WhisperForConditionalGeneration` 레이아웃 모두 변환 없이 로드됩니다. 체크포인트 디렉터리에는 `config.json`, 하나 이상의 SafeTensors 가중치 파일, `tokenizer.json`이 있어야 합니다.

Whisper 체크포인트를 로드하면 오디오 슬롯만 점유합니다. 서버는 해당 프로세스에서 `/v1/chat/completions`나 생성 요청을 처리하지 않습니다. 채팅과 STT는 별도의 서버 인스턴스입니다.

**지원되는 오디오 입력.** 전용 Whisper 워커는 현재 WAV만 허용합니다. 오디오는 공유 WAV 리더로 디코딩되고, 모노로 변환되며, log-mel 프런트엔드 이전에 16 kHz로 리샘플링됩니다. MP3와 FLAC는 아래에 설명된 Phi-4 Multimodal 및 Gemma 3n 채팅 모델 경로에서 지원되지만, 독립 실행형 Whisper 워커에 도달하기 전에는 변환되지 않습니다.

**긴 오디오.** 30초보다 긴 오디오는 연속된 30초 윈도우로 분할되며 각각 독립적으로 전사됩니다. 결과는 순서대로 이어 붙입니다. 단어 수준 타임스탬프, 세그먼트 수준 타임스탬프, VAD 기반 청킹은 후속 작업입니다.

**현재 제한 사항.** 비양자화(fp16/f32) 체크포인트만 지원하며, 양자화된 Whisper 가중치는 아직 지원되지 않습니다. Greedy 디코딩만 지원하며, beam search는 후속 작업입니다.

## Kokoro TTS 설정

`-m`에 Kokoro-82M 체크포인트 디렉터리를 전달하십시오. 서버는 `config.json`의 `istftnet` config 블록 또는 `kokoro-v1_0.safetensors`의 존재로 체크포인트를 감지하고, 모든 MLX 그래프 평가를 위한 전용 워커 스레드를 소유하는 `KokoroTtsProvider`를 통해 TTS 슬롯을 채웁니다.

```sh
mlxcel-server -m models/kokoro-82m
```

체크포인트 디렉터리에는 `config.json`(`vocab` 및 아키텍처 블록 포함), `kokoro-v1_0.safetensors`, 그리고 음성별 safetensors 팩이 들어 있는 `voices/` 하위 디렉터리가 있어야 합니다. Hugging Face의 Kokoro 체크포인트(`hexgrad/Kokoro-82M`)에는 54개의 음성 팩이 포함되어 있습니다.

Kokoro 체크포인트를 로드하면 오디오 슬롯만 점유합니다. 서버는 해당 프로세스에서 `/v1/chat/completions`나 생성 요청을 처리하지 않습니다. TTS와 텍스트 생성은 별도의 서버 인스턴스입니다.

**음성.** 요청의 `voice` 필드는 `voices/<name>.safetensors`에서 팩을 선택합니다. 사용 가능한 음성 이름은 파일 이름(확장자 제외)입니다(예: `af_heart`, `bm_lewis`). 기본값은 `af_heart`입니다. 존재하지 않거나 안전하지 않은 문자(`[A-Za-z0-9_-]` 이외의 모든 문자)를 포함하는 음성을 요청하면 조용히 `af_heart`로 폴백합니다.

**언어 범위.** 내장 g2p 프런트엔드는 미국 영어 전용입니다. 체크포인트의 비영어 음성도 로드되고 합성되지만 음소는 여전히 영어 프런트엔드에서 나오므로, 비영어 텍스트의 발음 품질은 제한적입니다.

**입력 길이.** 입력 텍스트는 g2p 실행 전에 4096자로 제한됩니다. 더 긴 입력은 거부되지 않고 잘리므로 다소 긴 정상 요청도 여전히 합성됩니다. 음향 모델은 최대 510개의 음소 토큰을 처리하며, 그 이상의 음소는 버려집니다.

**현재 제한 사항.** `response_format`은 `wav`만 지원하며, 다른 컨테이너는 후속 작업입니다. g2p 프런트엔드는 미국 영어를 다루며, 언어별 phonemizer는 향후 작업입니다. 양자화된 Kokoro 체크포인트는 아직 테스트되지 않았습니다.

## POST /v1/audio/speech

**요청 본문 (JSON):**

| 필드 | 타입 | 필수 | 비고 |
|-------|------|----------|-------|
| `model` | string | 예 | TTS 모델의 식별자입니다. 라우트 레이어에서는 무시되며 어떤 문자열이든 허용됩니다. |
| `input` | string | 예 | 합성할 텍스트입니다. 4096자로 제한되며, 더 긴 입력은 g2p 전에 잘립니다. |
| `voice` | string | 아니요 | Kokoro 음성 이름입니다(예: `af_heart`, `bm_lewis`). 기본값은 `af_heart`이며, 알 수 없는 이름은 기본값으로 폴백합니다. |
| `response_format` | string | 아니요 | 출력 컨테이너입니다. `wav`만 지원하며, 필드를 생략하면 기본값은 `wav`입니다. 그 외의 값은 400을 반환합니다. |
| `speed` | float | 아니요 | 길이 스케일 계수입니다. 1.0보다 큰 값은 더 짧은(빠른) 오디오를, 1.0보다 작은 값은 더 긴(느린) 오디오를 생성합니다. 양수가 아니거나 유한하지 않은 값은 기본값 1.0이 됩니다. |

**성공 응답 (200):**

Content-Type: `audio/wav`. Content-Disposition: `attachment; filename=speech.wav`. 본문은 44바이트 RIFF WAV 헤더 뒤에 프로바이더가 반환한 샘플 레이트의 16비트 리틀 엔디언 PCM 샘플이 이어지는 형태입니다.

**오류 응답:**

| 상태 | 조건 |
|--------|-----------|
| 400 | 지원되지 않는 `response_format`(`wav`가 아니거나 생략되지 않은 모든 값). |
| 501 | 등록된 TTS 모델이 없습니다. 본문: `{"error":{"type":"not_implemented","message":"audio model kind not loaded: tts"}}` |
| 503 | 모든 슬롯이 사용 중입니다: 생성 배치 큐 또는 제한된 오디오 워커 큐(`--audio-queue-depth`)가 가득 찼습니다. |
| 504 | 오디오 워커가 요청별 타임아웃(`--audio-request-timeout-secs`) 내에 응답하지 않았습니다. |

**예제 (curl):**

```sh
curl -s -X POST http://localhost:8080/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"my-tts","input":"Hello world"}' \
  --output speech.wav
```

## POST /v1/audio/transcriptions

이 라우트는 llama-server도 마운트하는 유일한 오디오 라우트이므로 mlxcel의 의미론이 아니라 **llama-server b10621의 의미론**을 따릅니다(이슈 #1446). 전체 계약과 남아 있는 차이점은 [`llama-server-compat.md`](llama-server-compat.md)를 참고하십시오. 이 섹션은 운영자를 위한 요약입니다.

**어떤 모델이 전사하는가.** 서버가 가진 것 중 다음 순서로 선택됩니다.

1. **로드된 채팅 모델**: 오디오를 받을 수 있는 경우(`gemma3n`, omni 계열, 그리고 타워가 `input_audio` 파트를 허용하는 기타 체크포인트). 이는 llama-server가 하는 방식이며 llama-server가 표현할 수 있는 유일한 서버 형태입니다.
2. **전용 Whisper 워커**: `-m`이 Whisper 체크포인트를 지정한 경우. 이 형태에는 채팅 모델이 전혀 없으며 mlxcel 고유의 방식입니다. llama-server에는 이에 해당하는 것이 없습니다.

둘 다 없으면 이 라우트는 `The current model does not support audio input.`와 함께 `501 not_supported_error`로 응답합니다.

**요청 본문 (`multipart/form-data`):**

| 필드 | 타입 | 필수 | 비고 |
|-------|------|----------|-------|
| `file` | 파일 파트 | 예 | 매직 바이트로 감지되는 인코딩된 오디오 바이트입니다. Phi-4 Multimodal과 Gemma 3n은 WAV, MP3, FLAC를 허용하고, 현재의 다른 채팅 오디오 핸들러와 독립 실행형 Whisper는 WAV만 허용합니다. 반복된 `file` 파트는 마지막 것을 유지합니다. |
| `model` | text | 아니요 | 업스트림이 채팅 본문으로 전달하는 것과 같이 전달만 되고 무시됩니다. |
| `prompt` | text | 아니요 | ASR 프롬프트를 대체합니다. 기본값: `Transcribe audio to text`. 채팅 모델 경로에서 적용되며, Whisper 워커는 무시합니다. |
| `language` | text | 아니요 | 업스트림이 전달하는 방식대로 프롬프트 뒤에 `" (language: xx)"`로 덧붙여집니다. |
| `response_format` | text | 아니요 | `json`만 지원합니다(기본값). 그 외의 값은 400을 반환합니다. `text`와 `verbose_json`은 `/v1/audio/translations`에 있습니다. |
| `temperature` | text | 아니요 | `f32`로 파싱되며, 파싱되지 않는 값은 400을 반환합니다. |
| `max_tokens` | text | 아니요 | 정수로 파싱되며, 파싱되지 않는 값은 400을 반환합니다. |
| `stream` | text | 아니요 | 폼이 문자열을 전달하므로 리터럴 `"true"`와 비교됩니다. |

알 수 없는 텍스트 필드는 전달만 되고 무시됩니다. 반복된 텍스트 필드는 배열로 합쳐지며 해당 필드의 기본값으로 폴백하므로, 중복된 `prompt`는 무시되고 중복된 `response_format`은 `json`으로 결정됩니다.

**컨테이너 지원은 선택된 모델 경로에 따라 달라집니다.** Phi-4 Multimodal과 Gemma 3n이 사용하는 공유 waveform 전처리기는 WAV, MP3, FLAC를 디코딩합니다. 현재의 Gemma 4, Gemma 4 Unified, Qwen3-Omni, Nemotron-H Nano Omni, 독립 실행형 Whisper 핸들러는 여전히 인트리(in-tree) WAV 리더를 직접 호출합니다. 라우트는 디스패치 전에 세 가지 호환 컨테이너를 모두 판별(sniff)하고 범위를 제한하지만, 이러한 WAV 전용 핸들러로 라우팅된 MP3 또는 FLAC 요청은 모델 준비 중에 실패합니다. 이는 파일 이름 규칙이 아니라 백엔드의 경계입니다.

**성공 응답**(`json`, 기본값). 이는 OpenAI의 기존 `{"text": ...}` 객체가 **아니라** llama-server가 내보내는 transcript 이벤트 형태임에 유의하십시오.

```json
{"type":"transcript.text.done","text":"The quick brown fox jumps over the lazy dog.","usage":{"type":"tokens","input_tokens":206,"output_tokens":10,"total_tokens":216,"input_tokens_details":{"cached_tokens":0}}}
```

Whisper 워커 경로에서는 `usage` 카운트가 0입니다. 워커가 프롬프트 또는 디코딩된 토큰 수를 보고하지 않기 때문입니다.

**스트리밍 응답**(`stream=true`): `{"type":"transcript.text.delta","delta":"..."}` 프레임, 위의 `done` 프레임, 그리고 `data: [DONE]`을 담은 `text/event-stream`입니다. 채팅 모델 경로는 생성이 진행되는 대로 디코딩된 토큰마다 하나의 delta를 내보냅니다. 독립 실행형 Whisper 워커는 완성된 전사본만 가지고 있으므로 전체 결과를 담은 delta 하나를 내보냅니다.

**제한**은 모두 디코더가 클립을 보기 전에 적용됩니다. 최대 32개의 multipart 파트, 파트당 25 MiB, 그리고 WAV, MP3 또는 FLAC 헤더에서 읽은 컨테이너 형상(최대 192 kHz, 8채널, 600초). 압축된 출력은 디코딩 중에 다시 제한되며, 파일이 담고 있는 것보다 더 많은 오디오를 선언한 WAV `data` 청크는 실제 존재하는 만큼으로 조정됩니다.

**오류 응답:**

| 상태 | 조건 |
|--------|-----------|
| 400 | 잘못된 multipart, 너무 많은 파트, `file` 파트 없음(`No input file found for transcription`), `json`이 아닌 `response_format`(`Only 'json' response_format is supported for transcription`), 숫자가 아닌 `temperature` 또는 `max_tokens`, 지원되지 않는 컨테이너, 잘못된 컨테이너 메타데이터, 또는 형상 제한을 초과하는 클립. |
| 413 | 업로드 본문이 25 MiB를 초과합니다. |
| 501 | 오디오를 지원하는 채팅 모델도 STT 워커도 로드되지 않았습니다. 본문: `{"error":{"type":"not_supported_error","message":"The current model does not support audio input."}}` |
| 503 | 모든 슬롯이 사용 중입니다: 생성 배치 큐 또는 제한된 오디오 워커 큐(`--audio-queue-depth`)가 가득 찼습니다. |
| 504 | 오디오 워커가 요청별 타임아웃(`--audio-request-timeout-secs`) 내에 응답하지 않았습니다. |

**예제 (curl):**

```sh
curl -s -X POST http://localhost:8080/v1/audio/transcriptions \
  -F file=@recording.wav \
  -F model=my-stt \
  -F language=en \
  -F response_format=json
```

## POST /v1/audio/translations

mlxcel 전용 라우트입니다. llama-server는 이를 마운트하지 않으므로 mlxcel 고유의 의미론을 유지하며, 추가 응답 포맷이 있는 곳입니다. 전사 라우트와 동일한 multipart 형식이며, 소스 언어와 관계없이 로드된 모델이 영어를 출력하도록 요청하고 Whisper 워커가 처리합니다.

| `response_format` | 본문 |
|---|---|
| `json`(기본값) | `{"text":"..."}` |
| `text` | `Content-Type: text/plain`의 일반 UTF-8 |
| `verbose_json` | `{"text":"...","language":"...","duration":1.5}`, 프로바이더가 `language`와 `duration`을 반환하지 않으면 생략됩니다 |

`501` 메시지는 `stt`를 명시합니다(동일한 기본 기능 방향).

## 요청 검증 순서

501 응답은 요청이 완전히 파싱된 **후에** 반환됩니다. 이는 다음을 의미합니다.

- 모델이 로드되지 않았더라도 잘못된 multipart 본문은 501이 아니라 400을 반환합니다.
- speech 요청의 `response_format` 오타는 TTS 모델이 로드된 경우 400을 반환하고 로드되지 않은 경우 501을 반환합니다(포맷은 모델 확인 이후에 결정됨).
- `file` 필드는 전사/번역에 필수이며, 없으면 모델 상태와 관계없이 400을 반환합니다.

이 순서 덕분에 호출자는 로드된 모델 없이도 잘못된 요청과 모델 부재를 구별할 수 있습니다.

## 큐 제한 및 요청별 타임아웃

모든 오디오 요청은 MLX를 소유한 하나의 전용 워커 스레드를 통해 직렬화됩니다(모델 가중치는 스레드 종속적이므로 이를 로드한 스레드가 실행도 해야 합니다). 두 가지 서버 설정이 이 경로에 상한을 두어 폭주하는 요청이나 멈춘 요청이 가용성을 저하시키지 않도록 합니다.

- **큐 제한**(`--audio-queue-depth`, 환경 변수 `MLXCEL_AUDIO_QUEUE_DEPTH`, 기본값 `8`). 워커의 명령 채널은 제한되어 있습니다. 처리 중인 요청 뒤에서 대기할 수 있는 요청은 이 수만큼이며, 그다음 요청은 무제한으로 큐에 쌓이지 않고 구조화된 `503`("All slots are busy")으로 거부됩니다. 이는 큐에 쌓이는 메모리에 상한을 둡니다. 큐에 쌓인 각 음성 텍스트 변환 명령은 요청당 최대 25 MiB 페이로드를 보유하므로, 기본값 `8`은 큐에 쌓인 페이로드를 처리 중인 하나를 제외하고 대략 200 MiB로 제한합니다. `0`은 최소 하나의 큐 명령으로 조정됩니다(용량이 0인 rendezvous 채널은 우리가 원하는 수용 동작이 아닙니다).

- **요청별 타임아웃**(`--audio-request-timeout-secs`, 환경 변수 `MLXCEL_AUDIO_REQUEST_TIMEOUT_SECS`, 기본값 `120`). 호출자는 최대 이 시간만큼 워커 응답을 기다린 뒤 구조화된 `504`를 반환합니다. 타임아웃은 호출자의 블로킹 스레드를 해제하지만 워커에서 진행 중인 모델 작업을 취소하지는 않습니다(단일 워커는 한 번에 하나의 요청만 안전하게 처리할 수 있음). 워커가 결국 작업을 마치면 그 응답은 조용히 버려집니다. `0`은 즉시 타임아웃되는 대신 기본값으로 폴백합니다.

두 설정 모두 공유 워커에 적용되므로 STT(Whisper)와 TTS(Kokoro) 경로를 함께 다룹니다.

## 장애 격리

오디오 워커의 각 엔진 호출은 `catch_unwind` 경계(`src/server/audio_worker.rs`의 `run_guarded`) 아래에서 실행되므로, 합성 또는 전사 중 패닉은 요청별 `Inference` 오류가 되고 워커 스레드는 서버를 중단시키지 않고 계속 서비스합니다. 이슈 #375 이후로 릴리스 빌드에서도 이것이 유지됩니다. 릴리스 프로필은 `panic = "unwind"`를 사용합니다(이전에는 프로덕션에서 이 경계를 조용히 무력화하던 `panic = "abort"`를 사용했음). 핵심 텍스트 생성 워커 스레드는 반대로 의도적인 방침을 취합니다. 그곳에서 포착되지 않은 패닉은 불변 조건이 깨졌다는 뜻이므로, 언와인딩하는 대신 로그를 남기고 프로세스를 `abort`하여 감독되는 재시작에 맡깁니다(ADR 0003 참조).

MLX C++ FFI 예외 경로(이슈 #382)는 PR #384부터 Kokoro 합성 경로에서 차단됩니다. 정렬 확장(alignment-expansion) matmul과 최종 PCM 읽기는 cxx 브리지에서 `-> Result<..>`로 선언된 `try_matmul`과 `try_array_to_raw_bytes`를 거치므로, MLX가 throw하면 `std::terminate`가 아니라 요청별 `Err`가 됩니다. PR #434는 같은 패턴을 `try_conv2d`와 `try_conv1d`로 확장하여 Gemma 4 Conformer 오디오 인코더의 컨볼루션 경로(이슈 #427)를 다루는데, 이곳의 입력 형상은 런타임 오디오 길이에서 유도되므로 형상 오류가 발생하면 서버가 중단될 수 있었습니다. PR #439는 동일한 `try_conv2d`/`try_conv1d` 적용 범위를 Nemotron-H Nano Omni Conformer/Parakeet 오디오 인코더(이슈 #435)로 확장했습니다. 해당 인코더의 데이터 의존적인 conv 호출 네 개 모두가 이제 실패 가능한 변형을 거치므로 형상 오류는 서버를 종료하는 대신 요청별 `Err`를 반환합니다. Whisper는 텐서 형상이 고정되어 있고 MLX 호출 전에 검사되므로 영향을 받지 않습니다. `std::exception`이 아닌 타입을 throw하는 `cxx` 연산은 여전히 프로세스를 종료시키며, 현재 오디오 경로에는 그러한 throw가 없습니다.

## 오디오 모델 프로바이더 추가하기

`crate::server::AudioModelProvider`를 구현하고 시작 시 등록하십시오.

```rust
let state = AppState::new(...)
    .with_audio_model(Some(Arc::new(MyProvider::new(...))));
```

프로바이더는 지원하는 방향(`transcribe` 또는 `synthesize`)만 오버라이드하고 `supports(AudioModelKind)`로 기능을 보고합니다. 구현되지 않은 방향의 기본 구현은 `AudioModelError::KindNotLoaded`를 반환하며, 라우트 레이어가 이를 다시 501로 매핑합니다.

## WAV 인코딩

TTS 라우트는 `f32` PCM 샘플을 16비트 리틀 엔디언 PCM 데이터를 가진 표준 44바이트 RIFF WAV로 인코딩합니다(`src/audio/wav_writer.rs`의 `encode_wav_pcm16`). 인코더는 다음과 같이 동작합니다.

- `[-1.0, 1.0]` 밖의 값을 `i16` 범위로 제한(clamp)합니다(wrapping 없음).
- `NaN`을 0(무음)으로 매핑합니다.
- 샘플이 인터리브된 경우(`L, R, L, R, ...`) 멀티채널 오디오를 허용합니다.
- 손상된 헤더를 만드는 대신 표현 가능한 최대 32비트 RIFF 페이로드에서 잘라냅니다.

출력은 기존 WAV 리더(`load_wav_from_bytes`)를 통해 16비트 양자화 단계 하나 이내로 왕복됩니다.
