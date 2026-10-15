# Nemotron VoiceChat

NemotronLabs VoiceChat(`model_type: "nemotron_voicechat"`, `architectures: ["NemotronVoiceChatForConditionalGeneration"]`)은 전이중(full-duplex) 음성-음성 변환 모델입니다. 80 ms 프레임 단위로 16 kHz 오디오를 입력받고, 매 프레임마다 어시스턴트 텍스트 토큰, function 채널 토큰, 사용자 전사 업데이트, 그리고 22.05 kHz 어시스턴트 음성 80 ms를 하나의 공유 타임라인 위에서 출력합니다. 무음과 겹치는 발화도 타임라인의 일부이며, 음성 활동 감지(voice-activity) 게이트는 없습니다.

체크포인트: `mlx-community/NemotronLabs-VoiceChat-11B-bf16`, `-8bit`, `-4bit`. 양자화(quantization)된 변환본은 LLM(임베딩 테이블, 두 헤드, Nemotron-H mixer 프로젝션)만 양자화하며, 음성 인코더, RNNT 브랜치, EAR-TTS, 코덱은 bf16으로 유지됩니다.

## 아키텍처

| 네트워크 | 가중치 | 역할 |
|---|---|---|
| Log-mel 프런트엔드 | 없음 | Preemphasis 0.97, 25 ms / 10 ms 대칭 Hann STFT(n_fft 512, reflect 패딩), 128개 Slaney mel bin, `ln(x + 2^-24)`, 정규화 없음. |
| FastConformer 인코더 | `stt_model.perception.encoder` | 인과적(causal) depthwise-striding 서브샘플링(8배, 10 ms mel 프레임이 80 ms 프레임이 됨), 왼쪽 컨텍스트 70프레임으로 제한된 상대 위치 어텐션을 갖는 24개 conformer 블록, 인과적 depthwise 컨볼루션, LayerNorm conv norm. |
| Perception 프로젝션 | `stt_model.perception.proj` | 1024에서 4480으로, LLM의 오디오 채널입니다. |
| RNNT 브랜치 | `stt_model.rnnt_decoder`, `stt_model.rnnt_joint` | 2층 LSTM prediction net과 joint 네트워크이며, SentencePiece `rnnt_vocabulary`에 대한 greedy 디코딩으로 사용자 전사를 생성합니다. |
| Duplex LLM | `stt_model.embed_tokens`, `stt_model.llm`, `stt_model.lm_head`, `stt_model.function_head` | 위치마다 하나의 융합 임베딩 `E(prev_text) + audio + 2 * E(prev_function)`에 대해 실행되는 Nemotron-H(56개 레이어, Mamba2 / attention / MLP 패턴)이며, 텍스트 헤드와 function 헤드를 가지며 둘 다 greedy입니다. |
| EAR-TTS | `tts_model.tts_model`, `tts_model.audio_prompt_latents.Aria` | 문자 인식(character-aware) 서브워드 인코더와 게이트 융합을 통해 현재 텍스트 토큰을 조건으로 하는 28층 Gemma-3 스타일 백본(sliding 및 global 어텐션, 임베딩 스케일링 없음)이며, classifier-free guidance(batch 2)를 사용합니다. 가우시안 혼합(mixture-of-Gaussians) 헤드가 8회의 masked 반복으로 31개의 residual codebook을 정제합니다. |
| 코덱 | `tts_model.audio_codec` | 확률적 residual VQ(1024개씩 31개 codebook)를 사용하는 ConvNeXt 인코더/디코더이며, n_fft 16 / hop 4의 iSTFT 출력을 가집니다. 코드 프레임 하나는 1764 샘플(22.05 kHz에서 80 ms)입니다. |

mlxcel 모듈은 `src/audio/nemotron_mel.rs`, `src/audio/fastconformer/`, `src/audio/rnnt/`, `src/audio/nemotron_codec/`, `src/models/gemma3_backbone.rs`, `src/models/nemotron_voicechat/`(config, LLM glue, TTS, 모델 로더, 오프라인 세션)입니다. 이 포팅은 mlx-vlm / mlx-audio 참조 구현을 따릅니다.

## 오프라인 타임라인

1. 입력 WAV를 16 kHz 모노로 리샘플링하고 `--extra-decoding-seconds`만큼의 무음을 뒤에 덧붙입니다.
2. Log-mel과 FastConformer가 80 ms 프레임마다 4480 폭의 오디오 임베딩 하나를 생성합니다.
3. 비어 있지 않은 시스템 프롬프트는 `[BOS] + tokens + [EOS]`가 되며, 이 토큰 임베딩이 오디오 채널의 첫 타임라인 위치들을 차지합니다.
4. EAR-TTS를 `Aria` 프롬프트 latent와 무음의 코덱 인코딩으로 한 번 워밍업합니다.
5. 모든 위치에서 LLM이 텍스트 토큰과 function 토큰을 출력합니다(프롬프트 접두부 구간에서는 pad). 첫 위치 이후의 모든 위치는 EAR-TTS를 31개 코드로 이루어진 한 프레임만큼 진행시킵니다. EOS 텍스트 토큰은 피드백되는 코드를 무음으로 초기화합니다.
6. 프롬프트 접두부를 버리고, 제어 코드를 codebook별 무음 코드로 대체한 뒤, 코덱이 전체 답변을 디코딩합니다. RNNT 브랜치는 사용자 오디오를 전사합니다.

언어 모델은 위치 간에 지속되는 Nemotron-H 캐시를 유지합니다. 참조 구현은 오프라인 경로에서 기본적으로 매 위치마다 전체 히스토리를 재계산하는데, 두 모드는 참조 구현에서 동일한 토큰과 코드를 만들며, 캐시 모드는 위치당 작업량에 상한을 둡니다.

## CLI

```
mlxcel generate -m models/NemotronLabs-VoiceChat-11B-4bit \
  --audio question.wav --output-audio response.wav \
  -p "Be concise and answer in one sentence."
```

이 명령은 `[user] <transcript>`, 어시스턴트 텍스트, 그리고 function 채널이 출력을 생성한 경우 `[function] <text>`를 출력하고, `response.wav`(22050 Hz 모노 PCM16)를 기록합니다. `-p`는 선택 사항이며(생략하면 시스템 프롬프트 없음), `-n`은 출력 길이가 입력 타임라인과 같으므로 무시되고, `--extra-decoding-seconds`(기본값 3)는 덧붙이는 무음 길이를 설정하며, `--seed`(기본값 0)는 EAR-TTS 샘플링 노이즈의 시드를 정하므로 같은 시드로 두 번 실행하면 바이트 단위로 동일합니다. `--image`와 `--video`는 거부됩니다.

## 온라인 세션

`NemotronVoiceChatModel::create_streaming_session(StreamingOptions)`는 임의 크기의 PCM 청크를 1280 샘플(80 ms) 프레임으로 변환하고, 지속되는 상태로부터 프레임마다 모든 네트워크를 정확히 한 프레임씩 진행시키는 `VoiceChatStreamingSession`을 반환합니다.

| 단계 | 지속 상태 |
|---|---|
| Log-mel | centered STFT가 여전히 필요로 하는 샘플 꼬리(프레임은 입력 끝보다 1280 샘플 뒤처지면 출력됨) |
| FastConformer | 레이어별 마지막 70개 어텐션 입력과 마지막 8개 conv 입력, 그리고 인과적 서브샘플링 스택을 위한 16프레임 mel 캐시 |
| RNNT | 마지막으로 출력한 토큰과 LSTM 상태 |
| Nemotron-H | Mamba2 상태와 어텐션 KV cache |
| EAR-TTS | 백본 KV cache(sliding 레이어는 rotating) |
| 코덱 | ConvNeXt 블록별 인과적 overlap과 iSTFT overlap |

```rust
let mut session = model.create_streaming_session(StreamingOptions {
    system_prompt: Some("Be concise and answer in one sentence.".into()),
    seed: 0,
    ..StreamingOptions::default()
})?;
for chunk in pcm_chunks {
    for event in session.push_audio(chunk, 16_000)? { handle(event) }
}
for event in session.flush(true)? { handle(event) }
```

이벤트는 오디오 프레임 인덱스를 담고 있습니다(시스템 프롬프트 접두부는 타임라인을 진행시키지만 아무것도 출력하지 않습니다).

| 이벤트 | 필드 |
|---|---|
| `AssistantTextDelta` | `frame_index`, `token_id`, `delta`, 누적 `text` |
| `FunctionDelta` | 동일한 형태, function 채널 |
| `UserTranscriptDelta` | `frame_index`, `delta`, 누적 `text` |
| `Audio` | `frame_index`, 22.05 kHz의 1764개 `samples`, 해당 프레임의 31개 `audio_codes` |
| `Done` / `Cancelled` | `frame_index` |

텍스트 delta는 pad, 무음, BOS, EOS id를 건너뜁니다. `delta`는 누적 디코딩이 이전 결과를 확장하는 경우 새 접미부이고, 그렇지 않으면 단일 토큰 디코딩 결과이며, 누적 `text`가 기준입니다. `push_audio`는 16 kHz가 아닌 샘플 레이트, 유한하지 않은(non-finite) 샘플, 닫힌 세션을 거부합니다. `flush(pad_partial)`는 선택적으로 부분 프레임을 0으로 패딩하여 실행한 다음 `Done`으로 닫고, `cancel()`은 대기 중인 오디오를 버리고 `Cancelled`로 닫습니다. `max_streaming_seconds`는 세션 길이에 상한을 둡니다(`ContextLimit`). 세션은 하나의 스레드에서 구동되며(MLX 평가는 스레드 종속적), 모든 상태를 스스로 소유하므로 한 모델에서 연달아 만든 세션끼리는 서로 영향을 주지 않습니다.

`use_language_cache: false`는 매 프레임마다 전체 융합 입력 히스토리에 대해 언어 모델을 재계산하고, `use_perception_cache: false`는 슬라이딩 샘플 윈도우에 대해 log-mel과 인코더를 재계산합니다. 둘 다 참조 구현과 마찬가지로 진단용입니다.

`profile: true`를 사용하면 세션은 프레임별 벽시계 단계 타이밍을 기록하며, 각 값은 해당 단계의 출력을 강제로 평가한 뒤 측정됩니다. 단계는 `perception`, `rnnt`, `language`, `tts`, `codec`, `total`입니다. `profile().summary(drop_first)`는 단계별 mean / p50 / p95 / max를 제공하고, `processing_frames_per_second = 1000 / mean total`, `realtime_factor = mean total / 80 ms`입니다(1.0을 넘으면 실시간보다 느림).

CLI는 `--stream`으로 세션을 구동합니다.

```
mlxcel generate -m models/NemotronLabs-VoiceChat-11B-4bit --audio question.wav \
  --output-audio response_stream.wav -p "Be concise and answer in one sentence." --stream --profile
```

입력과 `--extra-decoding-seconds`만큼의 무음을 80 ms 프레임으로 밀어 넣고, 프레임이 생성하는 대로 어시스턴트 텍스트를 출력하며, 이어 붙인 오디오 이벤트를 기록하고, `--profile`을 사용하면 5개의 콜드 프레임을 제외한 뒤 요약 JSON을 출력합니다. `--max-streaming-seconds`는 `max_streaming_seconds`에 대응합니다.

스트리밍 인코더는 처음 몇 프레임에서는 오프라인 인코더와 같다가 이후 약간 벗어나는데, 참조 구현의 cache-aware 서브샘플링 윈도우가 오프라인의 전체 발화 컨볼루션과 다르기 때문입니다. 이 포팅은 참조 구현의 규칙을 유지하며, 스트리밍된 프레임은 참조 구현의 스트리밍 경로와 float 정밀도 수준에서 일치합니다.

## 실시간 WebSocket (`/v1/realtime`)

`mlxcel-server -m models/NemotronLabs-VoiceChat-11B-4bit --port 8080`(또는 `mlxcel serve`)은 전용 실시간 엔진 스레드에 체크포인트를 로드하고, `/v1/realtime`의 WebSocket으로 온라인 세션을 서빙합니다. 이 체크포인트에서는 채팅 워커, 배치 스케줄러, 텍스트 워밍업이 시작되지 않으며, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`는 `/v1/realtime`을 알려주는 메시지와 함께 `501 Not Implemented`로 응답합니다. `--api-key`는 업그레이드 요청에 적용되며(`Authorization: Bearer <key>` 전송), `--api-prefix`는 다른 모든 라우트와 마찬가지로 경로에 접두사를 붙입니다. `--cors-origins`와 `--allowed-origins`도 `Origin` 헤더로 업그레이드를 제어합니다. 정책이 허용하지 않는 origin의 브라우저 페이지는 `403 Forbidden`을 받고, `Origin`을 보내지 않는 클라이언트(비브라우저 클라이언트)는 수락됩니다. 프로토콜은 mlx-vlm 참조 서버(`mlx_vlm/server/realtime.py`)를 따릅니다.

모든 MLX 작업은 엔진 스레드에서 실행됩니다. 소켓 태스크는 채널을 통해 `open`, `push`, `flush`, `cancel`, `close` 명령을 보내고 one-shot 응답을 기다리며, 엔진은 응답하기 전에 각 이벤트(base64 오디오 포함)를 직렬화합니다. MLX 버퍼 캐시는 프레임마다가 아니라 세션이 닫힐 때 비워집니다.

### 클라이언트 메시지 (JSON 텍스트 프레임)

| `type` | 필드 | 동작 |
|---|---|---|
| `session.update` | `session.system_prompt`(문자열, 없으면 체크포인트 기본값 사용, 기본값은 빈 문자열), `session.seed`(음이 아닌 정수, 기본값 0), `session.max_streaming_seconds`(숫자, 선택), `session.model`(선택, 아래 참조) | 스트리밍 세션을 열고(EAR-TTS를 워밍업하고 시스템 프롬프트를 prefill), `session.updated`로 응답합니다. 이미 구성된 연결에서 두 번째 `session.update`는 `invalid_request` 오류입니다. |
| `input_audio_buffer.append` | `audio`(base64 리틀 엔디언 PCM16 모노), `sample_rate`(기본값 16000, 다른 레이트는 세션이 거부) | `/ 32768`로 f32로 디코딩하여 1280 샘플 조각 단위로 밀어 넣으며, 조각마다 엔진 호출 한 번입니다. 각 조각의 이벤트는 다음 조각이 실행되기 전에 전송됩니다. |
| `input_audio_buffer.commit` | `pad_partial`(기본값 true) | `input_audio_buffer.committed`로 응답합니다. 세션이 flush되고(`pad_partial`이면 부분 프레임을 0으로 패딩), 이벤트는 `response.done`으로 끝나며, 서버는 소켓을 닫습니다(코드 1000). |
| `session.cancel`, `response.cancel` | | `response.cancelled`를 내보내고 소켓을 닫습니다(코드 1000). |
| `session.ping` | | `session.pong`. |

`session.update` 이전에는 `session.update`와 `session.ping` 이외의 모든 메시지에 `invalid_request` 오류 "send session.update before audio"로 응답합니다. 그 이후에는 알 수 없는 `type`이 `invalid_request` 오류입니다. JSON 객체가 아닌 프레임은 `invalid_request` 오류이며 루프는 계속됩니다. 서버는 하나의 모델을 실행하므로 `session.model`은 생략할 수 있고, 어떤 값이든 서빙 중인 체크포인트를 열며 `session.updated`는 서빙 중인 id(`--alias`, 없으면 모델 디렉터리 이름)를 보고합니다.

### 서버 이벤트

모든 이벤트는 `event_id`(`event_` 뒤에 16자리 16진수)를 가집니다.

| `type` | 필드 |
|---|---|
| `session.created` | `session.id`(`sess_` 뒤에 16자리 16진수), `session.state: "configuring"`, `session.input_audio_format {type: "pcm16", sample_rate: 16000}`, `session.output_audio_format {type: "pcm16", sample_rate: 22050}` |
| `session.updated` | `session.id`, `session.state: "ready"`, `session.model`, `session.frame_samples: 1280`, 두 포맷 모두 |
| `conversation.item.input_audio_transcription.delta` | `frame_index`, `delta`, 누적 `transcript` |
| `response.text.delta` | `frame_index`, `token_id`, `delta`, 누적 `text` |
| `response.function.delta` | `frame_index`, `token_id`, `delta`, 누적 `text` |
| `response.audio.delta` | `frame_index`, `delta`(`round(clip(x, -1, 1) * 32767)`의 base64 PCM16, 1764 샘플), `format: "pcm16"`, `sample_rate: 22050`, `channels: 1`, `audio_codes`(해당 프레임의 31개 코덱 코드) |
| `input_audio_buffer.committed` | |
| `response.done` / `response.cancelled` | `frame_index` |
| `session.pong` | |
| `error` | `error.code`, `error.message` |

오류 코드:

| `error.code` | 발생 시점 |
|---|---|
| `invalid_request` | 알 수 없는 `type`, 객체가 아닌 프레임, `session.update` 이전의 오디오, 두 번째 `session.update`. |
| `server_busy` | 다른 연결이 세션을 점유 중입니다. 소켓은 코드 1013(나중에 다시 시도)으로 닫힙니다. |
| `session_initialization_failed` | `session.update`가 실패했습니다(음수 시드나 양수가 아닌 `max_streaming_seconds` 같은 잘못된 필드). 연결은 열린 채로 유지되며 재시도할 수 있습니다. |
| `inference_error` | `audio`가 유효한 base64가 아니거나 바이트 수가 홀수인 append, 정수가 아닌 `sample_rate`, 16 kHz가 아닌 레이트, `max_streaming_seconds`를 넘은 스트림, 또는 모델 실패. 루프는 계속되며, 실패한 append는 실패한 조각에서 중단됩니다. |

### 한 번에 하나의 세션

서버 제한이 단일 세션 슬롯을 보호합니다. WebSocket 메시지 또는 프레임은 최대 1 MiB(base64 PCM16 약 24초 분량, 클라이언트는 80 ms 단위로 append를 전송)이고, 세션은 최대 600초 분량의 오디오까지 지속되며(`session.max_streaming_seconds`로 낮출 수 있고 더 큰 값은 제한값으로 조정되며, 제한에 도달하면 `inference_error`로 스트림이 끝남), 연결은 30초 이내에 `session.update`를 보내야 하고, 구성된 세션에서 120초 동안 클라이언트 메시지가 없으면(WebSocket ping은 포함되지 않음) `invalid_request` 오류와 종료 코드 1008을 받습니다.

엔진은 하나의 연결만 허용합니다. 세션이 활성인 동안 들어온 두 번째 연결은 `server_busy`와 종료 코드 1013을 받습니다. 예약은 세션이 끝날 때 해제됩니다. `commit` 또는 `cancel` 이후(서버의 close 프레임 이전)와, 클라이언트가 연결을 끊은 경우이며, 이 경우 서버는 먼저 flush되지 않은 세션을 취소합니다. 그러면 다음 연결은 정상적으로 허용됩니다.

### 예제

`examples/voicechat_file_client.rs`는 소켓을 통해 WAV 파일을 구동하고(16 kHz 리샘플링, `--extra-seconds`만큼의 무음(기본값 3), 이후 `commit`), 전사와 답변을 출력하며, 수신한 오디오를 22.05 kHz PCM16 WAV로 기록합니다.

```
cargo run --release --example voicechat_file_client -- \
  ws://127.0.0.1:8080/v1/realtime question.wav response_ws.wav \
  --system-prompt "Be concise and answer in one sentence." --seed 0
```

4-bit 체크포인트를 사용한 실제 세션의 발췌입니다(파일 클라이언트가 "What is the capital of France?"와 3초의 무음을 위 시스템 프롬프트와 함께 구동했으며, 클라이언트 메시지는 `>`로 시작하고 `event_id`와 오디오 페이로드는 생략함). `websocat ws://127.0.0.1:8080/v1/realtime` 세션도 동일한 프레임을 주고받습니다.

```
{"type":"session.created","session":{"id":"sess_…","state":"configuring","input_audio_format":{"type":"pcm16","sample_rate":16000},"output_audio_format":{"type":"pcm16","sample_rate":22050}}}
> {"type":"session.update","session":{"system_prompt":"Be concise and answer in one sentence.","seed":0}}
{"type":"session.updated","session":{"id":"sess_…","state":"ready","model":"nemotronlabs-voicechat-11b-4bit","frame_samples":1280,"input_audio_format":{"type":"pcm16","sample_rate":16000},"output_audio_format":{"type":"pcm16","sample_rate":22050}}}
> {"type":"input_audio_buffer.append","audio":"…","sample_rate":16000}
{"type":"response.audio.delta","frame_index":0,"delta":"…","format":"pcm16","sample_rate":22050,"channels":1,"audio_codes":[…31 ints…]}
{"type":"conversation.item.input_audio_transcription.delta","frame_index":5,"delta":"What","transcript":"What"}
{"type":"conversation.item.input_audio_transcription.delta","frame_index":19,"delta":" France","transcript":"What is the capital of France"}
{"type":"response.text.delta","frame_index":24,"token_id":…,"delta":"The","text":"The"}
{"type":"response.text.delta","frame_index":30,"token_id":…,"delta":".","text":"The capital of France is Paris."}
> {"type":"input_audio_buffer.commit"}
{"type":"input_audio_buffer.committed"}
{"type":"response.done","frame_index":61}
```

전체 실행은 61개의 `response.audio.delta` 이벤트(프레임 0~60, 각 1764 샘플), 6개의 전사 delta, 7개의 텍스트 delta, 그리고 `response.done`을 생성했으며, 이후 서버가 코드 1000으로 소켓을 닫았습니다. 세션 도중 열린 두 번째 연결은 `server_busy`와 종료 코드 1013을 받았고, 첫 번째 연결이 닫힌 뒤 열린 연결은 정상적으로 구성되었습니다.

이 세션은 같은 입력, 시스템 프롬프트, 시드에 대해 `mlxcel generate --stream`과 동일한 계산입니다. 전사, 답변 텍스트, 프레임 수, 오디오가 일치합니다. 두 출력이 바이트 단위로 동일한 파일이 아닌 이유는 양자화 방식이 다르기 때문입니다. CLI의 WAV 기록기는 32768로 스케일링하는 반면 전송 포맷은 참조 구현처럼 32767로 스케일링하므로, 각 샘플이 양자화 단계 하나만큼 다를 수 있습니다.

`examples/voicechat_microphone.rs`는 라이브 대화를 수행합니다. 기본(또는 `--input-device`) 마이크를 캡처하여 16 kHz로 다운믹스 및 리샘플링하고, 80 ms마다 append를 보내며, `response.audio.delta` 스트림을 링 버퍼를 통해 기본(또는 `--output-device`) 스피커로 재생하고, 전사와 답변을 도착하는 대로 출력하며, Ctrl-C에서 commit합니다. `--list-devices`는 오디오 장치를 나열합니다. 이 예제에만 cpal 오디오 장치 크레이트를 컴파일하는 `voicechat-mic` 기능이 필요합니다.

```
cargo run --release --features voicechat-mic --example voicechat_microphone -- \
  ws://127.0.0.1:8080/v1/realtime --system-prompt "Be concise and answer in one sentence."
```

음향 에코 제거(acoustic echo cancellation)가 없고 모델은 말하는 동안에도 계속 듣기 때문에 헤드폰을 사용하십시오. 열린 스피커로는 자신의 답변을 듣게 됩니다.

macOS에서는 터미널에 마이크 접근 권한이 필요합니다(시스템 설정 > 개인정보 보호 및 보안 > 마이크). 권한이 없으면 입력 스트림을 여는 데 실패하고, 예제는 장치 이름과 이 안내를 보고합니다.

## 검증

`mlx-community/NemotronLabs-VoiceChat-11B-4bit`를 mlx-vlm 참조 구현과 비교하여, 합성한 "What is the capital of France?" 질문과 시스템 프롬프트 "Be concise and answer in one sentence."로 검증했습니다(`tests/nemotron_voicechat_*_real.rs`의 환경 변수로 제어되는 테스트이며 `MLXCEL_VOICECHAT_MODEL`과 `MLXCEL_VOICECHAT_REF`를 설정하여 실행).

- 전사는 "What is the capital of France?"이고 답변은 "The capital of France is Paris."로, 참조 구현과 같습니다.
- 텍스트 및 function id는 모든 타임라인 위치에서 참조 구현과 정확히 일치하며, `--seed 0`에서는 모든 프레임의 31개 EAR-TTS 코드도 참조 구현과 일치하므로(이 포팅은 동일한 MLX 전역 RNG 시퀀스를 추출함) 디코딩된 답변 오디오는 참조 파형과 일치합니다.
- 생성된 답변 오디오를 같은 체크포인트로 다시 전사하면 "The capital of France is Paris"가 나옵니다.
- FastConformer 출력은 참조 구현과 1e-6(최대 절대 오차) 이내로 일치하며, 코덱은 참조 코드를 정확히 재현하고 디코딩 결과는 1e-7 이내입니다.

코덱은 손실이 있는(lossy) 신경망 코덱입니다. 순수 톤(pure tone)은 참조 구현에서도 약 0.7 dB의 파형 SNR로 왕복(round-trip)되므로, 왕복 일치 여부는 SNR 임계값이 아니라 참조 재구성과 비교하여 확인합니다.

## 성능

`mlxcel generate ... --stream --profile`로 측정하십시오(요약의 `realtime_factor`는 프레임당 평균 처리 시간을 80 ms 프레임으로 나눈 값으로 정의되며, 1.0을 넘으면 실시간보다 느리다는 뜻입니다). 이 표는 그 외에는 유휴 상태인 검증 머신에서 나온 것입니다. 3초의 추가 디코딩을 포함한 "What is the capital of France?" 프롬프트(56프레임, 처음 5개는 콜드로 제외)로 워밍업 실행 한 번 후 체크포인트당 세 번 측정했습니다. 세 번의 실행은 0.003 이내로 일치했습니다.

| 체크포인트 | 실시간 계수 | 프레임 시간 p50 / p95 | 머신 |
|---|---|---|---|
| 4-bit | 0.87 | 69.0 ms / 70.4 ms | Apple M1 Ultra (128 GB), Metal |
| 8-bit | 0.94 | 75.3 ms / 76.4 ms | Apple M1 Ultra (128 GB), Metal |

두 체크포인트 모두 이 머신에서 실시간을 따라갑니다. 4-bit 체크포인트의 단계별 p50은 perception 17.0 ms, RNNT 0.4 ms, language 16.0 ms, TTS 28.1 ms, codec 7.3 ms이며, 8-bit 체크포인트는 language 단계(22.2 ms)만 다릅니다. 12초의 추가 디코딩(168프레임)을 사용한 4-bit 실행은 인코더의 70프레임 어텐션 윈도우가 채워진 이후에도 같은 속도를 유지합니다. 71번째 이후 프레임은 평균 69.8 ms입니다(실시간 계수 0.87).

인코더, EAR-TTS 백본, MoG 헤드는 참조 구현과 마찬가지로 체크포인트의 bf16 가중치에 대해 f32 활성값으로 실행됩니다. MLX는 매 프레임마다 모든 matmul과 컨볼루션 내부에서 이 가중치를 f32로 캐스팅하므로, 로더가 이를 대신 f32로 보관합니다. 변환은 정확하고 출력은 비트 단위로 동일하며, 대가로 상주 메모리가 약 2.7 GB 늘어납니다(인코더 1.2 GB, TTS 백본 1.2 GB, MoG 헤드 0.3 GB). Norm 가중치와 bf16 서브워드 경로는 저장된 그대로 유지됩니다.

요약은 또한 프레임당 호스트 왕복 횟수인 `host_syncs`(약 8)를 보고합니다. 단계 내부의 원인 분석을 위해서는 `MLXCEL_VOICECHAT_PROFILE_STAGES=1`을 설정하십시오. 요약에 `sub_stages` 맵(예: `perception.layers`, `tts.backbone`, `tts.mog_head`, `codec.decoder_istft`)이 추가됩니다. 이 타이머는 모든 경계에서 평가를 강제하므로 그 실행의 프레임 합계는 더 느리며 실시간 계수가 아닙니다. `MLXCEL_VOICECHAT_PROFILE_FRAMES=<path>`는 긴 입력의 정상 상태(steady-state) 분석을 위해 모든 프레임의 타이밍(콜드 프레임 포함)을 JSON으로 기록합니다.

## 제한 사항

- 체크포인트에 내장된 `Aria` 음성만 지원하며, `Aria`가 아닌 `speaker`는 로드 시 거부됩니다.
- 배치 크기 1.
- 오프라인 턴은 입력 20분(그리고 최대 600초의 `--extra-decoding-seconds`)으로 제한됩니다. 인코더가 참조 구현과 마찬가지로 전체 발화에 대한 밀집(dense) 어텐션 마스크를 만들기 때문입니다.
- 채팅 인터페이스 없음: `--audio` 없는 대화형 `mlxcel generate`와 `mlxcel chat`은 오프라인 명령을 안내하며 이 체크포인트를 거부하고, `mlxcel serve`는 `/v1/realtime` WebSocket만 한 번에 하나의 세션으로 서빙합니다.
- 변환된 MLX safetensors 레이아웃만 지원하며, 원본 NeMo `.nemo` 체크포인트는 로드되지 않습니다.
