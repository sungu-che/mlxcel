# 오디오 입력 전처리

이 문서는 모델 입력 오디오를 위한 공유 요청 경계를 설명합니다. 별도의 `/v1/audio`
speech-to-text 및 text-to-speech API는 다루지 않으며, 해당 영역은
[Audio API](audio-api.md)를 참고하십시오.

## 파형 및 정책 계약

CLI 파일과 서버가 해석한 오디오 바이트는 하나의 컨테이너 경계를 사용합니다. WAV, MP3,
FLAC은 파일 이름이나 요청 레이블이 아니라 매직 바이트로 식별합니다. WAV는 PCM16 또는
float32를 위한 엄격한 트리 내 RIFF 디코더를 유지하며, MP3와 FLAC은 제한된 Symphonia
경로를 사용합니다. 이 경계는 비어 있거나, 형식이 잘못되었거나, 유한하지 않거나,
정책을 벗어난 입력을 거부하고, 채널을 평균하여 mono로 만들며, 샘플을 `[-1, 1]` 범위의
유한한 `f32` 값으로 클램프합니다. 클립 순서와 1부터 시작하는 플레이스홀더 서수는 보존됩니다.
소유된 결과에는 소스 샘플링 레이트, 채널 수, 샘플 수, 길이, 인코딩된 바이트, 정규화된 샘플 수,
feature 프레임 추정치, 유효 오디오 token 수가 각각 별도로 기록됩니다.

로드된 체크포인트 정책은 `config.json`과 `processor_config.json` 또는
`preprocessor_config.json`에서 도출됩니다. 인식되는 키는 다음과 같습니다.

- 샘플링 레이트: `audio_processor.sampling_rate`, `audio_processor.sample_rate`,
  `feature_extractor.sampling_rate`, 또는 최상위 `sampling_rate`;
- 샘플/길이 제한: `audio_processor.max_audio_samples`,
  `feature_extractor.max_length`, 최상위 `max_audio_samples`, 그리고
  이에 대응하는 `max_audio_duration` 또는 `max_duration_seconds` 키;
- 프레임 형태: `audio_processor.frame_length_samples`,
  `audio_processor.frame_hop_samples`, `feature_extractor.frame_length`,
  `feature_extractor.hop_length`;
- 채널/dtype: `audio_processor.channels`, `audio_processor.dtype`,
  `feature_extractor.channels`, `feature_extractor.dtype`;
- mlxcel 한도: `mlxcel_audio_limits.max_encoded_bytes_per_clip`,
  `max_clips`, `maximum_source_sample_rate`,
  `max_encoded_bytes_per_request`, `max_source_samples_per_request`,
  `max_normalized_samples_per_request`,
  `max_source_duration_micros_per_request`, `max_frames_per_request`,
  `max_waveform_result_bytes_per_request`,
  `max_waveform_working_bytes_per_request`,
  `max_prepared_result_bytes_per_request`, 그리고
  `mlxcel_audio_limits.resampling_algorithm`.

알려진 키의 JSON 타입이 잘못되었거나, 음수 정수이거나, 한도가 0이거나, 지원되지 않는
프런트엔드 형태이거나, 고정된 패밀리 최댓값을 넘는 값이면 모델 시작이 실패합니다.
메타데이터가 없으면 불변의 레퍼런스 기본값을 사용합니다.

| 패밀리 | 고정된 기본값 |
| --- | --- |
| Phi-4 Multimodal | Transformers 리비전 `93f923e1a7727d1c4f446756212d9d3e8fcc5d81`; mono float32; 공개된 SpeechLib 정수비 리샘플러; 총 30분 / 총 정규화 샘플 28,800,000개 / 총 feature 프레임 179,998개; 최대 소스 레이트 192 kHz; 클립당 및 요청당 인코딩 256 MiB; 유지되는 파형 128 MiB; 파형 작업 세트 512 MiB; 준비된 결과 256 MiB; 클립 16개; 번호가 매겨진 플레이스홀더. |
| Gemma 3n | Transformers 레퍼런스 리비전 `181beb3ba4c47098ed8cbc97ee250d1d45ae0107`; 결정적 선형 리샘플링을 사용하는 16 kHz mono float32; 총 30초 / 총 정규화 샘플 480,000개 / 총 feature 프레임 2,997개; 최대 소스 레이트 192 kHz; 클립당 및 요청당 인코딩 64 MiB; 유지되는 파형 2 MiB; 파형 작업 세트 32 MiB; 준비된 결과 64 MiB; 클립 16개; 클립당 soft token 188개. |

Phi 정책은 공개된 프로세서의 특이한 정수비 동작을 의도적으로 보존합니다.
이는 일반적인 선형 리샘플러로 대체되지 않습니다.

## 한도 및 취소

서버의 획득 단계는 먼저 클립당 및 요청당 클립 16개와 인코딩된 오디오 500 MiB라는
절대 상한을 적용합니다. 이후의 각 소스는 남은 요청 예산에 대해 읽히므로, URL
`Content-Length`와 base64 문자 수 사전 점검이 집계 상한을 넘겨 할당하기 전에 거부할 수 있습니다.
모델 패밀리가 알려지면, 더 엄격한 로드된 인코딩 바이트, 길이, 샘플, 프레임, 클립, 유지되는 결과,
작업 메모리, 준비된 결과 한도가 feature 할당 전에 적용됩니다. 이 값들은 요청 집계이며,
클립당 값에 클립 한도를 곱한 것이 아닙니다. 컨테이너 헤더, 압축 디코드 증가량, Phi4MM
polyphase 필터/출력 작업은 디코드되거나 리샘플된 벡터가 정책을 넘어 커지기 전에 제한됩니다.
FLAC 메타데이터 블록은 디코더가 공격자가 제어하는 구조를 보기 전에 검증됩니다.

잘못된 오디오는 조용히 버려져 텍스트 전용 요청이 되는 것이 아니라 요청 전체를 실패시킵니다.
개별 소스를 해석하는 동안 이미지 획득은 계속 관대하게 동작하지만, 요청 준비 단계에서는
raw 페이로드를 만들지 못한 선언된 이미지를 거부합니다.

스케줄러 admission 이전에는 프로덕션 HTTP 취소가 Axum 핸들러 future의 수명을 따릅니다.
핸들러를 drop하면 획득 future, 부분 버퍼, 원격 응답 스트림이 drop됩니다. 명시적인 청크
체크포인트 프로브는 향후 XLA 패밀리 admission 경로를 위한 협력적 이음새이며, 현재는
실제 연결 해제 토큰이 아닙니다. MLX 스케줄러 admission 이후에는 요청의 실시간 취소
플래그를 파형 디코드/리샘플링과 Phi4MM/Gemma3n CPU feature 루프 내부에서 폴링합니다.
제한된 호스트 워커에도 큐, 디코드, 리샘플, feature 단계별로 구분된 취소 체크포인트가 있습니다.

인코딩된 메모리와 전체 호스트 메모리 예약은 큐 admission부터 처리, 제한된 결과 채널
인계까지 하나의 RAII 가드가 소유합니다. 이 가드는 결과가 소비될 때, 또는 제출, 취소,
panic, 연결 해제, 결과 크기 거부, 단계 종료로 인해 drop될 때 정확히 한 번 해제됩니다.
결과 채널은 자체적으로 제한된 깊이를 가지므로, 소비자가 멈추면 준비된 텐서가 누적되는
대신 feature 생성에 backpressure가 걸립니다.

오디오를 포함하는 OpenAI 호환 요청의 경우, 비공개의 번호가 매겨진 센티널이 메시지와
템플릿 렌더링 전반에 걸쳐 텍스트/이미지/오디오 파트의 정확한 위치를 보존합니다.
Phi4MM은 이를 번호가 매겨진 패밀리 플레이스홀더로 구체화하고, Gemma3n은 제자리 확장 전에
이를 이미지/오디오 id로 다시 토큰화합니다. 누락되거나, 중복되거나, 순서가 바뀌거나,
사용자가 주입한 센티널은 fail closed 처리됩니다. 오디오가 없는 요청은 기존의 텍스트/이미지
평탄화 및 토큰화 동작을 그대로 유지합니다.

## 메트릭 및 XLA 상태

실시간 MLX 서버 경로는 요청당 정확히 한 번 공유 파형 결과와 최종 확장된 prefill을 기록합니다.
`/metrics`는 누적 소스 초, 소스 샘플, 정규화된 샘플, feature 프레임, 내부 오디오 token,
유효한 준비된 prefill 위치, 전처리 지연 시간, 취소 횟수, 현재 인코딩된 바이트, 그리고
큐 대기, 처리 중, 결과 인계 상태 전반에 걸쳐 유지되는 전체 호스트 바이트를 내보냅니다.
`mlxcel_audio_preprocess_rejections_total`은 상호 배타적인 `reason` 레이블을 사용합니다:
`queue_full`, `memory_limit`, `worker_unavailable`, `overflow`,
`waveform`, `feature`, `feature_panic`, `context_limit`. 이러한 내부
카운터는 공개 OpenAI usage-token 집계를 변경하지 않습니다.

일회성 MLX CLI는 패밀리 준비 요약을 보고할 때 동일한 `AudioWaveformBatch` 메타데이터를
사용하며, 명령이 종료된 후에는 프로세스 전역 Prometheus 누산기를 유지하지 않습니다.

OpenXLA/XLA 백엔드는 여전히 오디오 기능을 사용할 수 없다고 보고하며, 텍스트 폴백 이전에
오디오 및 비디오 선언을 거부합니다. 호스트 파형, 큐, 취소, 관측성 기반은 마련되어 있지만,
Phi-4 Multimodal 또는 Gemma 3n XLA feature 생성기는 아직 구현되거나 연결되지 않았습니다.
