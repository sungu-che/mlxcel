# Nemotron-H 요청 메모리 admission

Metal 텍스트 전용 Nemotron-H 서빙은 각 요청을 시작하기 전에 보수적인 메모리 추정치를 확인합니다. 스케줄러는 이 패밀리를 활성 슬롯 하나로 제한하고 활성 할당을 샘플링하기 전에 선행 작업이 끝나기를 기다립니다. 이 가드에 의해 거부된 요청은 GPU forward 작업을 전혀 실행하지 않으며, 스트리밍 요청은 SSE 헤더가 확정되기 전에 HTTP 503을 받습니다. `/health`는 계속 사용할 수 있으며, 이후에는 더 작은 요청이 실행될 수 있습니다.

## 예산과 오류

기본 예산은 디바이스의 권장 working-set 크기, **실제로 설정된** wired-memory 상한, 현재 MLX allocator 상한 중에서 양의 유한한 값 중 최솟값입니다. MLX의 `get_wired_limit()` 쿼리는 설정된 wired 상한이 아니라 디바이스 권장값을 보고하며, 런타임은 후자를 별도로 기록합니다.

`MLXCEL_REQUEST_MEMORY_BUDGET=24G`는 해당 예산을 낮출 수는 있지만 높이거나 누락된 런타임 예산을 제공할 수는 없습니다. 크기 접미사는 기존 이진 단위 문법(예: `1.5G`)을 따르며, 명시적인 0, 빈 값, `none`, 음수, 유한하지 않은 값, 오버플로하는 값은 서버 시작을 실패시킵니다. 알 수 없는 차원, 알 수 없는 예산, 산술 오버플로는 요청 admission을 fail closed로 실패시킵니다.

```json
{
  "error": {
    "message": "request_memory_budget_exceeded: estimated_bytes=..., budget_bytes=...; shorten the prompt/output or lower retained cache usage",
    "type": "request_memory_budget_exceeded",
    "code": "request_memory_budget_exceeded"
  }
}
```

오류 메시지는 추정 바이트와 예산 바이트를 보고합니다(알 수 없는 추정치는 `18446744073709551615`를, 알 수 없는 예산은 `0`을 사용합니다). 기존의 컨텍스트 길이 검사가 우선하며 HTTP 400으로 유지됩니다. 메모리 admission 로그에는 프롬프트/출력 길이, 가장 큰 forward 구간, 상주 할당, 구성 요소별 내역이 포함됩니다.

## 추정되는 것

- 체크포인트별 크기 표가 아니라, 로드되고 정규화된 모델 차원과 임베딩 출력 dtype.
- 모델 가중치와 유지되는 프롬프트 스냅샷을 포함한 현재 MLX **활성** 메모리. 공유 Metal Nemotron-H 생성자는 `--no-warmup`을 사용하는 경우를 포함하여 서빙 전에 sanitize된 가중치(fused adapter 그래프 포함)를 제자리에서 구체화하므로, lazy-weight 그래프가 상주량 0으로 위장할 수 없습니다. 회수 가능한 allocator free-cache는 영구 상주량으로 청구되지도, 활성 메모리에서 차감되지도 않습니다.
- 프롬프트와 계획된 출력에 대한 전체 cold attention KV로, 할당 단위로 반올림되며 증가 복사본과 스냅샷을 위한 여유가 있습니다. 또한 float32 recurrent 상태와 convolution 상태. 컨텍스트 shifting이 비활성화된 경우 컨텍스트 정지가 출력 증가를 제한합니다. Shifting이 활성화된 경우 Nemotron-H의 모델 소유 캐시에 대해 지원되지 않는 trim 크레딧은 적용되지 않습니다.
- 모델의 chunking opt-out과 history 경계 분할을 고려하여, 실제 `PrefillPlan`에서 가장 큰 패딩된 forward로, residual, MLP/MoE 작업, logits, 보수적인 float32 attention-score 폴백을 위한 것.
- Float32 SSD 선형 버퍼와 제한된 내부 청크 scratch. SSD 내부 청크는 서버 prefill 청크와 독립적입니다.

보호되는 요청은 현재 프롬프트 스냅샷을 복원하거나 기증하지 않고 cold로 prefill하며 백그라운드 prompt-cache warmup을 건너뜁니다. 그렇지 않으면 이전 요청이 아직 활성인 동안 스냅샷 복원이 요청 작업을 수행할 수 있기 때문입니다. 유지되는 스냅샷은 상주 기준선에 계속 계산됩니다. 이는 캐시 적중 지연 시간보다 안전성을 의도적으로 우선하며, 낙관적인 캐시 재사용이나 fused 커널이 있었다면 들어맞았을 요청을 거부할 수 있습니다. 오버라이드가 디바이스 상한을 우회하기를 기대하지 말고, 유지되는 프롬프트 캐시를 비우거나 줄이거나 프롬프트/출력 한도를 줄이십시오.

## 한계

이것은 범용 OOM 보장이나 디바이스 전체 예약이 아닙니다. 다른 프로세스, 드라이버 할당, 단편화, 예기치 않은 커널 워크스페이스가 admission 이후에 메모리를 소비할 수 있습니다. CUDA/ROCm/CPU, 다른 모델 패밀리, 멀티모달 경로, speculative decoding, 분산 handoff, 다중 슬롯 예약은 이 정책의 범위 밖입니다. 스케줄러 기반 서빙과 레거시 `--no-batch` 서빙 모두 이 가드를 사용합니다. 그 밖의 지원되지 않는 경로는 기존 동작을 유지합니다. Metal Nemotron-H 런타임(비 fused) LoRA는 모델 생성 시 명확한 오류와 함께 실패합니다. 임의 rank의 float32 activation 워크스페이스가 모델링되지 않았기 때문입니다. 대신 fused adapter 체크포인트를 사용하십시오. 공유 Nemotron-H 생성자는 또한 가중치를 즉시 구체화하며, VLM/VoiceChat에 내장될 때 staged 런타임 adapter를 거부합니다. 해당 모델들은 이 요청 admission 정책을 적용받지 않습니다. 이 가드는 모델 수치를 바꾸지 않으며 할당 실패 후 요청을 재시도하지도 않습니다.
