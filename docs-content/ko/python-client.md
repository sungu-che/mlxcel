# Python 클라이언트

`mlxcel`은 Python에서 OpenAI 호환 서버(`mlxcel serve`)를 구동하는 순수 Python 클라이언트를 제공합니다. 로컬 서버 프로세스를 직접 실행하고 감독하거나(managed 모드), 이미 실행 중인 서버에 연결하며(connect 모드), 서비스 중인 모델 id를 자동으로 찾아내고, 전체 API 표면(tools, `response_format`, logprobs, 멀티모달)을 위해 원본 `openai` 클라이언트를 노출합니다.

이는 Python 통합의 Phase 1입니다. 기존 HTTP 서버 위에 전적으로 구축되어 있으며, 네이티브 확장이 없고 Rust 추론 코어도 변경하지 않습니다. 패키지는 [`python/`](../python)에 있으며 import 이름 `mlxcel`로 배포됩니다.

## 설치

```bash
pip install ./python          # from a repo checkout
pip install "./python[dev]"     # adds pytest, ruff, mypy
```

Python 3.9 이상이 필요합니다. 런타임 의존성은 `openai>=1.40`과 `httpx>=0.27`입니다. managed 모드에는 추가로 `mlxcel` 바이너리가 필요합니다. 클라이언트는 `binary=` 인자, `MLXCEL_BIN` 환경 변수, `PATH`의 `mlxcel` 순서로 바이너리를 찾습니다. 바이너리 빌드는 [설치](installation.md)를 참조하십시오.

## Managed 모드

모델을 전달하면 클라이언트가 `mlxcel serve`를 실행하고, 서버가 준비되었다고 보고할 때까지 기다리며, 종료 시 서버를 중지합니다. 모델은 리스너가 바인딩되기 전에 로드되고 워밍업되므로 가중치를 다운로드하는 첫 실행은 시간이 걸릴 수 있습니다. 클라이언트는 넉넉한 `startup_timeout`(기본 600초)으로 `/health`를 폴링하고, 자식 프로세스가 일찍 종료하는지 감시합니다.

```python
import mlxcel

with mlxcel.LLM("mlx-community/Qwen3-4B-4bit") as llm:
    text = llm.generate("def fib(n):", max_tokens=128, temperature=0.7)
    print(text)

    reply = llm.chat([{"role": "user", "content": "Hello"}], max_tokens=64)
    print(reply)

    print(llm.model)       # resolved model id, auto-discovered from /v1/models
    print(llm.models())    # ["<id>"]
    ids = llm.tokenize("hello world")
    print(llm.detokenize(ids))
```

컨텍스트 매니저(`with`)가 권장되는 형태이며, 종료를 보장합니다. 클라이언트는 `atexit` 정리와 최선 노력 방식의 finalizer도 등록하므로, 핸들이 누수되더라도 서버는 중지됩니다.

managed 모드에서 유용한 인자: `binary=`, `host=`/`port=`(TCP를 강제하고 해당 위치에 바인딩), `socket=`(명시적인 Unix 소켓 경로), `api_key=`, `ctx_size=`, `n_predict=`, `alias=`, `warmup=`, `extra_args=[...]`(`mlxcel serve`에 그대로 전달됨), `startup_timeout=`.

마지막 그룹(`ctx_size=`, `n_predict=`, `alias=`, `warmup=`, `extra_args=`)은 `**server_kwargs`를 통해 전달됩니다. 추가 키워드 인자는 클라이언트가 아니라 실행되는 `mlxcel serve` 프로세스를 구성합니다.

```python
with mlxcel.LLM(
    "mlx-community/Qwen3-4B-4bit",
    ctx_size=8192,
    warmup=False,
    extra_args=["--parallel", "1"],
) as llm:
    print(llm.generate("hello", max_tokens=32))
```

## Connect 모드

직접 시작한 서버와 통신하려면 `base_url=` 또는 `socket=`을 전달하십시오(모델은 전달하지 않음). 하위 프로세스는 실행되거나 관리되지 않습니다.

```python
# TCP
llm = mlxcel.LLM(base_url="http://localhost:8080/v1")

# Unix domain socket
llm = mlxcel.LLM(socket="/tmp/mlxcel.sock")

print(llm.generate("Hello"))
llm.close()
```

CLI로 대응하는 서버를 시작합니다.

```bash
# TCP
mlxcel serve -m mlx-community/Qwen3-4B-4bit --host 127.0.0.1 --port 8080

# Unix domain socket: a --host ending in .sock selects socket mode
mlxcel serve -m mlx-community/Qwen3-4B-4bit --host /tmp/mlxcel.sock
```

모델과 연결 대상을 모두 전달하면 모드가 모호해지므로 `MlxcelError`가 발생합니다.

## 보안: 다중 사용자 호스트

공유 머신에서는 `/tmp` 아래의 기본 소켓 경로가 누구나 읽을 수 있습니다. 로컬의 어떤 사용자든 실행한 서버에 연결하여 요청을 보낼 수 있습니다. 배포 환경에서 이것이 문제가 된다면, 본인만 읽을 수 있는 디렉터리 아래의 명시적인 `socket=` 경로를 전달하십시오. 예를 들어 `$XDG_RUNTIME_DIR`(모드 `0700`, 본인 uid 소유) 아래의 경로입니다.

```python
import os, pathlib, mlxcel

runtime_dir = pathlib.Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

with mlxcel.LLM("mlx-community/Qwen3-4B-4bit", socket=str(runtime_dir / "mlxcel.sock")) as llm:
    print(llm.generate("hello"))
```

CLI에서는 `mlxcel serve --host "$XDG_RUNTIME_DIR/mlxcel.sock"`에 해당합니다.

macOS에서는 `$TMPDIR`가 이미 `/var/folders` 아래의 사용자별 경로로 확장되므로 기본 소켓이 비공개입니다. 활성 로그인 세션이 없는 Linux에서는 `$XDG_RUNTIME_DIR`가 없을 수 있으며, 필요하면 홈 디렉터리 아래의 `0700` 하위 디렉터리로 대체하십시오.

## 스트리밍

`stream`과 `chat_stream`은 도착하는 대로 텍스트 delta를 yield합니다.

```python
with mlxcel.LLM("mlx-community/Qwen3-4B-4bit") as llm:
    for delta in llm.stream("Write a haiku about autumn"):
        print(delta, end="", flush=True)

    for delta in llm.chat_stream([{"role": "user", "content": "List three uses for a Pi."}]):
        print(delta, end="", flush=True)
```

이 예제의 실행 가능한 버전은 [`python/examples/streaming.py`](../python/examples/streaming.py)에 있습니다.

## 채팅

`chat`은 assistant 메시지 내용을 문자열로 반환합니다.

```python
messages = [
    {"role": "system", "content": "You are concise."},
    {"role": "user", "content": "What is MLX?"},
]
print(llm.chat(messages, max_tokens=128, temperature=0.3))
```

## 샘플링 매개변수

생성 메서드는 OpenAI 샘플링 필드를 직접 받습니다: `max_tokens`, `temperature`, `top_p`, `stop`, `seed`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `response_format`. OpenAI 스키마에 속하지 않는 서버 전용 옵션(`top_k`, `min_p`, `top_n_sigma`, `typical_p`, `repetition_penalty`, DRY 설정, `xtc_probability` / `xtc_threshold`, 그리고 vLLM 호환 루프 감지 필드인 `max_pattern_size` / `min_pattern_size` / `min_count`)은 요청 본문에 그대로 전달됩니다. 모델의 어휘 크기 이상인 `top_k`는 llama-server와 마찬가지로 `0`(비활성)과 같이 모든 토큰을 유지합니다. `xtc_probability`(`0.0`-`1.0`, 기본값 `0.0`, 비활성)는 XTC(Exclude Top Choices)가 확률이 `xtc_threshold`(`0.0`-`0.5`, 기본값 `0.1`)를 초과하는 토큰들 중 확률이 가장 낮은 하나를 제외한 모두를 제거할 step별 확률입니다. 임계값은 원본 모델 분포가 아니라 `top_k`, `top_p`, `min_p`가 실행된 뒤 남은 재정규화된 분포에 대해 검사되며, 이는 llama-server 체인과 일치합니다. 두 값 모두 `/v1/chat/completions`, `/v1/completions`, `/v1/responses`에서 검증되며, 범위를 벗어난 값은 생성이 시작되기 전에 400을 반환합니다. `top_n_sigma`(기본값 `0.0`, 비활성)는 temperature / top-k / top-p / min-p 체인이 실행되기 전에 원시 logit이 행 최댓값으로부터 N 표준편차 이내인 토큰만 유지하며, 높은 temperature에서 샘플링을 안정적으로 유지합니다. OpenAI 형태의 엔드포인트에서는 유한하고 `>= 0.0`이어야 합니다(음수는 400을 반환). 네이티브 `/completion` 표기는 해당 llama-server 호환성 매니페스트 항목(#1436)과 함께 도입됩니다. `typical_p`(기본값 `1.0`, 비활성)는 locally typical 샘플링을 활성화합니다. surprisal이 행 엔트로피에 가장 가까운 토큰을 typicality 순서로 `typical_p` 확률 질량이 누적될 때까지 유지하며, OpenAI 형태의 엔드포인트에서는 유한하고 `(0.0, 1.0]` 범위여야 하고(그렇지 않으면 400), 네이티브 `/completion` 필드는 llama-server의 제한 없는 스키마를 따라 범위를 벗어난 값이 있으면 명시적인 비활성 형태로 해석하며, 서버 전체 `--typical` / `--typical-p` 플래그는 요청이 이 필드를 생략했을 때 적용되는 기본값을 설정합니다. #1436 이후 repetition, frequency, presence 페널티는 llama-server의 `repeat_last_n` 윈도우 의미론을 따릅니다. `repetition_context_size`(별칭 `repeat_last_n`)는 페널티를 마지막 N개 토큰으로 제한하고, `0`은 비활성화하며, 필드가 없으면 서버 전체 `--repeat-last-n` 기본값 64로 폴백합니다(이전에는 항상 전체 이력을 스캔했습니다). DRY도 마찬가지로 b10621의 sentinel 값을 채택합니다. `dry_penalty_last_n: 0`은 DRY를 비활성화하고, CLI의 `-1`은 전체 이력을 스캔하며, `1.0` 미만의 `dry_base`는 서버 기본값으로 폴백합니다. 서버 전체 기본값은 `top_n_sigma`(`--top-nsigma`, `-1.0` 또는 `0.0`이면 비활성), XTC(`--xtc-probability`, `--xtc-threshold`), 시퀀스 종료 억제(`--ignore-eos`), 중지 문자열(`-r` / `--reverse-prompt`, 반복 가능)에도 존재합니다. 임의의 서버 필드를 위해 명시적인 `extra_body={...}`를 전달할 수도 있으며, 여기서 설정한 값이 충돌 시 우선합니다.

```python
llm.generate("Once upon a time", max_tokens=200, top_p=0.9, top_k=40, min_p=0.05)
llm.generate("...", extra_body={"repetition_penalty": 1.1})
```

## 구조화된 출력

서버의 llguidance 기반 제약 디코딩은 `response_format`을 따르므로, 스키마에 유효한 JSON을 요구할 수 있습니다.

```python
import json

schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        },
    },
}

reply = llm.chat(
    [{"role": "user", "content": "Invent a person as JSON."}],
    response_format=schema,
    max_tokens=128,
)
person = json.loads(reply)
```

이 예제의 실행 가능한 버전은 [`python/examples/structured_output.py`](../python/examples/structured_output.py)에 있으며, 동일한 요청을 원본 `openai_client`를 통해서도 보냅니다.

## `openai_client` 탈출구

편의 메서드가 다루지 않는 기능(tools, logprobs, 비전 및 오디오 입력, Responses API)에는 구성된 OpenAI 클라이언트를 직접 사용하십시오. 동일한 transport에 연결되어 있으므로 TCP와 Unix 소켓 구성 모두에서 동작합니다.

```python
oai = llm.openai_client   # openai.OpenAI (or openai.AsyncOpenAI for AsyncLLM)
oai.chat.completions.create(
    model=llm.model,
    messages=[{"role": "user", "content": "Weather in SF?"}],
    tools=[...],
)
```

## 비동기 사용

`mlxcel.AsyncLLM`은 `async`/`await`와 비동기 이터레이터로 동기 API를 그대로 반영하며, `openai.AsyncOpenAI`와 `httpx.AsyncClient`를 기반으로 합니다.

```python
import asyncio
import mlxcel

async def main():
    async with mlxcel.AsyncLLM("mlx-community/Qwen3-4B-4bit") as llm:
        print(await llm.generate("def fib(n):", max_tokens=128))
        async for delta in llm.stream("Write a haiku"):
            print(delta, end="", flush=True)

asyncio.run(main())
```

managed 서버 수명 주기(실행 및 준비 상태 폴링)는 일회성 블로킹 설정이므로 생성자 안에서 동기적으로 실행되며, 모든 생성 호출은 비동기입니다. 모델 id는 첫 요청에서 지연 해석되므로, 호출이 한 번 실행된 뒤에 `llm.model`을 읽거나 명시적인 `model=` 재정의를 전달하십시오.

## 오류

| 예외 | 발생 시점 |
|-----------|-------------|
| `MlxcelError` | 모든 클라이언트 오류의 기본 클래스(모호한 인자에도 사용됨) |
| `MlxcelServerError` | managed 서버가 실행, 준비, 또는 유지에 실패함(캡처된 stderr 끝부분을 포함) |
| `MlxcelTimeoutError` | managed 서버가 `startup_timeout` 내에 준비되지 않음 |

서버의 HTTP 및 API 오류는 네이티브 `openai` SDK 예외(예: `openai.APIStatusError`)로 전파되므로, 상태 코드와 응답 본문이 그대로 보입니다. 수명 주기 관련 문제만 `Mlxcel*` 타입으로 감싸집니다.

## 문제 해결

- **서버가 준비되지 않는 경우.** 첫 실행은 리스너를 바인딩하기 전에 가중치를 다운로드합니다. `startup_timeout`을 늘리고, 로깅을 활성화하여 진행 상황을 확인하십시오: `logging.getLogger("mlxcel.server").setLevel(logging.INFO)`. 서버 stderr(다운로드 및 로드 진행 상황 포함)는 해당 로거로 전달됩니다.
- **시작 시 stderr 끝부분과 함께 `MlxcelServerError`가 발생하는 경우.** 자식 프로세스가 준비되기 전에 종료되었습니다. 첨부된 stderr에 보통 원인이 나와 있습니다(모델을 찾을 수 없음, 메모리 부족, 잘못된 플래그).
- **바이너리를 찾을 수 없는 경우.** `binary="/path/to/mlxcel"`을 전달하거나, `MLXCEL_BIN`을 설정하거나, `mlxcel`을 `PATH`에 두십시오.
- **Unix 소켓 경로가 너무 긴 경우.** `sun_path`는 macOS에서 약 104바이트, Linux에서 108바이트입니다. macOS에서는 `$TMPDIR`가 긴 `/var/folders/...` 경로로 해석되므로, 클라이언트는 대신 `/tmp` 아래의 짧은 이름을 소켓 기본값으로 사용합니다. 명시적인 `socket=` 경로가 제한을 넘으면 클라이언트가 명확한 오류를 발생시키므로, 더 짧은 경로를 전달하십시오.
- **포트가 이미 사용 중인 경우(TCP).** 명시적인 `port=` 없이 managed 모드를 사용하면 클라이언트가 사용 가능한 임시 포트를 선택합니다. 고정 포트가 필요할 때만 `port=`를 전달하십시오.
