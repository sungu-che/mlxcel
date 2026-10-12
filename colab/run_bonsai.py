# -*- coding: utf-8 -*-
"""run_bonsai.py — Google Colab(CPU)에서 Hugging Face 모델 prism-ml/Bonsai-27B-mlx-1bit 을 받아
Rust mlxcel 바이너리를 Python 래퍼로 구동해 JSON 응답(스키마 강제)을 받습니다.

run.py 와의 차이
  - 모델을 Hugging Face 에서 직접 내려받습니다(huggingface_hub). 이미 받은 파일은 건너뜁니다.
  - 받은 모델은 드라이브 MyDrive/colab/Bonsai-27B-mlx-1bit 에 저장해 다음 실행에서 재사용합니다(드라이브 미마운트 시 /content/models).
  - 나머지 흐름(CPU 바이너리 캐시, system/user 프롬프트, thinking 끄기, JSON 스키마 강제)은 run.py 와 같습니다.

실행: Colab 에서 `!python run_bonsai.py` 또는 셀에 붙여넣고 실행.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# 설정 (환경변수로 덮어쓸 수 있음 - 로컬 테스트용)
# ---------------------------------------------------------------------------
WORKDIR = os.environ.get("MLX_WORKDIR", "/content")                 # 작업 공간
DRIVE_ROOT = os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/colab")
MODEL_DIR = os.environ.get("MODEL_DIR")                              # 지정 시 최우선
MODEL_NAME = os.environ.get("MODEL_NAME", "Bonsai-27B-mlx-1bit")   # 저장 폴더 이름
MODEL_REPO = os.environ.get("MODEL_REPO", "prism-ml/Bonsai-27B-mlx-1bit")  # Hugging Face 저장소
MODEL_FILES = [
    "chat_template.jinja",
    "config.json",
    "merges.txt",
    "model.safetensors",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]
REQUIRED_FILES = ["config.json", "model.safetensors", "tokenizer.json"]  # 이 파일들이 있으면 완성된 모델로 봅니다
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "세차장이 50m 떨어져있다")   # system 메시지
PROMPT = os.environ.get("PROMPT", "걸어가? 운전해?")                       # user 메시지
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "200"))
# 디코딩 단계에서 이 스키마를 강제합니다(서버의 response_format json_schema). 필드를 바꾸면 SYSTEM_PROMPT 도 맞춰 주세요.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["walk", "drive"]},
        "distance_m": {"type": "integer"},
        "reason": {"type": "string"},
    },
    "required": ["decision", "distance_m", "reason"],
    "additionalProperties": False,
}
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.3"))
STARTUP_TIMEOUT = float(os.environ.get("STARTUP_TIMEOUT", "900"))    # 첫 로드 가중치 변환 대기
REPO_URL = "https://github.com/lablup/mlxcel.git"

REPO_DIR = os.path.join(WORKDIR, "mlxcel")
LOCAL_BIN = os.path.join(WORKDIR, "bin", "mlxcel")
DRIVE_BIN_CACHE = os.path.join(DRIVE_ROOT, "mlxcel_cache", "mlxcel-cpu")


def log(msg: str) -> None:
    print(msg, flush=True)


def run(cmd: str, cwd: str | None = None) -> None:
    """명령을 실행하고 출력을 그대로 흘려보냅니다(빌드처럼 오래 걸리는 작업의 진행상황 확인용)."""
    log(f"$ {cmd}")
    proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        tail.append(line)
        tail = tail[-80:]
        if "Compiling" not in line and "warning:" not in line:  # 컴파일 로그 노이즈 줄이기
            print(line, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"명령 실패 (exit {proc.returncode}): {cmd}\n" + "".join(tail[-40:]))


# ---------------------------------------------------------------------------
# 1) 구글 드라이브 마운트 (Colab 에서만)
# ---------------------------------------------------------------------------
def mount_drive_if_needed() -> None:
    if not DRIVE_ROOT.startswith("/content/drive"):
        return
    try:
        from google.colab import drive  # type: ignore
    except ImportError:
        log("[drive] Colab 환경이 아니므로 드라이브 마운트를 건너뜁니다.")
        return
    if not os.path.isdir("/content/drive/MyDrive"):
        log("[drive] 구글 드라이브를 마운트합니다...")
        drive.mount("/content/drive")
    else:
        log("[drive] 이미 마운트되어 있습니다.")


# ---------------------------------------------------------------------------
# 2) 모델 위치 찾기
# ---------------------------------------------------------------------------
def is_complete_model(path: str) -> bool:
    return all(os.path.isfile(os.path.join(path, f)) for f in REQUIRED_FILES)


def download_from_hub(target: str) -> None:
    """Hugging Face 에서 MODEL_FILES 를 target 폴더로 받습니다."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        run(f"{sys.executable} -m pip install -q huggingface_hub")
        from huggingface_hub import hf_hub_download
    log(f"[download] {MODEL_REPO} ({len(MODEL_FILES)}개 파일)")
    os.makedirs(target, exist_ok=True)
    for i, name in enumerate(MODEL_FILES, 1):
        log(f"[download] ({i}/{len(MODEL_FILES)}) {name}")
        hf_hub_download(repo_id=MODEL_REPO, filename=name, local_dir=target)


def resolve_model() -> str:
    """순서: MODEL_DIR 지정 > 드라이브(또는 /content/model)에 있는 모델 > Hugging Face 받기 후 드라이브 저장."""
    if MODEL_DIR:
        if is_complete_model(MODEL_DIR):
            log(f"[model] {MODEL_DIR}")
            return MODEL_DIR
        raise FileNotFoundError(f"MODEL_DIR 에 모델 파일이 없습니다: {MODEL_DIR}")

    # 1) 드라이브 모델명 폴더 확인
    drive_dest = os.path.join(DRIVE_ROOT, MODEL_NAME)
    if is_complete_model(drive_dest):
        log(f"[model] 드라이브에 있는 모델을 사용합니다: {drive_dest}")
        return drive_dest
    local_copy = os.path.join(WORKDIR, "model")
    if is_complete_model(local_copy):
        log(f"[model] 업로드된 모델을 사용합니다: {local_copy}")
        return local_copy

    # 2) 허깅페이스에서 임시 폴더로 받기
    tmp = os.path.join(WORKDIR, "hf_download", MODEL_NAME)
    if not is_complete_model(tmp):
        download_from_hub(tmp)

    # 3) 받은 파일을 드라이브 모델명 폴더에 저장 (드라이브 미마운트면 WORKDIR/models 에 저장)
    dest = drive_dest if DRIVE_ROOT.startswith("/content/drive") else os.path.join(WORKDIR, "models", MODEL_NAME)
    log(f"[save] {tmp} -> {dest}")
    shutil.copytree(tmp, dest, dirs_exist_ok=True)
    if not is_complete_model(dest):
        raise RuntimeError(f"저장 후 모델 파일이 완성되지 않았습니다: {dest}")
    log(f"[model] {dest}")
    return dest


# ---------------------------------------------------------------------------
# 3) 바이너리 준비: 로컬 > 드라이브 캐시 > 새로 빌드
# ---------------------------------------------------------------------------
def ensure_cargo() -> str:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        log("[rust] Rust 툴체인을 설치합니다(rustup)...")
        run('curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal')
        cargo = os.path.expanduser("~/.cargo/bin/cargo")
    os.environ["PATH"] = os.path.dirname(cargo) + os.pathsep + os.environ.get("PATH", "")
    return cargo


def build_binary() -> None:
    log("[build] 시스템 패키지(BLAS/LAPACK)를 설치합니다...")
    run("apt-get update -qq && apt-get install -y -qq libopenblas-dev liblapack-dev "
        "liblapacke-dev gfortran pkg-config libssl-dev")
    ensure_cargo()
    ensure_repo()
    log("[build] CPU 전용 release 빌드를 시작합니다. 첫 빌드는 30분 안팎 걸립니다...")
    # cuda/metal 미사용(CPU), webui 제외(추론에 불필요). rust-toolchain.toml 의 버전은 rustup 이 자동 설치.
    run("cargo build --release --bin mlxcel --no-default-features --features surgery", cwd=REPO_DIR)
    os.makedirs(os.path.dirname(LOCAL_BIN), exist_ok=True)
    shutil.copy2(os.path.join(REPO_DIR, "target", "release", "mlxcel"), LOCAL_BIN)


def ensure_binary() -> str:
    if os.path.isfile(LOCAL_BIN) and os.access(LOCAL_BIN, os.X_OK):
        log(f"[bin] 로컬 바이너리 사용: {LOCAL_BIN}")
    elif os.path.isfile(DRIVE_BIN_CACHE):
        log(f"[bin] 드라이브 캐시에서 복사: {DRIVE_BIN_CACHE}")
        os.makedirs(os.path.dirname(LOCAL_BIN), exist_ok=True)
        shutil.copy2(DRIVE_BIN_CACHE, LOCAL_BIN)
        os.chmod(LOCAL_BIN, 0o755)
    else:
        build_binary()
        if os.path.isdir(os.path.dirname(DRIVE_BIN_CACHE)) or DRIVE_ROOT.startswith("/content/drive"):
            try:
                os.makedirs(os.path.dirname(DRIVE_BIN_CACHE), exist_ok=True)
                shutil.copy2(LOCAL_BIN, DRIVE_BIN_CACHE)
                log(f"[bin] 드라이브에 캐시했습니다: {DRIVE_BIN_CACHE}")
            except OSError as e:
                log(f"[bin] 드라이브 캐시 저장은 건너뜁니다: {e}")
    return LOCAL_BIN


# ---------------------------------------------------------------------------
# 4) Python 래퍼 로드 (pip 패키지 소스를 직접 import: 의존성은 openai, httpx)
# ---------------------------------------------------------------------------
def ensure_repo() -> None:
    """파이썬 래퍼 소스(python/src)를 쓰기 위해 저장소는 항상 있어야 합니다(얕은 복제, 빠름)."""
    if not os.path.isdir(os.path.join(REPO_DIR, ".git")):
        log("[repo] mlxcel 저장소를 복제합니다...")
        run(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")


def load_wrapper():
    try:
        import httpx  # noqa: F401
        import openai  # noqa: F401
    except ImportError:
        run(f"{sys.executable} -m pip install -q 'openai>=1.40' 'httpx>=0.27'")
    src = os.path.join(REPO_DIR, "python", "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from mlxcel import LLM  # noqa: E402
    return LLM


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def main() -> int:
    t0 = time.time()
    mount_drive_if_needed()
    model_dir = resolve_model()

    # 이 노트북은 GPU가 없으므로 CPU 실행을 강제합니다. 자식 프로세스(mlxcel serve)가 상속합니다.
    os.environ["MLXCEL_DEVICE"] = "cpu"
    binary = ensure_binary()
    ensure_repo()
    LLM = load_wrapper()

    log("[run] mlxcel 서버를 띄우고 모델을 로드합니다...")
    with LLM(model_dir, binary=binary, startup_timeout=STARTUP_TIMEOUT) as llm:
        log(f"[run] 준비 완료. model id = {llm.model}")
        log(f"[system] {SYSTEM_PROMPT}")
        log(f"[prompt] {PROMPT}")
        t1 = time.time()
        # chat template 을 거치도록 chat() 사용 (generate() 는 raw 프롬프트라 instruct 모델이 빈 응답을 내는 경우가 있음)
        text = llm.chat(
            [{"role": "system", "content": SYSTEM_PROMPT + "\n\n반드시 JSON 객체 하나로만 답하세요."},
             {"role": "user", "content": PROMPT}],
            max_tokens=MAX_TOKENS, temperature=TEMPERATURE,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "answer", "schema": RESPONSE_SCHEMA}},
            # Qwen3.5 는 chat template 기본값이 thinking 모드라 MAX_TOKENS 를 추론에 다 써 답이 비는 문제가 있어 끕니다.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        log("[output raw]")
        log(text if text.strip() else "(빈 응답: MAX_TOKENS 를 늘리거나 모델/템플릿 설정을 확인하세요)")
        if text.strip():
            result = json.loads(text)  # 스키마로 강제했으므로 항상 파싱되어야 합니다
            log("[output json]")
            log(json.dumps(result, ensure_ascii=False, indent=2))
        log(f"[done] 생성 {time.time() - t1:.1f}s / 전체 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
