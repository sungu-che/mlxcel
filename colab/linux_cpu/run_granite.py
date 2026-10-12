# -*- coding: utf-8 -*-
"""Linux(CPU) 에서 Rust mlxcel 바이너리를 Python 래퍼로 구동해 Granite 모델 추론을 수행합니다.

흐름
  1. 바이너리(mlxcel)와 모델 디렉터리를 찾습니다. (환경변수 또는 기본 경로)
  2. MLXCEL_DEVICE=cpu 로 CPU 실행을 강제합니다. (이 머신에는 GPU가 없음)
  3. mlxcel.LLM(managed mode) 이 `mlxcel serve -m <model>` 을 띄우고 준비될 때까지 기다립니다.
  4. generate() 로 프롬프트를 실행하고 결과를 출력합니다.

사용 예
  python run_granite.py
  python run_granite.py --prompt "Hello" --max-tokens 64
  MLXCEL_BIN=/path/to/mlxcel MODEL_DIR=/path/to/model python run_granite.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# 기본 경로 (setup_linux.sh 가 이 위치에 맞춰 준비합니다)
DEFAULT_REPO = HERE / "mlxcel"
# 모델 탐색 순서: MODEL_DIR 환경변수 > ../model (colab/model) > ./models/granite-4.0-h-350m-8bit
MODEL_CANDIDATES = [
    HERE.parent / "model",
    HERE / "models" / "granite-4.0-h-350m-8bit",
]
DEFAULT_PROMPT = "Explain what Artificial Intelligence (AI) is in one short sentence."


def resolve_binary() -> str:
    """mlxcel 바이너리 경로를 찾습니다. MLXCEL_BIN > PATH > 저장소 빌드 산출물 순서."""
    env_bin = os.environ.get("MLXCEL_BIN")
    if env_bin and Path(env_bin).is_file():
        return str(Path(env_bin).resolve())
    on_path = shutil.which("mlxcel")
    if on_path:
        return on_path
    built = DEFAULT_REPO / "target" / "release" / "mlxcel"
    if built.is_file():
        return str(built)
    raise SystemExit(
        "mlxcel 바이너리를 찾지 못했습니다. setup_linux.sh 로 빌드하거나 MLXCEL_BIN 을 지정하세요."
    )


def resolve_model() -> str:
    candidates = [Path(os.environ["MODEL_DIR"])] if os.environ.get("MODEL_DIR") else MODEL_CANDIDATES
    for model in candidates:
        if (model / "config.json").is_file():
            return str(model.resolve())
    raise SystemExit(f"모델 디렉터리를 찾지 못했습니다(config.json 없음). 확인 경로: {candidates}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Granite mlxcel CPU 추론 (Python 래퍼 구동)")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--startup-timeout", type=float, default=900.0,
                        help="서버 준비 대기 시간(초). 첫 로드 시 가중치 변환이 있을 수 있어 넉넉히 둡니다.")
    args = parser.parse_args()

    # CPU 실행 강제. 자식 프로세스(mlxcel serve)가 상속받습니다.
    os.environ.setdefault("MLXCEL_DEVICE", "cpu")

    binary = resolve_binary()
    model_dir = resolve_model()
    print(f"[info] binary : {binary}")
    print(f"[info] model  : {model_dir}")
    print(f"[info] device : {os.environ['MLXCEL_DEVICE']}")

    # Python 래퍼 import (pip install 된 mlxcel 패키지)
    try:
        from mlxcel import LLM
    except ImportError:
        print("mlxcel 파이썬 패키지가 없습니다. setup_linux.sh 를 먼저 실행하세요.", file=sys.stderr)
        return 1

    t0 = time.time()
    with LLM(model_dir, binary=binary, startup_timeout=args.startup_timeout) as llm:
        print(f"[info] server ready in {time.time() - t0:.1f}s, model id = {llm.model}")
        print(f"[prompt] {args.prompt}")

        t1 = time.time()
        text = llm.generate(
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        elapsed = time.time() - t1

    print("[output]")
    print(text)
    print(f"[done] generation {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
