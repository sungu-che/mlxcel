#!/usr/bin/env bash
# Linux(CPU) 환경 준비: 시스템 패키지 -> mlxcel 저장소 -> 바이너리 빌드 -> Python 래퍼 설치
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HERE/mlxcel"
VENV="$HERE/.venv"

# 1) 시스템 패키지 (BLAS/LAPACK 은 MLX CPU 백엔드용)
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update -qq || apt-get update -qq
sudo apt-get install -y -qq libopenblas-dev liblapack-dev liblapacke-dev gfortran pkg-config libssl-dev \
  || apt-get install -y -qq libopenblas-dev liblapack-dev liblapacke-dev gfortran pkg-config libssl-dev

# 2) 저장소
[ -d "$REPO/.git" ] || git clone --depth 1 https://github.com/lablup/mlxcel.git "$REPO"

# 3) 바이너리 빌드 (CPU: cuda/metal 미사용, webui 제외)
#    rust-toolchain.toml 의 1.97.1 을 받을 수 없는 환경이면 RUSTUP_TOOLCHAIN=stable 로 우회합니다.
( cd "$REPO" && cargo build --release --bin mlxcel --no-default-features --features surgery )

# 4) Python 래퍼 설치 (venv)
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q "$REPO/python"

echo "setup done. 실행: $VENV/bin/python $HERE/run_granite.py"
