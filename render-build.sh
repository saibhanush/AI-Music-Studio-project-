#!/usr/bin/env bash
set -euo pipefail

# Install FFmpeg system libraries required to build PyAV (av package) from source.
# These are needed by audiocraft's transitive dependency on av==11.0.0.
apt-get update -y
apt-get install -y \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config

# Upgrade pip toolchain and install Python dependencies.
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
