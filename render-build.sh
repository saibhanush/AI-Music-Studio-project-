#!/usr/bin/env bash
set -e

export DEBIAN_FRONTEND=noninteractive

# Install FFmpeg system libraries required to build PyAV (av package) from source.
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

# Make sure pkg-config can find the installed FFmpeg .pc files.
export PKG_CONFIG_PATH="/usr/lib/x86_64-linux-gnu/pkgconfig:/usr/lib/pkgconfig:/usr/share/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

# Verify pkg-config can find libavformat before proceeding.
if ! pkg-config --exists libavformat; then
    echo "ERROR: pkg-config still cannot find libavformat after apt-get install."
    echo "PKG_CONFIG_PATH=$PKG_CONFIG_PATH"
    pkg-config --list-all | grep -i av || true
    exit 1
fi

echo "FFmpeg libraries found: $(pkg-config --modversion libavformat)"

# Upgrade pip toolchain and install Python dependencies.
pip install --upgrade pip setuptools wheel

# Install av separately first so the env var is in scope during its build.
PKG_CONFIG_PATH="$PKG_CONFIG_PATH" pip install "av==11.0.0"

pip install -r requirements.txt
