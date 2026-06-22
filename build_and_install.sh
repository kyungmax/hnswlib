#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
PYTHON_BIN="${PYTHON:-python}"

echo "▶ Configuring CMake build"
cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}"

echo "▶ Cleaning previous CMake build outputs"
cmake --build "${BUILD_DIR}" --target clean

echo "▶ Building CMake targets"
cmake --build "${BUILD_DIR}" --parallel

echo "▶ Installing Python package with current environment"
"${PYTHON_BIN}" -m pip install --no-build-isolation -e "${PROJECT_ROOT}"

echo "✅ Build & install completed successfully"
