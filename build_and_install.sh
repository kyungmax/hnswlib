#!/usr/bin/env bash
set -e  # 중간에 에러 나면 즉시 종료

echo "▶ Entering build directory"
cd build

echo "▶ make clean"
make clean

echo "▶ cmake .."
cmake ..

echo "▶ make all -j"
make all -j

echo "▶ Returning to project root"
cd ..

echo "▶ pip install -e ."
pip install -e .

echo "✅ Build & install completed successfully"

