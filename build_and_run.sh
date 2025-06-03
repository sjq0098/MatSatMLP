#!/bin/bash
set -e

# 编译选项
CXX=g++
CXXFLAGS="-O3 -mavx2 -mfma -fopenmp -std=c++17"

# 源文件与输出可执行文件
SRC="mmm.cpp"
OUT="matmul_opt"

# 检查源文件是否存在
if [ ! -f "$SRC" ]; then
  echo "Error: Source file $SRC not found."
  exit 1
fi

# 编译
echo "Compiling $SRC with flags: $CXXFLAGS"
$CXX $CXXFLAGS "$SRC" -o "$OUT"

echo "Compilation finished. Executable: $OUT"

echo "Running $OUT..."
# 执行并显示输出
./"$OUT"
echo "Execution finished."
# 清理生成的可执行文件
echo "Cleaning up..."
rm -f "$OUT"
echo "Cleanup finished."