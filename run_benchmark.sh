#!/usr/bin/env bash

# 清空（或新建）result.txt
: > result.txt

# 打印标题
echo "====== 编译并运行 matmul_benchmark ======" | tee -a result.txt
echo "" >> result.txt

# 编译：使用 hipcc 将 matmul_benchmark.cpp 编译成可执行文件 matmul_benchmark
# 这里传递 -fopenmp -O3 -march=native 给主机端优化
echo "-> 正在编译 matmul_benchmark.cpp ..." | tee -a result.txt
hipcc -fopenmp -O3 -march=native matmul_benchmark.cpp -o matmul_benchmark
if [ $? -ne 0 ]; then
    echo "[Error] 编译失败，请检查错误信息。" | tee -a result.txt
    exit 1
else
    echo "[Done] 生成可执行文件：matmul_benchmark" | tee -a result.txt
fi
echo "" >> result.txt

# 运行可执行文件并将所有输出（stdout & stderr）追加到 result.txt
echo "-> 运行 matmul_benchmark，将日志写入 result.txt ..." | tee -a result.txt
./matmul_benchmark &>> result.txt
echo "" >> result.txt

echo "====== 全部测试完成，结果保存在 result.txt ======" | tee -a result.txt
