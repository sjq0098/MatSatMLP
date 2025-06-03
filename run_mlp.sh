#!/usr/bin/env bash

# 清空旧的日志文件
: > mlp_result.txt

echo "====== 编译 mlp_forward.cpp ======" | tee -a mlp_result.txt
hipcc mlp_forward_dcu.cpp -o mlp_forward_dcu
if [ $? -ne 0 ]; then
    echo "[Error] 编译失败，请检查 mlp_forward_dcu.cpp 中的错误。" | tee -a mlp_result.txt
    exit 1
else
    echo "[Done] 生成可执行文件：mlp_forward_dcu" | tee -a mlp_result.txt
fi
echo "" >> mlp_result.txt

echo "====== 运行 mlp_forward ======" | tee -a mlp_result.txt
./mlp_forward_dcu &>> mlp_result.txt
if [ $? -ne 0 ]; then
    echo "[Error] 运行 mlp_forward 时出错。" | tee -a mlp_result.txt
    exit 1
else
    echo "[Done] mlp_forward 执行完毕，输出已追加到 mlp_result.txt" | tee -a mlp_result.txt
fi
echo "" >> mlp_result.txt

echo "====== 如果需要使用性能分析，请执行：hipprof ./mlp_forward ======" | tee -a mlp_result.txt
