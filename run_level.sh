#!/usr/bin/env bash

# 清空旧的日志文件
: > mlp_forward_level.txt

echo "====== 编译 mlp_forward.cpp ======" | tee -a mlp_forward_level.txt
hipcc mlp_forward_level.cpp -o mlp_forward_level
if [ $? -ne 0 ]; then
    echo "[Error] 编译失败，请检查 mlp_forward_level.cpp 中的错误。" | tee -a mlp_forward_level.txt
    exit 1
else
    echo "[Done] 生成可执行文件：mlp_forward_level" | tee -a mlp_forward_level.txt
fi
echo "" >> mlp_forward_level.txt

echo "====== 运行 mlp_forward ======" | tee -a mlp_forward_level.txt
./mlp_forward_level &>> mlp_forward_level.txt
if [ $? -ne 0 ]; then
    echo "[Error] 运行 mlp_forward 时出错。" | tee -a mlp_forward_level.txt
    exit 1
else
    echo "[Done] mlp_forward 执行完毕，输出已追加到 mlp_forward_level.txt" | tee -a mlp_forward_level.txt
fi
echo "" >> mlp_forward_level.txt

echo "====== 如果需要使用性能分析，请执行：hipprof ./mlp_forward ======" | tee -a mlp_forward_level.txt
