#!/usr/bin/env bash

# 清空旧日志
: > mlp_dcu_result.txt

echo "====== 编译 mlp_dcu.cpp ======" | tee -a mlp_dcu_result.txt
hipcc mlp_dcu.cpp -o mlp_dcu
if [ $? -ne 0 ]; then
    echo "[Error] 编译失败，请检查 mlp_dcu.cpp 中的错误。" | tee -a mlp_dcu_result.txt
    exit 1
else
    echo "[Done] 生成可执行文件：mlp_dcu" | tee -a mlp_dcu_result.txt
fi
echo "" >> mlp_dcu_result.txt

echo "====== 运行 mlp_dcu ======" | tee -a mlp_dcu_result.txt
./mlp_dcu &>> mlp_dcu_result.txt
if [ $? -ne 0 ]; then
    echo "[Error] 运行 mlp_dcu 时出错。" | tee -a mlp_dcu_result.txt
    exit 1
else
    echo "[Done] mlp_dcu 执行完毕，输出已追加到 mlp_dcu_result.txt" | tee -a mlp_dcu_result.txt
fi
echo "" >> mlp_dcu_result.txt

echo "====== 若要进一步做性能剖析，请执行：hipprof ./mlp_dcu ======" | tee -a mlp_dcu_result.txt

