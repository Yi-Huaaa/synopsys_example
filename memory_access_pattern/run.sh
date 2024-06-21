#!/bin/bash

clear && g++ cpu_memory_access.cpp -o cpu_memory_access

echo ""
echo "CPU Runtime"

# Outer loop for different values of round
for round in 10 100 1000 10000
do
    echo "Running with round = $round"
    # Inner loop for different values of n
    for i in {10..14} 
    do
        n=$((2**i))
        echo "N = $n" 
        ./cpu_memory_access $n $round
    done
done


nvcc memory_access.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std=c++17 -o memory_access

echo ""
echo "GPU runtime"

# Outer loop for different values of round
for round in 10 100 1000 10000
do
    echo "Running with round = $round"
    # Inner loop for different values of n
    for i in {10..14} 
    do
        n=$((2**i))
        echo "N = $n" 
        ./memory_access $n $round
    done
done
