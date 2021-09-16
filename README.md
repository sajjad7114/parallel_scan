# parallel_scan
Using GPU to scan a large array. Means every element of array will become the sum of all elements from first to where they are.
# Compile in ubuntu:
nvcc scan2.cu scan2_main.cu -o scan2
