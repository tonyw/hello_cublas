#!/bin/bash

rm -f hello
nvcc -g -G -arch=sm_89 -lcublas -o hello hello.cu
./hello
