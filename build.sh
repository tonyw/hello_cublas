#!/bin/bash

rm -f a
nvcc -arch=sm_89 -o a hello.cu
./a
