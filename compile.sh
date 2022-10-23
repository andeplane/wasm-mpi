#!/bin/bash
if [[ -z ${MPI_DEBUG} ]]; then
  echo "Compiling production build"
  emcc -std=c++17 \
  -O3 --bind \
  -s SINGLE_FILE=1 \
  -s PTHREAD_POOL_SIZE=8 \
  -s USE_PTHREADS=1 \
  -s ASSERTIONS=1 \
  -s EXPORT_NAME=LibModule \
  -s MODULARIZE=1 \
  -s WASM=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -o lib.js \
  main.cpp mpi.cpp
else
  echo "Compiling debug build"
  emcc -std=c++17 \
  --bind \
  -s SINGLE_FILE=1 \
  -s PTHREAD_POOL_SIZE=8 \
  -s USE_PTHREADS=1 \
  -s ASSERTIONS=1 \
  -s EXPORT_NAME=LibModule \
  -s MODULARIZE=1 \
  -s WASM=1 \
  -s ALLOW_MEMORY_GROWTH=1 \
  -gsource-map \
  -o lib.js \
  main.cpp mpi.cpp
fi
