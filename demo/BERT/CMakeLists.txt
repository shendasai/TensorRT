# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  \
--expt-relaxed-constexpr \
--expt-extended-lambda \
-std=c++11 \
-g \
-gencode arch=compute_70,code=sm_70 \
-gencode arch=compute_75,code=sm_75 \
-Wno-deprecated-declarations")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11 -Wno-deprecated-declarations")

set(BERT_LIBS
    cudart
    cublas
    nvinfer
    nvinfer_plugin
    pthread
    z
)

include_directories(
    ./
    ./bert
    ./layers
    ./plugins
    ./util
    ../../include/
    ../../samples/common/
    ../../third_party/cub/
    /usr/include/x86_64-linux-gnu
    /usr/local/cuda-10.0/include
    /workspace/tensorrt/include
    /workspace/tensorrt/samples/common
    /workspace/cub/
    /workspace/cutlass/
)

link_directories(
    /usr/lib/x86_64-linux-gnu
    /usr/local/cuda-10.0/lib64
    /workspace/tensorrt/lib
    /home/odin/shendasai/TensorRT6/TensorRT-6.0.1.5/lib
)

add_library(common SHARED
    ../../samples/common/logger.cpp
)

add_library(bert_plugins SHARED
    plugins/embLayerNormPlugin.cu
    plugins/geluPlugin.cu
    plugins/qkvToContextPlugin.cu
    plugins/skipLayerNormPlugin.cu
)

target_link_libraries(bert_plugins
    common
    ${BERT_LIBS}
)

target_link_libraries(common
    ${BERT_LIBS}
)

#add_library(sample_bert 
add_library(bert SHARED
#add_executable(sample_bert
    bert/bert.cpp
    bert/driver.cpp
    util/dataUtils.cpp
    BertQA.cpp
    BertFactory.cpp
)

add_executable(sample_bert
    sampleBERT.cpp
)

target_compile_features(sample_bert PUBLIC cxx_std_11)

target_link_libraries(sample_bert
    common
    bert_plugins
    bert
)
