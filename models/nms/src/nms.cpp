/*
* ==============================================================================
* Binaries and/or source for the following packages or projects are presented under one or more of the following open
* source licenses:
* nms.cpp       The PersFormer Authors        Apache License, Version 2.0
*
* Contact simachonghao@pjlab.org.cn if you have any issue
* 
* See:
* https://github.com/lucastabelini/LaneATT/tree/main/lib/nms/src/nms.cpp
*
* Copyright (c) 2022 The PersFormer Authors. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ==============================================================================
*/

#include <torch/extension.h>
#include <torch/types.h>
#include <iostream>

std::vector<at::Tensor> nms_cuda_forward(
        at::Tensor boxes,
        at::Tensor idx,
        float nms_overlap_thresh,
        unsigned long top_k);

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<at::Tensor> nms_forward(
        at::Tensor boxes,
        at::Tensor scores,
        float thresh,
        unsigned long top_k) {


    auto idx = std::get<1>(scores.sort(0,true));

    CHECK_INPUT(boxes);
    CHECK_INPUT(idx);

    return nms_cuda_forward(boxes, idx, thresh, top_k);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms_forward", &nms_forward, "NMS");
}

