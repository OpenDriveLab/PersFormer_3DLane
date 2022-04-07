/*
* ==============================================================================
* Binaries and/or source for the following packages or projects are presented under one or more of the following open
* source licenses:
* vision.cpp       The PersFormer Authors        Apache License, Version 2.0
*
* Contact simachonghao@pjlab.org.cn if you have any issue
* 
* See:
* https://github.com/fundamentalvision/Deformable-DETR/tree/main/models/ops/src/vision.cpp
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

#include "ms_deform_attn.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}
