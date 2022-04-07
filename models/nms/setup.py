# ==============================================================================
# Binaries and/or source for the following packages or projects are presented under one or more of the following open
# source licenses:
# setup.py       The PersFormer Authors        Apache License, Version 2.0
#
# Contact simachonghao@pjlab.org.cn if you have any issue
# 
# See:
# https://github.com/lucastabelini/LaneATT/tree/main/lib/nms/setup.py
#
# Copyright (c) 2022 The PersFormer Authors. All Rights Reserved.
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
# ==============================================================================

from setuptools import setup

from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(name='nms', packages=['nms'],
        package_dir={'':'src'},
        ext_modules=[CUDAExtension('nms.details', ['src/nms.cpp', 'src/nms_kernel.cu'])],
        cmdclass={'build_ext': BuildExtension})
