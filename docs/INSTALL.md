# Installation
This docs provides the instructions to get started.
## Environment Setup
- We develop with PyTorch 1.8, and the higher versions are supported. Versions lower than (but not included) PyTorch 1.7 are not supported because we use some DDP function like `all_reduce()` / `all_gather()`.
```
git clone https://github.com/OpenPerceptionX/PersFormer_3DLane.git
cd PersFormer_3DLane/

conda create -n lanemd_torch18 python=3.8 -y

conda activate lanemd_torch18

pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install -r requirements.txt

cd models/nms/
python setup.py install

cd ../ops/
bash make.sh
```
## Known Issues
- If you couldn't compile the `nms` operator because of error like `nvcc fatal   : Unsupported gpu architecture 'compute_86'`, try specifying your cuda path to some environment variable.
    ```
    export LIBRARY_PATH=/usr/local/cuda-11.0/lib64/:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64/:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-11.0/bin/:$PATH
    export CUDA_HOME=/usr/local/cuda-11.0/
    ```