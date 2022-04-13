# Installation
This docs provides the instructions to get started.
## Environment Setup
- We develop with PyTorch 1.8, and the higher versions are supported. Versions lower than (but not included) PyTorch 1.7 are not supported because we use some DDP function like `all_reduce()` / `all_gather()`.
```
git clone https://github.com/OpenPerceptionX/PersFormer_3DLane.git
cd PersFormer_3DLane/

conda create -n lanemd_torch18 python=3.8 -y

conda activate lanemd_torch18

# for cuda 11.1
pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# for cuda 10.1
# conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch

pip3 install -r requirements.txt

cd models/nms/
python setup.py install

cd ../ops/
bash make.sh
```
## Known Issues
- If you couldn't compile the `nms` operator because of error like `nvcc fatal   : Unsupported gpu architecture 'compute_86'`, try specifying your cuda path to some environment variable. Following is an example for cuda-11.4
    ```
    export LIBRARY_PATH=/usr/local/cuda-11.4/lib64/:$LIBRARY_PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.4/lib64/:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-11.4/bin/:$PATH
    export CUDA_HOME=/usr/local/cuda-11.4/
    ```
    or you can try to compile `nms` following installation of pytorch under cuda-10.1