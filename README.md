# PersFormer
**PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark** [[Paper]](https://arxiv.org/abs/2203.11089)  
Li Chen<sup>1∗†</sup>, Chonghao Sima<sup>1,2∗</sup>, Yang Li<sup>1∗</sup>, Zehan Zheng<sup>1</sup>, Jiajie Xu<sup>3</sup>, Xiangwei Geng<sup>4</sup>,  Hongyang Li<sup>1,5†</sup>, Conghui He<sup>1</sup>, Jianping Shi<sup>4</sup>, Yu Qiao<sup>1</sup>, Junchi Yan<sup>5</sup>.   
  
<sup>1</sup> Shanghai AI Laboratory, Shanghai, China  
<sup>2</sup> Purdue University, West Lafayette, IN, USA  
<sup>3</sup> Carnegie Mellon University, Pittsburgh, PA, USA  
<sup>4</sup> SenseTime Research, Beijing, China  
<sup>5</sup> Shanghai Jiao Tong University, Shanghai, China  
<sup>∗</sup> equal contributions.  
<sup>†</sup> corresponding authors  
  


## Introduction
  This repository is the PyTorch implementation for **PersFormer**.  
  
![](pipeline.png)  

We present PersFormer: an end-to-end monocular 3D lane detector with a novel Transformer-based spatial feature transformation module. Our model generates BEV features by attending to related front-view local regions with camera parameters as a reference. PersFormer adopts a unified 2D/3D anchor design and an auxiliary task to detect 2D/3D lanes simultaneously, enhancing the feature consistency and sharing the benefits of multi-task learning.
  
- [Changelog](#changelog)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Demo](#demo)
- [Benchmark](#benchmark)
- [Citation](#citation)
- [Acknowledge](#acknowledge)
- [License](#license)  
  
## Changelog
**2022-4-7**: We released the v1.0 code for PersFormer.  

## Get Started
  
### Installation
```
git clone https://github.com/OpenPerceptionX/PersFormer_3DLane.git
cd PersFormer_3DLane/

conda create -n lanemd_torch18 python=3.8 -y

conda activate lanemd_torch18

conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt

cd models/nms/
python setup.py install

cd ../ops/
bash make.sh
```
  
### Dataset
Please refer to [OpenLane](https://github.com/OpenPerceptionX/OpenLane) for downloading.

### Demo 
Please specify your config in `config/persformer_openlane.py`, especially the `args.dataset_dir` and `args.data_dir` after downloading the [OpenLane](https://github.com/OpenPerceptionX/OpenLane) Dataset.  
Other config details please see `utils/utils.py`.  
- For **training**, set `args.evaluate = False` , and run:  
```
python main_persformer.py --batch_size=$BATCH_SIZE$ --nepochs=$EPOCHS$
```
- For **evaluation**, set `args.evaluate = True` , and run:
```
python main_persformer.py --batch_size=$BATCH_SIZE$
``` 

## Benchmark
  - 3D Lane Detection Results (**F-Score**) in [OpenLane](https://github.com/OpenPerceptionX/OpenLane)
  
| Method     | All  | Up &<br>Down | Curve | Extreme<br>Weather | Night | Intersection | Merge&<br>Split |  
| :----:     |:----:|:----:|:----:|:----:|:----:|:----:|:----:|  
| GenLaneNet | 29.7 | 24.2 | 31.1 | 26.4 | 17.5 | 19.7 | 27.4 |  
| 3DLaneNet  | 40.2 | 37.7 | 43.2 | 43.0 | 39.3 | 29.3 | 36.5 |  
|**PersFormer**|**47.8**|**42.4**|**52.8**|**48.7**|**46.0**|**37.9**|**44.6**|  
  
## Citation
  Please use the following citation when referencing [PersFormer](https://arxiv.org/abs/2203.11089):

    @article{chen2022persformer,
      title={PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark},
      author={Chen, Li and Sima, Chonghao and Li, Yang and Zheng, Zehan and Xu, Jiajie and Geng, Xiangwei and Li, Hongyang and He, Conghui and Shi, Jianping and Qiao, Yu and Yan, Junchi},
      journal={arXiv preprint arXiv:2203.11089},
      year={2022}
    }  

## Acknowledge
  We would like to acknowledge the great support from SenseBee labelling team at SenseTime Research, and the fruitful discussions and comments for this project from Zhiqi Li, Yuenan Hou, Yu Liu.


## License
  All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).