# PersFormer: a New Baseline for 3D Laneline Detection
![pipeline](imgs/pipeline.svg)  

> **PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark**  
> Li Chen<sup>∗†</sup>, Chonghao Sima<sup>∗</sup>, Yang Li<sup>∗</sup>, Zehan Zheng, Jiajie Xu, Xiangwei Geng,  Hongyang Li<sup>†</sup>, Conghui He, Jianping Shi, Yu Qiao, Junchi Yan. <sup>∗</sup> equal contributions.  <sup>†</sup> corresponding authors 
> 
> - Paper: [arXiv 2203.11089](https://arxiv.org/abs/2203.11089), ECCV 2022 **Oral** Presentation (2.7% acceptance rate)  
> - Third-party In-depth [Blog on Persformer](https://patrick-llgc.github.io/Learning-Deep-Learning/paper_notes/persformer.html) (recommended) 
> - Our blog | Slides | Presentation video


## Introduction

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/persformer-3d-lane-detection-via-perspective-1/3d-lane-detection-on-openlane)](https://paperswithcode.com/sota/3d-lane-detection-on-openlane?p=persformer-3d-lane-detection-via-perspective-1)

This repository is the PyTorch implementation for **PersFormer**.  

PersFormer is an end-to-end monocular 3D lane detector with a novel Transformer-based spatial feature transformation module. Our model generates BEV features by attending to related front-view local regions with camera parameters as a reference. It adopts a unified 2D/3D anchor design and an auxiliary task to detect 2D/3D lanes simultaneously, enhancing the feature consistency and sharing the benefits of multi-task learning.
  
- [Changelog](#changelog)
- [Get Started](#get-started)
  - [Installation](#installation)
  - [Dataset](#dataset)
  - [Training and evaluation](#training-and-evaluation)
- [Benchmark](#benchmark)
- [Visualization](#visualization)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)  
  
## Changelog
2022-11-3: Fix the evaluation pipeline bug and upload the best model of PersFormer on openlaneV1.1 for reproducibility. Other models are coming soon.   
2022-9-27: Update evaluation metrics, prune gt points by visibility before evaluation, detail can be found in related issue [A question about prune_3d_lane_by_visibility](https://github.com/OpenPerceptionX/OpenLane/issues/18); support Gen-LaneNet on OpenLane; support PersFormer on once dataset.    
2022-5-9: We compared our method on [ONCES_3DLanes](https://github.com/once-3dlanes/once_3dlanes_benchmark) Dataset, where PersFormer also **outperforms** other methods.  
2022-4-12: We released the v1.0 code for PersFormer.  

## Get Started
  
### Installation
- To run PersFormer, make sure you are using a machine with **at least** one GPU.
- Please follow [INSTALL.md](docs/INSTALL.md) to setup the environment.
  
### Dataset
- Please refer to [OpenLane](https://github.com/OpenPerceptionX/OpenLane) for downloading OpenLane Dataset.
- Please refer to [Gen-LaneNet](https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection) for downloading Apollo 3D Lane Synthetic Dataset.

### Training and evaluation 
- Please follow [TRAIN_VAL.md](docs/TRAIN_VAL.md) to train and evaluate the model.

## Benchmark
  - 3D Lane Detection Results (**F-Score**) in [OpenLane](https://github.com/OpenPerceptionX/OpenLane).
  
| Method     |Version| All  | Up &<br>Down | Curve | Extreme<br>Weather | Night | Intersection | Merge&<br>Split |  Best model |x-c|x-f|z-c|z-f|Category Accuracy|
| :----:     |:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| GenLaneNet |1.1| 32.3 | 25.4 | 33.5 | 28.1 | 18.7 | 21.4 | 31.0 |    [model](https://docs.google.com/forms/d/e/1FAIpQLScLRUvr14taMIZzGZs7ZDRIOchPQQ50rdIWyipQeRl2bsO9dQ/viewform?usp=sharing)    |0.59294| 0.494231|0.139849|0.1952|-|
| 3DLaneNet  |1.1| 44.1 | 40.8 | 46.5 | 47.5 | 41.5 | 32.1 | 41.7 |    -    | -|-| -|-|-|
|**PersFormer**|1.1|**50.5**|**45.6**|**58.7**|**54.0**|**50.0**|**41.6**|**53.1**|[model](https://drive.google.com/file/d/1TwjmFB3p3luCG8ytZ4MEI-TMoDT2Vn3G/view?usp=share_link)  | 0.31897|0.3252|0.1115|0.14078|89.51|
|**PersFormer**|1.2|**52.9**|**47.5**|**58.4**|**51.8**|**47.4**|**42.1**|**50.9**|[model](https://drive.google.com/file/d/1jtDfnxcNNbefgpYGfue1XlvcvtmfPZj7/view?usp=share_link)  | 0.2909|0.29429| 0.0800|0.11565|89.24|

  
  - 2D Lane Detection Results (**F-Score**) in [OpenLane](https://github.com/OpenPerceptionX/OpenLane). Note that the baseline of 2D branch in PersFormer is **LaneATT**.
  
| Method     | All  | Up&<br>Down | Curve | Extreme<br>Weather | Night | Intersection | Merge&<br>Split |
| :----:     |:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| LaneATT-S  | 28.3 | 25.3 | 25.8 | 32.0 | 27.6 | 14.0 | 24.3 | 
| LaneATT-M  | 31.0 | 28.3 | 27.4 | 34.7 | 30.2 | 17.0 | 26.5 | 
| PersFormer | 42.0 | 40.7 | 46.3 | 43.7 | 36.1 | 28.9 | 41.2 |  
| CondLaneNet-S | 52.3 | 55.3 | 57.5 | 45.8 | 46.6 | 48.4 | 45.5 | 
| CondLaneNet-M | 55.0 | 58.5 | 59.4 | 49.2 | 48.6 | 50.7 | 47.8 | 
|**CondLaneNet-L**|**59.1**|**62.1**|**62.9**|**54.7**|**51.0**|**55.7**|**52.3**| 
  
  - 3D Lane Detection Results in [ONCE_3DLanes](https://github.com/once-3dlanes/once_3dlanes_benchmark).
  
| Method     | F1(%)  | Precision(%) | Recall(%) | CD error(m) |  Best model |
| :----:     |:----:|:----:|:----:|:----:|:----:|    
| 3DLaneNet  | 44.73 | 61.46 | 35.16 | 0.127 | - | 
| GenLaneNet | 45.59 | 63.95 | 35.42 | 0.121 |  - | 
| SALAD ([paper](https://arxiv.org/pdf/2205.00301.pdf) of ONCE 3DLanes )     | 64.07 | 75.90 | 55.42 | 0.098 | - | 
|**PersFormer**|**72.07**|**77.82**|**67.11**|**0.086**|  [model](https://drive.google.com/file/d/1jtDfnxcNNbefgpYGfue1XlvcvtmfPZj7/view?usp=share_link)| 
  

## Visualization
Following are the visualization results of PersFormer on OpenLane dataset and Apollo dataset.
- OpenLane visualization results  
<img src=imgs/openlane_vis.png width="720" height="720" alt="openlane_vis"/><br/> 
- Apollo 3D Synthetic visualization results  
<img src=imgs/apollo_vis.png width="720" height="500" alt="apollo_vis"/><br/> 
  

## Citation
  Please use the following citation if you find our repo or our paper [PersFormer](https://arxiv.org/abs/2203.11089) useful:
```bibtex
    @inproceedings{chen2022persformer,
      title={PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark},
      author={Chen, Li and Sima, Chonghao and Li, Yang and Zheng, Zehan and Xu, Jiajie and Geng, Xiangwei and Li, Hongyang and He, Conghui and Shi, Jianping and Qiao, Yu and Yan, Junchi},
      booktitle={European Conference on Computer Vision (ECCV)},
      year={2022}
    }  
```

## Acknowledgements
  We would like to acknowledge the great support from SenseBee labelling team at SenseTime Research, constructive suggestion from Zihan Ding at BUAA, and the fruitful discussions and comments for this project from Zhiqi Li, Yuenan Hou, Yu Liu, Jing Shao, Jifeng Dai. We thank for the code implementation from [Gen-LaneNet](https://github.com/yuliangguo/Pytorch_Generalized_3D_Lane_Detection), [LaneATT](https://github.com/lucastabelini/LaneATT) and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). 


## License
  All code within this repository is under [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
