# PG-RCNN: Semantic Surface Point Generation for 3D Object Detection (ICCV 2023)

<p align="center">
  <img src="docs/framework.png" width="95%">
</p>

This is the official implementation of "PG-RCNN: Semantic Surface Point Generation for 3D Object Detection" (ICCV 2023).

[[ArXIv](http://arxiv.org/abs/2307.12637)]

Thanks to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), our implementation is based of pcdet v0.5.2.

## Overview
- [Model Zoo](#model-zoo)
- [Installation](docs/INSTALL.md)
- [Getting Started](docs/GETTING_STARTED.md)


## Model Zoo

The results are the 3D detection performance of moderate difficulty on the *val* set of KITTI dataset.

|                                             | training time | Car@R40 | Pedestrian@R40 | Cyclist@R40  | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:---------:|
| [PGRCNN](tools/cfgs/kitti_models/pgrcnn.yaml) |~4.5 hours| 85.25 | 58.37 | 75.04 | [model-8.8M](https://drive.google.com/file/d/1tOQM4SKJsirc_WUxxAZrdl2RNuPZkVv8/view?usp=sharing) | 

Note that the performance may vary a little due to sampling in PointNet++ encoder.

## Installation

Please refer to [INSTALL.md](docs/INSTALL.md) for the installation of `OpenPCDet`.

To train `PG-RCNN`, You need to additionally install [`pytorch3d`](https://github.com/facebookresearch/pytorch3d) for utilizing Chamfer Distance. 
We recommend using pytorch3d ver0.7.0.

## Getting Started

Please refer to [GETTING_STARTED.md](docs/GETTING_STARTED.md) to learn more usage about this project.

### Important: Generate approximated complete object points
Under `pcdet` directory, execute:
```python 
python -m pcdet.datasets.multifindbestfit
```


## License

`PG-RCNN` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

We would like to thank the authors of [`OpenPCDet`](https://github.com/open-mmlab/OpenPCDet) and [`BtcDet`](https://github.com/Xharlie/BtcDet) for their open source release of their codebase.