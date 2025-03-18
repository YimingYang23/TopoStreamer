# TopoStreamer
## Prerequisites
- 4 x 32G memory V100 GPU (for batch size = 2)
- Linux
- Python 3.8.18
- NVIDIA GPU + CUDA 11.1
- PyTorch 1.9.1
- TorchVision: 0.10.1+cu111
- OpenCV: 4.8.0
- MMCV: 1.5.2
- MMCV Compiler: GCC 7.3
- MMCV CUDA Compiler: 11.1
- MMDetection: 2.26.0
- MMSegmentation: 0.30.0
- MMDetection3D: 1.0.0rc6+
- spconv2.0: False
## Installation
It is recommended to download the [environment](https://pan.baidu.com/s/1TLdPFQ-rQlzc8l9LlknNQQ?pwd=9ibi) directly.


cd {Your anaconda path}/envs


mkdir lanesegnet2roadnet


cd lanesegnet2roadnet


Download the [environment](https://pan.baidu.com/s/1TLdPFQ-rQlzc8l9LlknNQQ?pwd=9ibi) and unpack the file in current path.


conda activate lanesegnet2roadnet

## Train

We recommend using 4 GPUs for training. The training logs will be saved to `work_dirs/stream`.
```bash
mkdir -p work_dirs/stream
./tools/dist_train.sh 4 && ./tools/dist_train_stage2.sh 4
```

## Evaluate
```bash
./tools/dist_test.sh 4 
```

For per frame visualization, you can run:
```bash
./tools/dist_test.sh 4 --show
```
## Related resources

We acknowledge all the open-source contributors for the following projects to make this work possible:

- [Openlane-V2](https://github.com/OpenDriveLab/OpenLane-V2)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet)
- [Roadnet](https://github.com/fudan-zvg/RoadNet)
