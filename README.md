<div align="center">
  
# TopoStreamer: Temporal Lane Segment Topology Reasoning in Autonomous Driving


[![arXiv](https://img.shields.io/badge/arXiv-2312.16108-479ee2.svg)](xxxxx)
  
</div>
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

## Prepare Dataset
Following [OpenLane-V2 repo](https://github.com/OpenDriveLab/OpenLane-V2/blob/v2.1.0/data) to download the **Image** and the **Map Element Bucket** data. Run the following script to collect data for this repo. 

```bash
cd TopoStreamer
mkdir data

ln -s {Path to OpenLane-V2 repo}/data/OpenLane-V2 ./data/
python ./tools/data_process.py
python ./tools/tracking/dist_track.sh
```

After setup, the hierarchy of folder `data` is described below:
```
data/OpenLane-V2
├── train
|   └── ...
├── val
|   └── ...
├── test
|   └── ...
├── data_dict_subset_A_train_lanesegnet.pkl
├── data_dict_subset_A_val_lanesegnet.pkl
├── data_dict_subset_A_train_lanesegnet_gt_tracks.pkl
├── data_dict_subset_A_val_lanesegnet_gt_tracks.pkl
├── ...
```
## Installation

We recommend using [conda](https://docs.conda.io/en/latest/miniconda.html) to run the code.
```bash
conda create -n topostreamer python=3.8 -y
conda activate topostreamer

# (optional) If you have CUDA installed on your computer, skip this step.
conda install cudatoolkit=11.1.1 -c conda-forge

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Install mm-series packages.
```bash
pip install mmcv-full==1.5.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.26.0
pip install mmsegmentation==0.29.1
pip install mmdet3d==1.0.0rc6
```

Install other required packages.
```bash
pip install -r requirements.txt
```

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

- [SQD-MapNet](https://github.com/shuowang666/SQD-MapNet)
- [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet)
- [Openlane-V2](https://github.com/OpenDriveLab/OpenLane-V2)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [LaneSegNet](https://github.com/OpenDriveLab/LaneSegNet)
- [MapTracker](https://github.com/woodfrog/maptracker)
