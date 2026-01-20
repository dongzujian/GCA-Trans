# GCA-Trans

**GCA-Trans: Global Context Aware Transformer for Segmenting Transparent Objects of Arbitrary Scales**, [[paper](https:)].


## Installation

Create environment:

```bash
conda create -n gca_trans python=3.7
conda activate gca_trans
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge
conda install pyyaml pillow requests tqdm ipython scipy opencv-python thop tabulate
```
Install Apex (Required for mixed precision training):
```bash
git clone [https://github.com/NVIDIA/apex](https://github.com/NVIDIA/apex)
cd apex
# Check if your cuda version matches pytorch cuda version before installing
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
cd ..
```
And install:

```bash
python setup.py develop --user
```

## Datasets
Create `datasets` directory and prepare [GDD](https://mhaiyang.github.io/CVPR2020_GDNet/), [GSD](https://jiaying.link/cvpr2021-gsd/), [RGBDGSD](https://jiaying.link/AAAI25-RGBDGlass/), and [Trans10K](https://github.com/xieenze/Trans2Seg#data-preparation) datasets as the structure below:

```text
./datasets
├── GDD
│   ├── test
│   └── train
├── GSD
│   ├── test
│   └── train
├── RGBDGSD
│   ├── test
│   └── train
├── Trans10K_cls12
│   ├── test
│   ├── train
│   ├── validation
│
```

Create `pretrained` direcotry and prepare [pretrained models](https://github.com/whai362/PVT#image-classification) as:

```text
./pretrained
|
└── v2
    ├── pvt_v2_b0.pth
    ├── pvt_v2_b1.pth
    ├── pvt_v2_b2.pth
    ├── pvt_v2_b3.pth
    ├── pvt_v2_b4.pth
    └── pvt_v2_b5.pth
```

## Training 

Before training, please modify the [config](./configs) file to match your own paths.

We train our models on 2 2080Ti GPUs, for example:

```bash
python -m torch.distributed.launch --nproc_per_node=2 tools/train.py --config-file configs/trans10kv2/gca_trans/b4/pvt2_b4_mcm.yaml
```

We recommend to use the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) framework to train models with higher resolutions (e.g. 768x768).

## Evaluation

Before testing the model, please change the `TEST_MODEL_PATH` of the config file.

```bash
python -m torch.distributed.launch --nproc_per_node=2 tools/eval.py --config-file configs/trans10kv2/gca_trans/b4/pvt2_b4_mcm.yaml
```
The weights can be downloaded from [BaiduDrive](https://pan.baidu.com/s/1oCswRAgiIRZkOzFPiQnutw?pwd=1145).


(Visualization) Please check [`demo.py`](./demo.py) to customize the configurations, for example, the speech volumn and frequency.

Code is largely based on [Segmentron](https://github.com/LikeLy-Journey/SegmenTron), [Trans2Seg](https://github.com/xieenze/Trans2Seg), [Trans4Trans](https://github.com/InSAI-Lab/Trans4Trans)
## References
* [Segmentron](https://github.com/LikeLy-Journey/SegmenTron)
* [Trans2Seg](https://github.com/xieenze/Trans2Seg)
* [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
* [Trans4Trans](https://github.com/InSAI-Lab/Trans4Trans)

## License

This repository is under the Apache-2.0 license. For commercial use, please contact with the authors.


## Citations

If you are interested in this work, please cite the following work:

```text

```