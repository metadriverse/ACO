# Learning to Drive by Watching YouTube videos: Action-Conditioned Contrastive Policy Pretraining (ECCV22)

[**Webpage**](https://metadriverse.github.io/ACO) | [**Code**](https://github.com/metadriverse/ACO) |  [**Paper**](https://arxiv.org/pdf/2204.02393.pdf) | [**YouTube Driving Dataset**](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155165194_link_cuhk_edu_hk/ErrNZZuZPuJOoX75o7Lo45YBB-bwbLHbD1GSenfnf4-xzQ?e=Xx4RRS) | [**Pretrained ResNet34**](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155165194_link_cuhk_edu_hk/EUF_AbjmGEVKoJSNSpF0cYwBm9hJ1qSEMSYanGOqxBeUPQ)

![](./docs/images/teaser.jpg) 

## Installation

Our codebase is based on [MoCo](https://github.com/facebookresearch/moco). A simple PyTorch environment will be enough (see [MoCo's installation instruction](https://github.com/facebookresearch/moco#preparation)). 

## Dataset

We collect driving videos from YouTube. Here we provide the [:link: video list](https://docs.google.com/spreadsheets/d/1KNFFrfEE5q4d40uBR6MN9YtTggnv2o2AHRxGRZMgs3E/edit?usp=sharing) we used. You could also download the frames directly via [:link: this OneDrive link](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155165194_link_cuhk_edu_hk/ErrNZZuZPuJOoX75o7Lo45YBB-bwbLHbD1GSenfnf4-xzQ?e=Xx4RRS) and run:

```bash
cat sega* > frames.zip
```
to get the zip file.
For training ACO, you should also download `label.pt` and `meta.txt`, and put them under `{aco_path}/code` and `{your_dataset_directory}/` respectively.


## Training

We provide `main_label_moco.py` for training. To perform ACO training of a ResNet-34 model in an 8-gpu machine, run:

```python
python main_label_moco.py -a resnet34 --mlp -j 16 --lr 0.003 \
			 --batch-size 256 --moco-k 40960 --dist-url 'tcp://localhost:10001' \
			 --multiprocessing-distributed --world-size 1 --rank 0 {your_dataset_directory} 
```

Some important arguments:

+ `--aug_cf`: whether to use Cropping and Flipping augmentations in pre-training. In ACO, we do not use these two augmentations by default.
+ `--thres`: action similarity threshold.  

## Pretrained weights

We also provide [:link: pretrained ResNet34 checkpoint](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155165194_link_cuhk_edu_hk/EUF_AbjmGEVKoJSNSpF0cYwBm9hJ1qSEMSYanGOqxBeUPQ). After downloading, you can load this checkpoint via:

```python
import torch
from torchvision.models import resnet34
net = resnet34()
net.load_state_dict(torch.load('ACO_resnet34.ckpt'), strict=False) 
```

## Bibtex

```
@article{zhang2022learning,
  title={Learning to Drive by Watching YouTube videos: Action-Conditioned Contrastive Policy Pretraining},
  author={Zhang, Qihang and Peng, Zhenghao and Zhou, Bolei},
  journal={European Conference on Computer Vision (ECCV)},
  year={2022}
}
```
