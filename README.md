# Action-Conditioned Contrastive Policy Pretraining (ECCV22)

[**Webpage**](https://metadriverse.github.io/ACO) | [**Code**](https://github.com/metadriverse/ACO) |  [**Paper**](https://arxiv.org/pdf/2204.02393.pdf) 

![](./docs/images/teaser.jpg)

## Installation

Our codebase is based on [MoCo](https://github.com/facebookresearch/moco). A simple PyTorch environment will be enough (see [MoCo's installation instruction](https://github.com/facebookresearch/moco#preparation)). 

## Dataset

We collect driving videos from YouTube. Here we provide the [:link: video list](https://docs.google.com/spreadsheets/d/1KNFFrfEE5q4d40uBR6MN9YtTggnv2o2AHRxGRZMgs3E/edit?usp=sharing) we used.

## Training

We provide `main_label_moco.py` for training. To perform ACO training of a ResNet-34 model in an 8-gpu machine, run:

```python
python main_label_moco.py -a resnet34 --mlp -j 16 --lr 0.003 \
			 --batch-size 256 --moco-k 40960 --dist-url 'tcp://localhost:10001' \
			 --multiprocessing-distributed --world-size 1 --rank 0 [your-dataset-directory] 
```

Some important arguments:

+ `--aug_cf`: whether to use Cropping and Flipping augmentations in pre-training. In ACO, we do not use these two augmentations by default.
+ `--thres`: action similarity threshold.  

## Bibtex

