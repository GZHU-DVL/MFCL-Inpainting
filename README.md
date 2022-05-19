# MFCL-Inpainting

**Multi-feature Co-learning for Image Inpainting**

*[Jiayu Lin](linjy@e.gzhu.edu.cn), Yuan-Gen Wang<sup>\*</sup>, Wenzhi Tang, Aifeng Li*.

In ICPR'2022.

## Installation

Clone this repo.

```
git clone https://github.com/GZHU-DVL/MFCL-Inpainting.git
```

Prerequisites

- Python=3.8
- Pytorch=1.4
- Torchvision=0.5.0
- Torchaudio=0.4.0
- Tensorboard=2.9.0
- Pillow=8.2.0
- Cudatookit=10.1

## Dataset Preparation

**Image Dataset.**

We evaluate the proposed method on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html), [Paris StreetView](https://github.com/pathak22/context-encoder), and [Places2](http://places2.csail.mit.edu/) datasets, download the datasets from the official website.

For Structure image of datasets used in this paper follows the [structure flow](https://github.com/RenYurui/StructureFlow) and utlize the [RTV smooth method](http://www.cse.cuhk.edu.hk/~leojia/projects/texturesep/). Run generation function [data/Matlab/generate_structre_images.m](https://github.com/GZHU-DVL/MFCL-Inpainting/blob/main/data/Matlab/generate_structure_images.m) in your matlab to get this dataset. For example, if you want to generate smooth images for CelebA, you can run the following code:

```
generate_structure_images("path to CelebA dataset root folder", "path to output folder");
```

**Mask Dataset.** 

Irregular masks are obtained from [Irregular Masks](https://nv-adlr.github.io/publication/partialconv-inpainting) and classified based on their hole sizes relative to the entire image with an increment of 10%.

### Training

To train the model, you run the following code.

```
python train.py \
  --de_root [the path of ground truth images] \
  --st_root [the path of structure images] \
  --mask_root [the path of mask images] \
  --checkpoints_dir [models are saved here] \
  --log_dir [the path to record log]
```

### Testing

To test the model, you modify the following code in [test.py](https://github.com/GZHU-DVL/MFCL-Inpainting/blob/main/test.py).

```
model.netEN.module.load_state_dict(torch.load("")['net'])
model.netDE.module.load_state_dict(torch.load("")['net'])
model.netMEDFE.module.load_state_dict(torch.load("")['net'])
```

### License

This source code is made available for research purpose only.

### Acknowledgement

Our code is built upon [**Rethinking-Inpainting-MEDFE**](https://github.com/KumapowerLIU/Rethinking-Inpainting-MEDFE).
