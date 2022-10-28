# ArbSR-PyTorch

## Overview

This repository contains an op-for-op PyTorch reimplementation
of [Learning A Single Network for Scale-Arbitrary Super-Resolution](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_a_Single_Network_for_Scale-Arbitrary_Super-Resolution_ICCV_2021_paper.pdf)
.

## Table of contents

- [ArbSR-PyTorch](#arbsr-pytorch)
    - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [Download weights](#download-weights)
    - [Download datasets](#download-datasets)
    - [How Test and Train](#how-test-and-train)
        - [Test ArbSR_RCAN](#test-arbsr_rcan)
        - [Train ArbSR_RCAN model](#train-arbsr_rcan-model)
        - [Resume train ArbSR_RCAN model](#resume-train-arbsr_rcan-model)
    - [Result](#result)
    - [Contributing](#contributing)
    - [Credit](#credit)
        - [Learning A Single Network for Scale-Arbitrary Super-Resolution](#learning-a-single-network-for-scale-arbitrary-super-resolution)

## Download weights

- [Google Driver](https://drive.google.com/drive/folders/17ju2HN7Y6pyPK2CC_AqnAfTOe9_3hCQ8?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1yNs4rqIb004-NKEdKBJtYg?pwd=llot)

## Download datasets

Contains DIV2K, DIV8K, Flickr2K, OST, T91, Set5, Set14, BSDS100 and BSDS200, etc.

- [Google Driver](https://drive.google.com/drive/folders/1A6lzGeQrFMxPqJehK9s37ce-tPDj20mD?usp=sharing)
- [Baidu Driver](https://pan.baidu.com/s/1o-8Ty_7q6DiS3ykLU09IVg?pwd=llot)

Please refer to `README.md` in the `data` directory for the method of making a dataset.

## How Test and Train

Both training and testing only need to modify the `config.py` file.

### Test ArbSR_RCAN

modify the `config.py`

- line 31: `model_arch_name` change to `arbsr_rcan`.
- line 40: `upscale_factor` change to `4`.
- line 50: `mode` change to `test`.
- line 52: `exp_name` change to `ArbSR_RCAN_x1_x4-DIV2K`.
- line 101: `model_weights_path` change to `./results/pretrained_models/ArbSR_RCAN_x1_x4-DIV2K-8c206342.pth.tar`.
-

```bash
python3 test.py
```

### Train ArbSR_RCAN model

modify the `config.py`

- line 31: `model_arch_name` change to `arbsr_rcan`.
- line 40: `upscale_factor` change to `4`.
- line 50: `mode` change to `train`.
- line 52: `exp_name` change to `ArbSR_RCAN_x1_x4-DIV2K`.

```bash
python3 train.py
```

### Resume train ArbSR_RCAN model

modify the `config.py`

- line 31: `model_arch_name` change to `arbsr_rcan`.
- line 40: `upscale_factor` change to `4`.
- line 50: `mode` change to `train`.
- line 52: `exp_name` change to `ArbSR_RCAN_x1_x4-DIV2K`.
- line 57: `resume_model_weights_path` change to `./results/ArbSR_RCAN_x1_x4-DIV2K/epoch_xxx.pth.tar`.

```bash
python3 train.py
```

## Result

Source of original paper
results: [https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_a_Single_Network_for_Scale-Arbitrary_Super-Resolution_ICCV_2021_paper.pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_a_Single_Network_for_Scale-Arbitrary_Super-Resolution_ICCV_2021_paper.pdf)

In the following table, the psnr value in `()` indicates the result of the project, and `-` indicates no test.

- None

```bash
# Download `ArbSR_RCAN_x1_x4-DIV2K-8c206342.pth.tar` weights to `./results/pretrained_models/ArbSR_RCAN_x1_x4-DIV2K-8c206342.pth.tar`
# More detail see `README.md<Download weights>`
python3 ./inference.py
```

Input:

<span align="center"><img width="480" height="312" src="figure/119082_lr.png"/></span>

Output:

<span align="center"><img width="480" height="312" src="figure/119082_sr.png"/></span>

```text
Build `arbsr_rcan` model successfully.
Load `arbsr_rcan` model weights `./results/pretrained_models/ArbSR_RCAN_x1_x4-DIV2K-8c206342.pth.tar` successfully.
SR image save to `./figure/119082_lr.png`
```

## Contributing

If you find a bug, create a GitHub issue, or even better, submit a pull request. Similarly, if you have questions,
simply post them as GitHub issues.

I look forward to seeing what the community does with these models!

## Credit

### Learning A Single Network for Scale-Arbitrary Super-Resolution

_Longguang Wang, Yingqian Wang, Zaiping Lin, Jungang Yang, Wei An, Yulan Guo_ <br>

**Abstract** <br>
Recently, the performance of single image super-resolution (SR) has been significantly improved with powerful networks.
However, these networks are developed for image SR with a single specific integer scale (e.g., x2;x3,x4), and cannot be
used for non-integer and asymmetric SR. In this paper, we propose to learn a scale-arbitrary image SR network from
scale-specific networks. Specifically, we propose a plug-in module for existing SR networks to perform scale-arbitrary
SR, which consists of multiple scale-aware feature adaption blocks and a scale-aware upsampling layer. Moreover, we
introduce a scale-aware knowledge transfer paradigm to transfer knowledge from scale-specific networks to the
scale-arbitrary network. Our plug-in module can be easily adapted to existing networks to achieve scale-arbitrary SR.
These networks plugged with our module can achieve promising results for non-integer and asymmetric SR while maintaining
state-of-the-art performance for SR with integer scale factors. Besides, the additional computational and memory cost of
our module is very small.

[[Paper]](https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Learning_a_Single_Network_for_Scale-Arbitrary_Super-Resolution_ICCV_2021_paper.pdf) [[Code]](https://github.com/The-Learning-And-Vision-Atelier-LAVA/ArbSR)

```bibtex
@InProceedings{Wang2020Learning,
  title={Learning A Single Network for Scale-Arbitrary Super-Resolution},
  author={Longguang Wang, Yingqian Wang, Zaiping Lin, Jungang Yang, Wei An, and Yulan Guo},
  booktitle={ICCV},
  year={2021}
}
```
