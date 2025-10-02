# UNETRppCMBSeg
Official implementation of **Enhancing Cerebral Microbleed Segmentation with Pretrained UNETR++**.

## Citation
If you find this code useful in your research, please consider citing:

```
@inproceedings{kwon2024unetrpp,
    author={Kwon, Junmo and Seo, Sang Won and Park, Hyunjin},
    title={Enhancing Cerebral Microbleed Segmentation with Pretrained UNETR++},
    booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
    pages={3372--3377},
    year={2024},
    doi={10.1109/BIBM62325.2024.10822393},
    organization={IEEE}
}
```

## Implementations

UNETR++ implementation: [unetr_pp_cmb.py](https://github.com/junmokwon/UNETRppCMBSeg/blob/main/unetrpp/unetr_pp_cmb.py)

UNETR++ pre-training: [UNETRppPretrain.py](https://github.com/junmokwon/UNETRppCMBSeg/blob/main/nnunet/training/network_training/UNETRppPretrain.py)

UNETR++ fine-tuning: [UNETRppFinetune.py](https://github.com/junmokwon/UNETRppCMBSeg/blob/main/nnunet/training/network_training/UNETRppFinetune.py)

Vision Transformer (ViT) implementation for SimMIM: [vit.py](https://github.com/junmokwon/UNETRppCMBSeg/blob/main/simmim/vit.py)

ViT pre-training: [ViTPretrain.py](https://github.com/junmokwon/UNETRppCMBSeg/blob/main/nnunet/training/network_training/ViTPretrain.py)

UNETR fine-tuning: [UNETRFinetune.py](https://github.com/junmokwon/UNETRppCMBSeg/blob/main/nnunet/training/network_training/UNETRFinetune.py)

## Simple Usage

For each pre-training script, there are class variants designed for masked image modeling (MIM) configurations. The naming convention is as follows:

- `PatchN` indicates a masked patch size of N×N.
- `RatioR` indicates a mask ratio of R/10.

For example, `UNETRppPretrainPatch32Ratio6` refers to pre-training UNETR++ with a patch size of 32×32 and a mask ratio of 0.6.

### Example Commands

- **Pre-train UNETR++** for task ID 401 and fold index 1 with patch size 32×32 and mask ratio 0.6:

```python
!nnUNet_train 2d UNETRppPretrainPatch32Ratio6 401 1
```

- **Fine-tune UNETR++** for task ID 401 and fold index 1:

```python
!nnUNet_train 2d UNETRppFinetunePatch32Ratio6 401 1
```

The naming convention is the same for pre-training ViT and fine-tuning UNETR.

- **Pre-train ViT** for task ID 402 and fold index 0 with a patch size 16×16 and mask ratio 0.5:

```python
!nnUNet_train 2d ViTPretrainPatch16Ratio5 402 0
```

- **Fine-tune UNETR** for task ID 402 and fold index 0:

```python
!nnUNet_train 2d UNETRFinetunePatch16Ratio5 402 0
```
