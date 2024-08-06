import numpy as np


class MaskGenerator:
    """
    MaskGenerator based on: "Xie et al.,
    SimMIM: A Simple Framework for Masked Image Modeling"
    """
    def __init__(self, height=384, width=320, mask_patch_size=64, model_patch_size=32, mask_ratio=0.6):
        self.height = height
        self.width = width
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.height % self.mask_patch_size == 0
        assert self.width % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_height = self.height // self.mask_patch_size
        self.rand_width = self.width // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_width * self.rand_height
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_height, self.rand_width))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask
