from torch import nn
from typing import Tuple, Union
from torch.nn.functional import interpolate
from nnunet.network_architecture.neural_network import SegmentationNetwork
from unetrpp.dynunet_block import UnetOutBlock, UnetResBlock
from unetrpp.model_components import UnetrPPEncoder, UnetrUpBlock


class UNETR_PP(SegmentationNetwork):
    """
    UNETR++ based on: "Shaker et al.,
    UNETR++: Delving into Efficient and Accurate 3D Medical Image Segmentation"
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            feature_size: int = 16,
            hidden_size: int = 256,
            num_heads: int = 4,
            pos_embed: str = "perceptron",
            norm_name: Union[Tuple, str] = "instance",
            dropout_rate: float = 0.0,
            depths=None,
            dims=None,
            conv_op=nn.Conv2d,
            do_ds=True,
            do_pretrain=False,
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimensions of  the last encoder.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            dropout_rate: faction of the input units to drop.
            depths: number of blocks for each stage.
            dims: number of channel maps for the stages.
            conv_op: type of convolution operation.
            do_ds: use deep supervision to compute the loss.
        """

        super().__init__()
        if depths is None:
            depths = [3, 3, 3, 3]
        self.do_ds = do_ds
        self.do_pretrain = do_pretrain
        self.conv_op = conv_op
        self.num_classes = out_channels
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.feat_size = (12, 10,)
        self.hidden_size = hidden_size

        self.unetr_pp_encoder = UnetrPPEncoder(dims=dims, depths=depths, num_heads=num_heads)

        self.encoder1 = UnetResBlock(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 16,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=24*20,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=48*40,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            out_size=96*80,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=2,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=(4, 4),
            norm_name=norm_name,
            out_size=384*320,
            conv_decoder=True,
        )
        # Segmentation head
        self.out_seg1 = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=out_channels)
        if self.do_ds:
            self.out_seg2 = UnetOutBlock(spatial_dims=2, in_channels=feature_size * 2, out_channels=out_channels)
            self.out_seg3 = UnetOutBlock(spatial_dims=2, in_channels=feature_size * 4, out_channels=out_channels)
        # Reconstruction head
        self.out_rec1 = UnetOutBlock(spatial_dims=2, in_channels=feature_size, out_channels=in_channels)
        if self.do_ds:
            self.out_rec2 = UnetOutBlock(spatial_dims=2, in_channels=feature_size * 2, out_channels=in_channels)
            self.out_rec3 = UnetOutBlock(spatial_dims=2, in_channels=feature_size * 4, out_channels=in_channels)

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x_in, mask=None):
        if self.do_pretrain:
            x_output, hidden_states = self.unetr_pp_encoder(x_in, mask)
        else:
            x_output, hidden_states = self.unetr_pp_encoder(x_in)

        convBlock = self.encoder1(x_in)

        # Four encoders
        enc1 = hidden_states[0]
        enc2 = hidden_states[1]
        enc3 = hidden_states[2]
        enc4 = hidden_states[3]

        # Four decoders
        dec4 = self.proj_feat(enc4, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc3)
        dec2 = self.decoder4(dec3, enc2)
        dec1 = self.decoder3(dec2, enc1)

        out = self.decoder2(dec1, convBlock)
        if self.do_pretrain:
            logit = self.out_rec1(out)
            if self.do_ds:
                logits = [
                    logit, 
                    interpolate(self.out_rec2(dec1), logit.shape[2:]),
                    interpolate(self.out_rec3(dec2), logit.shape[2:])
                ]
            else:
                logits = logit
        else:
            if self.do_ds:
                logits = [self.out_seg1(out), self.out_seg2(dec1), self.out_seg3(dec2)]
            else:
                logits = self.out_seg1(out)

        return logits
