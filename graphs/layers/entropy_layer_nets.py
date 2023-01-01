import torch
from torch import nn
from torch.nn import functional as F

from compressai.entropy_models import EntropyBottleneck, GaussianConditional

from typing import Any, Callable, List, Optional, Tuple, Union    # following 3 lines for derived Lossless entropy mdls
from torch import Tensor
from compressai.ops import LowerBound   # to lowerbound GMM weights by zero


class EntropyBottleneckLossless(EntropyBottleneck):
    r""" Modified Compressai's EntropyBottleneck class so that it works also with lossless compression.
    Modifications :
    1. Since there is now quantizatin (i.e. latents are already discrete due to lifting), do not perform
    quantization i.e. rounding during VALIDATION, do not add noise during TRAINING.
    2. In actual ENTROPY CODING, the code of this class (and its parents) may be modified because they do stuff based
    on lossy coding and quantization (and subtract mean, round, add mean).
    """
    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        **kwargs: Any,
    ):
        super().__init__(channels=channels, *args, tail_mass=tail_mass, init_scale=init_scale, filters=filters, **kwargs)

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            # half = float(0.5)
            # noise = torch.empty_like(inputs).uniform_(-half, half)
            # inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        # if means is not None:
        #     outputs -= means
        #
        # outputs = torch.round(outputs)

        if mode == "dequantize":
            # if means is not None:
            #     outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs


class GaussianConditionalLossless(GaussianConditional):
    r""" Modified Compressai's GaussianConditional class so that it works also with lossless compression.
    Modifications :
    1. Since there is now quantizatin (i.e. latents are already discrete due to lifting), do not perform
    quantization i.e. rounding during VALIDATION, do not add noise during TRAINING.
    2. In actual ENTROPY CODING, the code of this class (and its parents) may be modified because they do stuff based
    on lossy coding and quantization (and subtract mean, round, add mean).
    """
    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.11 / 255.0,
        tail_mass: float = 1e-9 / 255.0,
        **kwargs: Any,
    ):
        super().__init__(scale_table=scale_table, *args, scale_bound=scale_bound, tail_mass=tail_mass, **kwargs)

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            # half = float(0.5)
            # noise = torch.empty_like(inputs).uniform_(-half, half)
            # inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        # if means is not None:
        #     outputs -= means
        #
        # outputs = torch.round(outputs)

        if mode == "dequantize":
            # if means is not None:
            #     outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Optional[Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        # outputs = self.quantize(inputs, "noise" if training else "dequantize", means)
        outputs = inputs
        # likelihood = self._likelihood(outputs, scales, means)
        likelihood = self._likelihood_fk(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def _likelihood_fk(
        self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        # half = float(0.5)
        half = float(0.5 / 255.0)  # take 255.0 as an input ?

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower

        return likelihood


class GaussianConditionalLosslessGMM(GaussianConditionalLossless):
    r""" GMM version
    """
    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.11 / 255.0,
        tail_mass: float = 1e-9 / 255.0,
        num_mix: int,  # X
        num_chn: int,  # M
        **kwargs: Any,
    ):
        super().__init__(scale_table=scale_table, *args, scale_bound=scale_bound, tail_mass=tail_mass, **kwargs)
        self.X = num_mix
        self.M = num_chn
        self.lower_bound_weights = LowerBound(1e-6)

    def forward(
        self,
        inputs: Tensor,                  # B x M  x H x W
        weights: Tensor,                 # B x MX x H x W   # mixture weights
        scales: Tensor,                  # B x MX x H x W
        means: Optional[Tensor] = None,  # B x MX x H x W
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        B, M, H, W = inputs.shape
        if training is None:
            training = self.training
        # outputs = self.quantize(inputs, "noise" if training else "dequantize", means)
        outputs = inputs
        # likelihood = self._likelihood(outputs.permute(0, 2, 3, 1).repeat_interleave(self.X, dim=3), scales.permute(0, 2, 3, 1), means.permute(0, 2, 3, 1))
        likelihood_mixs = self._likelihood_fk(outputs.permute(0, 2, 3, 1).repeat_interleave(self.X, dim=3),
                                         scales.permute(0, 2, 3, 1),
                                         means.permute(0, 2, 3, 1))
        # weights = F.softmax(weights.permute(0, 2, 3, 1).view(B, H, W, self.M, self.X), dim=4)
        weights = self.lower_bound_weights(weights.permute(0, 2, 3, 1).view(B, H, W, self.M, self.X))
        weights = weights / torch.sum(weights, dim=4, keepdim=True)
        likelihood = torch.sum(weights * likelihood_mixs.view(B, H, W, self.M, self.X), dim=4).permute(0, 3, 1, 2)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def get_cdfs(
        self,
        cdf_sampling_pts,                # 257 or 512 (1-D tensor)
        weights: Tensor,                 # B x MX x H x W   # mixture weights
        scales: Tensor,                  # B x MX x H x W
        means: Optional[Tensor] = None   # B x MX x H x W
    ) -> Tuple[Tensor, Tensor]:
        P = cdf_sampling_pts.shape[0]  # 257 or 512 depending on Y or Co, Cg channel
        B, MX, H, W = means.shape
        X = self.X
        M = MX // X

        scales = self.lower_bound_scale(scales)
        # weights = F.softmax(weights.permute(0, 2, 3, 1).view(B, H, W, self.M, self.X), dim=4)
        weights = self.lower_bound_weights(weights.permute(0, 2, 3, 1).view(B, H, W, M, X))
        weights = weights / (1e-9 + torch.sum(weights, dim=4, keepdim=True))   # weights = weights / torch.sum(weights, dim=4, keepdim=True)

        cdfs_mixs = self._standardized_cumulative((cdf_sampling_pts - means.unsqueeze(dim=4)) / scales.unsqueeze(dim=4))  # B x MX x H x W x 257
        cdfs = torch.sum(weights.unsqueeze(dim=5) * cdfs_mixs.permute(0, 2, 3, 1, 4).view(B, H, W, M, X, P), dim=4).permute(0, 3, 1, 2, 4)
        return cdfs  # B x M x H x W x 257/512

    # Note: not tested !
    def get_cdfs_lessmemory(  # iterates over each mixture component with for loop, decreases mem usage by #mixs
        self,
        weights: Tensor,                 # B x MX x H x W   # mixture weights
        scales: Tensor,                  # B x MX x H x W
        means: Optional[Tensor] = None,  # B x MX x H x W
    ) -> Tuple[Tensor, Tensor]:
        B, _, H, W = means.shape
        cdf_sampling_pts = torch.linspace(-0.5, 255.5, steps=257, device=means.device) / 255.0
        cdf_sampling_pts[0] = -20 / 255.0
        cdf_sampling_pts[-1] = 275 / 255.0

        scales = self.lower_bound_scale(scales)
        # weights = F.softmax(weights.permute(0, 2, 3, 1).view(B, H, W, self.M, self.X), dim=4)
        weights = self.lower_bound_weights(weights.permute(0, 2, 3, 1).view(B, H, W, self.M, self.X))
        weights = weights / torch.sum(weights, dim=4, keepdim=True)

        for x in range(self.X):  # loop to find each mixture's cdf and weighted-add the cdfs as you loop to get finalcdf
            cdfs_1mix = self._standardized_cumulative((cdf_sampling_pts - means[:, x::self.X, :, :].unsqueeze(dim=4)) /
                                                      scales[:, x::self.X, :, :].unsqueeze(dim=4))  # B x M x H x W x 257
            cdfs_1mix_wgtd = weights[:, :, :, :, x:x].unsqueeze(dim=5) * cdfs_1mix.permute(0, 2, 3, 1, 4).view(B, H, W, self.M, 1, 257)
            if x == 0:
                cdfs_accu = cdfs_1mix_wgtd
            else:
                cdfs_accu += cdfs_1mix_wgtd
        return cdfs_accu.view(B, H, W, self.M, 257).permute(0, 3, 1, 2, 4)  # B x M x H x W x 257


class LogisticConditionalLossless(GaussianConditionalLossless):
    r""" This distribution does not decay as fast as the Normal. Used in PixelCNN++ etc.
    """
    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.04, # Note, need different bound than Normal since this decays slower than Normal
        tail_mass: float = 1e-9,
        **kwargs: Any,
    ):
        super().__init__(scale_table=scale_table, *args, scale_bound=scale_bound, tail_mass=tail_mass, **kwargs)
        # get sigmoid function
        self.sig = nn.Sigmoid()

    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Optional[Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        # outputs = self.quantize(inputs, "noise" if training else "dequantize", means)
        outputs = inputs
        likelihood = self._likelihood_logistic(outputs, scales, means)
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)
        return outputs, likelihood

    def _likelihood_logistic(
            self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        # half = float(0.5)
        half = float(0.5 / 255.0)

        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = self.lower_bound_scale(scales)

        upper = self.sig((values + half) / scales)
        lower = self.sig((values - half) / scales)
        likelihood = upper - lower

        return likelihood
