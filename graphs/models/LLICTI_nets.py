import torch
from torch import nn
from torch.nn import functional as F

from compressai.entropy_models import EntropyBottleneck
from graphs.layers.entropy_layer_nets import EntropyBottleneckLossless, GaussianConditionalLossless, \
    GaussianConditionalLosslessGMM, LogisticConditionalLossless
from compressai.layers import GDN1
import torchac
import numpy as np


class LLICTIBaseNet(nn.Module):
    """ (L)earned (L)ossless (I)mage (C)ompression (T)hrough (I)nterpolation Net."""
    def __init__(self, config):
        """ Set up sizes/lengths for models, inputs etc """
        super(LLICTIBaseNet, self).__init__()
        # YCoCg-R reversible lifting based color transform related..
        self.ycocg = config.ycocg
        self.clrchs = config.clrchs
        self.clrjnt = config.clr_joint_mode  # 0: all clr channels indep. 1: ch0 indep, ch12 joint with pixcnn++ method  2: ch012 joint with pixcnn++ method
        # self.clrseq = config.clrchs_process_sequentially
        # lifting precision bits; also used in quantization or rounding operations
        self.precision_bits = config.lif_prec_bits
        self.RNDFACTOR = 255 * (2 ** (self.precision_bits - 8))
        self.mean_y_ycocg = ((2 ** (self.precision_bits - 1)) - 1) / ((2 ** self.precision_bits) - 1)  # i.e. subtract out - 127/255

    def display(self, x):
        pass

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def get_YCoCg_R_from_RGB(self, x):
        assert x.shape[1] == 3  # ensure x has 3 color channels
        R, G, B = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        # apply lifting based equations in JVT-I014r3
        Co = R - B
        t = B + torch.round(Co * self.RNDFACTOR / 2) / self.RNDFACTOR
        Cg = G - t
        Y = t + torch.round(Cg * self.RNDFACTOR / 2) / self.RNDFACTOR
        YCoCg = torch.cat((Y, Co, Cg), dim=1)
        return YCoCg

    def get_RGB_from_YCoCg_R(self, x):
        assert x.shape[1] == 3  # ensure x has 3 color channels
        Y, Co, Cg = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        t = Y - torch.round(Cg * self.RNDFACTOR / 2) / self.RNDFACTOR
        G = Cg + t
        B = t - torch.round(Co * self.RNDFACTOR / 2) / self.RNDFACTOR
        R = B + Co
        RGB = torch.cat((R, G, B), dim=1)
        return RGB

    @staticmethod
    def get_YCoCg_R_from_RGB__intOps(x):
        assert x.shape[1] == 3  # ensure x has 3 color channels
        R, G, B = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        R, G, B = (R*255).round().to(torch.int16), (G*255).round().to(torch.int16), (B*255).round().to(torch.int16),
        # apply lifting based equations in JVT-I014r3
        Co = R - B
        # t = B + torch.round(Co / 2).to(torch.int16)
        t = B + Co // 2
        Cg = G - t
        # Y = t + torch.round(Cg / 2).to(torch.int16)
        Y = t + Cg // 2
        YCoCg = torch.cat((Y, Co, Cg), dim=1)
        return YCoCg

    @staticmethod
    def get_RGB_from_YCoCg_R__intOps(x):
        assert x.shape[1] == 3  # ensure x has 3 color channels
        Y, Co, Cg = x[:, 0:1, :, :], x[:, 1:2, :, :], x[:, 2:3, :, :]
        Y, Co, Cg = (Y*255).round().to(torch.int16), (Co*255).round().to(torch.int16), (Cg*255).round().to(torch.int16),
        # t = Y - torch.round(Cg / 2).to(torch.int16)
        t = Y - Cg // 2
        G = Cg + t
        # B = t - torch.round(Co / 2).to(torch.int16)
        B = t - Co // 2
        R = B + Co
        RGB = torch.cat((R, G, B), dim=1) / 255
        return RGB


class LLICTI(LLICTIBaseNet):
    """ Auto-encoder based on : LAZY Discrete Wavelet Transform """
    def __init__(self, config):
        super(LLICTI, self).__init__(config)
        # Set up sizes/lengths for models, inputs etc
        self.list_scales = config.dwtlevels
        self.num_scales = len(self.list_scales)
        # Entropy Layer
        self.entropymodel = LLICTIEntropyLayer(config)

    def forward(self, x):
        """
        See notes if can not easily understand architecture
        :param x: original images/patches  # B x C x H x W
        :return: reconstructed image, -log2(quantized latent tensor probability)
        """
        # convert to YCoCg-R
        if self.ycocg:
            x = self.get_YCoCg_R_from_RGB(x)
            x[:, 0, :, :] = x[:, 0, :, :] - self.mean_y_ycocg  # make mean of Y channel zero mean like Co, Cg channels, i.e. subtract out - 127/255
        else:
            x[:, :, :, :] = x[:, :, :, :] - self.mean_y_ycocg  # make mean of R,G and B channel zero mean, i.e. subtract out - 127/255
        # lazy DWT
        if self.clrchs == 3:
            y_list = self.lazyDWT(x, levels=self.list_scales, clrchs=self.clrchs, clrjnt=self.clrjnt)
        elif self.clrchs < 3:  # 0, 1, 2
            y_list = self.lazyDWT(x, levels=self.num_scales, clrchs=self.clrchs)
        else:
            y_list = []
            pass
        # entropy
        self_informations_y_list = self.entropymodel.forward(y_list)
        return self_informations_y_list

    def compress(self, x):
        """
        See notes if can not easily understand architecture
        :param x: original images/patches  # B x C x H x W  (comes directly from dataloader which performs int->flt32/255)
        :return: byte_stream
        """
        # get last level/scales x00 band in RGB; these will be coded in header with FLC
        x00_lastlevel_RGB_uint8 = self._get_lastlevel_x00_flt_to_uint8(x, max(self.list_scales))
        # convert to YCoCg-R
        if self.ycocg:
            x = self.get_YCoCg_R_from_RGB__intOps(x)
            # --- find/analyze min and max values of color channels for image ---
            minY, minCo, minCg =   0, x[:, 1, :, :].min().item(), x[:, 2, :, :].min().item()
            maxY, maxCo, maxCg = 255, x[:, 1, :, :].max().item(), x[:, 2, :, :].max().item()
            minYCoCg_maxYCoCg_list = [minY, minCo, minCg,  maxY, maxCo, maxCg]
            # print("{:4d} {:4d} {:4d}   {:4d} {:4d} {:4d}  ".format(minY, minCo, minCg, maxY, maxCo, maxCg), end='')
            # print("   #Values: {:4d} {:4d} {:4d}".format(maxY-minY+1, maxCo-minCo+1, maxCg-minCg+1))
            #  --- find/analyze min and max values of color channels for image ---
            x[:, 0, :, :] = x[:, 0, :, :] - 127  # make mean of Y ch zero like Co Cg chs, i.e. subtract out - 127/255
            x = x / 255
        else:
            minYCoCg_maxYCoCg_list = [0, 0, 0,  255, 255, 255]
            x[:, :, :, :] = x[:, :, :, :] - self.mean_y_ycocg  # make mean of R,G,B chs zero, i.e. subtract - 127/255
        # lazy DWT
        if self.clrchs == 3:
            y_list, padHW_lev, padHW_int = self.lazyDWT(x, levels=self.list_scales, clrchs=self.clrchs, clrjnt=self.clrjnt, pad=True)
        elif self.clrchs < 3:  # 0, 1, 2
            y_list = self.lazyDWT(x, levels=self.num_scales, clrchs=self.clrchs)
            padHW_lev, padHW_int = [], 0
        else:
            y_list = []
            padHW_lev, padHW_int = [], 0
        # entropy coding
        bytestream_list = self.entropymodel.compress(y_list, x00_lastlevel_RGB_uint8, minYCoCg_maxYCoCg_list, padHW_lev, padHW_int)
        return bytestream_list, x

    def decompres(self, bytestream_list, devc, xorg=None):
        """
        :return:
        """
        # decode and obtain y_list
        y_3ch = self.entropymodel.decompress(bytestream_list, devc=devc)
        # check if decoded pixels (not converted to RGB yet) match with the original pixels
        if xorg is not None:
            maxx_abserr = ((xorg - y_3ch) * 255).abs().max()
            if maxx_abserr >= 1.0:
                print(maxx_abserr)
        # convert back to RGB
        if self.ycocg:
            y_3ch[:, 0, :, :] = y_3ch[:, 0, :, :] + self.mean_y_ycocg
            y_3ch_RGB = self.get_RGB_from_YCoCg_R__intOps(y_3ch)
            return y_3ch_RGB
        else:
            y_3ch[:, :, :, :] = y_3ch[:, :, :, :] + self.mean_y_ycocg
            return y_3ch

    @staticmethod
    def lazyDWT(x, levels, clrchs=3, clrjnt=0, pad=False):
        """ Return lazy DWT coefs in a list"""
        y_list = []
        padHW_lev = []  # list to hold padding info for H and W for each level/scale
        padHW_int = 0  # integer to have H and W for each level/scale in its bits. Will be written into header...
        # modify input (ddd channel of 0's to) input if clrjnt is 1
        if clrjnt in [0, 2]:
            x_ = x
        elif clrjnt == 1:
            zrs = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
            x_ = torch.cat((zrs, x), dim=1)  # make channels  0 Y Co Cg. will be processed with groups=2
        else:
            x_ = None
        # extract scales/levels from input and form a list containing each scale's data
        if clrchs != 3:  # old version
            for lev in range(levels):
                # get the 4 sub pictures _ij from even/odd samples
                if clrchs == 3:
                    x_00 = x_[:, :, 0::2, 0::2]
                    x_01 = x_[:, :, 0::2, 1::2]
                    x_10 = x_[:, :, 1::2, 0::2]
                    x_11 = x_[:, :, 1::2, 1::2]
                else:  # take only 0th, 1st or 2nd channel for now
                    if lev == 0:
                        x_00 = x_[:, clrchs:clrchs+1, 0::2, 0::2]
                        x_01 = x_[:, clrchs:clrchs+1, 0::2, 1::2]
                        x_10 = x_[:, clrchs:clrchs+1, 1::2, 0::2]
                        x_11 = x_[:, clrchs:clrchs+1, 1::2, 1::2]
                    else:
                        x_00 = x_[:, 0:1, 0::2, 0::2]
                        x_01 = x_[:, 0:1, 0::2, 1::2]
                        x_10 = x_[:, 0:1, 1::2, 0::2]
                        x_11 = x_[:, 0:1, 1::2, 1::2]
                y_list.append(torch.cat((x_00, x_11, x_01, x_10), dim=1))
                x_ = x_00
        else:  # clrchs==3  new version. works for arbitrarty levels or dwtlevels
            for lev in range(0, max(levels)+1):
                if lev in levels:
                    st = 2**(lev+1)
                    of = st // 2
                    x_00 = x_[:, :,  0::st,  0::st]
                    x_01 = x_[:, :,  0::st, of::st]
                    x_10 = x_[:, :, of::st,  0::st]
                    x_11 = x_[:, :, of::st, of::st]
                    if pad:
                        padH = True if x_00.shape[2] > x_11.shape[2] else False
                        padW = True if x_00.shape[3] > x_11.shape[3] else False
                        padHW_lev.append([padH, padW])
                        padHW_int = 4 * padHW_int + 2 * padH + 1 * padW
                        if padH and padW:
                            x_01 = F.pad(x_01, (0, 1, 0, 0), mode='replicate')  # (padding_left,padding_right, padding_top,padding_bottom)
                            x_10 = F.pad(x_10, (0, 0, 0, 1), mode='replicate')
                            x_11 = F.pad(x_11, (0, 1, 0, 1), mode='replicate')
                        elif padW:
                            x_01 = F.pad(x_01, (0, 1, 0, 0), mode='replicate')
                            x_11 = F.pad(x_11, (0, 1, 0, 0), mode='replicate')
                        elif padH:
                            x_10 = F.pad(x_10, (0, 0, 0, 1), mode='replicate')
                            x_11 = F.pad(x_11, (0, 0, 0, 1), mode='replicate')
                    y_list.append(torch.cat((x_00, x_11, x_01, x_10), dim=1))
        if not pad:
            return y_list
        else:
            return y_list, padHW_lev, padHW_int

    @staticmethod
    def _get_lastlevel_x00_flt_to_uint8(x, lastlevel):
        st = 2**(lastlevel+1)
        of = st // 2
        x_00 = x[:, :,  0::st,  0::st]
        return (x_00*255).round().to(torch.uint8)


class LLICTIEntropyLayer(nn.Module):
    """  Entropy Layer for LLICTI """
    def __init__(self, config):
        super(LLICTIEntropyLayer, self).__init__()
        # Set up sizes/lengths for models, inputs etc
        self.list_scales = config.dwtlevels
        self.num_scales = len(self.list_scales)
        self.useprevlevNN = config.useprevlevNN  # whther to use previous levels NN also for this level
        self.ent_mdl_num = config.ent_mdl_num
        self.clrchs = config.clrchs
        self.clrjnt = config.clr_joint_mode  # 0: all clr ch indep 1: ch0 indep, ch12 joint with pixcnn++ method  2: ch012 joint with pixcnn++ method
        # self.clrseq = config.clrchs_process_sequentially
        self.combine_layers1toL = config.combine_layers1toL
        # lifting precision bits; also used in quantization or rounding operations, color space and mean shifts
        self.precision_bits = config.lif_prec_bits
        self.RNDFACTOR = 255 * (2 ** (self.precision_bits - 8))
        self.mean_y_ycocg = ((2 ** (self.precision_bits - 1)) - 1) / ((2 ** self.precision_bits) - 1)  # i.e. subtract out - 127/255
        self.ycocg = config.ycocg
        c = 3 if (self.clrchs == 3 and self.clrjnt in [0, 2]) else 4 if (self.clrchs == 3 and self.clrjnt == 1) else 1
        self.mean_shifts_ycocg = torch.zeros(1, 4 * c, 1, 1)  #, device=devc)  # entire model is moved to gpu anyway, so this should also be moved ?
        self.mean_shifts_ycocg[0, :, 0, 0] = torch.tensor([self.mean_y_ycocg, 1.0, 1.0]).repeat(4)
        self.mean_shifts_ycocg_int = torch.zeros(1, 4 * c, 1, 1, dtype=torch.int16)  #, device=devc)  # entire model is moved to gpu anyway, so this should also be moved ?
        self.mean_shifts_ycocg_int[0, :, 0, 0] = torch.tensor([127, 255, 255], dtype=torch.int16).repeat(4)
        self.num_mixtures = config.num_mixtures
        # ent models
        self.entmdls_scale_band = nn.ModuleList()
        scl_no = 0
        for scl in self.list_scales:  # for scl in range(self.num_scales):
            if scl_no > 0 and self.useprevlevNN[scl_no] is True:
                scl_no += 1
                continue
            entmdls_band = nn.ModuleList()
            for b in range(4 - 1):
                if self.ent_mdl_num == 0:
                    pass
                elif self.ent_mdl_num == 3:
                    pass
                elif self.ent_mdl_num == 4:
                    ev = config.Evens[scl]
                    od = config.Odds[scl]
                    if self.clrchs == 3:
                        if self.clrjnt == 2:
                            chs = config.chs
                        elif self.clrjnt == 0:
                            chs = config.chs
                        else:  # 1
                            chs = config.chs
                    else:  # i.e. 0,1,2
                        chs = [48, 32, 24, 24]  # for Y ch
                        if self.clrchs in [1, 2]:
                            chs = [int(i*0.75) for i in chs]  # for Co Cg chs
                    ch = chs[scl]
                    ly = config.conv_layers  # 1+3  # 1+4  # 1+3
                    if not self.combine_layers1toL:
                        entmdls_band.append(LLICTIEntropyModel4(scale=scl, band=b, config=config,
                                                                Ev=ev, Od=od, Ch=ch, Ly=ly))
                    else:
                        entmdls_band.append(LLICTIEntropyModel4(scale=scl, band=-1, config=config,
                                                                Ev=ev, Od=od, Ch=ch, Ly=ly))
                        break  # don't call for other bands !!!
            self.entmdls_scale_band.append(entmdls_band)
            scl_no += 1

    def forward(self, y_list):
        # opmode = 'train'
        # c = 3 if self.clrchs == 3 else 1
        c = 3 if (self.clrchs == 3 and self.clrjnt in [0, 2]) else 4 if (self.clrchs == 3 and self.clrjnt == 1) else 1
        self_informations_y_list = []
        scl_no, mdl_scl = 0, 0
        for scl in range(self.num_scales):  # no need to change this now we use self.list_scales
            y_list_lev = y_list[scl]
            if scl_no > 0 and self.useprevlevNN[scl_no] is False:
                mdl_scl = mdl_scl + 1
            b = 0
            bb = 0 if self.combine_layers1toL else 0
            self_infos_b0 = self.entmdls_scale_band[mdl_scl][bb].forward(y_list_lev[:, 0:c*(b+1), :, :],
                                                                         y_list_lev[:, c*(b+1):c*(b+2), :, :])
            b = 1
            bb = 0 if self.combine_layers1toL else 1
            self_infos_b1 = self.entmdls_scale_band[mdl_scl][bb].forward(y_list_lev[:, 0:c*(b+1), :, :],
                                                                         y_list_lev[:, c*(b+1):c*(b+2), :, :])
            b = 2
            bb = 0 if self.combine_layers1toL else 2
            self_infos_b2 = self.entmdls_scale_band[mdl_scl][bb].forward(y_list_lev[:, 0:c*(b+1), :, :],
                                                                         y_list_lev[:, c*(b+1):c*(b+2), :, :])
            self_informations_y_list.append(torch.cat((self_infos_b0, self_infos_b1, self_infos_b2), dim=1))
            scl_no += 1
        return self_informations_y_list

    def compress(self, y_list, x00_lastlevel_RGB_uint8, minYCoCg_maxYCoCg_list, padHW_lev, padHW_int):
        # First, encode header info and the last level's x00 band in RGB with FLC (8 bpp)
        lastscale_hgt, lastscale_wdt = y_list[self.num_scales-1].shape[2], y_list[self.num_scales-1].shape[3]
        header_bytestream = bytes(np.array([self.num_scales, lastscale_hgt, lastscale_wdt], dtype=np.uint8))
        header_minmaxYCoCg_bytestream = bytes(np.array(minYCoCg_maxYCoCg_list, dtype=np.int16))
        header_padHW_bytestream = bytes(np.array([padHW_int], dtype=np.int16))
        lastscale_initband_bytestream = bytes(x00_lastlevel_RGB_uint8.to('cpu', non_blocking=True).numpy())
        em = b''  # empty_bytestream
        byte_stream_list = [[header_bytestream, header_minmaxYCoCg_bytestream, header_padHW_bytestream,
                             lastscale_initband_bytestream, em, em,
                             em, em, em]]
        # init some parameters
        c = 3 if (self.clrchs == 3 and self.clrjnt in [0, 2]) else 4 if (self.clrchs == 3 and self.clrjnt == 1) else 1
        M = self.num_mixtures
        self._adjust_mean_shifts(minYCoCg_maxYCoCg_list)
        # adjust mdl_scl
        scl_no, mdl_scl = self.num_scales-1, self.num_scales-1
        while mdl_scl > 0 and self.useprevlevNN[mdl_scl] is True:
            mdl_scl = mdl_scl - 1
        # loop over scales/levels
        for scl in range(self.num_scales-1, -1, -1):  # code in reverse order since that must be the decoding order
            y_list_lev = y_list[scl]
            # convert pixels back to integers in 0-255 and move to cpu
            y_list_lev_int16_cpu = self._convert_float32gpu_to_int16cpu(c=c, tensor_flt32_gpu=y_list_lev, devc=y_list_lev.device)
            # will hold 9 bytestreams, each for one band and clrchannel
            byte_stream_list_lev = []
            # get padding info for this scale
            padH, padW = padHW_lev[scl][0], padHW_lev[scl][1]
            # update mdl_scl so that the correct model is used for each scale
            if scl_no < mdl_scl:
                mdl_scl -= mdl_scl
                while mdl_scl > 0 and self.useprevlevNN[mdl_scl] is True:
                    mdl_scl = mdl_scl - 1
            # loop over each subband (0,1,2): get cdfs and loop over clr channels to encode each
            for b in range(3):
                bb = 0 if self.combine_layers1toL else b
                params = self.entmdls_scale_band[mdl_scl][bb].get_params(y_list_lev[:, 0:c*(b+1), :, :])
                aw, bw, dw = params[:, 9 * M:10 * M, :, :], params[:, 10 * M:11 * M, :, :], params[:, 11 * M:12 * M, :, :]
                # encode y,co,cg channels sequentially by getting their cdfs also sequentially
                for clr in range(c):
                    # get stdevs, means, weights
                    stdevs  = params[:, (0*c+clr)*M:(0*c+clr+1)*M, :, :]
                    means   = params[:, (1*c+clr)*M:(1*c+clr+1)*M, :, :]
                    weights = params[:, (2*c+clr)*M:(2*c+clr+1)*M, :, :]
                    # update means of this color channel from previous color channels
                    if clr == 1:  # update means of this color channel from previous color channels
                        means += aw * y_list_lev[:, c*(b+1)+0:c*(b+1)+1, :, :]
                    elif clr == 2:
                        means += bw * y_list_lev[:, c*(b+1)+0:c*(b+1)+1, :, :] + dw * y_list_lev[:, c*(b+1)+1:c*(b+1)+2, :, :]
                    # get cdf of this clr channel and encode this color channel
                    minsamp = -127 if clr == 0 else minYCoCg_maxYCoCg_list[clr]
                    maxsamp = 128 if clr == 0 else minYCoCg_maxYCoCg_list[3 + clr]
                    crpdH = y_list_lev_int16_cpu.shape[2] - padH if b in [0, 2] else y_list_lev_int16_cpu.shape[2]
                    crpdW = y_list_lev_int16_cpu.shape[3] - padW if b in [0, 1] else y_list_lev_int16_cpu.shape[3]
                    if False:  # with encode_float_cdf
                        cdfs_b_clr = self.entmdls_scale_band[mdl_scl][bb].get_cdfs(stdevs, means, weights, clrch=clr, int_cdf=False, minVal=minsamp, maxVal=maxsamp)
                        byte_stream_b_clr = torchac.encode_float_cdf(cdfs_b_clr[:, :, :crpdH, :crpdW].to('cpu').clip(min=0.0, max=1.0),
                                                                     y_list_lev_int16_cpu[:, c*(b+1)+clr:c*(b+1)+clr+1, :crpdH, :crpdW],
                                                                     needs_normalization=True,
                                                                     check_input_bounds=False)  # check_input_bounds=True
                    else:  # with encode_int16_normalized_cdf
                        cdfs_b_clr = self.entmdls_scale_band[mdl_scl][bb].get_cdfs(stdevs, means, weights, clrch=clr, int_cdf=True, minVal=minsamp, maxVal=maxsamp)
                        byte_stream_b_clr = torchac.encode_int16_normalized_cdf(cdfs_b_clr[:, :, :crpdH, :crpdW].to('cpu'),
                                                                                y_list_lev_int16_cpu[:, c*(b+1)+clr:c*(b+1)+clr+1, :crpdH, :crpdW])
                    # append byte_stream_b_clr to bitstream list of this scale/lev
                    byte_stream_list_lev.append(byte_stream_b_clr)
            # combine list of bytestreams into another list of scales' bytestream
            byte_stream_list.append(byte_stream_list_lev)
            scl_no -= 1
        return byte_stream_list

    def decompress(self, bytestream_list, devc):
        """
        :return:
        """
        # First, decode header info and the last scales/levels band 0 with FLC
        header_bytestream, header_minmaxYCoCg_bytestream, header_padHW_bytestream = bytestream_list[0][0], bytestream_list[0][1], bytestream_list[0][2]
        lastscale_initband_bytestream = bytestream_list[0][3]  # (8 bpp for RGB)
        # self_num_scales, lastscale_hgt, lastscale_wdt = np.frombuffer(header_bytestream, dtype=np.int16)
        self_num_scales, lastscale_hgt, lastscale_wdt = np.frombuffer(header_bytestream, dtype=np.uint8)
        assert self.num_scales == self_num_scales
        minYCoCg_maxYCoCg_list = np.frombuffer(header_minmaxYCoCg_bytestream, dtype=np.int16)
        self._adjust_mean_shifts(minYCoCg_maxYCoCg_list)
        padHW_int = np.frombuffer(header_padHW_bytestream, dtype=np.int16)
        padHW_lev = self._get_padHW_lev_list(padHW_int, self_num_scales)
        lastscale_initband_uint8_cpu = torch.from_numpy(np.frombuffer(lastscale_initband_bytestream, dtype=np.uint8)).view(1, 3, lastscale_hgt, lastscale_wdt)  # to get back tensor from bytestream
        lastscale_initband_int16_cpu = self.get_int16YCoCg_R_from_uint8RGB__intOps(lastscale_initband_uint8_cpu)  # convert from uint8 RGB to YCoCg int16
        len_bytestream_list = len(bytestream_list)
        # init some parameters
        c = 3 if (self.clrchs == 3 and self.clrjnt in [0, 2]) else 4 if (self.clrchs == 3 and self.clrjnt == 1) else 1
        M = self.num_mixtures
        # adjust mdl_scl
        scl_no, mdl_scl = self.num_scales-1, self.num_scales-1
        while mdl_scl > 0 and self.useprevlevNN[mdl_scl] is True:
            mdl_scl = mdl_scl - 1
        # loop over scales/levels
        for scl in range(self.num_scales-1, -1, -1):  # code in reverse order since that must be the decoding order
            # prepare tensor y_list_lev and assign to it the initial band
            if scl == self.num_scales-1:  # decode from bitstream : the initial band for this scale
                y_list_lev = torch.zeros(1, 4*c, lastscale_hgt, lastscale_wdt, device=devc)
                y_list_lev[:, 0:c, :, :] = (lastscale_initband_int16_cpu.to(devc) - torch.tensor([127, 0, 0], dtype=torch.int16, device=devc).view(1, c, 1, 1)) / 255
            else:  # prepare from next scale's pixels : the initial band x00 for this scale
                yB, yC, yH, yW = y_list_lev.shape
                temp = torch.zeros(yB, yC, yH*2, yW*2, device=y_list_lev.device)
                # now apply inverse lazyDWT  (works only for clrchs == 3 clrjnt == 2 !?)
                temp[:, 0:c, 0::2, 0::2] = y_list_lev[:, 0*c:1*c, :, :]
                temp[:, 0:c, 0::2, 1::2] = y_list_lev[:, 2*c:3*c, :, :]
                temp[:, 0:c, 1::2, 0::2] = y_list_lev[:, 3*c:4*c, :, :]
                temp[:, 0:c, 1::2, 1::2] = y_list_lev[:, 1*c:2*c, :, :]
                crpdH, crpdW = temp.shape[2] - padHW_lev[scl+1][0], temp.shape[3] - padHW_lev[scl+1][1]
                y_list_lev = temp[:, :, :crpdH, :crpdW]
            # get padding info for this scale
            padH, padW = padHW_lev[scl][0], padHW_lev[scl][1]
            # update mdl_scl so that the correct model is used for each scale
            if scl_no < mdl_scl:
                mdl_scl -= mdl_scl
                while mdl_scl > 0 and self.useprevlevNN[mdl_scl] is True:
                    mdl_scl = mdl_scl - 1
            # loop over each subband (0,1,2): get cdfs and loop over clr channels to decode each
            for b in range(3):
                bb = 0 if self.combine_layers1toL else b
                params = self.entmdls_scale_band[mdl_scl][bb].get_params(y_list_lev[:, 0:c*(b+1), :, :])
                aw, bw, dw = params[:, 9 * M:10 * M, :, :], params[:, 10 * M:11 * M, :, :], params[:, 11 * M:12 * M, :, :]
                # decode y,co,cg channels sequentially by getting their cdfs also sequentially
                for clr in range(c):
                    # get stdevs, means, weights
                    stdevs  = params[:, (0*c+clr)*M:(0*c+clr+1)*M, :, :]
                    means   = params[:, (1*c+clr)*M:(1*c+clr+1)*M, :, :]
                    weights = params[:, (2*c+clr)*M:(2*c+clr+1)*M, :, :]
                    # update means of this color channel from previous color channels
                    if clr == 1:  # update means of this color channel from previous color channels
                        means += aw * y_list_lev[:, c*(b+1)+0:c*(b+1)+1, :, :]
                    if clr == 2:
                        means += bw * y_list_lev[:, c*(b+1)+0:c*(b+1)+1, :, :] + dw * y_list_lev[:, c*(b+1)+1:c*(b+1)+2, :, :]
                    # get cdf of this clr channel and decode this color channel from bitstream
                    minsamp = -127 if clr == 0 else minYCoCg_maxYCoCg_list[clr]
                    maxsamp =  128 if clr == 0 else minYCoCg_maxYCoCg_list[3 + clr]
                    if False:  # with decode_float_cdf
                        cdfs_b_clr = self.entmdls_scale_band[mdl_scl][bb].get_cdfs(stdevs, means, weights, clrch=clr, int_cdf=False, minVal=minsamp, maxVal=maxsamp)
                        crpdH = cdfs_b_clr.shape[2] - padH if b in [0, 2] else cdfs_b_clr.shape[2]
                        crpdW = cdfs_b_clr.shape[3] - padW if b in [0, 1] else cdfs_b_clr.shape[3]
                        y_list_lev_b_clr_int16_cpu = torchac.decode_float_cdf(cdfs_b_clr[:, :, :crpdH, :crpdW].to('cpu').clip(min=0.0, max=1.0),
                                                                              bytestream_list[len_bytestream_list-1-scl][c*b+clr],
                                                                              needs_normalization=True)  # check_input_bounds=True
                    else:  # with decode_int16_normalized_cdf
                        cdfs_b_clr = self.entmdls_scale_band[mdl_scl][bb].get_cdfs(stdevs, means, weights, clrch=clr, int_cdf=True, minVal=minsamp, maxVal=maxsamp)
                        crpdH = cdfs_b_clr.shape[2] - padH if b in [0, 2] else cdfs_b_clr.shape[2]
                        crpdW = cdfs_b_clr.shape[3] - padW if b in [0, 1] else cdfs_b_clr.shape[3]
                        y_list_lev_b_clr_int16_cpu = torchac.decode_int16_normalized_cdf(cdfs_b_clr[:, :, :crpdH, :crpdW].to('cpu'),
                                                                                         bytestream_list[len_bytestream_list-1-scl][c*b+clr])
                    # pad decoded tensor if needed
                    # y_list_lev_b_clr_int16_cpu = self._pad_decoded_tensor(y_list_lev_b_clr_int16_cpu, b, padH, padW)
                    y_list_lev_b_clr_int16_cpu = self._pad_decoded_tensor(y_list_lev_b_clr_int16_cpu.to(devc, dtype=torch.float32), b, padH, padW)
                    # convert decoded pixels to float32, move to gpu (like _convert_int16cpu_to_float32gpu)
                    y_list_lev[:, c*(b+1)+clr, :, :] = self._convert_int16cpu_to_float32gpu(y_list_lev_b_clr_int16_cpu, devc=devc, clr=clr)
            scl_no -= 1
        # now apply inverse lazyDWT one last time, convert and move and return
        yB, _, yH, yW = y_list_lev.shape
        temp = torch.zeros(yB, 3, yH*2, yW*2, device=y_list_lev.device)
        # now apply inverse lazyDWT  (works only for clrchs == 3 clrjnt == 2 !?)
        temp[:, :, 0::2, 0::2] = y_list_lev[:, 0*c:1*c, :, :]
        temp[:, :, 0::2, 1::2] = y_list_lev[:, 2*c:3*c, :, :]
        temp[:, :, 1::2, 0::2] = y_list_lev[:, 3*c:4*c, :, :]
        temp[:, :, 1::2, 1::2] = y_list_lev[:, 1*c:2*c, :, :]
        crpdH, crpdW = temp.shape[2] - padHW_lev[0][0], temp.shape[3] - padHW_lev[0][1]
        return temp[:, :, :crpdH, :crpdW]

    @staticmethod
    def _pad_decoded_tensor(x_, b, padH, padW):
        if padH and padW:
            if b == 1:  # x_01
                x_ = F.pad(x_, (0, 1, 0, 0), mode='replicate')  # (padleft, padright, padtop, padbottom)
            if b == 2:  # x_10
                x_ = F.pad(x_, (0, 0, 0, 1), mode='replicate')
            if b == 0:  # x_11
                x_ = F.pad(x_, (0, 1, 0, 1), mode='replicate')
        elif padW:
            if b == 1:  # x_01
                x_ = F.pad(x_, (0, 1, 0, 0), mode='replicate')
            if b == 0:  # x_11
                x_ = F.pad(x_, (0, 1, 0, 0), mode='replicate')
        elif padH:
            if b == 2:  # x_10
                x_ = F.pad(x_, (0, 0, 0, 1), mode='replicate')
            if b == 0:  # x_11
                x_ = F.pad(x_, (0, 0, 0, 1), mode='replicate')
        return x_

    @staticmethod
    def _get_padHW_lev_list(padHW_int, num_scales):
        padHW_lev = []
        for i in range(num_scales):
            padW = True if padHW_int % 2 == 1 else False
            padHW_int = padHW_int // 2
            padH = True if padHW_int % 2 == 1 else False
            padHW_int = padHW_int // 2
            padHW_lev.append([padH, padW])
        padHW_lev.reverse()  # reverse order so that initial scale is at index 0 of list
        return padHW_lev

    def _adjust_mean_shifts(self, minYCoCg_maxYCoCg_list):
        minY, minCo, minCg,  maxY, maxCo, maxCg = minYCoCg_maxYCoCg_list
        self.mean_shifts_ycocg_int[0, :, 0, 0] = torch.tensor([127, -minCo, -minCg], dtype=torch.int16).repeat(4)
        self.mean_shifts_ycocg[0, :, 0, 0] = self.mean_shifts_ycocg_int[0, :, 0, 0] / 255

    def _convert_float32gpu_to_int16cpu(self, c, tensor_flt32_gpu, devc):
        # convert pixels back to integers in int16 and move to cpu
        if self.ycocg:
            # map Y ch to ints 0-255 and Co Cg chs to ints 0-511 by shifting min value to 0 and multiplying with 255
            y_list_lev_int16_cpu = torch.round(((tensor_flt32_gpu + self.mean_shifts_ycocg.to(devc)) * 255)).to(torch.int16).to('cpu', non_blocking=True)
        else:
            # map R,G and B channels to integers 0-255 by shifting min value to 0 and multiplying with 255
            y_list_lev_int16_cpu = torch.round(((tensor_flt32_gpu + self.mean_y_ycocg.to(devc)) * 255)).to(torch.int16).to('cpu', non_blocking=True)
        return y_list_lev_int16_cpu

    def _convert_int16cpu_to_float32gpu(self, tensor_int16_cpu, devc, clr=None):
        # convert pixels back to float32, move to gpu (!!! apply inverse operations of _convert_float32gpu_to_int16cpu)
        if self.ycocg:
            if clr is None:
                tensor_flt32_gpu = (tensor_int16_cpu.to(devc) - self.mean_shifts_ycocg_int[0:1, 0:3, 0:1, 0:1].to(devc)) / 255
            else:
                tensor_flt32_gpu = (tensor_int16_cpu.to(devc) - self.mean_shifts_ycocg_int[0:1, clr:clr+1, 0:1, 0:1].to(devc)) / 255
        else:
            tensor_flt32_gpu = tensor_int16_cpu.to(devc).to(torch.float32).div(255) - self.mean_y_ycocg.to(devc)
        return tensor_flt32_gpu

    @staticmethod
    def get_int16YCoCg_R_from_uint8RGB__intOps(x):
        assert x.shape[1] == 3  # ensure x has 3 color channels
        R, G, B = x[:, 0:1, :, :].to(torch.int16), x[:, 1:2, :, :].to(torch.int16), x[:, 2:3, :, :].to(torch.int16)
        # apply lifting based equations in JVT-I014r3
        Co = R - B
        # t = B + torch.round(Co / 2).to(torch.int16)
        t = B + Co // 2
        Cg = G - t
        # Y = t + torch.round(Cg / 2).to(torch.int16)
        Y = t + Cg // 2
        YCoCg = torch.cat((Y, Co, Cg), dim=1)
        return YCoCg


class LLICTIEntropyModel4(nn.Module):
    """  Like LLICTIEntropyModel3 but NN parameters for each scale and band can be adjusted separately """
    def __init__(self, scale, band, config, Ev, Od, Ch, Ly):
        super(LLICTIEntropyModel4, self).__init__()
        # Set up sizes/lengths for models, inputs etc
        self.net_type = config.net_type
        self.clrchs = config.clrchs
        self.clrjnt = config.clr_joint_mode  # 0: all clr ch indep 1: ch0 indep, ch12 joint with pixcnn++ method  2: ch012 joint with pixcnn++ method
        self.clrjnt0seqmd = config.clrjnt0seqmd
        self.mwsa_joint = config.mwsa_joint
        self.distribution = config.distribution
        self.num_mixtures = config.num_mixtures
        assert self.num_mixtures > 1, "Use GMM with num_mixtures > 1 please!"
        # self.opmode = config.opmode  # operation mode : train or encdec
        # lifting precision bits; also used in quantization or rounding operations
        self.ycocg = config.ycocg
        self.precision_bits = config.lif_prec_bits
        self.RNDFACTOR = 255 * (2 ** (self.precision_bits - 8))
        self.mean_y_ycocg = ((2 ** (self.precision_bits - 1)) - 1) / ((2 ** self.precision_bits) - 1)  # i.e. subtract out - 127/255
        # prepare cdf sampling points for cdfs of all color channels
        cdf_sampling_pts = torch.linspace(-127.5, 128.5, steps=257) / 255
        cdf_sampling_pts[0], cdf_sampling_pts[-1] = -147.5 / 255, 148.5 / 255  # to get probs in tails...
        self.cdf_sampling_pts__Y_RGB = cdf_sampling_pts
        cdf_sampling_pts = torch.linspace(-255.5, 255.5, steps=512) / 255
        cdf_sampling_pts[0], cdf_sampling_pts[-1] = -275.5 / 255, 275.5 / 255  # to get probs in tails...
        self.cdf_sampling_pts__CoCg = cdf_sampling_pts
        # probability model for y
        if self.distribution == "normal":
            if self.num_mixtures == 1:
                self.conditional_prob_model = GaussianConditionalLossless(None)   # lossless version of GaussianConditional
            else:
                num_chn = 3 if (self.clrchs == 3 and self.clrjnt in [2, 0]) else 2 if (self.clrchs == 3 and self.clrjnt == 1) else 1  # !!!
                self.conditional_prob_model = GaussianConditionalLosslessGMM(scale_table=None, num_mix=self.num_mixtures, num_chn=num_chn)
                if self.clrchs == 3 and self.clrjnt == 1:
                    self.conditional_prob_model_Yonly = GaussianConditionalLosslessGMM(scale_table=None, num_mix=2*self.num_mixtures, num_chn=1)
        elif self.distribution == "logistic":
            self.conditional_prob_model = LogisticConditionalLossless(None)
        # Set up NN that will process available pixels/tensors
        if self.clrchs == 3:
            if self.clrjnt == 2:  # 2: ch012 joint with pixcnn++ method
                grps = 1 if self.mwsa_joint else 4  # 4  # mean, stdev, weights, abc (i.e. separete NN for each of these groups)
                Ch = grps * Ch  # num channles in intermediate layers
                Co = (3 * self.num_mixtures) * 3 + (1 + 2) * self.num_mixtures  # mean, stdev, weights, abc (output layer num of channels)
            elif self.clrjnt == 1:  # 1: ch0 indep, ch12 joint with pixcnn++ method
                grps = 4 * 2  # mean, stdev, weights, abd for Y and then for CoCg
                Ch = grps * Ch  # num channles in intermediate layers
                Co = self.num_mixtures * (4 * 2 * 2)   # see your notes: will use M mixtures for Co, Cg and 2M for Y
            elif self.clrjnt == 0:  # 0: all clr ch indep 1: ch0 indep
                grps = 3 if self.mwsa_joint else (3*3)  # 3 * 3  # mean, stdev, weights (i.e. separete NN for each of these groups)
                Ch = grps * Ch  # num channles in intermediate layers
                Co = self.num_mixtures * grps                    # mean, stdev, weights (output layer num of channels)
            else:
                grps, Co = 1, 0
        elif self.clrchs < 3:  # 0th, 1st or 2nd channel individually
            grps = 3  # mean, stdev, weights (i.e. separete NN for each of these groups)
            Ch = grps * Ch  # num channles in intermediate layers
            Co = self.num_mixtures * 3                             # mean, stdev, weights (output layer num of channels)
        else:
            grps, Co = 1, 0
        # set up layer0, i.e. initial layer that will take input all previous bands
        self.band = band
        self.submean = config.subtract_mean
        # c = 3 if (self.clrchs == 3 and self.clrjnt in [0, 2]) else 1
        c = 3 if (self.clrchs == 3 and self.clrjnt in [0, 2]) else 4 if (self.clrchs == 3 and self.clrjnt == 1) else 1
        grp0 = 1 if (self.clrchs < 3 or self.clrjnt == 2) else (3 if self.clrjnt == 0 else 2)  # be careful !
        if self.band == 0 or self.band == -1:
            self.layer0_00_11 = nn.Conv2d(c,     Ch, kernel_size=(Ev, Ev), stride=1, padding=0, groups=grp0)
            self.paddd0_00_11 = (Ev//2-1, Ev//2, Ev//2-1, Ev//2)  # (padding_left, padding_right, padding_top, padding_bottom)
            if self.submean:
                self.mnfltr_00_11 = self._get_mean_filters(Ev, Ev)
            if self.clrchs == 3 and self.clrjnt == 0 and self.clrjnt0seqmd:
                self.layer0_to_11_toCo = nn.Conv2d(1,     Ch//3, kernel_size=1, stride=1, padding=0, groups=1)
                self.layer0_to_11_toCg = nn.Conv2d(2,     Ch//3, kernel_size=1, stride=1, padding=0, groups=1)
        if self.band == 1 or self.band == -1:
            self.layer0_00_01 = nn.Conv2d(c,     Ch, kernel_size=(Od, Ev), stride=1, padding=0, groups=grp0)
            self.paddd0_00_01 = (Ev//2-1, Ev//2, Od//2, Od//2)
            self.layer0_11_01 = nn.Conv2d(c,     Ch, kernel_size=(Ev, Od), stride=1, padding=0, groups=grp0)
            self.paddd0_11_01 = (Od//2, Od//2, Ev//2, Ev//2-1)
            if self.submean:
                self.mnfltr_00_01 = self._get_mean_filters(Od, Ev)
                self.mnfltr_11_01 = self._get_mean_filters(Ev, Od)
            if self.clrchs == 3 and self.clrjnt == 0 and self.clrjnt0seqmd:
                self.layer0_to_01_toCo = nn.Conv2d(1,     Ch//3, kernel_size=1, stride=1, padding=0, groups=1)
                self.layer0_to_01_toCg = nn.Conv2d(2,     Ch//3, kernel_size=1, stride=1, padding=0, groups=1)
        if self.band == 2 or self.band == -1:
            self.layer0_00_10 = nn.Conv2d(c,     Ch, kernel_size=(Ev, Od), stride=1, padding=0, groups=grp0)
            self.paddd0_00_10 = (Od//2, Od//2, Ev//2-1, Ev//2)
            self.layer0_11_10 = nn.Conv2d(c,     Ch, kernel_size=(Od, Ev), stride=1, padding=0, groups=grp0)
            self.paddd0_11_10 = (Ev//2, Ev//2-1, Od//2, Od//2)
            self.layer0_01_10 = nn.Conv2d(c,     Ch, kernel_size=(Ev, Ev), stride=1, padding=0, groups=grp0)
            self.paddd0_01_10 = (Ev//2, Ev//2-1, Ev//2-1, Ev//2)
            if self.submean:
                self.mnfltr_00_10 = self._get_mean_filters(Ev, Od)
                self.mnfltr_11_10 = self._get_mean_filters(Od, Ev)
                self.mnfltr_01_10 = self._get_mean_filters(Ev, Ev)
            if self.clrchs == 3 and self.clrjnt == 0 and self.clrjnt0seqmd:
                self.layer0_to_10_toCo = nn.Conv2d(1,     Ch//3, kernel_size=1, stride=1, padding=0, groups=1)
                self.layer0_to_10_toCg = nn.Conv2d(2,     Ch//3, kernel_size=1, stride=1, padding=0, groups=1)
        # activation func after layer0
        if config.activfun == "ReLU":
            self.activfun = nn.ReLU(inplace=True)
        elif config.activfun == "LeakyReLU":
            self.activfun = nn.LeakyReLU(inplace=True)
        elif config.activfun == "PReLU":
            self.activfun = nn.PReLU(num_parameters=Ch)
        elif config.activfun == "GDN1":
            self.activfun = GDN1(in_channels=Ch)
        else:
            self.activfun = nn.Identity()
        # remaining layers of the NN that will output GMM parameters
        layers1toL_list = []
        for i in range((Ly-1)-1):
            layers1toL_list.append(nn.Conv2d(Ch, Ch, kernel_size=1, stride=1, padding=1 // 2, padding_mode='replicate', groups=grps))
            if config.activfun == "ReLU":
                actf = nn.ReLU(inplace=True)
            elif config.activfun == "LeakyReLU":
                actf = nn.LeakyReLU(inplace=True)
            elif config.activfun == "PReLU":
                actf = nn.PReLU(num_parameters=Ch)
            elif config.activfun == "GDN1":
                actf = GDN1(in_channels=Ch)
            else:
                actf = nn.Identity()
            layers1toL_list.append(actf)
        # for i in range((Ly-1)-1):  # just add 2 more res layers before final layer
        #     layers1toL_list.append(ResBlock2dSingleConv(Ch, grps))
        layers1toL_list.append(nn.Conv2d(Ch,             Co, kernel_size=1, stride=1, padding=1 // 2, padding_mode='replicate', groups=grps))  # final/output layer
        self.layers1toL = nn.Sequential(*layers1toL_list)

    def _get_mean_filters(self, ks0, ks1):
        c = 3 if self.clrchs == 3 else 1
        mnfltr = nn.Conv2d(c,      c, kernel_size=(ks0, ks1), stride=1, padding=0, groups=c, bias=False)
        mnfltr.weight.data.fill_(1.0/(ks0*ks1))
        mnfltr.weight.requires_grad = False
        return mnfltr

    def _layer0_forward(self, y_condition, y_topredict=None):
        # c = 3 if self.clrchs == 3 else 1
        c = 3 if (self.clrchs == 3 and self.clrjnt in [0, 2]) else 4 if (self.clrchs == 3 and self.clrjnt == 1) else 1
        if self.band == 0 or (self.band == -1 and y_condition.shape[1] == 1*c):
            y0 = F.pad(y_condition[:, 0:c, :, :], pad=self.paddd0_00_11, mode='replicate')
            out0 = self.layer0_00_11(y0)
            if self.clrchs == 3 and self.clrjnt == 0 and self.clrjnt0seqmd:
                K = out0.shape[1] // (3*3)
                out0[:, 3*K:6*K, :, :] += self.layer0_to_11_toCo(y_topredict[:, 0:1, :, :])  # use also Y channel of curr pix
                out0[:, 6*K:9*K, :, :] += self.layer0_to_11_toCg(y_topredict[:, 0:2, :, :])  # use also YCo channells of ...
            return self.activfun(out0)
        if self.band == 1 or (self.band == -1 and y_condition.shape[1] == 2*c):
            y0 = F.pad(y_condition[:, 0*c:1*c, :, :], pad=self.paddd0_00_01, mode='replicate')
            out0 = self.layer0_00_01(y0)
            y1 = F.pad(y_condition[:, 1*c:2*c, :, :], pad=self.paddd0_11_01, mode='replicate')
            out1 = self.layer0_11_01(y1)
            if self.clrchs == 3 and self.clrjnt == 0 and self.clrjnt0seqmd:
                K = out0.shape[1] // (3*3)
                out0[:, 3*K:6*K, :, :] += self.layer0_to_01_toCo(y_topredict[:, 0:1, :, :])  # use also Y channel of curr pix
                out0[:, 6*K:9*K, :, :] += self.layer0_to_01_toCg(y_topredict[:, 0:2, :, :])  # use also YCo channells of ...
            return self.activfun(out0 + out1)
        if self.band == 2 or (self.band == -1 and y_condition.shape[1] == 3*c):
            y0 = F.pad(y_condition[:, 0*c:1*c, :, :], pad=self.paddd0_00_10, mode='replicate')
            out0 = self.layer0_00_10(y0)
            y1 = F.pad(y_condition[:, 1*c:2*c, :, :], pad=self.paddd0_11_10, mode='replicate')
            out1 = self.layer0_11_10(y1)
            y2 = F.pad(y_condition[:, 2*c:3*c, :, :], pad=self.paddd0_01_10, mode='replicate')
            out2 = self.layer0_01_10(y2)
            if self.clrchs == 3 and self.clrjnt == 0 and self.clrjnt0seqmd:
                K = out0.shape[1] // (3*3)
                out0[:, 3*K:6*K, :, :] += self.layer0_to_10_toCo(y_topredict[:, 0:1, :, :])  # use also Y channel of curr pix
                out0[:, 6*K:9*K, :, :] += self.layer0_to_10_toCg(y_topredict[:, 0:2, :, :])  # use also YCo channells of ...
            return self.activfun(out0 + out1 + out2)

    def _layer0_submean_forward(self, y_condition):
        c = 3 if self.clrchs == 3 else 1
        if self.band == 0 or (self.band == -1 and y_condition.shape[1] == 3):
            # band 0
            yp0 = F.pad(y_condition[:, 0*c:1*c, :, :], pad=self.paddd0_00_11, mode='replicate')
            mnq0 = torch.round(self.mnfltr_00_11(yp0) * self.RNDFACTOR) / self.RNDFACTOR
            y0 = F.pad(y_condition[:, 0*c:1*c, :, :] - mnq0, pad=self.paddd0_00_11, mode='replicate')
            out0 = self.layer0_00_11(y0)
            # band 0
            return self.LeakyReLU(out0), mnq0
        if self.band == 1 or (self.band == -1 and y_condition.shape[1] == 6):
            # band 0
            yp0 = F.pad(y_condition[:, 0*c:1*c, :, :], pad=self.paddd0_00_01, mode='replicate')
            mn0 = self.mnfltr_00_01(yp0)
            mnq0 = torch.round(mn0 * self.RNDFACTOR) / self.RNDFACTOR
            y0 = F.pad(y_condition[:, 0*c:1*c, :, :] - mnq0, pad=self.paddd0_00_01, mode='replicate')
            out0 = self.layer0_00_01(y0)
            # band 1
            yp1 = F.pad(y_condition[:, 1*c:2*c, :, :], pad=self.paddd0_11_01, mode='replicate')
            mn1 = self.mnfltr_11_01(yp1)
            mnq1 = torch.round(mn1 * self.RNDFACTOR) / self.RNDFACTOR
            y1 = F.pad(y_condition[:, 1*c:2*c, :, :] - mnq1, pad=self.paddd0_11_01, mode='replicate')
            out1 = self.layer0_11_01(y1)
            # band 0+1
            return self.LeakyReLU(out0 + out1), torch.round((mn0 + mn1)/2 * self.RNDFACTOR) / self.RNDFACTOR
        if self.band == 2 or (self.band == -1 and y_condition.shape[1] == 9):
            # band 0
            yp0 = F.pad(y_condition[:, 0*c:1*c, :, :], pad=self.paddd0_00_10, mode='replicate')
            mn0 = self.mnfltr_00_10(yp0)
            mnq0 = torch.round(mn0 * self.RNDFACTOR) / self.RNDFACTOR
            y0 = F.pad(y_condition[:, 0*c:1*c, :, :] - mnq0, pad=self.paddd0_00_10, mode='replicate')
            out0 = self.layer0_00_10(y0)
            # band 1
            yp1 = F.pad(y_condition[:, 1*c:2*c, :, :], pad=self.paddd0_11_10, mode='replicate')
            mn1 = self.mnfltr_11_10(yp1)
            mnq1 = torch.round(mn1 * self.RNDFACTOR) / self.RNDFACTOR
            y1 = F.pad(y_condition[:, 1*c:2*c, :, :] - mnq1, pad=self.paddd0_11_10, mode='replicate')
            out1 = self.layer0_11_10(y1)
            # band 2
            yp2 = F.pad(y_condition[:, 2*c:3*c, :, :], pad=self.paddd0_01_10, mode='replicate')
            mn2 = self.mnfltr_01_10(yp2)
            mnq2 = torch.round(mn2 * self.RNDFACTOR) / self.RNDFACTOR
            y2 = F.pad(y_condition[:, 2*c:3*c, :, :] - mnq2, pad=self.paddd0_01_10, mode='replicate')
            out2 = self.layer0_01_10(y2)
            # band 0+1+2
            return self.LeakyReLU(out0 + out1 + out2), torch.round((mn0 + mn1 + mn2)/3 * self.RNDFACTOR) / self.RNDFACTOR

    def forward(self, y_condition, y_topredict):
        if not self.submean:
            # get prob model params from NN
            if self.clrchs == 3 and self.clrjnt == 0 and self.clrjnt0seqmd:
                outlayer0 = self._layer0_forward(y_condition, y_topredict)
            else:
                outlayer0 = self._layer0_forward(y_condition)
            params = self.layers1toL(outlayer0)
            # get probs and then self_infos
            return self.get_self_infos(params, y_topredict, opmode='train')
        else:
            # get prob model params from NN
            outlayer0, mean_y_topredict = self._layer0_submean_forward(y_condition)
            params = self.layers1toL(outlayer0)
            # get probs and then self_infos
            return self._get_self_infos(params, y_topredict - mean_y_topredict)

    # to be used during compress only not training.
    # assumes   not self.submean:
    # and not  self.clrchs == 3 and self.clrjnt == 0 and self.clrjnt0seqmd:
    def get_params(self, y_condition):
        outlayer0 = self._layer0_forward(y_condition)
        params = self.layers1toL(outlayer0)
        return params

    def get_self_infos(self, params, y_topredict, opmode='train', clrch=0):
        # c = 3 if self.clrchs == 3 else 1
        c = 3 if (self.clrchs == 3 and self.clrjnt in [0, 2]) else 4 if (self.clrchs == 3 and self.clrjnt == 1) else 1
        M = self.num_mixtures
        # get probabilities/likelihoods from prob model parameters
        if self.distribution == "normal":
            if self.num_mixtures == 1:
                y_stdev, y_mean = params[:, 0:c, :, :], params[:, c:2*c, :, :]  # torch.chunk(params, 2, dim=1)
                if self.clrchs == 3:
                    a, b, d = params[:, 6:7, :, :], params[:, 7:8, :, :], params[:, 8:9, :, :]  # weights for mean updates
                    # update means of G and B
                    y_mean[:, 1:2, :, :] = y_mean[:, 1:2, :, :] + a * y_topredict[:, 0:1, :, :]
                    y_mean[:, 2:3, :, :] = y_mean[:, 2:3, :, :] + b * y_topredict[:, 0:1, :, :] + d * y_topredict[:, 1:2, :, :]
                _, pmf_values_y = self.conditional_prob_model(y_topredict,  # * self.RNDFACTOR,
                                                              y_stdev,
                                                              means=y_mean)
            else:  # GMM
                # --------OLD VERSION. KEEP IT.-------------------------------------------------------------------------
                # # y_stdev, y_mean, y_weights = torch.chunk(params, 3, dim=1)
                # y_stdev =   params[:, 0*c * M:1*c * M, :, :]
                # y_mean  =   params[:, 1*c * M:2*c * M, :, :]
                # y_weights = params[:, 2*c * M:3*c * M, :, :]
                # if self.clrchs == 3:
                #     a = params[:,  9 * M:10 * M, :, :]  # weights for mean updates
                #     b = params[:, 10 * M:11 * M, :, :]
                #     d = params[:, 11 * M:12 * M, :, :]
                #     # update means of G and B
                #     y_mean[:, 1 * M:2 * M, :, :] = y_mean[:, 1 * M:2 * M, :, :] + a * y_topredict[:, 0:1, :, :]
                #     y_mean[:, 2 * M:3 * M, :, :] = y_mean[:, 2 * M:3 * M, :, :] + b * y_topredict[:, 0:1, :, :] \
                #                                                                 + d * y_topredict[:, 1:2, :, :]
                # _, pmf_values_y = self.conditional_prob_model(inputs=y_topredict,  # * self.RNDFACTOR,
                #                                               scales=y_stdev,
                #                                               means=y_mean,
                #                                               weights=y_weights)
                # ------------------------------------------------------------------------------------------------------
                if self.clrchs == 3 and self.clrjnt == 2:
                    c = 3
                    # get stdevs, means, weights
                    y_stdev =   params[:, 0*c * M:1*c * M, :, :]
                    y_mean  =   params[:, 1*c * M:2*c * M, :, :]
                    y_weights = params[:, 2*c * M:3*c * M, :, :]
                    # weights for mean updates
                    a = params[:,  9 * M:10 * M, :, :]
                    b = params[:, 10 * M:11 * M, :, :]
                    d = params[:, 11 * M:12 * M, :, :]
                    # update means of G and B
                    y_mean[:, 1 * M:2 * M, :, :] = y_mean[:, 1 * M:2 * M, :, :] + a * y_topredict[:, 0:1, :, :]
                    y_mean[:, 2 * M:3 * M, :, :] = y_mean[:, 2 * M:3 * M, :, :] + b * y_topredict[:, 0:1, :, :] \
                                                                                + d * y_topredict[:, 1:2, :, :]
                    # get probabilities
                    _, pmf_values_y = self.conditional_prob_model(inputs=y_topredict,  # * self.RNDFACTOR,
                                                                    scales=y_stdev,
                                                                    means=y_mean,
                                                                    weights=y_weights)
                elif self.clrchs == 3 and self.clrjnt == 0:
                    # get stdevs, means, weights
                    y_stdev =   torch.cat((params[:, 0*M:1*M, :, :], params[:, 3*M:4*M, :, :], params[:, 6*M:7*M, :, :]), dim=1)  # Y Co Cg
                    y_mean  =   torch.cat((params[:, 1*M:2*M, :, :], params[:, 4*M:5*M, :, :], params[:, 7*M:8*M, :, :]), dim=1)  # Y Co Cg
                    y_weights = torch.cat((params[:, 2*M:3*M, :, :], params[:, 5*M:6*M, :, :], params[:, 8*M:9*M, :, :]), dim=1)  # Y Co Cg
                    # now weights since no mean updates
                    # get probabilities
                    _, pmf_values_y = self.conditional_prob_model(inputs=y_topredict,  # * self.RNDFACTOR,
                                                                    scales=y_stdev,
                                                                    means=y_mean,
                                                                    weights=y_weights)
                elif self.clrchs == 3 and self.clrjnt == 1:
                    # get stdevs, means, weights
                    throw_away = params[:, 0*M:2*M, :, :]
                    y_stdev_Y =   params[:, 2*M:4*M, :, :]  # Y  (2M)
                    y_mean_Y  =   params[:, 4*M:6*M, :, :]  # Y  (2M)
                    y_weights_Y = params[:, 6*M:8*M, :, :]  # Y  (2M)
                    y_stdev_CoCg =   params[:,  8*M:10*M, :, :]  # CoCg (M + M)
                    y_mean_CoCg  =   params[:, 10*M:12*M, :, :]  # CoCg (M + M)
                    y_weights_CoCg = params[:, 12*M:14*M, :, :]  # CoCg (M + M)
                    # weights for mean updates of Cg only
                    a = params[:, 14*M:15*M, :, :]
                    throw_awa2 = params[:, 15*M:16*M, :, :]
                    # update means of Cg only
                    y_mean_CoCg[:, 1*M:2*M, :, :] = y_mean_CoCg[:, 1*M:2*M, :, :] + a * y_topredict[:, 2:3, :, :]  # 0 Y Co Cg
                    # get probaiblities
                    _, pmf_values_y_CoCg = self.conditional_prob_model(inputs=y_topredict[:, 2:4, :, :],  # * self.RNDFACTOR,
                                                                    scales=y_stdev_CoCg,
                                                                    means=y_mean_CoCg,
                                                                    weights=y_weights_CoCg)
                    _, pmf_values_y_Y = self.conditional_prob_model_Yonly(inputs=y_topredict[:, 1:2, :, :],  # * self.RNDFACTOR,
                                                                    scales=y_stdev_Y,
                                                                    means=y_mean_Y,
                                                                    weights=y_weights_Y)
                    pmf_values_y = torch.cat((pmf_values_y_Y, pmf_values_y_CoCg), dim=1)
                elif self.clrchs < 3:
                    c = 1
                    # get stdevs, means, weights
                    y_stdev =   params[:, 0*c * M:1*c * M, :, :]
                    y_mean  =   params[:, 1*c * M:2*c * M, :, :]
                    y_weights = params[:, 2*c * M:3*c * M, :, :]
                    # get probabilities
                    _, pmf_values_y = self.conditional_prob_model(inputs=y_topredict,  # * self.RNDFACTOR,
                                                                    scales=y_stdev,
                                                                    means=y_mean,
                                                                    weights=y_weights)
                else:
                    pmf_values_y = torch.tensor([0.0], device=y_topredict.device, requires_grad=False)
                    pass
        else:
            pmf_values_y = torch.tensor([0.0], device=y_topredict.device, requires_grad=False)
            pass
        # get self infos
        self_informations_y = -torch.log2(pmf_values_y)
        return self_informations_y

    # assumes GMM, self.clrchs == 3 and self.clrjnt == 2
    def get_cdfs(self, stdevs, means, weights, clrch=0, int_cdf=False, minVal=0, maxVal=255):
        c, M = 3, self.num_mixtures
        # prepare cdf sampling points for cdfs with given minVal and maxVal values dynamic range
        cdf_sampling_pts = torch.linspace(minVal-0.5, maxVal+0.5, steps=maxVal-minVal+1+1, device=stdevs.device) / 255
        cdf_sampling_pts[0], cdf_sampling_pts[-1] = (minVal-0.5-20)/255, (maxVal+0.5+20)/255  # to get probs in tails...
        # now get cdfs of this channel
        cdfs = self.conditional_prob_model.get_cdfs(cdf_sampling_pts=cdf_sampling_pts,
                                                    scales=stdevs,
                                                    means=means,
                                                    weights=weights)
        if int_cdf is False:
            return cdfs
        else:
            cdfs_int = _convert_to_int_and_normalize(cdfs, needs_normalization=True)
            return cdfs_int


def _convert_to_int_and_normalize(cdf_float, needs_normalization):
    """Convert floatingpoint CDF to integers. See README for more info.
    The idea is the following:
    When we get the cdf here, it is (assumed to be) between 0 and 1, i.e,
    cdf \in [0, 1)
    (note that 1 should not be included.)
    We now want to convert this to int16 but make sure we do not get
    the same value twice, as this would break the arithmetic coder
    (you need a strictly monotonically increasing function).
    So, if needs_normalization==True, we multiply the input CDF
    with 2**16 - (Lp - 1). This means that now,
    cdf \in [0, 2**16 - (Lp - 1)].
    Then, in a final step, we add an arange(Lp), which is just a line with
    slope one. This ensure that for sure, we will get unique, strictly
    monotonically increasing CDFs, which are \in [0, 2**16)
    """
    PRECISION = 16
    Lp = cdf_float.shape[-1]
    factor = torch.tensor(2, dtype=torch.float32, device=cdf_float.device).pow_(PRECISION)
    new_max_value = factor
    if needs_normalization:
        new_max_value = new_max_value - (Lp - 1)
    cdf_float = cdf_float.mul(new_max_value)
    cdf_float = cdf_float.round()
    cdf = cdf_float.to(dtype=torch.int16, non_blocking=True)
    if needs_normalization:
        r = torch.arange(Lp, dtype=torch.int16, device=cdf.device)
        cdf.add_(r)
    return cdf
