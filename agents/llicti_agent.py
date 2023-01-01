import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from agents.base import BaseAgent
from graphs.models.LLICTI_nets import LLICTI
from graphs.losses.rate_dist import TrainRLoss, TrainRLossList, CompressionRLossList
from dataloaders.image_dl import ImageDataLoader
from loggers.rate import RateLogger
import time


class LLICTIAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(config)
        # wavelet transform type
        self.wtr_type = config.wtr_type
        assert self.wtr_type in ('lazydwt', "x")  # see notebook for meanings
        if self.wtr_type == 'lazydwt':
            self.model = LLICTI(config)
            self.train_loss = TrainRLossList()
            self.valid_loss = TrainRLossList()
            self.compr_loss = CompressionRLossList()
        else:  # in case we do also other splitting for interpolation
            pass
        self.model = self.model.to(self.device)
        self.lr = self.config.learning_rate
        self.optimizer = optim.Adam([{'params': self.model.parameters(), 'lr':self.lr}])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.5, patience=16,
                                                              threshold=0.0001, threshold_mode='rel',
                                                              cooldown=15, min_lr=2.5e-05, eps=1e-08, verbose=False)  # min_lr=1e-5
        self.grad_acc_iters = config.grad_acc_iters
        self.loss_prnt_iters = config.loss_prnt_iters
        self.data_loader  = ImageDataLoader(config)
        self.lambda_      = config.lambda_
        self.train_logger = RateLogger()
        self.trnit_logger = RateLogger()  # to report for every 1000 iterations inside and epoch
        self.valid_logger = RateLogger()
        self.test_logger  = RateLogger()
        if config.mode in ['test', 'validate', 'debug', 'eval_model']:
            self.load_checkpoint('model_best.pth.tar')
        elif config.resume_training:
            self.load_checkpoint(self.config.checkpoint_file)
        # print model size briefly
        self.model_size_estimation()

    def train_one_epoch(self):
        torch.backends.cudnn.benchmark = True  # https://discuss.pytorch.org/t/does-pytorch-optimize-the-group-parameter-in-convs/3060/5
        self.model.train()
        for batch_idx, x in enumerate(self.data_loader.train_loader):
            x = x.to(self.device)
            if x.dim() == 5:  # rearrange into 4 dimensions without copy
                x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
            # run through model, calculate loss, back-prop etc.
            if self.wtr_type == 'lazydwt':
                self_infos_y_list = self.model(x)
                r_loss, rate1_list = self.train_loss.forward(torch.numel(x), self_infos_y_list)
            else:
                pass
            (r_loss / self.grad_acc_iters).backward()
            # gradeint accumulation of grad_acc_iters
            if (self.current_iteration + 1) % self.grad_acc_iters == 0:
                # apply gradient clipping/scaling (if loss has been switched to R+lD)
                nn.utils.clip_grad_value_(self.model.parameters(), clip_value=5.0)  # 0.1
                # update weights, iteration number and log losses
                self.optimizer.step()
                self.optimizer.zero_grad()
            self.current_iteration += 1
            if self.wtr_type == 'lazydwt':
                self.train_logger(rate1_list)
                self.trnit_logger(rate1_list)
            else:
                pass
            if not (self.current_iteration+1) % self.config.loss_prnt_iters:
                trnit_rate_loss, trnit_rate2_loss = self.trnit_logger.display(lr=self.optimizer.param_groups[0]['lr'], typ='it')
                valid_loss = self.validate()
                self.model.train()  # put back to train mode
                is_best = valid_loss < self.best_valid_loss
                if is_best:
                    self.best_valid_loss = valid_loss
                self.save_checkpoint(is_best=is_best)
        train_rate_loss, train_rate2_loss = self.train_logger.display(lr=self.optimizer.param_groups[0]['lr'], typ='tr')

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        with torch.no_grad():
            for batch_idx, x in enumerate(self.data_loader.valid_loader):
                x = x.to(self.device)
                numel_x_orig = torch.numel(x)
                x = self._pad_img(x)  # make H and W multiple of 2**dwtlevels
                # run through model, calculate loss
                if self.wtr_type == 'lazydwt':
                    self_infos_y_list = self.model(x)
                    r_loss, rate1_list = self.train_loss.forward(torch.numel(x), self_infos_y_list)
                    self.valid_logger(rate1_list)
                else:
                    pass
            valid_rate_loss, valid_rate2_loss = self.valid_logger.display(lr=0.0, typ='va')
            self.scheduler.step(valid_rate_loss + valid_rate2_loss)
            # self.scheduler.step()
            return valid_rate_loss + valid_rate2_loss

    def _pad_img(self, x):
        B = 2**(max(self.config.dwtlevels)+1)  # B = 2**self.config.dwtlevels
        # B = 2**(self.config.dwtlevels+1)
        # pad img on right and bottom to make H,W mulitple of self.block_size
        h, w = x.size(2), x.size(3)
        new_h, new_w = (h + B - 1) // B * B, (w + B - 1) // B * B
        padding_bottom, padding_right = new_h - h, new_w - w
        x = F.pad(x, (0, padding_right, 0, padding_bottom), mode='replicate')  # mode="constant",value=0)
        return x

    # test should be modified to have actual entorpy coding....
    @torch.no_grad()
    def test(self):
        self.model.eval()
        with torch.no_grad():
            pass

    @torch.no_grad()
    def eval_model(self):
        ##torch.backends.cudnn.benchmark = True  # https://discuss.pytorch.org/t/does-pytorch-optimize-the-group-parameter-in-convs/3060/5
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        self.model.eval()
        with torch.no_grad():
            for batch_idx, x in enumerate(self.data_loader.test_loader):
                x = x.to(self.device)
                numel_x_orig = torch.numel(x)
                ###x = self._pad_img(x)  # make H and W multiple of 2**dwtlevels
                print_text = "{:3d} {:3d}x{:3d} ".format(batch_idx, x.shape[2], x.shape[3])
                # ****** COMPRESS: run through model, get bistream list, *****
                start_time = time.time()
                bytestream_list, xorg = self.model.compress(x)
                enc_time = time.time() - start_time
                # calculate bits per pixel for each scale, band etc
                rate1_list = self.compr_loss.forward(torch.numel(x), bytestream_list)
                self.test_logger(rate1_list)
                # also print stats for each image being compressed
                total = 0
                for i in range(len(bytestream_list)):
                    for j in range(len(bytestream_list[i])):
                        total += len(bytestream_list[i][j]) * 8
                # ****** DECOMPRESS:  ******
                start_time = time.time()
                x_reco = self.model.decompres(bytestream_list, self.device)  #, xorg)
                dec_time = time.time() - start_time
                # check if decoded pixels match with the original pixels
                maxx_abserr = ((x - x_reco) * 255).abs().max()
                if maxx_abserr >= 0.5:
                    # print information about this image
                    self.logger.info(print_text + "bpsp= {:.3f} Enc/Dec-Times:{:.3f}/{:.3f} "
                                                  "(Error: Decoded img does NOT match original image perfectly! "
                                                  "The maximum of absolute error is {:.4f})"
                                                  .format(total / torch.numel(x), enc_time, dec_time, maxx_abserr))
                else:
                    # print information about this image
                    self.logger.info(print_text + "bpsp= {:.3f} Enc/Dec-Times:{:.3f}/{:.3f} "
                                                  "(Check: Decoded img matches original)".format(total / torch.numel(x),
                                                                                                 enc_time, dec_time))
            # print out average bitrates for each scale band colorch
            comp_rate_loss, comp_rate2_loss = self.test_logger.display(lr=0.0, typ='te')


    def model_size_estimation(self, print_params=False):
        model = self.model

        if print_params:
            print('---------------Printing paramters--------------------------')
        param_size = 0
        for name, param in model.named_parameters():
            if print_params:
                print(name, type(param), param.size())
            param_size += param.nelement() * param.element_size()
        if print_params:
            print('---------------Printing buffers--------------------------')
        buffer_size = 0
        for name, buffer in model.named_buffers():
            if print_params:
                print(name, type(buffer), buffer.size())
            buffer_size += buffer.nelement() * buffer.element_size()
        param_size_mb = param_size / 1024 ** 2
        buffer_size_mb = buffer_size / 1024 ** 2
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        # print('------------------TOT----------------------------------------------')
        # print(' model param+buffer=total size: {:.2f}+{:.2f}={:.2f}MB'.format(param_size_mb, buffer_size_mb, size_all_mb))
        # print('------------------END----------------------------------------------')
        self.logger.info('------------------TOT----------------------------------------------')
        self.logger.info(' model param+buffer=total size: {:.3f}+{:.3f}={:.3f}MB'.format(param_size_mb, buffer_size_mb, size_all_mb))
        self.logger.info('------------------END----------------------------------------------')

    def flops_estimation(self):
        from ptflops import get_model_complexity_info
        with torch.cuda.device(self.config.gpu_device):
            net = self.model
            macs, params = get_model_complexity_info(net, (3, 512, 512), as_strings=True, print_per_layer_stat=True, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)
