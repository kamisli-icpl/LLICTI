import logging
import sys
import shutil

import torch
from torch.backends import cudnn
from utils.mailer import Mailer

##cudnn.benchmark = True
##cudnn.enabled = True


class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        self.best_valid_loss = float('inf')  # 0
        self.current_epoch = 0
        self.current_iteration = 0
        self.manual_seed = self.config.seed
        self.cuda = torch.cuda.is_available() & self.config.cuda
        if self.cuda:
            self.device = torch.device("cuda")
            torch.cuda.manual_seed(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
        
    def train_one_epoch(self):
        """
        One epoch of training
        :return:
        """
        raise NotImplementedError

    def validate(self):
        """
        One cycle of model validation without recursive reconstrcution, where zhat is simply a noisy version of original
        :return:
        """
        raise NotImplementedError

    def test(self):
        """
        One cycle of model test
        :return:
        """
        raise NotImplementedError

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            # checkpoint = torch.load(filename)
            checkpoint = torch.load(filename, map_location=self.device)
            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.best_valid_loss = checkpoint['best_valid_loss']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.train_logger.load_state_dict(checkpoint['train_logger'])
            self.trnit_logger.load_state_dict(checkpoint['trnit_logger'])
            self.valid_logger.load_state_dict(checkpoint['valid_logger'])
            #self.test_logger.load_state_dict(checkpoint['test_logger'])
            # self.amp_scaler.load_state_dict(checkpoint['amp_scaler'])
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})"
                            .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
            # self.model.to(self.device)   # no need here ?
            # Fix the optimizer cuda error
            if self.cuda:  # this if statement is added by fatih later but not tested 
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()

        except OSError as e:
            self.logger.info("!!! No checkpoint exists from '{}'. Continuing with available parameters..."
                             .format(self.config.checkpoint_dir))
            ##self.logger.info("**First time to train**")
            
    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        state = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'best_valid_loss': self.best_valid_loss,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'train_logger': self.train_logger.state_dict(),
            'trnit_logger': self.trnit_logger.state_dict(),
            'valid_logger': self.valid_logger.state_dict(),
            #'test_logger': self.test_logger.state_dict(),
            # 'amp_scaler' : self.amp_scaler.state_dict()
        }
        torch.save(state, self.config.checkpoint_dir + filename)
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def run(self):
        try:
            if self.config.mode == 'test':
                self.test()
            elif self.config.mode == 'validate':
                # with torch.autograd.detect_anomaly():  # !!! comment his line if you don't need to debug validation !
                #     self.validate()
                self.validate()
            elif self.config.mode == 'train':
                self.train()
            elif self.config.mode == 'debug':
                with torch.autograd.detect_anomaly():
                    self.train()
            elif self.config.mode == 'eval_model':
                self.eval_model()
            elif self.config.mode == 'model_size':
                self.model_size_estimation(print_params=True)
            elif self.config.mode == 'flops_est':
                self.flops_estimation()
            else:
                raise NameError("'" + self.config.mode + "'" 
                                + ' is not a valid training mode.' )
        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")
        except AssertionError as e:
            raise e
        except Exception as e:
            self.save_checkpoint()
            raise e

    def train(self):
        #torch.backends.cudnn.benchmark = True  # https://discuss.pytorch.org/t/does-pytorch-optimize-the-group-parameter-in-convs/3060/5
        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            self.train_one_epoch()
            if not (self.current_epoch+1) % self.config.validate_every:
                valid_loss = self.validate()
                is_best = valid_loss < self.best_valid_loss
                if is_best:
                    self.best_valid_loss = valid_loss
                self.save_checkpoint(is_best=is_best)
            # validation code above is commented because lock of training desktop may be caused due to saving quickly two times ?
            # if not (self.current_epoch+1) % self.config.test_every:
            #     test_loss = self.test()
            self.current_epoch += 1

    def finalize(self):
        self.logger.info("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
