import logging
from statistics import mean
import numpy as np
from datetime import datetime


class RateMeter():
    def __init__(self):
        self.rate = []
        # self.rate2 = []
        self.current_iteration = 0
        self.current_epoch = 0

    def append(self, rate):
        self.current_iteration += 1
        self.rate.append(rate)
        # self.rate2.append(rate2)

    def reset(self):
        self.rate = []
        # self.rate2 = []

    # def mean(self):
    #     self.current_epoch += 1
    #     if isinstance(self.rate[0], list):
    #         rate = [sum(sub_list) / len(sub_list) for sub_list in zip(*self.rate)]  # average of i^th item of all lists in the list
    #     else:
    #         rate = mean(self.rate)
    #     if isinstance(self.rate2[0], list):
    #         rate2 = [sum(sub_list) / len(sub_list) for sub_list in zip(*self.rate2)]
    #     else:
    #         rate2 = mean(self.rate2)
    #     self.reset()
    #     return rate, rate2
    def mean(self):
        self.current_epoch += 1
        npaaa = np.array(self.rate)
        npaa = npaaa.mean(axis=0)
        self.reset()
        return npaa

    def state_dict(self):
        # return {'rate': self.rate, 'rate2': self.rate2, 'it': self.current_iteration, 'ep': self.current_epoch}
        return {'rate': self.rate, 'it': self.current_iteration, 'ep': self.current_epoch}

    def load_state_dict(self, info):
        self.rate = info['rate']
        # self.rate2 = info['rate2']
        self.current_iteration = info['it']
        self.current_epoch = info['ep']


class RateLogger(RateMeter):
    def __init__(self):
        super(RateLogger, self).__init__()
        self.logger = logging.getLogger("Rate Loss")

    def __call__(self, *args):
        self.append(*args)

    # def display(self, lr=0.0, typ='tr'):
    #     rate, rate2 = self.mean()
    #     if isinstance(rate, list):
    #         self.text_log_list(self.current_epoch, rate, rate2, lr, typ)
    #         if isinstance(rate2, list):
    #             return sum(rate), sum(rate2)
    #         else:
    #             return sum(rate), rate2
    #     else:
    #         self.text_log(self.current_epoch, rate, rate2, lr, typ)
    #         return rate, rate2
    def display(self, lr=0.0, typ='tr'):
        rate = self.mean()
        self.text_log_list(self.current_epoch, rate, lr, typ)
        return np.sum(rate), 0.0

    def text_log(self, cur_iter, rate, rate2, lr, typ):
        timedifstr = self._get_time_now_str()
        if typ == 'tr':
            self.logger.info('  Train Epoch: {:3d}  Rate: {:.3f}+{:.3f}={:.3f}  (lr: {:.6f}) ({})'.format(cur_iter, rate, rate2, rate+rate2, lr, timedifstr))
        elif typ == 'te':
            self.logger.info('   Test Epoch: {:3d}  Rate: {:.3f}+{:.3f}={:.3f} ({})'.format(cur_iter, rate, rate2, rate+rate2, timedifstr))
        elif typ == 'va':
            self.logger.info('  Valid Epoch: {:3d}  Rate: {:.3f}+{:.3f}={:.3f} ({})'.format(cur_iter, rate, rate2, rate+rate2, timedifstr))
        elif typ == 'it':
            self.logger.info('Train Itera: {:3d}  Rate: {:.3f}+{:.3f}={:.3f}  (lr: {:.6f}) ({})'.format(cur_iter, rate, rate2, rate+rate2, lr, timedifstr))

    def _get_time_now_str(self):
        self.t2 = datetime.now()
        return self.t2.strftime("%H:%M:%S")

    # def text_log_list(self, cur_iter, rate, rate2, lr, typ):
    #     timedifstr = self._get_time_now_str()
    #     log_text = ""
    #     # front of text
    #     if typ == 'tr':
    #         log_text = "  Train Epoch: {:3d}  Rate: ".format(cur_iter)
    #     elif typ == 'te':
    #         log_text = "   Test Epoch: {:3d}  Rate: ".format(cur_iter)
    #     elif typ == 'va':
    #         log_text = "  Valid Epoch: {:3d}  Rate: ".format(cur_iter)
    #     elif typ == 'it':
    #         log_text = "Train Itera: {:3d}  Rate: ".format(cur_iter)
    #     # write rates from lists to text
    #     if not isinstance(self.rate2, list):
    #         for i in range(len(rate)):
    #             log_text = log_text + "{:.3f}+{:.3f} ".format(rate[i], rate2[i])
    #         log_text = log_text + "={:.3f}+{:.3f} ={:.3f} ".format(sum(rate), sum(rate2), sum(rate)+sum(rate2))
    #     else:
    #         for i in range(len(rate)):
    #             log_text = log_text + "{:.3f}+{:.3f} ".format(rate[i], rate2)
    #         log_text = log_text + "={:.3f}+{:.3f} ={:.3f} ".format(sum(rate), rate2, sum(rate)+rate2)
    #     # finish text
    #     if typ == 'tr' or typ == 'it':
    #         log_text = log_text + "  (lr: {:.6f}) ({})".format(lr, timedifstr)
    #     elif typ == 'te' or typ == 'va':
    #         log_text = log_text + " ({})".format(timedifstr)
    #     # log final text now
    #     self.logger.info(log_text)
    def text_log_list(self, cur_iter, rate, lr, typ):
        timedifstr = self._get_time_now_str()
        log_text = ""
        # front of text
        if typ == 'tr':
            log_text = "  Train Epoch: {:3d}  Rates: scl".format(cur_iter)
        elif typ == 'te':
            log_text = "   Test Epoch: {:3d}  Rates: hdr ".format(cur_iter)
        elif typ == 'va':
            log_text = "  Valid Epoch: {:3d}  Rates: scl".format(cur_iter)
        elif typ == 'it':
            log_text = "Train Itera: {:3d}  Rates: scl".format(cur_iter)
        # write rates from lists to text
        assert rate.shape[1] == (3*3)  # i.e subbands * clrchannels
        # assert rate.shape[1] == (3 * 1)  # i.e subbands * (clrchannels=1)
        sum_all = 0.0
        for s in range(rate.shape[0]):
            sum_scl = 0.0
            temp = "{:d}-> ".format(s-1) if typ == 'te' and s > 0 else "-> " if typ == 'te' and s == 0 else "{:d}-> ".format(s)
            log_text = log_text + temp
            for b in range(3):  # over subbands in scale
                rr, gg, bb = rate[s][3*b+0], rate[s][3*b+1], rate[s][3*b+2]
                # rr, gg, bb = rate[s][1 * b + 0], 0, 0
                sumrgb = rr + gg + bb
                log_text = log_text + "{:.2f}+{:.2f}+{:.2f}(b{:d}={:.3f}) ".format(rr, gg, bb, b, sumrgb)
                sum_scl += sumrgb
            temp = "(s{:d}={:.3f}) ".format(s-1, sum_scl) if typ == 'te' and s > 0 else "(hd={:.3f}) ".format(sum_scl) if typ == 'te' and s == 0 else "(s{:d}={:.3f}) ".format(s, sum_scl)
            log_text = log_text + temp
            sum_all += sum_scl
            if s < rate.shape[0] - 1:  # not last scale
                log_text = log_text + "\n"
                # front of text
                if typ == 'tr':
                    log_text = log_text + "                                   scl"
                elif typ == 'te':
                    log_text = log_text + "                                   scl"
                elif typ == 'va':
                    log_text = log_text + "                                   scl"
                elif typ == 'it':
                    log_text = log_text + "                                 scl"
            else:
                log_text = log_text + "(({:.3f})) ".format(sum_all)
        # finish text
        if typ == 'tr' or typ == 'it':
            log_text = log_text + "  (lr: {:.6f}) ({})".format(lr, timedifstr)
        elif typ == 'te' or typ == 'va':
            log_text = log_text + " ({})".format(timedifstr)
        # log final text now
        self.logger.info(log_text)
