
from plato.servers import fedavg
from plato.config import Config
from model_n_quantize import Processor


import numpy as np
from types import SimpleNamespace


class Server(fedavg.Server):
    def __init__(self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        with open("./factor", "w+") as f:
            f.write("1")


    def weights_received(self, deltas_received):
        reports = [update.report for update in self.updates]
        n = len(reports)
        loss_0 = sum([report.loss_0 for report in reports]) / n
        loss_ = sum([report.loss_ for report in reports]) / n
        loss = sum([report.loss for report in reports]) / n
        t = max([report.t for report in reports])
        t_ = max([report.t_ for report in reports])
        r = loss_0 - loss / t
        r_ = loss_0 - loss_ / t_
        sign = np.sign((r_ - r).cpu())
        multi_factor = 1
        if sign == 1:
            multi_factor = 0.5
        else:
            multi_factor = 2

        with open("./factor", "w+") as f:
            f.write(str(multi_factor))

        # 使用特定bit解压
        decompressed_deltas = [
            Processor(n=int(report.quantize_n * multi_factor)).process(delta)
            for delta, report in zip(deltas_received, reports)
        ]
        return super().weights_received(decompressed_deltas)
