import os
from plato.servers import fedavg
from plato.config import Config
from model_n_quantize import Processor


import numpy as np
from types import SimpleNamespace


class Server(fedavg.Server):
    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        with open("./factor", "w+") as f:
            f.write("1")

        os.makedirs("./results/cost", exist_ok=True)
        self.record_file = f"./results/cost/{os.getpid()}.csv"
        with open(self.record_file, "w") as f:
            print(
                "round,total_time,compute_time,communication_cost,compute_cost", file=f
            )

    def weights_received(self, deltas_received):
        reports = [update.report for update in self.updates]
        n = len(reports)
        loss_0 = sum([report.loss_0 for report in reports]) / n
        loss_ = sum([report.loss_ for report in reports]) / n
        loss = sum([report.loss for report in reports]) / n
        t = max([report.t for report in reports])
        t_ = max([report.t_ for report in reports])
        r = (loss_0 - loss) / t
        r_ = (loss_0 - loss_) / t_
        if type(r_ - r) == float:
            sign = np.sign((r_ - r))
        else:
            sign = np.sign((r_ - r).cpu())
        multi_factor = 1
        if sign == 1:
            multi_factor = 0.5
        else:
            multi_factor = 2

        self.multi_factor = multi_factor
        self.sign = sign
        with open("./factor", "w+") as f:
            f.write(str(multi_factor))

        self.record(reports)
        # 使用特定bit解压
        decompressed_deltas = [
            Processor(n=int(report.quantize_n * multi_factor)).process(delta)
            for delta, report in zip(deltas_received, reports)
        ]
        return super().weights_received(decompressed_deltas)

    def record(self, reports):
        # 记录总时间、计算开销（论文里的）、通信开销（上传的梯度总大小
        total_time = 0  # 总时间
        # 计算时间
        compute_time = np.array([report.t_compute for report in reports])
        compute_time_sum = sum(compute_time)
        # 计算开销
        compute_cost = sum(map(lambda x: x.compute_cost, reports))

        t_arr = np.array(list(map(lambda x: x.t, reports)))
        t_arr_ = np.array(list(map(lambda x: x.t_, reports)))

        # 两次量化等级的时间开销上的差值
        delta_t = t_arr - t_arr_
        if self.sign == 1:
            total_time = max(t_arr + delta_t * 2)
        else:
            total_time = max(t_arr_)

        # 通信开销
        communication_cost = sum(
            [
                (
                    report.model_size
                    if report.quantize_n >= 32
                    else report.model_size * self.multi_factor
                )
                for report in reports
            ]
        )

        # 通信时间
        communication_time = np.array(
            [
                min(32, report.quantize_n * self.multi_factor) * report.each_bit_time
                for report in reports
            ]
        )
        total_time = max(communication_time + compute_time)
        with open(self.record_file, "a") as f:
            print(
                f"{self.current_round},{total_time},{compute_time_sum},{communication_cost},{compute_cost}",
                file=f,
            )
