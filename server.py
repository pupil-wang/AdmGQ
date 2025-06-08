import logging
import os
import math
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
        self.s: int = Config().server.s_init
        # 当前记录的损失值
        self.loss = 0

        os.makedirs("./results/cost", exist_ok=True)
        self.record_file = f"./results/cost/{os.getpid()}.csv"
        with open(self.record_file, "w") as f:
            print(
                "round,total_time,compute_time,communication_cost,compute_cost", file=f
            )

    def get_quantize_level(self):
        return math.ceil(math.log2(self.s))

    def weights_received(self, deltas_received):
        reports = [update.report for update in self.updates]

        # 使用特定bit解压
        decompressed_deltas = [
            Processor(n=self.get_quantize_level()).process(delta)
            for delta, report in zip(deltas_received, reports)
        ]

        n = len(reports)
        loss = sum([report.loss for report in reports]) / n

        self.s = math.sqrt(self.loss / loss) * self.s

        self.loss = loss

        self.record(reports)

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
                min(32, report.quantize_n) * report.each_bit_time
                for report in reports
            ]
        )
        total_time = max(communication_time + compute_time)
        with open(self.record_file, "a") as f:
            print(
                f"{self.current_round},{total_time},{compute_time_sum},{communication_cost},{compute_cost}",
                file=f,
            )

    def customize_server_payload(self, payload):
        """
            Customizes the server payload before sending to the client.
            添加量化等级
        """
        logging.debug(f"Server send quantize_n: {self.get_quantize_level()}")
        return payload, self.get_quantize_level()
