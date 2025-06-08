import logging
import os
import math
from plato.servers import fedavg
from plato.config import Config
from model_n_quantize import Processor
import csv
import numpy as np
from types import SimpleNamespace


class Server(fedavg.Server):
    def __init__(
            self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        self.s: int = Config().server.s_init
        logging.info(f"Server init s {self.s}")
        # 当前记录的损失值
        self.loss = 0

        os.makedirs("./results/cost", exist_ok=True)

        self.record_file = f"./results/cost/{os.getpid()}.csv"
        with open(self.record_file, "w") as f:
            csv.writer(f).writerow(
                ["round", "quantize_level", "s", "total_time", "compute_time", "communication_cost", "compute_cost"]
            )

    def get_quantize_level(self):
        n = math.ceil(math.log2(self.s))
        if n >= 32:
            return 32
        elif n <= 2:
            return 2
        return n

    def weights_received(self, deltas_received):
        reports = [update.report for update in self.updates]

        # 使用特定bit解压
        decompressed_deltas = [
            Processor(n=self.get_quantize_level()).process(delta)
            for delta, report in zip(deltas_received, reports)
        ]

        n = len(reports)
        loss = sum([report.loss for report in reports]) / n
        logging.info("loss: {}".format(loss))
        self.record(reports)

        # 更新
        if self.loss != 0:
            self.s = math.sqrt(self.loss / loss) * self.s

        self.loss = loss

        return super().weights_received(decompressed_deltas)

    def record(self, reports):
        # 记录总时间、计算开销（论文里的）、通信开销（上传的梯度总大小
        # 计算时间
        compute_time = np.array([report.t_compute for report in reports])
        compute_time_sum = sum(compute_time)
        # 计算开销
        compute_cost = sum(map(lambda x: x.compute_cost, reports))

        # 通信开销
        communication_cost = sum(report.model_size for report in reports)

        # 通信时间
        communication_time = np.array(
            [
                min(32, report.quantize_n) * report.each_bit_time
                for report in reports
            ]
        )
        total_time = max(communication_time + compute_time)
        with open(self.record_file, "a") as f:
            csv.writer(f).writerow(
                [self.current_round, self.get_quantize_level(), self.s, total_time, compute_time_sum,
                 communication_cost, compute_cost]
            )

    def customize_server_payload(self, payload):
        """
            Customizes the server payload before sending to the client.
            添加量化等级
        """
        logging.info(f"Server send quantize_n: {self.get_quantize_level()}")
        return payload, self.get_quantize_level()
