import copy
import logging
import math
import pickle
import sys
from types import SimpleNamespace
import numpy as np
from plato.clients import simple
from plato.config import Config
from plato.processors.base import Processor
from plato.processors import model_quantize_qsgd
import torch
import model_n_quantize
import model_n_dequantize

from plato.datasources import registry as datasources_registry
from plato.samplers import registry as samplers_registry

import random

# from multiple_processor import MultipleProcessor

M = 1024 * 1024


class Client(simple.Client):
    """
    client节点会有进行两种方式的通信压缩(无压缩和QSDG)
    然后把两种量化结果进行上传

    上传:
        report: [super, data_size, cost_time]
        weight: [raw_weight, QSGD_weight]
    """

    def __init__(
        self, model=None, datasource=None, algorithm=None, trainer=None, callbacks=None
    ):
        super().__init__(model, datasource, algorithm, trainer, callbacks)
        client_config = Config().clients._asdict()

        self.random = random.Random()

        self.cpu_freq = 0

        # TODO,需要具体确定数值
        # 样本大小
        self.sample_size = client_config["sample_size"] * M

        # max_speed, min_speed = (
        #     client_config["max_up_speed"],
        #     client_config["min_up_speed"],
        # )
        # __init__函数只会调用一次初始化数组
        self.bandwidth_arr = client_config["bandwidth"]
        self.cpu_freq__arr = client_config["cpu_freq"]
        self.cn_arr = client_config["cn"]
        self.up_speed = 0
        self.cpu_freq = 0
        self.cn = 0
        self.freq_cost_sample = 0
        self.pre_weight = None

        # 量化等级
        self.quantize_n = 8

        self.loss = 0
        self.loss_ = 0
        self.loss_0 = 0
        self.t = 0
        self.t_ = 0

    def do_test(self, weight) -> float:
        """根据weight计算在当前节点上的loss值
        Args:
            weight: dict: 模型权重
        Returns:
            loss: float: 当前模型权重的loss值
        """
        datasource = datasources_registry.get(self.client_id)
        sampler = samplers_registry.get(datasource, self.client_id)
        model = copy.deepcopy(self.trainer.model)
        batch_size = Config().trainer._asdict()["batch_size"]
        device = self.trainer.device
        test_loader = torch.utils.data.DataLoader(
            dataset=datasource.get_train_set(),
            shuffle=False,
            batch_size=batch_size,
            sampler=sampler.get(),
        )

        model.load_state_dict(weight, strict=True)
        model.eval()
        model.to(device)
        loss, count = 0, 0
        with torch.no_grad():
            for examples, labels in test_loader:
                examples, labels = examples.to(device), labels.to(device)

                outputs = model(examples)
                loss += self.trainer.get_loss_criterion()(outputs, labels)
                count += 1
        return loss / count

    @staticmethod
    def add(weight, deltas):
        ret = {}
        for name, current_weight in weight.items():
            ret[name] = current_weight + deltas[name]
        return ret

    def quantize(self, weight, deltas, n):
        """量化deltas并求和

        Args:
            weight (dict): 原模型参数
            deltas (dict): 变化值
            n (int): 量化等级

        Returns:
            dict: 将deltas量化然后去量化，最后和原模型参数进行相加的结果
        """
        quantize_processor = model_n_quantize.Processor(n=n)
        dequantize_processor = model_n_dequantize.Processor(n=n)

        deltas = dequantize_processor.process(quantize_processor.process(deltas))
        return Client.add(weight, deltas)

    def configure(self) -> None:
        super().configure()
        self.model_size = (
            sys.getsizeof(pickle.dumps(self.trainer.model.cpu().state_dict())) / M
        )

        self.up_speed = self.bandwidth_arr[self.client_id - 1]
        self.cpu_freq = self.cpu_freq__arr[self.client_id - 1]
        self.cn = self.cn_arr[self.client_id - 1]
        self.freq_cost_sample = self.sample_size * self.cn

        self.base_comm_time = self.model_size / self.up_speed

        if self.pre_weight != None and self.quantize_n >= 4:
            deltas = self.calcu_delta_weight(self.trainer.model.cpu().state_dict())
            w = self.quantize(
                self.trainer.model.cpu().state_dict(), deltas, self.quantize_n
            )
            w_ = self.quantize(
                self.trainer.model.cpu().state_dict(), deltas, self.quantize_n // 2
            )
            loss_0 = self.do_test(self.trainer.model.cpu().state_dict())
            loss = self.do_test(w)
            loss_ = self.do_test(w_)
            self.t = self.base_comm_time * math.log2(self.quantize_n) / 32
            self.t_ = self.base_comm_time * math.log2(self.quantize_n) / 2 / 32

            self.loss = loss
            self.loss_ = loss_
            self.loss_0 = loss_0

            # 读取上一轮的情况
            with open("./factor", "r") as f:
                l = f.readline()
                self.quantize_n = min(int(self.quantize_n * float(l)), 32)

        logging.info("[Client #%d]: quantize num %d", self.client_id, self.quantize_n)
        self.processor = model_n_quantize.Processor(n=self.quantize_n)

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """添加当前训练的量化等级"""

        report.quantize_n = self.quantize_n
        # 正常量化
        report.loss = self.loss
        # 量化等级下降
        report.loss_ = self.loss_
        # 聚合前的损失值
        report.loss_0 = self.loss_0
        report.t_compute = (self.freq_cost_sample * report.num_samples) / self.cpu_freq
        print(f"client {self.client_id}: {report.t_compute}")
        report.t = self.t + report.t_compute
        report.t_ = self.t_ + report.t_compute

        report.quantize_n = self.quantize_n

        report.model_size = self.model_size / 32 * self.quantize_n

        report.compute_cost = (
            2
            * (10 ** (-28))
            / 2
            * ((self.cpu_freq) ** 2)
            * (self.freq_cost_sample * report.num_samples)
        )

        report.each_bit_time = self.model_size / 32 / self.up_speed
        self.pre_weight = self.trainer.model.cpu().state_dict()
        return report

    def calcu_delta_weight(self, weight) -> dict[str, torch.Tensor]:
        deltas = {}
        for name, current_weight in weight.items():
            baseline = self.pre_weight[name]

            # Calculate update
            _delta = current_weight - baseline
            deltas[name] = _delta
        return deltas
