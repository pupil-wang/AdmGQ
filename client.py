import copy
import logging
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
        self.random = random.Random()

        # TODO,需要具体确定数值
        self.t_compute = self.random.randrange(0, 10)
        self.t_communication = self.random.randrange(0, 10)
        self.t_down = self.random.randrange(0, 10)
        self.t_server = self.random.randrange(0, 10)

        self.pre_weight = None


        # 量化等级
        self.quantize_n = 8 


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
            dataset=datasource.get_train_set(), shuffle=False, batch_size=batch_size, sampler=sampler.get()
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
        if self.pre_weight != None and self.quantize_n >= 4:
            deltas = self.calcu_delta_weight(self.trainer.model.cpu().state_dict())
            w = self.quantize(self.trainer.model.cpu().state_dict(), deltas, self.quantize_n)
            w_ = self.quantize(self.trainer.model.cpu().state_dict(), deltas, self.quantize_n // 2)
            loss_0 = self.do_test(self.trainer.model.cpu().state_dict())
            loss = self.do_test(w)
            loss_ = self.do_test(w_)
            t = self.t_compute + self.t_communication + self.t_down
            t_ = self.t_compute + self.t_communication / 2 + self.t_down
            r = loss_0 - loss / t
            r_ = loss_0 - loss_ / t_
            sign_s = np.sign((r_.cpu() - r.cpu()) / (loss.cpu() - loss_.cpu()))
            if sign_s > 1:
                self.quantize_n = max(self.quantize_n // 2, 32)
            else:
                self.quantize_n = self.quantize_n * 2
            
        
        logging.info('[Client #%d]: quantize num %d', self.client_id, self.quantize_n)
        self.processor = model_n_quantize.Processor(n=self.quantize_n)

    def customize_report(self, report: SimpleNamespace) -> SimpleNamespace:
        """添加当前训练的量化等级"""

        report.quantize_n = self.quantize_n
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
