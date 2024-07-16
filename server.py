
from plato.servers import fedavg
from plato.config import Config
from model_n_dequantize import Processor



from types import SimpleNamespace


class Server(fedavg.Server):

    def weights_received(self, deltas_received):
        reports = [update.report for update in self.updates]
        # 使用特定bit解压
        decompressed_deltas = [
            Processor(n=report.quantize_n).process(delta)
            for delta, report in zip(deltas_received, reports)
        ]
        return super().weights_received(decompressed_deltas)
