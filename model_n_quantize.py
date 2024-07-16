"""
Implements a Processor for quantizing model parameters.
"""

import torch

from plato.processors import model


class Processor(model.Processor):
    """
    Implements a Processor to quantize model parameters to 16-bit floating points.
    """

    def __init__(self, client_id=0, server_id=0, n: int = 16, **kwargs) -> None:
        super().__init__(client_id, server_id, **kwargs)
        self.n = n

    def _process_layer(self, layer: torch.Tensor) -> torch.Tensor:
        """Dequantizes each individual layer of the model."""
        if self.n == 32:
            return layer
        if self.n == 16:
            return layer.to(torch.bfloat16)
        elif self.n == 1:
            return layer.mean() * layer.sign()
        elif self.n == 8: 
            return torch.quantize_per_tensor_dynamic(
                layer, torch.qint8, reduce_range=False
            )
        else:
            print(self.n)
            delta = (layer.max() - layer.min()) / (1 << self.n) 
            # 没有实际进行压缩，只是进行量化
            return ((layer - layer.min()) // delta)* delta + layer.min()