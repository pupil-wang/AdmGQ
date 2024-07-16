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
        if self.n == 16:
            return layer.to(layer.to(torch.float32))
        elif self.n == 8:
            return torch.dequantize(layer)
        else:
            return layer