import torch
from torch import nn

from utils import problem


class MSELossLayer(nn.Module):
    @problem.tag("hw3-A")
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calculate MSE between predictions and truth values.

        Args:
            y_pred (torch.Tensor): More specifically a torch.FloatTensor, with some shape.
                Input data.
            y_true (torch.Tensor): More specifically a torch.FloatTensor, with the same shape as y_pred.
                Input data.

        Returns:
            torch.Tensor: More specifically a SINGLE VALUE torch.FloatTensor (i.e. with shape (1,)).
                Result.

        Note:
            - YOU ARE NOT ALLOWED to use torch.nn.MSELoss (or it's functional counterparts) in this class
            - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
        """
        result = torch.square(y_pred - y_true)
        #result = result ** 2
        mean = torch.mean(result, dtype=float).float()
        #print(output)
        return mean

