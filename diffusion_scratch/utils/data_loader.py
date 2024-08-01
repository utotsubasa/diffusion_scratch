import torch.utils.data.dataloader as torchloader
import torch


class DataLoader(torchloader.DataLoader[torch.Tensor]):
    """A base class of DataLoader"""

    ...
