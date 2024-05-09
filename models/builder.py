import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import get_root_logger

logger = get_root_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg, ) -> None:
        super().__init__()