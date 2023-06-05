import torch.nn as nn
from model.EventEncoder_ours_gray import EventEncoder_ours
from model.FrameEncoder_ours_gray import FrameEncoder_ours, EventFrameDecoder_ours
import torch


class model(nn.Module):
    def __init__(self, netParams, inChannels):
        super().__init__()
        self.EventEncoder = EventEncoder_ours(netParams, embed_dim=[24, 48, 96], depth = [1, 2, 3], mlp_ratios=[2., 2., 4.], hidden_number=32)
        self.FrameEncoder = FrameEncoder_ours(inChannels, embed_dim=[24, 48, 96], depth = [1, 2, 3], mlp_ratios=[2., 2., 4.])
        self.EventFrameDecoder = EventFrameDecoder_ours(inChannels=96, embed_dim=[48, 24], depth = [2, 1], mlp_ratios=[2., 2.])

    def forward(self, event, frames):
        f_e = self.EventEncoder(event)
        f_f = self.FrameEncoder(frames)
        output = self.EventFrameDecoder(f_e, f_f)
        return output

