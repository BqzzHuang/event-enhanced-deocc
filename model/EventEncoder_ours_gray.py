from torch import nn
import slayerSNN as snn
import torch
from model import down_light, SizeAdapter, CondConv2D
import torch.nn.functional as F


def getNeuronConfig(type: str='SRMALPHA', theta: float=10., tauSr: float=1., tauRef: float=1., scaleRef: float=2., tauRho: float=0.3, scaleRho: float=1.):
    """
    :param type:     neuron type
    :param theta:    neuron threshold
    :param tauSr:    neuron time constant
    :param tauRef:   neuron refractory time constant
    :param scaleRef: neuron refractory response scaling (relative to theta)
    :param tauRho:   spike function derivative time constant (relative to theta)
    :param scaleRho: spike function derivative scale factor
    :return: dictionary
    """
    return {
        'type': type,
        'theta': theta,
        'tauSr': tauSr,
        'tauRef': tauRef,
        'scaleRef': scaleRef,
        'tauRho': tauRho,
        'scaleRho': scaleRho,
    }


class SnnEncoder(nn.Module):
    def __init__(self, netParams, hidden_number=32, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4], scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100]):
        super(SnnEncoder, self).__init__()

        self.neuron_config = []
        self.neuron_config.append(getNeuronConfig(theta=theta[0], tauSr=tauSr[0], tauRef=tauRef[0], scaleRef=scaleRef[0], tauRho=tauRho[0], scaleRho=scaleRho[0]))
        self.neuron_config.append(getNeuronConfig(theta=theta[1], tauSr=tauSr[1], tauRef=tauRef[1], scaleRef=scaleRef[1], tauRho=tauRho[1], scaleRho=scaleRho[1]))
        self.neuron_config.append(getNeuronConfig(theta=theta[2], tauSr=tauSr[2], tauRef=tauRef[2], scaleRef=scaleRef[2], tauRho=tauRho[2], scaleRho=scaleRho[2]))

        self.slayer1 = snn.layer(self.neuron_config[0], netParams)
        self.slayer2 = snn.layer(self.neuron_config[1], netParams)
        self.slayer3 = snn.layer(self.neuron_config[2], netParams)

        self.conv1 = self.slayer1.conv(2, 16, kernelSize=3, padding=1)
        self.conv2 = self.slayer2.conv(18, 16, kernelSize=3, padding=1)
        self.conv3 = self.slayer3.conv(18, hidden_number, kernelSize=1, padding=0)

    def forward(self, spikeInput):
        psp0 = self.slayer1.psp(spikeInput)
        psp1 = self.conv1(psp0)
        spikes_1 = self.slayer1.spike(psp1)

        psp2 = torch.cat([self.slayer2.psp(spikes_1), psp0], dim=1)
        psp2 = self.conv2(psp2)
        spikes_2 = self.slayer2.spike(psp2)

        psp3 = torch.cat([self.slayer3.psp(spikes_2), psp0], dim=1)
        psp3 = self.conv3(psp3)
        return psp3


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, kernel_size=3, drop=0.):
        super(ConvFFN, self).__init__()
        out_features = in_features
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            act_layer()
        )
        self.DWConv = nn.Sequential(nn.Conv2d(hidden_features, hidden_features,
                                              kernel_size=kernel_size, padding=kernel_size // 2,
                                              groups=hidden_features, padding_mode='reflect'),
                                    act_layer())
        self.mlp2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1),
        )

    def forward(self, x):
        x = self.mlp1(x)
        x = self.DWConv(x)
        x = self.mlp2(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size - patch_size + 1) // 2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x


class DownSample(nn.Module):
    """
    DownSample: Conv
    B*H*W*C -> B*(H/2)*(W/2)*(2*C)
    """

    def __init__(self, input_dim, output_dim, kernel_size=4, stride=2):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = output_dim

        self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, stride=stride,
                              padding=(kernel_size - stride + 1) // 2, padding_mode='reflect')
        # Todo:relu
        self.proj2 = nn.ReLU()

    def forward(self, x):
        x = self.proj2(self.proj(x))
        return x


class SKFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(SKFusion, self).__init__()

        self.height = height
        d = max(int(dim / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim * height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1)

        attn = self.mlp(self.avg_pool(feats_sum))
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats * attn, dim=1)
        return out


class TokenMixer(nn.Module):
    def __init__(
            self,
            dim,
            kernel_size=[1, 3, 5, 7],
            embed_kernel_size=3,
            se_ratio=4,
            scale_ratio=2,
            split_num=4
    ):
        super(TokenMixer, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.scale_ratio = scale_ratio
        self.split_num = split_num
        self.dim1 = dim * scale_ratio // split_num
        # PW first or DW first?
        self.conv_embed_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim * scale_ratio, kernel_size=1),
            nn.GELU()
        )
        self.conv_embed_2 = nn.Sequential(
            nn.Conv2d(self.dim * scale_ratio, self.dim, kernel_size=1),
            # nn.GELU()
        )
        # self.conv1_1 = nn.Sequential(#PW->DW->
        #     # nn.Conv2d(self.dim1, self.dim1, 1),
        #     # nn.GELU(),
        #     # nn.Conv2d(self.dim1, self.dim1, kernel_size=kernel_size[0], padding=kernel_size[0] // 2, groups=self.dim1, padding_mode='reflect'),
        #     # nn.GELU()
        # )
        self.conv1_2 = nn.Sequential(
            # nn.Conv2d(self.dim1, self.dim1, 1),
            # nn.GELU(),
            nn.Conv2d(self.dim1, self.dim1, kernel_size=kernel_size[1], padding=kernel_size[1] // 2, groups=self.dim1,
                      padding_mode='reflect'),
            nn.GELU()
        )
        self.conv1_3 = nn.Sequential(
            # nn.Conv2d(self.dim1, self.dim1, 1),
            # nn.GELU(),
            nn.Conv2d(self.dim1, self.dim1, kernel_size=kernel_size[2], padding=kernel_size[2] // 2, groups=self.dim1,
                      padding_mode='reflect'),
            nn.GELU()
        )
        self.conv1_4 = nn.Sequential(
            # nn.Conv2d(self.dim1, self.dim1, 1),
            # nn.GELU(),
            nn.Conv2d(self.dim1, self.dim1, kernel_size=kernel_size[3], padding=kernel_size[3] // 2, groups=self.dim1,
                      padding_mode='reflect'),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * scale_ratio, dim * scale_ratio // self.c_down_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * scale_ratio // self.c_down_ratio, dim * scale_ratio, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_embed_1(x)  # B*C*H*W
        x = list(torch.split(x, self.dim1, dim=1))
        # x[0] = self.conv1_1(x[0])
        x[1] = self.conv1_2(x[1])
        x[2] = self.conv1_3(x[2])
        x[3] = self.conv1_4(x[3])
        # x = self.fusion(x) #res?
        x = torch.cat(x, dim=1)
        x = self.conv2(x) * x
        x = self.conv_embed_2(x)

        return x


class OurBlock(nn.Module):
    def __init__(
            self,
            dim,
            norm_layer=nn.BatchNorm2d,
            token_mixer=TokenMixer,
            convffn=ConvFFN,
            mlp_ratio=4,
            mixer_kernel_size=[1, 3, 5, 7],
            ffn_kernel_size=3,
    ):
        super(OurBlock, self).__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.mixer = token_mixer(dim=self.dim, kernel_size=mixer_kernel_size,
                                 se_ratio=8)
        self.ffn = convffn(in_features=self.dim, hidden_features=self.dim * mlp_ratio,
                           kernel_size=ffn_kernel_size)

    def forward(self, x):
        copy = x
        x = self.norm1(x)
        x = self.mixer(x)
        x = x + copy

        copy = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + copy

        return x


class OurStage(nn.Module):
    def __init__(
            self,
            depth=int,
            in_channels=int,
            mlp_ratio=4,
            mixer_kernel_size=[1, 3, 5, 7],
            ffn_kernel_size=3,
    ) -> None:
        """ Constructor method """
        # Call super constructor
        super(OurStage, self).__init__()
        # Init blocks
        self.blocks = nn.Sequential(*[
            OurBlock(
                dim=in_channels,
                norm_layer=nn.BatchNorm2d,
                token_mixer=TokenMixer,
                convffn=ConvFFN,
                mlp_ratio=int(mlp_ratio),
                mixer_kernel_size=mixer_kernel_size,
                ffn_kernel_size=ffn_kernel_size,
            )
            for index in range(depth)
        ])

    def forward(self, input=torch.Tensor) -> torch.Tensor:
        output = self.blocks(input)
        return output


class EventEncoder_ours(nn.Module):
    def __init__(self, netParams, hidden_number , ber=32, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100], patch_size=1,
                 embed_dim=[48, 96, 192], depth = [1, 2, 3], mlp_ratios=[2., 2., 4.], embed_kernel_size=3, downsample_kernel_size=None):
        super(EventEncoder_ours, self).__init__()
        self.SNN = SnnEncoder(netParams, hidden_number, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho)
        self.IN = nn.BatchNorm2d(hidden_number)

        if downsample_kernel_size is None:
            downsample_kernel_size = 4
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=hidden_number,
                                      embed_dim=embed_dim[0], kernel_size=embed_kernel_size)
        self.layer1 = OurStage(depth=depth[0], in_channels=embed_dim[0], mlp_ratio=mlp_ratios[0],
                               mixer_kernel_size=[1, 3, 5, 7], ffn_kernel_size=3)
        self.skip1 = nn.Conv2d(embed_dim[0], embed_dim[0], 1)
        self.downsample1 = DownSample(input_dim=embed_dim[0], output_dim=embed_dim[1],
                                      kernel_size=downsample_kernel_size, stride=2)
        self.layer2 = OurStage(depth=depth[1], in_channels=embed_dim[1], mlp_ratio=mlp_ratios[1],
                               mixer_kernel_size=[1, 3, 5, 7], ffn_kernel_size=3)
        self.skip2 = nn.Conv2d(embed_dim[1], embed_dim[1], 1)
        self.downsample2 = DownSample(input_dim=embed_dim[1], output_dim=embed_dim[2],
                                      kernel_size=downsample_kernel_size, stride=2)
        self.layer3 = OurStage(depth=depth[2], in_channels=embed_dim[2], mlp_ratio=mlp_ratios[2],
                               mixer_kernel_size=[1, 3, 5, 7], ffn_kernel_size=3)

    def forward(self, events):
        bs, _, H, W, Ts = events.shape
        snn_fea = self.SNN(events)
        snn_fea = torch.mean(snn_fea, dim=-1)
        snn_fea = self.IN(snn_fea)

        x = self.patch_embed(snn_fea)
        x = self.layer1(x)
        copy1 = x

        x = self.downsample1(x)
        x = self.layer2(x)
        copy2 = x

        x = self.downsample2(x)
        x = self.layer3(x)

        return [copy1, copy2, x]


class EventEncoder(nn.Module):
    def __init__(self, netParams, hidden_number , ber=32, theta=[3, 5, 10], tauSr=[1, 2, 4], tauRef=[1, 2, 4],
                 scaleRef=[1, 1, 1], tauRho=[1, 1, 10], scaleRho=[10, 10, 100], layers=[128, 128, 128, 256, 256, 512, 512], norm=False):
        super(EventEncoder, self).__init__()
        self.SNN = SnnEncoder(netParams, hidden_number, theta, tauSr, tauRef, scaleRef, tauRho, scaleRho)
        self.IN = nn.BatchNorm2d(hidden_number)

        self.layers = layers
        self.conv1 = nn.Sequential(CondConv2D(hidden_number, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)) if norm else nn.Conv2d(hidden_number, 64, 3, stride=1, padding=1)
        self.down0 = down_light(64, self.layers[0], norm=norm)
        for k in range(1, len(self.layers)):
            setattr(self, 'down%d'%k, down_light(self.layers[k-1], self.layers[k], norm=norm))

    def forward(self, events):
        bs, _, H, W, Ts = events.shape
        snn_fea = self.SNN(events)
        snn_fea = torch.mean(snn_fea, dim=-1)
        snn_fea = self.IN(snn_fea)
        output = []

        x = F.leaky_relu(self.conv1(snn_fea), negative_slope=0.1)
        output.append(x)
        for k in range(len(self.layers)):
            x = getattr(self, 'down%d'%k)(x)
            output.append(x)
        return output


if __name__ == '__main__':
    nb_of_time_bin = 15
    netParams = {'Ts': 1, 'tSample': nb_of_time_bin * 2}
    m = EventEncoder(netParams)
    x = torch.zeros([1, 2, 256, 256, 30]).cuda()
    m.cuda()
    o = m(x)