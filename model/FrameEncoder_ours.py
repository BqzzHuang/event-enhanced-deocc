import torch.nn as nn
from model import up1, down_light, CondConv2D
import torch.nn.functional as F
import torch


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
							  padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

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
                              padding=(kernel_size-stride+1)//2, padding_mode='reflect')
        #Todo:relu
        self.proj2=nn.ReLU()

    def forward(self, x):
        x = self.proj2(self.proj(x))
        return x


class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            # nn.PixelShuffle(patch_size),
            nn.Conv2d(out_chans, out_chans, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
        )

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchUnEmbed_for_upsample(nn.Module):
    def __init__(self, patch_size=4, embed_dim=96, out_dim=64, kernel_size=None):
        super().__init__()
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim, padding_mode='reflect'),
            nn.Conv2d(embed_dim, out_dim*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size),
        )

    def forward(self, x):
        x = self.proj(x)
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
            kernel_size = [1, 3, 5, 7],
            embed_kernel_size = 3,
            se_ratio = 4,
            scale_ratio = 2,
            split_num = 4
            ):
        super(TokenMixer, self).__init__()
        self.dim = dim
        self.c_down_ratio = se_ratio
        self.scale_ratio = scale_ratio
        self.split_num = split_num
        self.dim1 = dim * scale_ratio // split_num
        # PW first or DW first?
        self.conv_embed_1 = nn.Sequential(
            nn.Conv2d(self.dim, self.dim*scale_ratio, kernel_size=1),
            nn.GELU()
        )
        self.conv_embed_2 = nn.Sequential(
            nn.Conv2d(self.dim*scale_ratio, self.dim, kernel_size=1),
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
            nn.Conv2d(self.dim1, self.dim1, kernel_size=kernel_size[1], padding=kernel_size[1] // 2, groups=self.dim1, padding_mode='reflect'),
            nn.GELU()
        )
        self.conv1_3 = nn.Sequential(
            # nn.Conv2d(self.dim1, self.dim1, 1),
            # nn.GELU(),
            nn.Conv2d(self.dim1, self.dim1, kernel_size=kernel_size[2], padding=kernel_size[2] // 2, groups=self.dim1, padding_mode='reflect'),
            nn.GELU()
        )
        self.conv1_4 = nn.Sequential(
            # nn.Conv2d(self.dim1, self.dim1, 1),
            # nn.GELU(),
            nn.Conv2d(self.dim1, self.dim1, kernel_size=kernel_size[3], padding=kernel_size[3] // 2, groups=self.dim1, padding_mode='reflect'),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim*scale_ratio, dim*scale_ratio // self.c_down_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim*scale_ratio // self.c_down_ratio, dim*scale_ratio, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_embed_1(x) #B*C*H*W
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
            norm_layer = nn.BatchNorm2d,
            token_mixer = TokenMixer,
            convffn = ConvFFN,
            mlp_ratio = 4,
            mixer_kernel_size = [1, 3, 5, 7],
            ffn_kernel_size = 3,
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


class FrameEncoder_ours(nn.Module):
    def __init__(self, inChannels, patch_size=1,
                 embed_dim=[48, 96, 192], depth = [1, 2, 3], mlp_ratios=[2., 2., 4.], embed_kernel_size=3, downsample_kernel_size=None):
        super().__init__()
        if downsample_kernel_size is None:
            downsample_kernel_size = 4
        self.patch_embed = PatchEmbed(patch_size=patch_size, in_chans=inChannels,
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

    def forward(self, x):

        x = self.patch_embed(x)
        x = self.layer1(x)
        copy1 = x

        x = self.downsample1(x)
        x = self.layer2(x)
        copy2 = x

        x = self.downsample2(x)
        x = self.layer3(x)

        return [copy1, copy2, x]



class FrameEncoder(nn.Module):
    def __init__(self, inChannels, size_adapter=None, layers=[128, 128, 256, 256, 512, 512, 512], norm='BN'):
        super().__init__()
        self._size_adapter = size_adapter
        self.conv1 = nn.Sequential(CondConv2D(inChannels, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64)) if norm else CondConv2D(inChannels, 64, 3, stride=1, padding=1)
        self.layers = layers

        self.down0 = down_light(64, self.layers[0], norm=norm)
        for k in range(1, len(self.layers)):
            setattr(self, 'down%d'%k, down_light(self.layers[k-1], self.layers[k], norm=norm))

    def forward(self, x):
        output = []
        x = F.leaky_relu(self.conv1(x), negative_slope=0.1)
        output.append(x)

        for k in range(len(self.layers)):
            x = getattr(self, 'down%d' % k)(x)
            output.append(x)

        return output


class EventFrameDecoder_ours(nn.Module):
    def __init__(self, inChannels=192, embed_dim=[96, 48], depth = [2, 1], mlp_ratios=[2., 2.], embed_kernel_size=3, upwnsample_kernel_size=None,
                 patch_size=1, outChannels=3):
        super(EventFrameDecoder_ours, self).__init__()
        self.upsample1 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=inChannels, out_dim=embed_dim[0])
        self.fusion0 = SKFusion(inChannels)
        self.fusion1 = SKFusion(embed_dim[0], height=3)
        self.layer4 = OurStage(depth=depth[0], in_channels=embed_dim[0], mlp_ratio=mlp_ratios[0],
                               mixer_kernel_size=[1, 3, 5, 7], ffn_kernel_size=3)
        self.upsample2 = PatchUnEmbed_for_upsample(patch_size=2, embed_dim=embed_dim[0],
                                                   out_dim=embed_dim[1])
        self.fusion2 = SKFusion(embed_dim[1], height=3)
        self.layer5 = OurStage(depth=depth[1], in_channels=embed_dim[1], mlp_ratio=mlp_ratios[1],
                               mixer_kernel_size=[1, 3, 5, 7], ffn_kernel_size=3)
        self.patch_unembed = PatchUnEmbed(patch_size=patch_size, out_chans=outChannels,
                                          embed_dim=embed_dim[1], kernel_size=3)

    def forward(self, x1, x2):
        x = self.fusion0([x1[-1], x2[-1]])

        x = self.upsample1(x)

        x = self.fusion1([x, x1[-2], x2[-2]]) + x
        x = self.layer4(x)
        x = self.upsample2(x)

        x = self.fusion2([x, x1[-3], x2[-3]]) + x
        x = self.layer5(x)
        x = self.patch_unembed(x)

        return x



class EventFrameDecoder(nn.Module):
    def __init__(self, outChannels, size_adapter=None, layers=[128,128,256,256,512,512,512], norm='BN'):
        super(EventFrameDecoder, self).__init__()
        self._size_adapter = size_adapter
        num_heads = [8,8,4,4,2,2,1]
        for k in range(1, len(layers)):
            setattr(self, 'up%d'%k, up1(layers[k], layers[k-1], num_heads[k], norm))

        self.up0 = up1(layers[0], 64, num_heads[0], norm)
        self.conv_out = CondConv2D(64, outChannels, kernel_size=3, stride=1, padding=1)
        self.layers = layers

        for k in range(1, len(layers)):
            if norm:
                setattr(self, 'fuse%d'%k, nn.Sequential(nn.Conv2d(layers[k-1]*2, int(layers[k-1]/2), 3, padding=1), nn.BatchNorm2d(int(layers[k-1]/2)), nn.LeakyReLU(negative_slope=0.1),
                                                        nn.Conv2d(int(layers[k-1]/2), layers[k-1], 3, padding=1), nn.BatchNorm2d(layers[k-1]), nn.LeakyReLU(negative_slope=0.1)))
            else:
                setattr(self, 'fuse%d'%k, nn.Sequential(nn.Conv2d(layers[k-1]*2, int(layers[k-1]/2), 3, padding=1), nn.LeakyReLU(negative_slope=0.1),
                                                        nn.Conv2d(int(layers[k-1]/2), layers[k-1], 3, padding=1), nn.LeakyReLU(negative_slope=0.1)))
        self.fuse0 = nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.1),
                                   nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(negative_slope=0.1)) if norm else \
            nn.Sequential(nn.Conv2d(128, 32, 3, padding=1), nn.LeakyReLU(negative_slope=0.1), nn.Conv2d(32, 64, 3, padding=1), nn.LeakyReLU(negative_slope=0.1))

    def forward(self, input1, input2):
        x = torch.mul(input1[-1], input2[-1])

        for k in range(1, len(self.layers)+1):
            x1 = torch.cat([input1[-1-k], input2[-1-k]], dim=1)
            x2 = getattr(self, 'fuse%d'%(len(self.layers)-k))(x1) + input1[-1-k] + input2[-1-k]
            x = getattr(self, 'up%d'%(len(self.layers)-k))(x, x2)

        x = self.conv_out(x)
        return x


# if __name__ == '__main__':
#     e = FrameEncoder(33)
#     d = SingleDecoder(3)
#
#     x = torch.zeros([2,33,256,256])
#     o = e(x)
#     op = d(o)
