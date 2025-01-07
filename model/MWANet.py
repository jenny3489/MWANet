import torch
import torch.nn as nn
import seaborn as sns
import mmcv.cnn
import matplotlib.pyplot as plt
try:
	from torch.hub import load_state_dict_from_url
except ImportError:
	from torch.utils.model_zoo import load_url as load_state_dict_from_url
from timm.models.resnet import Bottleneck
from torchvision.ops import deform_conv2d
import math
from functools import partial
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import pywt
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        # x: (B, C, H, W)
        x_avg = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x_max, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        x2 = torch.cat([x_avg, x_max], dim=1)  # (B, 2, H, W)

        sattn = self.sa(x2)  # (B, 1, H, W)
        sattn = torch.sigmoid(sattn)  # 归一化注意力权重到[0, 1]

        # 返回注意力权重
        return sattn  # (B, 1, H, W)
#Dynamic Spatial Attention Projection Module
class DSAP(nn.Module):
    def __init__(
            self,
            d_model,
            d_conv=3,
            expand=2,
            conv_bias=True,
            bias=False,
            device='cuda',
            dtype=torch.float32,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)

        # 多层投影结构（复杂投影）
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # 逐点卷积
        self.pointwise_conv = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,  # 或者其他你需要的输出通道数
            kernel_size=1,
            bias=conv_bias,
            **factory_kwargs
        )

        self.act = nn.SiLU()  # 使用 SiLU 激活函数
        self.spatial_attention = SpatialAttention()  # 使用空间注意力替换通道注意力

        # 多层投影结构（输出投影）
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, **factory_kwargs)  # 保持一致的通道数

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x) # 通过复杂投影层
        x = xz.permute(0, 3, 1, 2).contiguous()  # 调整为 (B, d_inner, H, W)
        x = self.act(self.pointwise_conv (self.conv2d(x))) # (B, d_inner, H, W)
        attention_weights = self.spatial_attention(x)  # (B, 1, H, W)
        mean_attn = torch.mean(attention_weights)
        std_attn = torch.std(attention_weights)
        adaptive_threshold = mean_attn - std_attn

        mask = (attention_weights < adaptive_threshold).float()
        y= x*mask
        # 调整 y 的形状以匹配预期输出
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        out = self.out_proj(y)  # 通过复杂投影层得到最终输出

        return out
#Adaptive Spatial Focus Masking Module
class ASFM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        expand: int = 2,
        num_groups: int = 1,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = nn.GroupNorm(num_groups, hidden_dim)
        self.self_attention = DSAP(d_model=hidden_dim, expand=expand, **kwargs)

    def forward(self, input: torch.Tensor):
        i = input.permute(0, 3, 1, 2)
        i = self.ln_1(i)
        i = i.permute(0, 2, 3, 1)
        x = input + self.self_attention(i)

        return x
def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1),], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)


    return dec_filters, rec_filters
def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x
def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x
# Wavelet Transform Conv(WTConv2d)
class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.wt_function = partial(wavelet_transform, filters=self.wt_filter)
        self.iwt_function = partial(inverse_wavelet_transform, filters=self.iwt_filter)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.stride_filter = nn.Parameter(torch.ones(in_channels, 1, 1, 1), requires_grad=False)
            self.do_stride = lambda x_in: F.conv2d(x_in, self.stride_filter, bias=None, stride=self.stride,
                                                   groups=in_channels)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = self.wt_function(curr_x_ll)
            curr_x_ll = curr_x[:, :, 0, :, :]
            curr_x_hh = curr_x[:, :, 3, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag =self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = self.iwt_function(curr_x)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x
class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=0.1, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)
#Wavelet multispectral feature extractor
class WMSFE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(WMSFE, self).__init__()

        # 深度卷积：使用 WTConv2d 替换 3x3 卷积
        self.depthwise = WTConv2d(in_channels, in_channels, kernel_size=kernel_size)

        # 逐点卷积：使用 1x1 卷积
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
#HybridFeatureExtractionModule 混合特征提取模块
class HFEModule(nn.Module):
    """
    LSS 模块中加入光谱特征提取的小波卷积分支
    """
    def __init__(self, hidden_dim: int = 0, depth: int = 2, use_wavelet: bool = True,
			norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
			expand: int = 2,**kwargs,):
        super(HFEModule, self).__init__()
        self.use_wavelet = use_wavelet
        #Progressive Contextual Attention Refinement
        self.PSAR = nn.ModuleList([
            ASFM(hidden_dim=hidden_dim, expand=expand,
                     norm_layer=norm_layer,
                      **kwargs)
            for i in range(depth)])
        # 定义小波卷积分支

        self.wavelet_branch = WMSFE(in_channels=hidden_dim, out_channels=hidden_dim)

        self.finalconv11 = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.pointwise = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)

        self.finalconv12 = nn.Conv2d(hidden_dim , hidden_dim, kernel_size=1)
    def forward(self, input,draw):
        # 处理输入并提取特征
        out_ssm = input
        #Progressive spatial attention mask refinement
        for blk in self.PSAR:
            out_ssm = blk(out_ssm)
            # if draw:
            #     heatmap_data = out_ssm.mean(dim=-1).detach().cpu().numpy()
            #     heatmap_data = heatmap_data.squeeze(0)
            #     plt.figure(figsize=(6, 6))
            #     plt.axis('off')
            #     plt.imshow(heatmap_data,  cmap='coolwarm')
            #     #plt.colorbar()
            #     #plt.title("Heatmap after block ")
            #     plt.savefig("Net3HL.pdf", format='pdf', pad_inches=0.0, bbox_inches='tight', dpi=1200)
            #     plt.show()
        input_conv = input.permute(0, 3, 1, 2).contiguous()
        #Spectral-Spatial Residual Fusion (SSRF)
        if self.use_wavelet:
            wavelet_output = self.wavelet_branch(input_conv)  # 原始小波变换结果
            # if draw:
            #     wavelet_output1 = wavelet_output.permute(0, 2, 3, 1)
            #     heatmap_data = wavelet_output1.mean(dim=-1).detach().cpu().numpy()
            #     heatmap_data = heatmap_data.squeeze(0)
            #     plt.figure(figsize=(6, 5))
            #     plt.axis('off')
            #     plt.imshow(heatmap_data,  cmap='coolwarm')
            #     # plt.colorbar()
            #     plt.savefig("Net3WHL.pdf", format='pdf', pad_inches=0.0, bbox_inches='tight', dpi=1200)
            #     plt.title("Heatmap after block ")
            #
            #     plt.show()
            # 只使用新提取的特征（跳连）
            output = torch.cat((out_ssm.permute(0, 3, 1, 2).contiguous(),  wavelet_output), dim=1)
            output = self.finalconv11(output).permute(0, 2, 3, 1).contiguous()

            #逐点相乘融合
            # output = out_ssm.permute(0, 3, 1, 2).contiguous() * wavelet_output
            # output = self.finalconv12(output).permute(0, 2, 3, 1).contiguous()


            # output = out_ssm.permute(0, 3, 1, 2) + wavelet_output
            # output = self.finalconv12(output).permute(0, 2, 3, 1).contiguous()

            output = output + input
            # if draw:
            #     heatmap_data = output.mean(dim=-1).detach().cpu().numpy()
            #     heatmap_data = heatmap_data.squeeze(0)
            #     # 绘制热力图
            #     plt.figure(figsize=(6, 5))
            #     plt.imshow(heatmap_data,  cmap='coolwarm')
            #     plt.colorbar()
            #     plt.title("Heatmap after block ")
            #     plt.show()
        else:
            # wavelet_output = self.wavelet_branch(input_conv)
            # output = self.finalconv12(wavelet_output).permute(0, 2, 3, 1).contiguous()



            output = self.finalconv12(out_ssm.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()
            output = output+input


        return output
class HFELayer(nn.Module):
	""" A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """
	def __init__(
			self,
			dim,
			depth,
			expand = 2,
			norm_layer=nn.LayerNorm,
			use_checkpoint=False,
			**kwargs,
	):
		super().__init__()
		self.dim = dim
		self.use_checkpoint = use_checkpoint

		if depth % 3 == 0:
			self.blocks = nn.ModuleList([
				HFEModule(
					hidden_dim=dim,
					norm_layer=norm_layer,
					expand=expand,
					depth=3,
				)
				for i in range(depth//3)])
		elif depth % 2 == 0:
			self.blocks = nn.ModuleList([
				HFEModule(
					hidden_dim=dim,
					norm_layer=norm_layer,
					expand=expand,
					depth=2,
				)
				for i in range(depth // 2)])

		if True:  # is this really applied? Yes, but been overriden later in VSSM!
			def _init_weights(module: nn.Module):
				for name, p in module.named_parameters():
					if name in ["out_proj.weight"]:
						p = p.clone().detach_()  # fake init, just to keep the seed ....
						nn.init.kaiming_uniform_(p, a=math.sqrt(5))

			self.apply(_init_weights)
	def forward(self, x,draw):

		for blk in self.blocks:
			if self.use_checkpoint:
				x = checkpoint.checkpoint(blk, x)
			else:
				x = blk(x, draw)
		return x
#128,128,128 160,160,160
#64,64,64  96,96,96  192,192,192
class Net(nn.Module):
	def __init__(self, dims_decoder=[64,64,64], depths_decoder=[3,4,3],
				 norm_layer = nn.LayerNorm, expand=3):
		super().__init__()
		self.layers_up = nn.ModuleList()
		for i_layer in range(len(depths_decoder)):
			layer = HFELayer(
				dim=dims_decoder[i_layer],
				depth=depths_decoder[i_layer],
				expand=expand,
				norm_layer=norm_layer,
			)
			self.layers_up.append(layer)
		self.apply(self._init_weights)

	def _init_weights(self, m: nn.Module):
		"""
        out_proj.weight which is previously initilized in HSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, HSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
		if isinstance(m, nn.Linear):
			trunc_normal_(m.weight, std=.02)
			if isinstance(m, nn.Linear) and m.bias is not None:
				nn.init.constant_(m.bias, 0)
		elif isinstance(m, nn.LayerNorm):
			nn.init.constant_(m.bias, 0)
			nn.init.constant_(m.weight, 1.0)
	def forward(self, x,draw):
		x = rearrange(x,'b c h w -> b h w c')
		for i, layer in enumerate(self.layers_up):
			x = layer(x,draw)
		return x

class UMAD(nn.Module):
	def __init__(self,):
		super(UMAD, self).__init__()
		self.net_s = Net()
		self.frozen_layers = ['net_t']

	def forward(self, imgs,draw):
		feats_s =self.net_s(imgs,draw)
		feats_s = rearrange(feats_s, 'b h w c->b c h w  ')
		return  feats_s

