# DANet
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class ACBlock(nn.Module):
    def __init__(self, in_channel, out_channel, dilation):
        super(ACBlock,self).__init__()

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel,
                kernel_size=(5, 1), stride=(1, 1), padding=(2 ** (dilation + 1), 0), dilation=(2 ** dilation, 1)))
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel,
                kernel_size=(1, 3), stride=(1, 1), padding=(0, 2 ** dilation), dilation=(1, 2 ** dilation)))


    def forward(self, input):
        output2 = self.conv2(input)
        output3 = self.conv3(input)
        output = output2 + output3

        return output

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, drop_rate, memory_efficient=False, dilation=1):
        super(_DenseLayer, self).__init__()

        self.conv_block = nn.Sequential(
            ACBlock(num_input_features, growth_rate, dilation),
            ACBlock(growth_rate, growth_rate, dilation+1),
            nn.Sequential(
                    nn.BatchNorm2d(growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(growth_rate, growth_rate,
                        kernel_size=(5, 3), stride=(2, 1), padding=2**(dilation+1), dilation=2**(dilation+1),
                        bias=False))
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(num_input_features, growth_rate, kernel_size=1),
            nn.BatchNorm2d(growth_rate),
        )

        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def _pad(self, x, target):
        if x.shape != target.shape:
            padding_1 = target.shape[2] - x.shape[2]
            padding_2 = target.shape[3] - x.shape[3]
            return F.pad(x, (padding_2, 0, padding_1, 0), 'replicate')
        else:
            return x

    def forward(self, input):
        prev_features = input
        residual = self.conv_skip(prev_features)
        new_features = self._pad(self.conv_block(prev_features), prev_features) + residual

        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)

        return new_features

class _DenseBlock(nn.Module):
    __constants__ = ['layers']

    def __init__(self, num_layers, num_input_features, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        self.layers = nn.ModuleDict()
        self.recurrence = 2
        self.num_input_features = num_input_features
        out_channel = num_input_features + num_layers * growth_rate
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                dilation=i
            )
            self.layers['denselayer%d' % (i + 1)] = layer
            num_input_features = growth_rate

        self.conv1x1 = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, growth_rate, 1)
        )

        self.conv_skip = nn.Sequential(
            nn.Conv2d(self.num_input_features, growth_rate, kernel_size=1),
            nn.BatchNorm2d(growth_rate),
        )

    def forward(self, init_features):
        features_list = [init_features]
        residual = self.conv_skip(init_features)
        for name, layer in self.layers.items():
            init_features = layer(init_features)
            features_list.append(init_features)
        output = self.conv1x1(torch.cat(features_list, dim=1)) + residual

        return output

class Multiscale_Module_F(nn.Module):
    def __init__(self, input_channel=2,
                     first_channel=32,
                     first_kernel=(3, 3),
                     scale=3,
                     kl=[(14, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4)],
                     drop_rate=0.1,
                     hidden=None,
                     in_size=None):
        super(Multiscale_Module_F, self).__init__()
        self.first_channel = 32
        self.dual_rnn = []

        self.En1 = _DenseBlock(kl[0][1], self.first_channel, kl[0][0], drop_rate)
        self.pool1 = nn.Sequential(
            nn.Conv2d(kl[0][0], kl[0][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
            )

        self.En2 = _DenseBlock(kl[1][1], kl[0][0], kl[1][0], drop_rate)
        self.pool2 = nn.Sequential(
            nn.Conv2d(kl[1][0], kl[1][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        self.En3 = _DenseBlock(kl[2][1], kl[1][0], kl[2][0], drop_rate)
        self.pool3 = nn.Sequential(
            nn.Conv2d(kl[2][0], kl[2][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        # enter part
        self.Enter = _DenseBlock(kl[3][1], kl[2][0], kl[3][0], drop_rate)

        # decoder part
        self.up3 = nn.ConvTranspose2d(kl[3][0], kl[3][0], kernel_size=(2, 1), stride=(2, 1))
        self.De3 = _DenseBlock(kl[-3][1], kl[3][0], kl[-3][0], drop_rate)

        self.up2 = nn.ConvTranspose2d(kl[-3][0], kl[-3][0], kernel_size=(2, 1), stride=(2, 1))
        self.De2 = _DenseBlock(kl[-2][1], kl[-3][0], kl[-2][0], drop_rate)

        self.up1 = nn.ConvTranspose2d(kl[-2][0], kl[-2][0], kernel_size=(2, 1), stride=(2, 1))
        self.De1 = _DenseBlock(kl[-1][1], kl[-2][0], kl[-1][0], drop_rate)

        self.hglayer1 = HGlayer_F(kl[3][0], kl[4][0], 2)
        self.hglayer2 = HGlayer_F(kl[4][0], kl[5][0], 2)

    def forward(self, input):
        x0 = input
        # encoder part
        x1 = self.En1(x0)
        x_1 = self.pool1(x1)
        x2 = self.En2(x_1)
        x_2 = self.pool2(x2)
        x3 = self.En3(x_2)
        x_3 = self.pool3(x3)

        xy_ = self.Enter(x_3)
        Hg1 = self.hglayer1(x_2, xy_)
        Hg2 = self.hglayer2(x_1, Hg1)

        # decoder part
        y3 = self.up3(xy_)
        y_3 = self.De3(y3)
        y_3 = y_3 + Hg1
        y2 = self.up2(y_3)
        y_2 = self.De2(y2)

        y_2 = y_2 + Hg2
        y1 = self.up1(y_2)
        y_1 = self.De1(y1)

        return y_1

class Multiscale_Module_T(nn.Module):
    def __init__(self, input_channel=2,
                     first_channel=32,
                     first_kernel=(3, 3),
                     scale=3,
                     kl=[(14, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4)],
                     drop_rate=0.1,
                     hidden=None,
                     in_size=None):
        super(Multiscale_Module_T, self).__init__()
        self.first_channel = 32
        self.dual_rnn = []

        self.En1 = _DenseBlock(kl[0][1], self.first_channel, kl[0][0], drop_rate)
        self.pool1 = nn.Sequential(
            nn.Conv2d(kl[0][0], kl[0][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
            )

        self.En2 = _DenseBlock(kl[1][1], kl[0][0], kl[1][0], drop_rate)
        self.pool2 = nn.Sequential(
            nn.Conv2d(kl[1][0], kl[1][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

        self.En3 = _DenseBlock(kl[2][1], kl[1][0], kl[2][0], drop_rate)
        self.pool3 = nn.Sequential(
            nn.Conv2d(kl[2][0], kl[2][0], kernel_size=(1, 1)),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        )

        # enter part
        self.Enter = _DenseBlock(kl[3][1], kl[2][0], kl[3][0], drop_rate)

        # decoder part
        self.up3 = nn.ConvTranspose2d(kl[3][0], kl[3][0], kernel_size=(1, 2), stride=(1, 2))
        self.De3 = _DenseBlock(kl[-3][1], kl[3][0], kl[-3][0], drop_rate)

        self.up2 = nn.ConvTranspose2d(kl[-3][0], kl[-3][0], kernel_size=(1, 2), stride=(1, 2))
        self.De2 = _DenseBlock(kl[-2][1], kl[-3][0], kl[-2][0], drop_rate)

        self.up1 = nn.ConvTranspose2d(kl[-2][0], kl[-2][0], kernel_size=(1, 2), stride=(1, 2))
        self.De1 = _DenseBlock(kl[-1][1], kl[-2][0], kl[-1][0], drop_rate)

        self.hglayer1 = HGlayer_T(kl[3][0], kl[4][0], 2)
        self.hglayer2 = HGlayer_T(kl[4][0], kl[5][0], 2)

    def forward(self, input):
        x0 = input
        # encoder part
        x1 = self.En1(x0)
        x_1 = self.pool1(x1)
        x2 = self.En2(x_1)
        x_2 = self.pool2(x2)
        x3 = self.En3(x_2)
        x_3 = self.pool3(x3)

        xy_ = self.Enter(x_3)
        Hg1 = self.hglayer1(x_2, xy_)
        Hg2 = self.hglayer2(x_1, Hg1)

        # decoder part
        y3 = self.up3(xy_)
        y_3 = self.De3(y3)
        y_3 = y_3 + Hg1
        y2 = self.up2(y_3)
        y_2 = self.De2(y2)

        y_2 = y_2 + Hg2
        y1 = self.up1(y_2)
        y_1 = self.De1(y1)

        return y_1

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, do_activation=True):
        super(Conv, self).__init__()
        if not do_activation:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2))
        else:
            self.model = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.ReLU(),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=(kernel_size - 1) // 2))

    def forward(self, x):
        x = self.model(x)
        return x

class TFattn3(nn.Module):
    def __init__(self):
        super(TFattn3, self).__init__()
        self.bn = nn.BatchNorm2d(16)
        self.t_conv1 = nn.Sequential(
            nn.Conv1d(16, 16, 3, padding=1),
            nn.ReLU()
        )
        self.t_conv2 = nn.Sequential(
            nn.Conv1d(16, 16, 3, padding=1),
            nn.Sigmoid()
        )
        self.f_conv1 = nn.Sequential(
            nn.Conv1d(16, 16, 5, padding=2),
            nn.ReLU()
        )
        self.f_conv2 = nn.Sequential(
            nn.Conv1d(16, 16, 5, padding=2),
            nn.Sigmoid()
        )

    def forward(self,x):
        x = self.bn(x)
        B, C, F, T = x.shape
        a_t = torch.mean(x, dim=-2)  # (b,c,128)
        a_f = torch.mean(x, dim=-1)  # (b,c,360)
        a_t = self.t_conv1(a_t)
        a_t = self.t_conv2(a_t)
        a_t = a_t.unsqueeze(dim=-2)  # (b,c,1,128)

        a_f = self.f_conv1(a_f)
        a_f = self.f_conv2(a_f)
        a_f = a_f.unsqueeze(dim=-1)  # (b,c,360,1)

        a_ft = a_t * a_f  # (b,c,360,128)
        a_f = a_f.repeat(1, 1, 1, T)  # (b,c,360,128)
        a_t = a_t.repeat(1, 1, F, 1)  # (b,c,360,128)

        f_ft = x * a_ft  # (b,c,360,128)
        f_f = x * a_f  # (b,c,360,128)
        f_t = x * a_t  # (b,c,360,128)

        return f_ft, f_f, f_t

class chanenl_attn3(nn.Module):
    def __init__(self, input_num, in_channel, reduction):
        super(chanenl_attn3, self).__init__()
        self.bn = nn.BatchNorm2d(16)
        self.FC1 = nn.ModuleList([
            nn.Sequential(nn.Linear(21 * in_channel, (21 * in_channel) // reduction),
            nn.ReLU6(),
            ) for _ in range(input_num)
        ])
        self.avgpool4 = nn.AdaptiveAvgPool2d(4)
        self.avgpool2 = nn.AdaptiveAvgPool2d(2)
        self.avgpool1 = nn.AdaptiveAvgPool2d(1)
        self.FC2 = nn.ModuleList([
            nn.Linear((21 * in_channel) // reduction, 21 * in_channel) for _ in range(input_num)
        ])

        self.channel_recover0 = nn.Sequential(
            nn.Linear(21 * in_channel, in_channel),
            nn.ReLU6(),
            nn.Linear(in_channel, in_channel)
        )
        self.channel_recover1 = nn.Sequential(
            nn.Linear(21 * in_channel, in_channel),
            nn.ReLU6(),
            nn.Linear(in_channel, in_channel)
        )
        self.channel_recover2 = nn.Sequential(
            nn.Linear(21 * in_channel, in_channel),
            nn.ReLU6(),
            nn.Linear(in_channel, in_channel)
        )

    def forward(self, x):
        # (3,b,c=16,360,128)
        fused = None
        for x_s in x:
            x_s = self.bn(x_s)
            if fused is None:
                fused = x_s
            else:
                fused = fused + x_s
        # (b,c,360,128)
        b, c, _, _ = fused.size()
        y1 = self.avgpool4(fused)  # (b,c,4,4)
        y1 = y1.view(b, 16 * c)  # (b,16c)
        y2 = self.avgpool2(fused)  # (b,c,2,2)
        y2 = y2.view(b, 4 * c)  # (b,4c)
        y3 = self.avgpool1(fused)  # (b,c,1,1)
        y3 = y3.view(b, c)  # (b,c)
        y = torch.concat([y1, y2, y3], dim= 1)  # (b,21c)
        masks = []
        for i in range(len(x)):
            y_squeeze = self.FC1[i](y)  # (b,21c/r)
            y_expansion = self.FC2[i](y_squeeze)  # (b,21c)
            if i == 0:
                y_expansion = self.channel_recover0(y_expansion)  # (b,21c) to (b,c)
            elif i == 1:
                y_expansion = self.channel_recover1(y_expansion)
            elif i == 2:
                y_expansion = self.channel_recover2(y_expansion)
            masks.append(y_expansion.squeeze(dim=-1))  # (b,c) to [(b,c),(b,c),(b,c)]

        # (3,b,c)
        mask_stack = torch.stack(masks, dim=-1)  # (b,c,3)
        mask_stack = nn.Softmax(dim=-2)(mask_stack)
        selected = None
        for i, x_s in enumerate(x):
            mask = mask_stack[:, :, i][:, :, None, None]  # (b,c,1,1)
            x_s = x_s * mask
            if selected is None:
                selected = x_s
            else:
                selected = selected + x_s
        return selected

class HGlayer_F(nn.Module):
    def __init__(self, in_channel, out_channel, rate_recover):
        super(HGlayer_F, self).__init__()
        self.channel_down1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            nn.SELU()
        )
        self.uppool1 =nn.ConvTranspose2d(16, 16, kernel_size=(rate_recover, 1), stride=(rate_recover, 1))
        self.activition = nn.Sigmoid()
        self.channel_down = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, 1),
            nn.SELU()
        )

    def forward(self, x, x_superior):
        x_superior = self.uppool1(self.channel_down1(x_superior))
        weight_superior = self.activition(x_superior)
        x_guidance = x * weight_superior
        x_out = torch.cat([x_guidance, x_superior], dim=1)  # new add
        x_out = self.channel_down(x_out)  # new add
        return x_out

class HGlayer_T(nn.Module):
    def __init__(self, in_channel, out_channel, rate_recover):
        super(HGlayer_T, self).__init__()
        self.channel_down1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1),
            # nn.Conv2d(in_channel, out_channel, 3, padding=1)
            nn.SELU()
        )
        self.uppool1 =nn.ConvTranspose2d(16, 16, kernel_size=(1, rate_recover), stride=(1, rate_recover))
        self.activition = nn.Sigmoid()
        self.channel_down = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, 1),
            nn.SELU()
        )

    def forward(self, x, x_superior):
        x_superior = self.uppool1(self.channel_down1(x_superior))
        weight_superior = self.activition(x_superior)
        x_guidance = x * weight_superior
        # x_out = x_guidance + x_superior
        x_out = torch.cat([x_guidance, x_superior], dim=1)  # new add
        x_out = self.channel_down(x_out)  # new add
        return x_out

class DAnet(nn.Module):
    def __init__(self, input_channel, drop_rate=0.1):
        super(DAnet, self).__init__()
        kl_low = [(16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4), (16, 4)]
        self.bm_layer = nn.Sequential(
            nn.Conv2d(3, 16, (4, 1), stride=(4, 1)),
            nn.SELU(),
            nn.Conv2d(16, 16, (3, 1), stride=(3, 1)),  # 3
            nn.SELU(),
            nn.Conv2d(16, 16, (6, 1), stride=(6, 1)),  # 6
            nn.SELU(),
            nn.Conv2d(16, 1, (5, 1), stride=(5, 1)),
            nn.SELU()
        )
        self.bn = nn.BatchNorm2d(3)
        self.hugnet_F = Multiscale_Module_F(input_channel=input_channel, first_channel=32, first_kernel=(4, 3), scale=3,
                                       kl=kl_low, drop_rate=drop_rate)
        self.hugnet_T = Multiscale_Module_T(input_channel=input_channel, first_channel=32, first_kernel=(4, 3), scale=3,
                                            kl=kl_low, drop_rate=drop_rate)
        self.conv_cfp1 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 1)),
            nn.SELU()
        )
        self.conv_cfp2 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 1)),
            nn.SELU()
        )
        self.conv_cfp3 = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.SELU()
        )
        self.conv_tcfp1 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 1)),
            nn.SELU()
        )
        self.conv_tcfp2 = nn.Sequential(
            nn.Conv2d(16, 16, (1, 1)),
            nn.SELU()
        )
        self.conv_tcfp3 = nn.Sequential(
            nn.Conv2d(16, 16, (3, 3), padding=1),
            nn.SELU()
        )
        self.TFA = TFattn3()
        self.chanenl_attn = chanenl_attn3(input_num=3, in_channel=16, reduction=16)
        self.channel_up = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=2),
            nn.SELU(),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.SELU()
        )

        self.channel_down = nn.Sequential(
            # self.pad,
            nn.Conv2d(32, 16, 5, padding=2),
            nn.SELU(),
            nn.Conv2d(16, 1, 5, padding=2),
            nn.SELU()
        )

    def forward(self, x):
        x = self.bn(x)  # (b,3,360,128)
        bm = x
        bm = self.bm_layer(bm)  # (b,1,1,128)

        x_pre = self.channel_up(x)  # (b,32,360,128)

        f_hugnet = self.hugnet_F(x_pre)  # (b,16,360,128)
        t_hugnet = self.hugnet_T(x_pre)  # (b,16,360,128)

        x_cfp1 = self.conv_cfp1(f_hugnet)
        x_cfp2 = self.conv_cfp2(f_hugnet)
        x_cfp3 = self.conv_cfp3(f_hugnet)

        x_tcfp1 = self.conv_tcfp1(t_hugnet)
        x_tcfp2 = self.conv_tcfp2(t_hugnet)
        x_tcfp3 = self.conv_tcfp3(t_hugnet)

        x_differentias1 = x_cfp2 - x_cfp3
        x_differentias2 = x_tcfp2 - x_tcfp3

        x_differentias = x_differentias1 + x_differentias2

        attn, a_ft, a_tf = self.TFA(x_differentias)
        x_channelattn = self.chanenl_attn([attn, a_ft, a_tf])

        x_preoutput = x_channelattn + x_cfp1
        x_output = torch.concat([x_preoutput, x_tcfp1], dim=1)  # (b,32,360,128)

        x_output = self.channel_down(x_output)  # (b,1,360,128)
        x_output = torch.concat([bm, x_output], dim=2)  # (b,1,361,128)

        output = nn.Softmax(dim=-2)(x_output)
        output_pre = x_output

        return output, output_pre

if __name__ == "__main__":
    input = torch.randn(1, 3, 360, 128)
    model = DAnet(input_channel=3)
    output = model(input)
    print(output[0].shape)
