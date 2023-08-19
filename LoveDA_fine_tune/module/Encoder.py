import torch.nn as nn
import torch.nn.functional as F
import torch
from ever.util import param_util

class PPMBilinear(nn.Module):
    def __init__(self, num_classes=7, fc_dim=2048,
                 use_aux=False, pool_scales=(1, 2, 3, 6),
                 norm_layer = nn.BatchNorm2d
                 ):
        super(PPMBilinear, self).__init__()
        self.use_aux = use_aux
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        if self.use_aux:
            self.cbr_deepsup = nn.Sequential(
                nn.Conv2d(fc_dim // 2, fc_dim // 4, kernel_size=3, stride=1,
                          padding=1, bias=False),
                norm_layer(fc_dim // 4),
                nn.ReLU(inplace=True),
            )
            self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_classes, 1, 1, 0)
            self.dropout_deepsup = nn.Dropout2d(0.1)


        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim+len(pool_scales)*512, 512,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )


    def forward(self, conv_out):
        #conv5 = conv_out[-1]
        input_size = conv_out.size()
        ppm_out = [conv_out]
        for pool_scale in self.ppm:
            ppm_out.append(F.interpolate(
                pool_scale(conv_out),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))
        
        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)


        if self.use_aux and self.training:
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.dropout_deepsup(_)
            _ = self.conv_last_deepsup(_)

            return x
        else:
            return x

class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, base_model, n_classes=2):
        super().__init__()
        self.resnet = base_model
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(self.resnet.children()))[:3]
        self.input_pool = list(self.resnet.children())[3]
        for bottleneck in list(self.resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1, stride=1)


    def _freeze(self):
        # # param_util.freeze_modules(self.resnet, nn.modules.batchnorm._BatchNorm)
        for m in self.input_block.modules():
            # if not isinstance(m, nn.modules.batchnorm._BatchNorm):
            # print("Freezed input_block")
            m.eval()
        for m in self.input_pool.modules():
            # if not isinstance(m, nn.modules.batchnorm._BatchNorm):
            # print("Freezed input_pool")
            m.eval()
        for m in self.down_blocks.modules():
            # if not isinstance(m, nn.modules.batchnorm._BatchNorm):
            # print("Freezed down_blocks")
            m.eval()
        # param_util.freeze_bn(self.resnet)
        # param_util.freeze_modules(self.resnet)


    def train(self, mode=True):
        super(UNetWithResnet50Encoder, self).train(mode)
        self._freeze()


    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        output_feature_map = x
        x = self.out(x)
        del pre_pools
        if with_output_feature_map:
            return x, output_feature_map
        else:
            return x




# def convrelu(in_channels, out_channels, kernel, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
#         nn.ReLU(inplace=True),
#     )

# class UNet(nn.Module):
#     def __init__(self, n_class, base_model, if_edge=False):
#         super().__init__()

#         self.if_edge = if_edge

#         self.base_model = base_model
#         self.base_layers = list(self.base_model.children())

#         self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
#         self.layer0_1x1 = convrelu(64, 64, 1, 0)
#         # self.layer0_edge = nn.Sequential(
#         #                         nn.Conv2d(1, 16, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
#         #                         nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
#         #                         nn.ReLU(inplace=True)
#         #                     )   
#         # self.layer0_1x1_edge = convrelu(16, 16, 1, 0)

#         self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
#         self.layer1_1x1 = convrelu(256, 256, 1, 0)
#         self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
#         self.layer2_1x1 = convrelu(512, 512, 1, 0)
#         self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
#         self.layer3_1x1 = convrelu(1024, 1024, 1, 0)
#         self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
#         self.layer4_1x1 = convrelu(2048, 2048, 1, 0)

#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.conv_up3 = convrelu(1024 + 2048, 2048, 3, 1)
#         self.conv_up2 = convrelu(512 + 2048, 1024, 3, 1)
#         self.conv_up1 = convrelu(256 + 1024, 1024, 3, 1)
#         self.conv_up0 = convrelu(64 + 1024, 512, 3, 1)

#         self.conv_original_size0 = convrelu(3, 64, 3, 1)
#         self.conv_original_size1 = convrelu(64, 64, 3, 1)
#         if (if_edge == True):
#             self.conv_original_size0_edge = convrelu(1, 8, 3, 1)
#             self.conv_original_size1_edge = convrelu(8, 8, 3, 1)
#             self.conv_original_size2 = convrelu(64 + 512 + 8, 64, 3, 1)      # Added edge 16 channel here
#         else:
#             self.conv_original_size2 = convrelu(64 + 512, 64, 3, 1)

#         self.conv_last = nn.Conv2d(64, n_class, 1)

#     def forward(self, input):

#         if (self.if_edge):
#             input_edge = input[:, 3:4, :, :]
#             input = input[:, 0:3, :, :]
        
#         x_original = self.conv_original_size0(input)
#         x_original = self.conv_original_size1(x_original)

#         if (self.if_edge):
#             edge_original = self.conv_original_size0_edge(input_edge)
#             edge_original = self.conv_original_size1_edge(edge_original)

#         layer0 = self.layer0(input)
#         layer1 = self.layer1(layer0)
#         layer2 = self.layer2(layer1)
#         layer3 = self.layer3(layer2)
#         layer4 = self.layer4(layer3)

#         layer4 = self.layer4_1x1(layer4)
#         x = self.upsample(layer4)
#         layer3 = self.layer3_1x1(layer3)
#         x = torch.cat([x, layer3], dim=1)
#         x = self.conv_up3(x)              # 2048

#         x = self.upsample(x)
#         layer2 = self.layer2_1x1(layer2)  # 512
#         x = torch.cat([x, layer2], dim=1)
#         x = self.conv_up2(x)              # 1024
 
#         x = self.upsample(x)
#         layer1 = self.layer1_1x1(layer1)  # 256
#         x = torch.cat([x, layer1], dim=1)
#         x = self.conv_up1(x)              # 1024

#         x = self.upsample(x)
#         layer0 = self.layer0_1x1(layer0)
#         x = torch.cat([x, layer0], dim=1)
#         x = self.conv_up0(x)              # 512

#         x = self.upsample(x)
#         if (self.if_edge):
#             x = torch.cat([x, x_original, edge_original], dim=1)
#         else:
#             x = torch.cat([x, x_original], dim=1)
#         x = self.conv_original_size2(x)

#         out = self.conv_last(x)

#         return out


from module.resnet import ResNetEncoder
import ever as er
class Deeplabv2(er.ERModule):
    def __init__(self, config):
        super(Deeplabv2, self).__init__(config)
        self.encoder = ResNetEncoder(self.config.backbone)
        if self.config.multi_layer:
            print('Use multi_layer!')
            if self.config.cascade:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm1)
                    self.layer6 = PPMBilinear(**self.config.ppm2)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels // 2, [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
            else:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm)
                    self.layer6 = PPMBilinear(**self.config.ppm)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
        else:
            if self.config.use_ppm:
                self.cls_pred = PPMBilinear(**self.config.ppm)
            else:
                self.cls_pred = self._make_pred_layer(Classifier_Module, self.config.inchannels, [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape
        if self.config.multi_layer:
            if self.config.cascade:
                c3, c4 = self.encoder(x)[-2:]
                x1 = self.layer5(c3)
                x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)
                x2 = self.layer6(c4)
                x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                if self.training:
                    return x1, x2
                else:
                    return (x2).softmax(dim=1)
            else:

                x = self.encoder(x)[-1]
                x1 = self.layer5(x)
                x1 = F.interpolate(x1, (H, W), mode='bilinear', align_corners=True)
                x2 = self.layer6(x)
                x2 = F.interpolate(x2, (H, W), mode='bilinear', align_corners=True)
                if self.training:
                    return x1, x2
                else:
                    return (x1+x2).softmax(dim=1)

        else:
            feat, x = self.encoder(x)[-2:]
            # print("***************************")
            # print(feat.shape)
            # print(x.shape)
            # print(self.training)
            #x = self.layer5(x)
            
            x = self.cls_pred(x)
            #x = self.cls_pred(x)
            x = F.interpolate(x, (H, W), mode='bilinear', align_corners=True)
            #feat = F.interpolate(feat, (H, W), mode='bilinear', align_corners=True)
            if self.training:
                return x, feat
            else:
                return x.softmax(dim=1)


    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
            ),
            multi_layer=False,
            cascade=False,
              use_ppm=False,
            ppm=dict(
                num_classes=7,
                use_aux=False,
                norm_layer=nn.BatchNorm2d,
                
            ),
            inchannels=2048,
            num_classes=7
        ))

