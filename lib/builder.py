import _init_paths

import lib
import torch
import torch.nn as nn
class VOSNet(nn.Module):
    def __init__(self, opt):
        super(VOSNet, self).__init__()
        self.opt = opt
        self.bn = nn.BatchNorm2d
        self.num_points = opt.num_points

        if opt.encoder == 'swin_tiny': 
            self.backbone_x = lib.swin_tiny()
            self.backbone_y = lib.swin_tiny()
        elif opt.encoder == 'mit_b0':
            self.backbone_x = lib.mit_b0()
            self.backbone_y = lib.mit_b0()
        elif opt.encoder == 'efficientnetV2':
            self.backbone_x = lib.efficientnetv2_s()
            self.backbone_y = lib.efficientnetv2_s()
        if opt.encoder == 'swin_tiny':
            feature_channels = [96, 192, 384, 768]
            embedding_dim = 192
        if opt.encoder == 'mit_b0':
            feature_channels = [32, 64, 160, 256]
            embedding_dim = 64
        if opt.encoder == 'efficientnetV2':
            feature_channels = [48, 64, 160, 256]
            embedding_dim = 64
        self.project1 = nn.Conv2d(192, 96, 1, 1)
        self.project2 = nn.Conv2d(384, 192, 1, 1)
        self.project3 = nn.Conv2d(768, 384, 1, 1)
        self.project4 = nn.Conv2d(1536, 768, 1, 1)
        self.mask2former = lib.MaskFormerModel()
        self.decode_head = lib.SegFormerHead(feature_channels, embedding_dim, self.bn, opt.seghead_dropout, 'bilinear', False)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.init_backbone()

    def init_backbone(self):
        global saved_state_dict
        if self.opt.encoder == 'swin_tiny':
            saved_state_dict = torch.load('/home/qdk/code/MaskVOS-master/tools/pretrained_model/swin_tiny_patch4_window7_224.pth', map_location='cpu')
        if self.opt.encoder == 'mit_b0':
            saved_state_dict = torch.load('./pretrained_model/mit_b0.pth', map_location='cpu')
        if self.opt.encoder == 'efficientnetV2':
            saved_state_dict = torch.load('/home/qdk/code/MaskVOS-master/tools/pretrained_model/pre_efficientnetv2-s.pth')
        if 'swin' in self.opt.encoder:
            self.backbone_x.load_state_dict(saved_state_dict['model'], strict=False)
            self.backbone_y.load_state_dict(saved_state_dict['model'], strict=False)
        elif 'mit' in self.opt.encoder:
            self.backbone_x.load_state_dict(saved_state_dict, strict=False)
            self.backbone_y.load_state_dict(saved_state_dict, strict=False)
        elif 'efficientnetV2' in self.opt.encoder:
            keys = list(saved_state_dict.keys())
            for key in keys:
                if 'head' in key:
                    del saved_state_dict[key]
            self.backbone_x.load_state_dict(saved_state_dict, strict=False)
            self.backbone_y.load_state_dict(saved_state_dict, strict=False)
    def forward(self, x, y):
        output_x = self.backbone_x(x)
        output_y = self.backbone_y(y)
        fuse1 = self.project1(torch.concat([output_x[0], output_y[0]], 1))
        fuse2 = self.project2(torch.concat([output_x[1], output_y[1]], 1))
        fuse3 = self.project3(torch.concat([output_x[2], output_y[2]], 1))
        fuse4 = self.project4(torch.concat([output_x[3], output_y[3]], 1))
        input_z = dict()
        input_z['res1'] = fuse1
        input_z['res2'] = fuse2
        input_z['res3'] = fuse3
        input_z['res4'] = fuse4

        output_z2 = self.mask2former(input_z)
        z = self.decode_head(output_z2)
        return z

