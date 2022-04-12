import torch
import torch.nn as nn
import torch.nn.functional as F

import models.resnet38_base


class Net(models.resnet38_base.Net):
    def __init__(self, args):
        super().__init__()

        self.dropout7 = torch.nn.Dropout2d(0.5)

        self.fc8 = nn.Conv2d(4096, args.num_classes + 1, 1, bias=False)
        self.f8_3 = torch.nn.Conv2d(512, 64, 1, bias=False)
        self.f8_4 = torch.nn.Conv2d(1024, 128, 1, bias=False)
        self.f9 = torch.nn.Conv2d(192 + 3, 192, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.kaiming_normal_(self.f8_3.weight)
        torch.nn.init.kaiming_normal_(self.f8_4.weight)
        torch.nn.init.xavier_uniform_(self.f9.weight, gain=4)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.fc8, self.f8_3, self.f8_4, self.f9]

    def forward(self, x):
        d = super().forward_as_dict(x)

        cam = self.fc8(d['conv6'])
        n, c, h, w = d['conv6'].size()

        f8_3 = self.f8_3(d['conv4'].detach())
        f8_4 = self.f8_4(d['conv5'].detach())
        x_s = F.interpolate(x, (h, w), mode='bilinear', align_corners=True)
        f = torch.cat([x_s, f8_3, f8_4], dim=1)

        x_att = self.PCM(d['conv6'], f)
        cam_att = self.fc8(x_att)

        self.featmap = cam + cam_att

        _, _, h, w = cam.size()
        pred = F.avg_pool2d(self.featmap[:, :-1], kernel_size=(h, w), padding=0)

        pred = pred.view(pred.size(0), -1)
        return self.featmap, pred

    def PCM(self, cam, f):
        n, c, h, w = f.size()
        cam = F.interpolate(cam, (h, w), mode='bilinear', align_corners=True).view(n, -1, h * w)
        f = self.f9(f)
        f = f.view(n, -1, h * w)
        f = f / (torch.norm(f, dim=1, keepdim=True) + 1e-5)

        aff = F.relu(torch.matmul(f.transpose(1, 2), f), inplace=True)
        aff = aff / (torch.sum(aff, dim=1, keepdim=True) + 1e-5)
        cam_rv = torch.matmul(cam, aff).view(n, -1, h, w)\

        return cam_rv

    def forward_cam(self, x):
        x = super().forward(x)
        cam = self.fc8(x)

        return cam

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups
