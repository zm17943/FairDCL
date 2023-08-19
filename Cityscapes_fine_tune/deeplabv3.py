import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import os
import collections

from aspp import ASPP, ASPP_Bottleneck


class DeepLabV3(nn.Module):
    def __init__(self, model_id, project_dir):
        super(DeepLabV3, self).__init__()

        self.num_classes = 20

        self.model_id = model_id
        self.project_dir = project_dir
        self.create_model_dirs()

        #self.resnet = nn.Sequential(*list(base_model.children()))[:-2]  # NOTE! specify the type of ResNet here
        #self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) # NOTE! if you use ResNet50-152, set self.aspp = ASPP_Bottleneck(num_classes=self.num_classes) instead
        
        base_model = models.resnet50(pretrained=True)
        old_state_dict = base_model.state_dict()
        checkpoint = torch.load('checkpoint_0024_DI.pth.tar')
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if 'encoder_k' in k:
                continue
            if 'module' in k:
                k = k.replace('module.', '')
            if 'encoder_q' in k:
                k = k.replace('encoder_q.', '')
            if 'fc.2.weight' in k:
                continue
                k = k.replace('fc.2.weight', 'fc.weight')
                v = old_state_dict['fc.weight']
            if 'fc.2.bias' in k:
                continue
                k = k.replace('fc.2.bias', 'fc.bias')
                v = old_state_dict['fc.bias']
            if (k in ["queue", "queue_ptr", "fc.0.weight", "fc.0.bias"]):
                continue
            new_state_dict[k]=v
        self.model = torchvision.models.segmentation.fcn_resnet50(num_classes=20)
        self.model.backbone.load_state_dict(new_state_dict)

    def forward(self, x):
        # (x has shape (batch_size, 3, h, w))

        #h = x.size()[2]
        #w = x.size()[3]

        #feature_map = self.resnet(x) # (shape: (batch_size, 512, h/16, w/16)) (assuming self.resnet is ResNet18_OS16 or ResNet34_OS16. If self.resnet is ResNet18_OS8 or ResNet34_OS8, it will be (batch_size, 512, h/8, w/8). If self.resnet is ResNet50-152, it will be (batch_size, 4*512, h/16, w/16))

        #output = self.aspp(feature_map) # (shape: (batch_size, num_classes, h/16, w/16))

        #output = F.upsample(output, size=(h, w), mode="bilinear") # (shape: (batch_size, num_classes, h, w))
        output = self.model(x)['out']

        return output

    def create_model_dirs(self):
        self.logs_dir = self.project_dir + "/training_logs"
        self.model_dir = self.logs_dir + "/model_%s" % self.model_id
        self.checkpoints_dir = self.model_dir + "/checkpoints"
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            os.makedirs(self.checkpoints_dir)
