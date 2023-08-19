import argparse
import torch
import torch.optim as optim
import os.path as osp
import sys
from module.Encoder import Deeplabv2
from torchvision import models
# from module.Encoder import UNet
from module.Encoder import UNetWithResnet50Encoder
from matplotlib import pyplot as plt
from data.loveda import LoveDALoader
from ever.core.iterator import Iterator
from utils.tools import *
from torch.nn.utils import clip_grad
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from eval import evaluate
from eval import evaluate_rural
from eval import test_target
from eval import test_source
from torchsummary import summary
from ever.util import param_util
import torch.nn as nn
#import cv2 as cv
import torch.backends.cudnn as cudnn
parser = argparse.ArgumentParser(description='Run Baseline methods.')

parser.add_argument('--config_path',  type=str,
                    help='config path')
parser.add_argument('--test_model_path',  type=str,
                    help='path of model to be tested')
args = parser.parse_args()
cfg = import_config(args.config_path)

num_classes = 8

def main():
    """Create the model and start the training."""
    os.makedirs(cfg.SNAPSHOT_DIR, exist_ok=True)
    logger = get_console_file_logger(name='Deeplabv2', logdir=cfg.SNAPSHOT_DIR)
    # Create Network
    
    # model = Deeplabv2(dict(
    #   backbone=dict(
    #             resnet_type='resnet50',
    #             output_stride=16,
    #             pretrained=True,
    #         ),
    #         multi_layer=False,
    #         cascade=False,
    #         use_ppm=False,
    #         ppm=dict(
    #             num_classes=7,
    #             use_aux=False,
    #             norm_layer=nn.BatchNorm2d,
    #         ),
    #         inchannels=2048,
    #         num_classes=7
    # ))
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    

    base_model = models.resnet50(pretrained=True)
    old_state_dict = base_model.state_dict()

    # model_state_dict = torch.load('log/baseline/2urban/URBAN20000.pth')
    # model.load_state_dict(model_state_dict,  strict=True)

    checkpoint = torch.load('checkpoint_0050_JSD.pth.tar')

    
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    
    
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
    
    '''
    names = ["resnet.conv1.weight", "resnet.bn1.weight", "resnet.bn1.bias", "resnet.bn1.running_mean", "resnet.bn1.running_var", "resnet.layer1.0.conv1.weight", "resnet.layer1.0.bn1.weight", "resnet.layer1.0.bn1.bias", "resnet.layer1.0.bn1.running_mean", "resnet.layer1.0.bn1.running_var", "resnet.layer1.0.conv2.weight", "resnet.layer1.0.bn2.weight", "resnet.layer1.0.bn2.bias", "resnet.layer1.0.bn2.running_mean", "resnet.layer1.0.bn2.running_var", "resnet.layer1.0.conv3.weight", "resnet.layer1.0.bn3.weight", "resnet.layer1.0.bn3.bias", "resnet.layer1.0.bn3.running_mean", "resnet.layer1.0.bn3.running_var", "resnet.layer1.0.downsample.0.weight", "resnet.layer1.0.downsample.1.weight", "resnet.layer1.0.downsample.1.bias", "resnet.layer1.0.downsample.1.running_mean", "resnet.layer1.0.downsample.1.running_var", "resnet.layer1.1.conv1.weight", "resnet.layer1.1.bn1.weight", "resnet.layer1.1.bn1.bias", "resnet.layer1.1.bn1.running_mean", "resnet.layer1.1.bn1.running_var", "resnet.layer1.1.conv2.weight", "resnet.layer1.1.bn2.weight", "resnet.layer1.1.bn2.bias", "resnet.layer1.1.bn2.running_mean", "resnet.layer1.1.bn2.running_var", "resnet.layer1.1.conv3.weight", "resnet.layer1.1.bn3.weight", "resnet.layer1.1.bn3.bias", "resnet.layer1.1.bn3.running_mean", "resnet.layer1.1.bn3.running_var", "resnet.layer1.2.conv1.weight", "resnet.layer1.2.bn1.weight", "resnet.layer1.2.bn1.bias", "resnet.layer1.2.bn1.running_mean", "resnet.layer1.2.bn1.running_var", "resnet.layer1.2.conv2.weight", "resnet.layer1.2.bn2.weight", "resnet.layer1.2.bn2.bias", "resnet.layer1.2.bn2.running_mean", 
         "resnet.layer1.2.bn2.running_var", "resnet.layer1.2.conv3.weight", "resnet.layer1.2.bn3.weight", "resnet.layer1.2.bn3.bias", "resnet.layer1.2.bn3.running_mean", "resnet.layer1.2.bn3.running_var", "resnet.layer2.0.conv1.weight", "resnet.layer2.0.bn1.weight", "resnet.layer2.0.bn1.bias", "resnet.layer2.0.bn1.running_mean", "resnet.layer2.0.bn1.running_var", "resnet.layer2.0.conv2.weight", "resnet.layer2.0.bn2.weight", "resnet.layer2.0.bn2.bias", "resnet.layer2.0.bn2.running_mean", "resnet.layer2.0.bn2.running_var", "resnet.layer2.0.conv3.weight", "resnet.layer2.0.bn3.weight", "resnet.layer2.0.bn3.bias", "resnet.layer2.0.bn3.running_mean", "resnet.layer2.0.bn3.running_var", "resnet.layer2.0.downsample.0.weight", "resnet.layer2.0.downsample.1.weight", "resnet.layer2.0.downsample.1.bias", "resnet.layer2.0.downsample.1.running_mean", "resnet.layer2.0.downsample.1.running_var", "resnet.layer2.1.conv1.weight", "resnet.layer2.1.bn1.weight", "resnet.layer2.1.bn1.bias", "resnet.layer2.1.bn1.running_mean", "resnet.layer2.1.bn1.running_var", "resnet.layer2.1.conv2.weight", "resnet.layer2.1.bn2.weight", "resnet.layer2.1.bn2.bias", "resnet.layer2.1.bn2.running_mean", "resnet.layer2.1.bn2.running_var", "resnet.layer2.1.conv3.weight", "resnet.layer2.1.bn3.weight", "resnet.layer2.1.bn3.bias", "resnet.layer2.1.bn3.running_mean", "resnet.layer2.1.bn3.running_var", "resnet.layer2.2.conv1.weight", "resnet.layer2.2.bn1.weight", "resnet.layer2.2.bn1.bias", 
         "resnet.layer2.2.bn1.running_mean", "resnet.layer2.2.bn1.running_var", "resnet.layer2.2.conv2.weight", "resnet.layer2.2.bn2.weight", "resnet.layer2.2.bn2.bias", "resnet.layer2.2.bn2.running_mean", "resnet.layer2.2.bn2.running_var", "resnet.layer2.2.conv3.weight", "resnet.layer2.2.bn3.weight", "resnet.layer2.2.bn3.bias", "resnet.layer2.2.bn3.running_mean", "resnet.layer2.2.bn3.running_var", "resnet.layer2.3.conv1.weight", "resnet.layer2.3.bn1.weight", "resnet.layer2.3.bn1.bias", "resnet.layer2.3.bn1.running_mean", "resnet.layer2.3.bn1.running_var", "resnet.layer2.3.conv2.weight", "resnet.layer2.3.bn2.weight", "resnet.layer2.3.bn2.bias", "resnet.layer2.3.bn2.running_mean", "resnet.layer2.3.bn2.running_var", "resnet.layer2.3.conv3.weight", "resnet.layer2.3.bn3.weight", "resnet.layer2.3.bn3.bias", "resnet.layer2.3.bn3.running_mean", "resnet.layer2.3.bn3.running_var", "resnet.layer3.0.conv1.weight", "resnet.layer3.0.bn1.weight", "resnet.layer3.0.bn1.bias", "resnet.layer3.0.bn1.running_mean", "resnet.layer3.0.bn1.running_var", "resnet.layer3.0.conv2.weight", "resnet.layer3.0.bn2.weight", "resnet.layer3.0.bn2.bias", "resnet.layer3.0.bn2.running_mean", "resnet.layer3.0.bn2.running_var", "resnet.layer3.0.conv3.weight", "resnet.layer3.0.bn3.weight", "resnet.layer3.0.bn3.bias", "resnet.layer3.0.bn3.running_mean", "resnet.layer3.0.bn3.running_var", "resnet.layer3.0.downsample.0.weight", "resnet.layer3.0.downsample.1.weight", 
         "resnet.layer3.0.downsample.1.bias", "resnet.layer3.0.downsample.1.running_mean", "resnet.layer3.0.downsample.1.running_var", "resnet.layer3.1.conv1.weight", "resnet.layer3.1.bn1.weight", "resnet.layer3.1.bn1.bias", "resnet.layer3.1.bn1.running_mean", "resnet.layer3.1.bn1.running_var", "resnet.layer3.1.conv2.weight", "resnet.layer3.1.bn2.weight", "resnet.layer3.1.bn2.bias", "resnet.layer3.1.bn2.running_mean", "resnet.layer3.1.bn2.running_var", "resnet.layer3.1.conv3.weight", "resnet.layer3.1.bn3.weight", "resnet.layer3.1.bn3.bias", "resnet.layer3.1.bn3.running_mean", "resnet.layer3.1.bn3.running_var", "resnet.layer3.2.conv1.weight", "resnet.layer3.2.bn1.weight", "resnet.layer3.2.bn1.bias", "resnet.layer3.2.bn1.running_mean", "resnet.layer3.2.bn1.running_var", "resnet.layer3.2.conv2.weight", "resnet.layer3.2.bn2.weight", "resnet.layer3.2.bn2.bias", "resnet.layer3.2.bn2.running_mean", "resnet.layer3.2.bn2.running_var", "resnet.layer3.2.conv3.weight", "resnet.layer3.2.bn3.weight", "resnet.layer3.2.bn3.bias", "resnet.layer3.2.bn3.running_mean", "resnet.layer3.2.bn3.running_var", "resnet.layer3.3.conv1.weight", "resnet.layer3.3.bn1.weight", "resnet.layer3.3.bn1.bias", "resnet.layer3.3.bn1.running_mean", "resnet.layer3.3.bn1.running_var", "resnet.layer3.3.conv2.weight", "resnet.layer3.3.bn2.weight", "resnet.layer3.3.bn2.bias", "resnet.layer3.3.bn2.running_mean", "resnet.layer3.3.bn2.running_var", "resnet.layer3.3.conv3.weight", 
         "resnet.layer3.3.bn3.weight", "resnet.layer3.3.bn3.bias", "resnet.layer3.3.bn3.running_mean", "resnet.layer3.3.bn3.running_var", "resnet.layer3.4.conv1.weight", "resnet.layer3.4.bn1.weight", "resnet.layer3.4.bn1.bias", "resnet.layer3.4.bn1.running_mean", "resnet.layer3.4.bn1.running_var", "resnet.layer3.4.conv2.weight", "resnet.layer3.4.bn2.weight", "resnet.layer3.4.bn2.bias", "resnet.layer3.4.bn2.running_mean", "resnet.layer3.4.bn2.running_var", "resnet.layer3.4.conv3.weight", "resnet.layer3.4.bn3.weight", "resnet.layer3.4.bn3.bias", "resnet.layer3.4.bn3.running_mean", "resnet.layer3.4.bn3.running_var", "resnet.layer3.5.conv1.weight", "resnet.layer3.5.bn1.weight", "resnet.layer3.5.bn1.bias", "resnet.layer3.5.bn1.running_mean", "resnet.layer3.5.bn1.running_var", "resnet.layer3.5.conv2.weight", "resnet.layer3.5.bn2.weight", "resnet.layer3.5.bn2.bias", "resnet.layer3.5.bn2.running_mean", "resnet.layer3.5.bn2.running_var", "resnet.layer3.5.conv3.weight", "resnet.layer3.5.bn3.weight", "resnet.layer3.5.bn3.bias", "resnet.layer3.5.bn3.running_mean", "resnet.layer3.5.bn3.running_var", "resnet.layer4.0.conv1.weight", "resnet.layer4.0.bn1.weight", "resnet.layer4.0.bn1.bias", "resnet.layer4.0.bn1.running_mean", "resnet.layer4.0.bn1.running_var", "resnet.layer4.0.conv2.weight", "resnet.layer4.0.bn2.weight", "resnet.layer4.0.bn2.bias", "resnet.layer4.0.bn2.running_mean", "resnet.layer4.0.bn2.running_var", "resnet.layer4.0.conv3.weight", 
         "resnet.layer4.0.bn3.weight", "resnet.layer4.0.bn3.bias", "resnet.layer4.0.bn3.running_mean", "resnet.layer4.0.bn3.running_var", "resnet.layer4.0.downsample.0.weight", "resnet.layer4.0.downsample.1.weight", "resnet.layer4.0.downsample.1.bias", "resnet.layer4.0.downsample.1.running_mean", "resnet.layer4.0.downsample.1.running_var", "resnet.layer4.1.conv1.weight", "resnet.layer4.1.bn1.weight", "resnet.layer4.1.bn1.bias", "resnet.layer4.1.bn1.running_mean", "resnet.layer4.1.bn1.running_var", "resnet.layer4.1.conv2.weight", "resnet.layer4.1.bn2.weight", "resnet.layer4.1.bn2.bias", "resnet.layer4.1.bn2.running_mean", "resnet.layer4.1.bn2.running_var", "resnet.layer4.1.conv3.weight", "resnet.layer4.1.bn3.weight", "resnet.layer4.1.bn3.bias", "resnet.layer4.1.bn3.running_mean", "resnet.layer4.1.bn3.running_var", "resnet.layer4.2.conv1.weight", "resnet.layer4.2.bn1.weight", "resnet.layer4.2.bn1.bias", "resnet.layer4.2.bn1.running_mean", "resnet.layer4.2.bn1.running_var", "resnet.layer4.2.conv2.weight", "resnet.layer4.2.bn2.weight", "resnet.layer4.2.bn2.bias", "resnet.layer4.2.bn2.running_mean", "resnet.layer4.2.bn2.running_var", "resnet.layer4.2.conv3.weight", "resnet.layer4.2.bn3.weight", "resnet.layer4.2.bn3.bias", "resnet.layer4.2.bn3.running_mean", "resnet.layer4.2.bn3.running_var"]
    
    names = [name.replace('resnet.', '') for name in names]
    new_state_dict = OrderedDict()
    count = 0
    for k, v in state_dict.items():
        if 'num_batches_tracked' in k or 'queue' in k or count >= len(names):
            continue
        new_state_dict[names[count]] = v
        print(k)
        count += 1
    '''



    #base_model.load_state_dict(new_state_dict, strict=True)

    #model = UNetWithResnet50Encoder(base_model=base_model, n_classes=num_classes)

    model = torchvision.models.segmentation.fcn_resnet50(num_classes=num_classes)
    model.backbone.load_state_dict(new_state_dict)


    #param_util.freeze_bn(model.backbone)
    #param_util.freeze_modules(model.backbone)

    for m in model.backbone.modules():
        for name, p in m.named_parameters():
            p.requires_grad = False
        m.eval()
    
    model.cuda()
    model.train()
    print(model)

    # print(summary(model, (4, 256, 256)))


    if (args.test_model_path != ''):
        logger.info("Starting evaluation on target domain...")
        # ckpt_path = args.test_model_path
        ckpt_path = 'log/baseline/2urban/URBAN11000.pth'
        #evaluate(model, cfg, False, ckpt_path, logger)
        #evaluate_rural(model, cfg, False, ckpt_path, logger)
        test_target(model, cfg, False, ckpt_path, logger)

        logger.info("***************************************")
        logger.info("Starting evaluation on target domain rural...")
        test_source(model, cfg, False, ckpt_path, logger)

        ckpt_path = 'log/baseline/2urban/URBAN12000.pth'
        test_target(model, cfg, False, ckpt_path, logger)

        logger.info("***************************************")
        logger.info("Starting evaluation on target domain rural...")
        test_source(model, cfg, False, ckpt_path, logger)

        ckpt_path = 'log/baseline/2urban/URBAN3000.pth'
        test_target(model, cfg, False, ckpt_path, logger)

        logger.info("***************************************")
        logger.info("Starting evaluation on target domain rural...")
        test_source(model, cfg, False, ckpt_path, logger)

        ckpt_path = 'log/baseline/2urban/URBAN4000.pth'
        test_target(model, cfg, False, ckpt_path, logger)

        logger.info("***************************************")
        logger.info("Starting evaluation on target domain rural...")
        test_source(model, cfg, False, ckpt_path, logger)

        ckpt_path = 'log/baseline/2urban/URBAN5000.pth'
        test_target(model, cfg, False, ckpt_path, logger)

        logger.info("***************************************")
        logger.info("Starting evaluation on target domain rural...")
        test_source(model, cfg, False, ckpt_path, logger)

        ckpt_path = 'log/baseline/2urban/URBAN6000.pth'
        test_target(model, cfg, False, ckpt_path, logger)

        logger.info("***************************************")
        logger.info("Starting evaluation on target domain rural...")
        test_source(model, cfg, False, ckpt_path, logger)

        ckpt_path = 'log/baseline/2urban/URBAN7000.pth'
        test_target(model, cfg, False, ckpt_path, logger)

        logger.info("***************************************")
        logger.info("Starting evaluation on target domain rural...")
        test_source(model, cfg, False, ckpt_path, logger)

        ckpt_path = 'log/baseline/2urban/URBAN8000.pth'
        test_target(model, cfg, False, ckpt_path, logger)

        logger.info("***************************************")
        logger.info("Starting evaluation on target domain rural...")
        test_source(model, cfg, False, ckpt_path, logger)

        ckpt_path = 'log/baseline/2urban/URBAN9000.pth'
        test_target(model, cfg, False, ckpt_path, logger)

        logger.info("***************************************")
        logger.info("Starting evaluation on target domain rural...")
        test_source(model, cfg, False, ckpt_path, logger)

        ckpt_path = 'log/baseline/2urban/URBAN10000.pth'
        test_target(model, cfg, False, ckpt_path, logger)

        logger.info("***************************************")
        logger.info("Starting evaluation on target domain rural...")
        test_source(model, cfg, False, ckpt_path, logger)

        ckpt_path = 'log/baseline/2urban/URBAN11000.pth'
        test_target(model, cfg, False, ckpt_path, logger)

        logger.info("***************************************")
        logger.info("Starting evaluation on target domain rural...")
        test_source(model, cfg, False, ckpt_path, logger)

        sys.exit(0)

    #cudnn.enabled = True
    #cudnn.benchmark = True
    logger.info('exp = %s'% cfg.SNAPSHOT_DIR)
    count_model_parameters(model, logger)

    print("=========== Training data config is: ===========")
    print(cfg.SOURCE_DATA_CONFIG)

    trainloader = LoveDALoader(cfg.SOURCE_DATA_CONFIG)
    epochs = cfg.NUM_STEPS_STOP / len(trainloader)
    logger.info('epochs ~= %.3f' % epochs)
    trainloader_iter = Iterator(trainloader)
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    # model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    optimizer.zero_grad()

    #loss_func = F.cross_entropy()

    for i_iter in tqdm(range(cfg.NUM_STEPS_STOP)):
        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, i_iter, cfg)
        # Train with Source
        batch = trainloader_iter.next()
        images_s, labels_s = batch[0]
        
        #print("******************")
        #print(images_s.shape)
        
        pred_source = model(images_s.cuda())
        #print(pred_source)
        #print(pred_source.shape)
        #pred_source = pred_source[0] if isinstance(pred_source, tuple) else pred_source
 
        pred_source = pred_source['out']

        # print("******************")
        # print(pred_source.shape)
        # print(pred_source)

        #Segmentation Loss
        loss = F.cross_entropy(pred_source, labels_s['cls'].long().cuda(), ignore_index=-1, reduction='mean')
        loss.backward()
        
        optimizer.step()
        if i_iter % 100 == 0:
            logger.info('exp = {}'.format(cfg.SNAPSHOT_DIR))
            text = 'iter = %d, loss_seg = %.3f, lr = %.3f'% (
                i_iter, loss, lr)
            logger.info(text)
        if i_iter >= cfg.NUM_STEPS_STOP - 1:
            print('save model ...')
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(cfg.NUM_STEPS_STOP) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path, logger)
            evaluate_rural(model, cfg, True, ckpt_path, logger)
            break
        if i_iter % cfg.EVAL_EVERY == 0 and i_iter != 0:
            print("learning rate is: " + str(lr))
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + str(i_iter) + '.pth')
            torch.save(model.state_dict(), ckpt_path)
            evaluate(model, cfg, True, ckpt_path, logger)
            evaluate_rural(model, cfg, True, ckpt_path, logger)
            model.train()



if __name__ == '__main__':
    seed_torch(42)
    main()
