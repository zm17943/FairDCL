import importlib
import logging
import time
import os
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import torch.nn as nn
import shutil
import numpy as np
import random
from skimage.io import imsave
from functools import reduce
from collections import OrderedDict
from module.viz import VisualizeSegmm
def import_config(config_name, prefix='configs'):
    cfg_path = '{}.{}'.format(prefix, config_name)
    m = importlib.import_module(name=cfg_path)
    os.makedirs(m.SNAPSHOT_DIR, exist_ok=True)
    shutil.copy(cfg_path.replace('.', '/')+'.py', os.path.join(m.SNAPSHOT_DIR, 'config.py'))
    return m

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def lr_warmup(base_lr, iter, warmup_iter):
    return base_lr * (float(iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter, cfg):
    if i_iter < cfg.PREHEAT_STEPS:
        lr = lr_warmup(cfg.LEARNING_RATE, i_iter, cfg.PREHEAT_STEPS)
    else:
        lr = lr_poly(cfg.LEARNING_RATE, i_iter, cfg.NUM_STEPS, cfg.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def adjust_learning_rate_D(optimizer, i_iter, cfg):
    if i_iter < cfg.PREHEAT_STEPS:
        lr = lr_warmup(cfg.LEARNING_RATE_D, i_iter, cfg.PREHEAT_STEPS)
    else:
        lr = lr_poly(cfg.LEARNING_RATE_D, i_iter, cfg.NUM_STEPS, cfg.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr

def get_console_file_logger(name, level=logging.INFO, logdir='./baseline'):
    logger = logging.Logger(name)
    logger.setLevel(level=level)
    logger.handlers = []
    BASIC_FORMAT = "%(asctime)s, %(levelname)s:%(name)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel(level=level)

    fhlr = logging.FileHandler(os.path.join(logdir, str(time.time()) + '.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    return logger
    

def bce_loss(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    return F.binary_cross_entropy_with_logits(pred, label)


def _masked_ignore(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int=-1):
    # usually used for BCE-like loss
    y_pred = y_pred.reshape((-1,))
    y_true = y_true.reshape((-1,))
    valid = y_true != ignore_index
    y_true = y_true.masked_select(valid).float()
    y_pred = y_pred.masked_select(valid).float()
    return y_pred, y_true

def select(y_pred: torch.Tensor, y_true: torch.Tensor, ignore_index: int):
    assert y_pred.ndim == 4 and y_true.ndim == 3
    c = y_pred.size(1)
    y_pred = y_pred.permute(0, 2, 3, 1).reshape(-1, c)
    y_true = y_true.reshape(-1)

    valid = y_true != ignore_index

    y_pred = y_pred[valid, :]
    y_true = y_true[valid]
    return y_pred, y_true

def dice_coeff(y_pred, y_true, weights: torch.Tensor, smooth_value: float = 1.0, ):
    y_pred = y_pred[:, weights]
    y_true = y_true[:, weights]
    inter = torch.sum(y_pred * y_true, dim=0)
    z = y_pred.sum(dim=0) + y_true.sum(dim=0) + smooth_value

    return ((2 * inter + smooth_value) / z).mean()

def binary_cross_entropy_with_logits(output: torch.Tensor, target: torch.Tensor, reduction: str = 'mean',
                                     ignore_index: int = -1):
    output, target = _masked_ignore(output, target, ignore_index)
    return F.binary_cross_entropy_with_logits(output, target, reduction=reduction)


def dice_loss_with_logits(y_pred: torch.Tensor, y_true: torch.Tensor,
                          smooth_value: float = 1.0,
                          ignore_index: int = -1,
                          ignore_channel: int = -1):
    c = y_pred.size(1)
    y_pred, y_true = select(y_pred, y_true, ignore_index)
    weight = torch.as_tensor([True] * c, device=y_pred.device)
    if c == 1:
        y_prob = y_pred.sigmoid()
        return 1. - dice_coeff(y_prob, y_true.reshape(-1, 1), weight, smooth_value)
    else:
        y_prob = y_pred.softmax(dim=1)
        y_true = F.one_hot(y_true, num_classes=c)
        if ignore_channel != -1:
            weight[ignore_channel] = False

        return 1. - dice_coeff(y_prob, y_true, weight, smooth_value)


def multi_binary_label(batched_mask: torch.Tensor, num_classes):
    labels = []
    # batched_mask here start from 0 (background)
    for cls in range(0, num_classes):
        labels.append((batched_mask == cls).to(torch.long))
    return labels


def multi_binary_loss(y_pred, y_true, num_classes, reduction='mean', label_smooth=0., dice_scaler=0.5, bce_scaler=0.7):
    labels = multi_binary_label(y_true, num_classes)
    losses = []
    # Skipping the background class
    for cls in range(1, num_classes):
        bipred = y_pred[:, cls, :, :]
        bipred = bipred.reshape(y_pred.size(0), 1, y_pred.size(2), y_pred.size(3)).contiguous()
        if label_smooth > 0:
            bce_loss = label_smoothing_binary_cross_entropy(bipred, labels[cls].reshape_as(bipred).float(),
                                                                     label_smooth)
            dice_loss = label_smoothing_dice_loss_with_logits(bipred, labels[cls], label_smooth)

        else:
            bce_loss = binary_cross_entropy_with_logits(bipred, labels[cls].reshape_as(bipred).float(), ignore_index=-1)
            dice_loss = dice_loss_with_logits(bipred, labels[cls], ignore_index=-1)

        losses.append(bce_loss*bce_scaler+dice_loss*dice_scaler)
        # losses.append(bce_loss+dice_loss)

    if 'sum' == reduction:
        tloss = sum(losses)
    elif 'mean' == reduction:
        tloss = sum(losses) / float(num_classes-1)
    else:
        raise ValueError()

    return tloss

    # if label_smooth > 0:
    #     return dict(mb_sm_loss=tloss)
    # else:
    #     return dict(mb_loss=tloss)


def loss_calc(pred, label, num_classes, reduction='mean'):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    loss = F.cross_entropy(pred, label.long(), ignore_index=-1, reduction=reduction)
    # loss = F.cross_entropy(pred, label.long(), ignore_index=-1, reduction=reduction) + multi_binary_loss(pred, label, num_classes)
    # loss = multi_binary_loss(pred, label, num_classes)
   
    return loss


def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1e-6) + inv_tgt * torch.log(inv_pred))

def bugged_cls_bal_bce(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))


def adjust_confidence(i_iter, max_iter, cfg):
    confi_max, confi_min = cfg['confidence_maxin']
    if cfg['schedule'] == 'ploy':
        confi = (confi_max - confi_min) * ((1 - float(i_iter) / max_iter) ** (cfg['power'])) + confi_min
    else:
        confi = confi_min
    return confi


def som(loss, ratio=0.5, reduction='none'):
    # 1. keep num
    num_inst = loss.numel()
    num_hns = int(ratio * num_inst)
    # 2. select loss
    top_loss, _ = loss.reshape(-1).topk(num_hns, -1)
    if reduction is 'none':
        return top_loss
    else:
        loss_mask = (top_loss != 0)
        # 3. mean loss
        return torch.sum(top_loss[loss_mask]) / (loss_mask.sum() + 1e-6)


def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True
    

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ias_thresh(conf_dict, n_class, alpha, w=None, gamma=1.0):
    if w is None:
        w = np.ones(n_class)
    # threshold
    cls_thresh = np.ones(n_class,dtype = np.float32)
    for idx_cls in np.arange(0, n_class):
        if conf_dict[idx_cls] != None:
            arr = np.array(conf_dict[idx_cls])
            cls_thresh[idx_cls] = np.percentile(arr, 100 * (1 - alpha * w[idx_cls] ** gamma))
    return cls_thresh


import ever as er
from tqdm import tqdm

COLOR_MAP = OrderedDict(
    Background=(255, 255, 255),
    Building=(255, 0, 0),
    Road=(255, 255, 0),
    Water=(0, 0, 255),
    Barren=(159, 129, 183),
    Forest=(0, 255, 0),
    Agricultural=(255, 195, 128),
)

palette = np.asarray(list(COLOR_MAP.values())).reshape((-1,)).tolist()

def generate_pseudo(model, target_loader, save_dir, n_class=7, pseudo_dict=dict(), logger=None):
    logger.info('Start generate pseudo labels: %s' % save_dir)
    viz_op = VisualizeSegmm(os.path.join(save_dir, 'vis'), palette)
    os.makedirs(os.path.join(save_dir, 'pred'), exist_ok=True)
    model.eval()
    cls_thresh = np.ones(n_class)*0.9
    for image, labels in tqdm(target_loader):
        out = model(image.cuda())
        logits = out[0] if isinstance(out, tuple) else out
        max_items = logits.max(dim=1)
        label_pred = max_items[1].data.cpu().numpy()
        logits_pred = max_items[0].data.cpu().numpy()

        logits_cls_dict = {c: [cls_thresh[c]] for c in range(n_class)}
        for cls in range(n_class):
            logits_cls_dict[cls].extend(logits_pred[label_pred == cls].astype(np.float16))
        # instance adaptive selector
        tmp_cls_thresh = ias_thresh(logits_cls_dict, n_class, pseudo_dict['pl_alpha'],  w=cls_thresh, gamma=pseudo_dict['pl_gamma'])
        beta = pseudo_dict['pl_beta']
        cls_thresh = beta*cls_thresh + (1-beta)*tmp_cls_thresh
        cls_thresh[cls_thresh>=1] = 0.999

        np_logits = logits.data.cpu().numpy()
        for _i, fname in enumerate(labels['fname']):
            # save pseudo label
            logit = np_logits[_i].transpose(1,2,0)
            label = np.argmax(logit, axis=2)
            logit_amax = np.amax(logit, axis=2)
            label_cls_thresh = np.apply_along_axis(lambda x: [cls_thresh[e] for e in x], 1, label)
            ignore_index = logit_amax < label_cls_thresh
            viz_op(label, fname)
            label += 1
            label[ignore_index] = 0
            imsave(os.path.join(save_dir, 'pred', fname), label.astype(np.uint8))

    return os.path.join(save_dir, 'pred')

def entropyloss(logits, weight=None):
    """
    logits:     N * C * H * W
    weight:     N * 1 * H * W
    """
    val_num = weight[weight>0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    entropy = -torch.softmax(logits, dim=1) * weight * logits_log_softmax
    entropy_reg = torch.sum(entropy) / val_num
    return entropy_reg

def kldloss(logits, weight):
    """
    logits:     N * C * H * W
    weight:     N * 1 * H * W
    """
    val_num = weight[weight>0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classes = logits.size()[1]
    kld = - 1/num_classes * weight * logits_log_softmax
    kld_reg = torch.sum(kld) / val_num
    return kld_reg

def count_model_parameters(module, _default_logger=None):
    cnt = 0
    for p in module.parameters():
        cnt += reduce(lambda x, y: x * y, list(p.shape))
    _default_logger.info('#params: {}, {} M'.format(cnt, round(cnt / float(1e6), 3)))

    return cnt
    
if __name__ == '__main__':
    seed_torch(2333)
    s = torch.randn((5, 5)).cuda()
    print(s)