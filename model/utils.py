# Copyright (c) Duowang Zhu.
# All rights reserved.

import os
from os.path import join as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from scipy import stats
from typing import Union, Type, List

from eval_func.bleu.bleu import Bleu
from eval_func.rouge.rouge import Rouge
from eval_func.cider.cider import Cider
from eval_func.meteor.meteor import Meteor


def weight_init(module):
    """
    Initialize weights for neural network modules using best practices.
    
    This function recursively initializes weights in a module:
    - Conv2D layers: Kaiming normal initialization (fan_in, relu)
    - BatchNorm and GroupNorm: Weights set to 1, biases to 0
    - Linear layers: Kaiming normal initialization (fan_in, relu)
    - Sequential containers: Each component initialized individually
    - Pooling, ModuleList, Loss functions: Skipped (no initialization needed)
    
    Args:
        module: PyTorch module whose weights will be initialized
    """
    # Process all named children in the module
    for name, child_module in module.named_children():
        # Skip modules that don't need initialization
        if isinstance(child_module, (nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, 
                                     nn.ModuleList, nn.BCELoss)):
            continue
            
        # Initialize convolutional layers
        elif isinstance(child_module, nn.Conv2d):
            nn.init.kaiming_normal_(child_module.weight, mode='fan_in', nonlinearity='relu')
            if child_module.bias is not None:
                nn.init.zeros_(child_module.bias)
                
        # Initialize normalization layers
        elif isinstance(child_module, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(child_module.weight)
            if child_module.bias is not None:
                nn.init.zeros_(child_module.bias)
                
        # Initialize linear layers
        elif isinstance(child_module, nn.Linear):
            nn.init.kaiming_normal_(child_module.weight, mode='fan_in', nonlinearity='relu')
            if child_module.bias is not None:
                nn.init.zeros_(child_module.bias)

        # Handle Sequential containers
        elif isinstance(child_module, nn.Sequential):
            for seq_name, seq_module in child_module.named_children():
                if isinstance(seq_module, nn.Conv2d):
                    nn.init.kaiming_normal_(seq_module.weight, mode='fan_in', nonlinearity='relu')
                    if seq_module.bias is not None:
                        nn.init.zeros_(seq_module.bias)

                elif isinstance(seq_module, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.ones_(seq_module.weight)
                    if seq_module.bias is not None:
                        nn.init.zeros_(seq_module.bias)

                elif isinstance(seq_module, nn.Linear):
                    nn.init.kaiming_normal_(seq_module.weight, mode='fan_in', nonlinearity='relu')
                    if seq_module.bias is not None:
                        nn.init.zeros_(seq_module.bias)
                else:
                    # Recursively initialize other modules in sequential container
                    weight_init(seq_module)

        # Recursively handle other module types
        elif len(list(child_module.children())) > 0:
            weight_init(child_module)

def adjust_learning_rate(args, optimizer, epoch=None, iter=None, max_batches=None, 
                         lr_factor=1.0, shrink_factor=None, verbose=True):
    """
    Adjust learning rate based on scheduler type, epoch, iteration, or explicit shrinking.
    
    This function supports multiple learning rate adjustment strategies:
    1. Step decay: Reduces LR at fixed intervals
    2. Polynomial decay: Smoothly reduces LR according to a polynomial function
    3. Manual shrinking: Explicitly shrinks LR by a specified factor
    4. Warm-up phase: Gradually increases LR at the beginning of training
    
    Args:
        args: Command line arguments containing lr_mode, lr, step_loss, max_epochs
        optimizer: Optimizer instance whose learning rate will be adjusted
        epoch: Current epoch (required for step and poly modes)
        iter: Current iteration (required for poly mode and warm-up)
        max_batches: Total batches per epoch (required for poly mode)
        lr_factor: Additional scaling factor for the learning rate (default: 1.0)
        shrink_factor: If provided, explicitly shrink LR by this factor (0-1)
        verbose: Whether to print the learning rate change (default: True)
        
    Returns:
        float: Current learning rate after adjustment
    """
    if shrink_factor is not None:
        # Manual shrinking mode (from the second implementation)
        if not 0 < shrink_factor < 1:
            raise ValueError(f"Shrink factor must be between 0 and 1, got {shrink_factor}")
            
        if verbose:
            print("\nDECAYING learning rate.")
            
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * shrink_factor
            
        if verbose:
            print(f"The new learning rate is {optimizer.param_groups[0]['lr']:.6f}\n")
            
        return optimizer.param_groups[0]['lr']
    
    # Scheduler-based learning rate adjustment
    if args.lr_mode == 'step':
        if epoch is None:
            raise ValueError("Epoch must be provided for step lr_mode")
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
        
    elif args.lr_mode == 'poly':
        if any(param is None for param in [epoch, iter, max_batches]):
            raise ValueError("Epoch, iter, and max_batches must be provided for poly lr_mode")
            
        cur_iter = iter
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
        
    else:
        raise ValueError(f'Unknown lr mode {args.lr_mode}')
    
    # Apply warm-up phase if we're in the first epoch
    if epoch == 0 and iter is not None and iter < 200:
        lr = args.lr * 0.9 * (iter + 1) / 200 + 0.1 * args.lr
    
    # Apply additional lr factor
    lr *= lr_factor
    
    # Update learning rate for all parameter groups
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def BCEDiceLoss(inputs, targets):
    """
    Combined BCE and Dice loss for binary segmentation.
    
    Args:
        inputs: Model predictions after sigmoid
        targets: Ground truth binary masks
        
    Returns:
        Combined loss value
    """
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    return bce + 1 - dice

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_index=-1):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss(weight=weight, ignore_index=ignore_index,
                                   reduction='mean')

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs, dim=1), targets)
    
class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)
        
    def forward(self, x1, x2, label_change):
        b,c,h,w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0,2,3,1)
        x2 = x2.permute(0,2,3,1)
        x1 = torch.reshape(x1,[b*h*w,c])
        x2 = torch.reshape(x2,[b*h*w,c])
        
        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target,[b*h*w])
        
        loss = self.loss_f(x1, x2, target)
        return loss

def load_checkpoint(args, model, save_path, max_batches):
    """
    Load checkpoint if resume is specified.
    
    Args:
        args: Command line arguments
        model: Model instance
        save_path: Path to save directory
        
    Returns:
        start_epoch, cur_iter
    """
    start_epoch = 0
    cur_iter = 0
    
    if args.resume is not None:
        checkpoint_path = osp(save_path, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_path):
            print(f"=> loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch']
            cur_iter = start_epoch * max_batches
            model.load_state_dict(checkpoint['state_dict'])
            print(f"=> loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{checkpoint_path}'")
    
    return start_epoch, cur_iter


def setup_logger(args, save_path):
    """
    Set up logging and log all command line arguments.
    
    Args:
        args: Command line arguments
        save_path: Path to save directory
        
    Returns:
        Logger file handle
    """
    log_file_loc = osp(save_path, args.log_file)
    logger = open(log_file_loc, 'a+')
    # Log all arguments
    logger.write("Model Configurations:\n")
    for arg, value in vars(args).items():
        logger.write(f"{arg}: {value}\n")
        print(f"{arg}: {value}")
    logger.write('\n' + '-' * 60)

    if args.dataset in ['LEVIR-CD', 'WHU-CD', 'CLCD']:
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s" % 
            ('Epoch', 'Kappa (val)', 'IoU (val)', 'F1 (val)', 'R (val)', 'P (val)'))

    elif args.dataset in ['HRSCD', 'SECOND']:
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" %
            ('epoch', 'train_loss', 'train_acc', 'val_Fscd', 'val_IoU_mean', 'val_Sek',\
                        'val_loss', 'val_acc'))
    
    elif args.dataset in ['xBD']:
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s" %
            ('epoch', 'loss_val', 'loc_f1_score', 'harmonic_mean_f1', 'oa_f1', 'damage_f1_scores'))

    elif args.dataset in ['LEVIR-CC', 'DUBAI-CC']:
        logger.write("\n%s\t%s\t%s\t%s\t%s\t%s" %
            ('epoch', 'loss_val', 'loc_f1_score', 'harmonic_mean_f1', 'oa_f1', 'damage_f1_scores')) # check
    
    else:
        assert False, r'setup_logger error, please check the input dataset!'

    logger.flush()
    return logger

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg


def accuracy(pred, label, ignore_zero=False):
    valid = (label >= 0)
    if ignore_zero: valid = (label > 0)
    acc_sum = (valid * (pred == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc, valid_sum
    
def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def get_hist(image, label, num_class):
    hist = np.zeros((num_class, num_class))
    hist += fast_hist(image.flatten(), label.flatten(), num_class)
    return hist

def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


def SCDD_eval_all(preds, labels, num_class):
    hist = np.zeros((num_class, num_class))
    for pred, label in zip(preds, labels):
        infer_array = np.array(pred)
        unique_set = set(np.unique(infer_array))
        assert unique_set.issubset(set([0, 1, 2, 3, 4, 5, 6])), "unrecognized label number"
        label_array = np.array(label)
        assert infer_array.shape == label_array.shape, "The size of prediction and target must be the same"
        hist += get_hist(infer_array, label_array, num_class)
    
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e
    
    pixel_sum = hist.sum()
    change_pred_sum  = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum/pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP/change_pred_sum
    SC_Recall = SC_TP/change_label_sum
    Fscd = stats.hmean([SC_Precision, SC_Recall])
    return Fscd, IoU_mean, Sek

class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2, dtype=np.longlong)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / (self.confusion_matrix.sum(axis=1) + 1e-7)
        mAcc = np.nanmean(Acc)
        return mAcc, Acc

    def Pixel_Precision_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Pre = self.confusion_matrix[1, 1] / (self.confusion_matrix[0, 1] + self.confusion_matrix[1, 1])
        return Pre

    def Pixel_Recall_Rate(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.confusion_matrix[1, 1] / (self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return Rec

    def Pixel_F1_score(self):
        assert self.confusion_matrix.shape[0] == 2
        Rec = self.Pixel_Recall_Rate()
        Pre = self.Pixel_Precision_Rate()
        F1 = 2 * Rec * Pre / (Rec + Pre)
        return F1


    def calculate_per_class_metrics(self):
        # Adjustments to exclude class 0 in calculations
        TPs = np.diag(self.confusion_matrix)[1:]  # Start from index 1 to exclude class 0
        FNs = np.sum(self.confusion_matrix, axis=1)[1:] - TPs
        FPs = np.sum(self.confusion_matrix, axis=0)[1:] - TPs
        return TPs, FNs, FPs
    
    def Damage_F1_socore(self):
        TPs, FNs, FPs = self.calculate_per_class_metrics()
        precisions = TPs / (TPs + FPs + 1e-7)
        recalls = TPs / (TPs + FNs + 1e-7)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-7)
        return f1_scores
    
    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix) + 1e-7)
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Intersection_over_Union(self):
        IoU = self.confusion_matrix[1, 1] / (
                self.confusion_matrix[0, 1] + self.confusion_matrix[1, 0] + self.confusion_matrix[1, 1])
        return IoU

    def Kappa_coefficient(self):
        # Number of observations (total number of classifications)
        # num_total = np.array(0, dtype=np.long)
        # row_sums = np.array([0, 0], dtype=np.long)
        # col_sums = np.array([0, 0], dtype=np.long)
        # total += np.sum(self.confusion_matrix)
        # # Observed agreement (i.e., sum of diagonal elements)
        # observed_agreement = np.sum(np.diag(self.confusion_matrix))
        # # Compute expected agreement
        # row_sums += np.sum(self.confusion_matrix, axis=0)
        # col_sums += np.sum(self.confusion_matrix, axis=1)
        # expected_agreement = np.sum((row_sums * col_sums) / total)
        num_total = np.sum(self.confusion_matrix)
        observed_accuracy = np.trace(self.confusion_matrix) / num_total
        expected_accuracy = np.sum(
            np.sum(self.confusion_matrix, axis=0) / num_total * np.sum(self.confusion_matrix, axis=1) / num_total)

        # Calculate Cohen's kappa
        kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)
        return kappa

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int64') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class ** 2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def caption_accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)

def eval_caption_score(references, hypotheses):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    hypo = [[' '.join(hypo)] for hypo in [[str(x) for x in hypo] for hypo in hypotheses]]
    ref = [[' '.join(reft) for reft in reftmp] for reftmp in
           [[[str(x) for x in reft] for reft in reftmp] for reftmp in references]]

    score = []
    method = []
    for scorer, method_i in scorers:
        score_i, scores_i = scorer.compute_score(ref, hypo)
        score.extend(score_i) if isinstance(score_i, list) else score.append(score_i)
        method.extend(method_i) if isinstance(method_i, list) else method.append(method_i)
        print("{} {}".format(method_i, score_i))
    score_dict = dict(zip(method, score))

    return score_dict