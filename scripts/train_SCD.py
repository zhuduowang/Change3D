# Copyright (c) Duowang Zhu.
# All rights reserved.

import os
import sys
import time
import numpy as np
from os.path import join as osp
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# Insert current path to sys.path for local imports
sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms

from model.trainer import Trainer
from model.utils import (
    adjust_learning_rate,
    BCEDiceLoss,
    CrossEntropyLoss2d,
    ChangeSimilarity,
    AverageMeter,
    load_checkpoint,
    setup_logger,
    accuracy,
    SCDD_eval_all
)


def create_data_loaders(args, train_transform, val_transform):
    """
    Creates DataLoaders for training, validation, and testing.

    Args:
        args: Command line arguments.
        train_transform: Transform pipeline for training data.
        val_transform: Transform pipeline for validation data.

    Returns:
        tuple: train_loader, val_loader, test_loader, and max_batches.
    """
    # Training data
    train_data = RSDataset.SCDDataset(
        file_root=args.file_root,
        split="train",
        transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Validation data
    val_data = RSDataset.SCDDataset(
        file_root=args.file_root,
        split="val",
        transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Test data
    test_data = RSDataset.SCDDataset(
        file_root=args.file_root,
        split="test",
        transform=val_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    max_batches = len(train_loader)
    print(f"For each epoch, we have {max_batches} batches.")

    return train_loader, val_loader, test_loader, max_batches


@torch.no_grad()
def val(args, val_loader, model, seg_loss, sim_loss):
    """
    Validates the model on the validation set.

    Args:
        args: Command line arguments.
        val_loader (DataLoader): DataLoader for validation set.
        model (nn.Module): Model to validate.
        seg_loss: Segmentation loss function.
        sim_loss: Similarity loss function.

    Returns:
        tuple: (Fscd, IoU_mean, Sek, acc_meter, val_loss)
    """
    model.eval()
    val_loss = AverageMeter()
    acc_meter = AverageMeter()

    preds_all = []
    labels_all = []

    start_time = time.time()
    
    for iter_idx, batched_inputs in enumerate(val_loader):
        imgs, labels = batched_inputs
        pre_img = imgs[:, 0:3].cuda().float()
        post_img = imgs[:, 3:6].cuda().float()

        pre_label = labels[:, 0].cuda().long()
        post_label = labels[:, 1].cuda().long()
        label_change = labels[:, 2].cuda().long()

        pre_label = pre_label * label_change
        post_label = post_label * label_change

        # Forward pass
        pre_mask, post_mask, change_mask = model.update_scd(pre_img, post_img)

        # Loss computation
        binary_loss = BCEDiceLoss(change_mask, label_change.unsqueeze(1).float())
        segm_loss = seg_loss(pre_mask, pre_label) + seg_loss(post_mask, post_label)
        similarity_loss = sim_loss(pre_mask[:, 1:], post_mask[:, 1:], label_change.unsqueeze(1))

        loss = binary_loss + segm_loss * 0.5 + similarity_loss
        val_loss.update(loss.cpu().detach().numpy())

        pre_label = pre_label.cpu().detach().numpy()
        post_label = post_label.cpu().detach().numpy()
        pre_mask = pre_mask.cpu().detach()
        post_mask = post_mask.cpu().detach()

        change_mask = change_mask.cpu().detach() > 0.5

        pre_mask = torch.argmax(pre_mask, dim=1)
        post_mask = torch.argmax(post_mask, dim=1)

        pre_mask = (pre_mask * change_mask.squeeze().long()).numpy()
        post_mask = (post_mask * change_mask.squeeze().long()).numpy()

        # Accuracy computation
        for (pred_A, pred_B, label_A, label_B) in zip(pre_mask, post_mask, pre_label, post_label):
            acc_A, _ = accuracy(pred_A, label_A)
            acc_B, _ = accuracy(pred_B, label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_meter.update(acc)

            preds_all.append(pred_A)
            preds_all.append(pred_B)
            labels_all.append(label_A)
            labels_all.append(label_B)

    # Evaluate performance
    Fscd, IoU_mean, Sek = SCDD_eval_all(preds_all, labels_all, args.num_class)
    elapsed_time = time.time() - start_time

    print(
        f"{elapsed_time:.1f}s Val loss: {val_loss.average():.2f} "
        f"Fscd: {Fscd * 100:.2f} IoU: {IoU_mean * 100:.2f} "
        f"Sek: {Sek * 100:.2f} Accuracy: {acc_meter.average() * 100:.2f}"
    )

    return Fscd, IoU_mean, Sek, acc_meter, val_loss


def train(args, train_loader, model, optimizer, epoch, max_batches, cur_iter, seg_loss, sim_loss):
    """
    Trains the model for one epoch.

    Args:
        args: Command line arguments.
        train_loader (DataLoader): Training set DataLoader.
        model (nn.Module): Model to train.
        optimizer: Optimizer instance.
        epoch (int): Current epoch index.
        max_batches (int): Maximum number of batches in one epoch.
        cur_iter (int): Current iteration count.
        seg_loss: Segmentation loss function.
        sim_loss: Similarity loss function.

    Returns:
        tuple: (train_sum_loss, acc_meter, lr)
    """
    model.train()
    acc_meter = AverageMeter()
    train_seg_loss = AverageMeter()
    train_bn_loss = AverageMeter()
    train_sc_loss = AverageMeter()
    train_sum_loss = AverageMeter()

    for iter_idx, batched_inputs in enumerate(train_loader):
        start_time = time.time()
        imgs, labels = batched_inputs
        pre_img = imgs[:, 0:3].cuda().float()
        post_img = imgs[:, 3:6].cuda().float()

        pre_label = labels[:, 0].cuda().long()
        post_label = labels[:, 1].cuda().long()
        label_change = labels[:, 2].cuda().long()

        pre_label = pre_label * label_change
        post_label = post_label * label_change

        # Adjust learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, iter_idx + cur_iter, max_batches)

        # Forward pass
        pre_mask, post_mask, change_mask = model.update_scd(pre_img, post_img)

        # Loss computation
        segm_loss = seg_loss(pre_mask, pre_label) + seg_loss(post_mask, post_label)
        binary_loss = BCEDiceLoss(change_mask, label_change.unsqueeze(1).float())
        similarity_loss = sim_loss(pre_mask[:, 1:], post_mask[:, 1:], label_change.unsqueeze(1))
        loss = segm_loss * 0.5 + binary_loss + similarity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        pre_label = pre_label.cpu().detach().numpy()
        post_label = post_label.cpu().detach().numpy()
        pre_mask = pre_mask.cpu().detach()
        post_mask = post_mask.cpu().detach()

        change_mask = change_mask.cpu().detach() > 0.5
        pred_mask = torch.argmax(pre_mask, dim=1)
        post_mask = torch.argmax(post_mask, dim=1)

        pred_mask = (pred_mask * change_mask.squeeze().long()).numpy()
        post_mask = (post_mask * change_mask.squeeze().long()).numpy()

        acc_curr_meter = AverageMeter()
        for (pred_A, pred_B, label_A, label_B) in zip(pred_mask, post_mask, pre_label, post_label):
            acc_A, _ = accuracy(pred_A, label_A)
            acc_B, _ = accuracy(pred_B, label_B)
            acc = (acc_A + acc_B) * 0.5
            acc_curr_meter.update(acc)

        acc_meter.update(acc_curr_meter.avg)
        train_seg_loss.update(segm_loss.cpu().detach().numpy())
        train_bn_loss.update(binary_loss.cpu().detach().numpy())
        train_sc_loss.update(similarity_loss.cpu().detach().numpy())
        train_sum_loss.update(loss.cpu().detach().numpy())

        time_taken = time.time() - start_time
        remaining_time = (
            (max_batches * args.max_epochs - iter_idx - cur_iter) *
            time_taken / 3600
        )

        if (iter_idx + 1) % 5 == 0:
            print(
                f"[epoch {epoch}] [iter {iter_idx + 1}/{len(train_loader)} {remaining_time:.2f}h] "
                f"[lr {optimizer.param_groups[0]['lr']:.6f}] "
                f"[train seg_loss {train_seg_loss.val:.4f} sim_loss {train_sc_loss.val:.4f} "
                f"bn_loss {train_bn_loss.val:.4f} sum_loss {train_sum_loss.val:.4f} "
                f"acc {acc_meter.val * 100:.2f}]"
            )

    return train_sum_loss, acc_meter, lr


def trainValidate(args):
    """
    Main training and validation pipeline.

    Args:
        args: Command line arguments.
    """
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Enable cudnn optimizations
    torch.backends.cudnn.benchmark = True
    cudnn.benchmark = True
    torch.manual_seed(seed=16)
    torch.cuda.manual_seed(seed=16)

    # Initialize model
    model = Trainer(args).cuda().float()

    # Create save directory
    save_path = osp(
        args.save_dir,
        f"{args.dataset}_iter_{args.max_steps}_lr_{args.lr}"
    )
    os.makedirs(save_path, exist_ok=True)

    # Create data transforms
    train_transform, val_transform = RSTransforms.SCDTransforms.get_transform_pipelines(args)

    # Create DataLoaders
    train_loader, val_loader, test_loader, max_batches = create_data_loaders(
        args, train_transform, val_transform
    )

    # Compute max epochs
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))

    # Load checkpoint if necessary
    start_epoch, cur_iter = load_checkpoint(args, model, save_path, max_batches)

    # Setup logger
    logger = setup_logger(args, save_path)

    # Define loss functions
    seg_loss = CrossEntropyLoss2d(ignore_index=0).cuda()
    sim_loss = ChangeSimilarity().cuda()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        (0.9, 0.99),
        eps=1e-08,
        weight_decay=1e-4
    )

    bestaccT = 0
    bestmIoU = 0.0
    bestloss = 1.0

    # Main training loop
    for epoch in range(start_epoch, args.max_epochs):
        torch.cuda.empty_cache()

        # Train for one epoch
        loss_train, score_tr, lr = train(
            args,
            train_loader,
            model,
            optimizer,
            epoch,
            max_batches,
            cur_iter,
            seg_loss,
            sim_loss
        )
        cur_iter += len(train_loader)

        # Skip validation for the first epoch
        if epoch == 0:
            continue

        # Validate
        torch.cuda.empty_cache()
        Fscd, IoU_mean, Sek, val_acc_meter, loss_val = val(
            args, test_loader, model, seg_loss, sim_loss
        )

        # Log validation
        logger.write(
            "\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                epoch,
                loss_train.average(),
                score_tr.avg,
                Fscd,
                IoU_mean,
                Sek,
                loss_val.avg,
                val_acc_meter.avg
            )
        )
        logger.flush()

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_train': loss_train.avg,
            'loss_val': loss_val.avg,
            'acc_train': score_tr.avg,
            'acc_val': val_acc_meter.avg,
            'lr': lr
        }, osp(save_path, 'checkpoint.pth.tar'))

        # Save best model if needed
        model_file_name = osp(save_path, 'best_model.pth')
        if score_tr.avg > bestaccT:
            bestaccT = score_tr.avg

        if IoU_mean > bestmIoU:
            bestmIoU = IoU_mean
            bestaccV = val_acc_meter.avg
            bestloss = loss_val.avg
            torch.save(model.state_dict(), model_file_name)

        # Report best results so far
        print(
            f"Epoch {epoch}: Details\n"
            f"Best rec: Train acc {bestaccT * 100:.2f}, "
            f"Val mIoU {bestmIoU * 100:.2f} acc {bestaccV * 100:.2f} "
            f"loss {bestloss:.4f}"
        )

    # Test with best model
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    # The original code's val() signature might differ; here we keep the logic consistent
    Fscd, IoU_mean, Sek, val_acc_meter, loss_val = val(
        args, test_loader, model, seg_loss, sim_loss
    )

    # Log validation
    logger.write(
        "\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
            'Test',
            loss_train.average(),
            score_tr.avg,
            Fscd,
            IoU_mean,
            Sek,
            loss_val.avg,
            val_acc_meter.avg
        )
    )
    logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    
    # Dataset configuration
    parser.add_argument(
        '--dataset', 
        default="HRSCD", 
        help='Choose between "HRSCD" or "SECOND".'
    )
    parser.add_argument(
        '--file_root',
        default="path/to/HRSCD",
        help='path to the dataset directory'
    )
    
    # Input dimensions
    parser.add_argument(
        '--in_height', 
        type=int, 
        default=256, 
        help='Height of RGB input.'
    )
    parser.add_argument(
        '--in_width', 
        type=int, 
        default=256, 
        help='Width of RGB input.'
    )
    
    # Model configuration
    parser.add_argument(
        '--num_perception_frame', 
        type=int, 
        default=3, 
        help='Number of perception frames.'
    )
    parser.add_argument(
        '--num_class', 
        type=int, 
        default=6, 
        help='Number of classes.'
    )
    
    # Training parameters
    parser.add_argument(
        '--max_steps', 
        type=int, 
        default=80000, 
        help='Maximum number of iterations.'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=8, 
        help='Batch size.'
    )
    parser.add_argument(
        '--num_workers', 
        type=int, 
        default=4, 
        help='Number of worker threads.'
    )
    
    # Learning rate settings
    parser.add_argument(
        '--lr', 
        type=float, 
        default=2e-4, 
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--lr_mode', 
        default='poly', 
        help='Learning rate policy: "step" or "poly".'
    )
    parser.add_argument(
        '--step_loss', 
        type=int, 
        default=100, 
        help='Decrease learning rate after how many epochs.'
    )
    
    # Weights and saving
    parser.add_argument(
        '--pretrained', 
        default='model/X3D_L.pyth', 
        type=str, 
        help='Path to pretrained weights.'
    )
    parser.add_argument(
        '--save_dir', 
        default='./exp', 
        help='Directory to save experiment results.'
    )
    parser.add_argument(
        '--resume', 
        default=None, 
        help='Resume training from a checkpoint.'
    )
    parser.add_argument(
        '--log_file', 
        default='train_val_log.txt', 
        help='Log file to store training and validation stats.'
    )
    
    # Hardware
    parser.add_argument(
        '--gpu_id', 
        default=0, 
        type=int, 
        help='GPU ID to use.'
    )

    args = parser.parse_args()
    trainValidate(args)
