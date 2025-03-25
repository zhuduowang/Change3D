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

# Insert current path for local module imports
sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms

from model.trainer import Trainer
from model.utils import (
    adjust_learning_rate,
    BCEDiceLoss,
    CrossEntropyLoss2d,
    load_checkpoint,
    setup_logger,
    Evaluator
)


def create_data_loaders(args, train_transform, val_transform):
    """
    Creates data loaders for training, validation, and testing.
    
    Args:
        args: Command line arguments.
        train_transform: Transform pipeline for training data.
        val_transform: Transform pipeline for validation and testing data.
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, max_batches).
    """
    # Training data
    train_data = RSDataset.BDADataset(
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
    val_data = RSDataset.BDADataset(
        file_root=args.file_root,
        split="hold",
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
    test_data = RSDataset.BDADataset(
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
def val(val_loader, model, seg_loss, evaluator_loc, evaluator_cls):
    """
    Validates the model on the validation set.

    Args:
        args: Command line arguments.
        val_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): The model to validate.
        
    Returns:
        tuple: (average_loss, scores).
    """
    model.eval()

    evaluator_cls.reset()
    evaluator_loc.reset()
    
    for iter_idx, batched_inputs in enumerate(val_loader):
        img, label = batched_inputs
        
        # Simplified data preparation
        pre_img = img[:, 0:3].cuda().float()
        post_img = img[:, 3:6].cuda().float()
        label_loc = label[:, 0].cuda().float()
        label_cls = torch.prod(label, dim=1).cuda().long()

        # Forward pass
        pred_cls, pred_loc = model.update_bda(pre_img, post_img)

        segment_loss = seg_loss(pred_cls, label_cls)
        binary_loss = BCEDiceLoss(pred_loc, label_loc.unsqueeze(1))
        loss = segment_loss + binary_loss

        # Binarize predictions
        pred_loc = pred_loc.cpu().numpy() > 0.5
        label_loc = label_loc.cpu().numpy()

        evaluator_loc.add_batch(label_loc, pred_loc.squeeze(1))
        
        pred_cls = torch.argmax(pred_cls, dim=1).cpu().numpy()
        label_cls = label_cls.cpu().numpy()

        pred_cls = pred_cls[label_loc > 0]
        label_cls = label_cls[label_loc > 0]
        evaluator_cls.add_batch(label_cls, pred_cls)

    loc_f1_score = evaluator_loc.Pixel_F1_score()
    damage_f1_score = evaluator_cls.Damage_F1_socore()
    harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
    oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1

    print(
        f"lofF1 is {loc_f1_score}, clfF1 is {harmonic_mean_f1}, oaF1 is {oaf1}, "
        f"sub class F1 score is {damage_f1_score} "
    )

    return loss, loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score


def train(args, train_loader, model, optimizer, epoch, max_batches, 
          cur_iter, seg_loss):
    """
    Trains the model for one epoch.
    
    Args:
        args: Command line arguments.
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): Model to train.
        optimizer: Optimizer instance.
        epoch (int): Current epoch index.
        max_batches (int): Number of batches per epoch.
        cur_iter (int): Current iteration count.
        lr_factor (float): Learning rate adjustment factor.
        
    Returns:
        tuple: (average_loss, scores, current_lr).
    """
    model.train()
    epoch_loss = []

    for iter_idx, batched_inputs in enumerate(train_loader):
        img, label = batched_inputs
        
        # Simplified data preparation
        pre_img = img[:, 0:3].cuda().float()
        post_img = img[:, 3:6].cuda().float()
        label_loc = label[:, 0].cuda().float()
        label_cls = torch.prod(label, dim=1).cuda().long()

        start_time = time.time()

        # Adjust learning rate
        lr = adjust_learning_rate(
            args,
            optimizer,
            epoch,
            iter_idx + cur_iter,
            max_batches,
        )

        # Forward pass
        pred_cls, pred_loc = model.update_bda(pre_img, post_img)
        segment_loss = seg_loss(pred_cls, label_cls)
        binary_loss = BCEDiceLoss(pred_loc, label_loc.unsqueeze(1))
        loss = segment_loss + binary_loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record loss
        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter_idx - cur_iter) * time_taken / 3600

        if (iter_idx + 1) % 5 == 0:
            print(
                f"[epoch {epoch}] [iter {iter_idx + 1}/{len(train_loader)} {res_time:.2f}h] "
                f"[lr {optimizer.param_groups[0]['lr']:.6f}] "
                f"[seg_loss {segment_loss.data.item():.4f} "
                f"bn_loss {binary_loss.data.item():.4f} "
                f"sum_loss {loss.data.item():.4f}] "
            )

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    return average_epoch_loss_train, lr


def trainValidate(args):
    """
    Main training and validation routine.
    
    Args:
        args: Command line arguments.
    """
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Enable CUDA optimizations and fix random seed
    torch.backends.cudnn.benchmark = True
    cudnn.benchmark = True
    torch.manual_seed(seed=16)
    torch.cuda.manual_seed(seed=16)

    # Initialize model
    model = Trainer(args).cuda().float()

    seg_loss = CrossEntropyLoss2d(ignore_index=0).cuda()

    evaluator_loc = Evaluator(num_class=2)
    evaluator_cls = Evaluator(num_class=args.num_class)

    # Create experiment save directory
    save_path = osp(
        args.save_dir,
        f"{args.dataset}_iter_{args.max_steps}_lr_{args.lr}"
    )
    os.makedirs(save_path, exist_ok=True)

    # Data transformations
    train_transform, val_transform = RSTransforms.BDATransforms.get_transform_pipelines(args)

    # Data loaders
    train_loader, val_loader, test_loader, max_batches = create_data_loaders(
        args, train_transform, val_transform
    )

    # Compute maximum epochs
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    
    # Load checkpoint if needed
    start_epoch, cur_iter = load_checkpoint(args, model, save_path, max_batches)
    
    # Set up logger
    logger = setup_logger(args, save_path)
    
    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        args.lr,
        (0.9, 0.99),
        eps=1e-08,
        weight_decay=1e-4
    )
    
    # Track the best score
    best_oa = 0

    # Main training loop
    for epoch in range(start_epoch, args.max_epochs):
        torch.cuda.empty_cache()

        # Train one epoch
        loss_train, lr = train(
            args,
            train_loader,
            model,
            optimizer,
            epoch,
            max_batches,
            cur_iter,
            seg_loss
        )
        cur_iter += len(train_loader)

        # Skip validation for the first epoch
        if epoch == 0:
            continue
        
        # Validation (using test set as validation)
        torch.cuda.empty_cache()
        loss_val, loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score \
              = val(test_loader, model, seg_loss, evaluator_loc, evaluator_cls)
        
        # Log validation results
        damage_scores = "\t\t".join(["%.4f"]*len(damage_f1_score))
        logger.write(
            ("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t" + damage_scores) % (
                epoch,
                loss_val,
                loc_f1_score,
                harmonic_mean_f1,
                oaf1,
                *damage_f1_score,
            )
        )
        logger.flush()

        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_train': loss_train,
            'loss_val': loss_val,
            'loc_f1_score': loc_f1_score,
            'harmonic_mean_f1': harmonic_mean_f1,
            'lr': lr
        }, osp(save_path, 'checkpoint.pth.tar'))

        # Save the best model
        model_file_name = osp(save_path, 'best_model.pth')
        if oaf1 > best_oa: 
            best_oa = oaf1
            torch.save(model.state_dict(), model_file_name)

        # Print summary
        print(f"\nEpoch {epoch}: Details")
        print(
            f"\nEpoch No. {epoch}:\tTrain Loss = {loss_train:.4f}\t"
            f"Val Loss = {loss_val:.4f}\tloc_f1_score = {loc_f1_score:.4f}\t"
            f"harmonic_mean_f1 = {harmonic_mean_f1:.4f}\t"
            f"oaf1 = {oaf1:.4f}\t"
            f"damage_f1_score = {damage_f1_score}"
        )

    # Test with the best model
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    loss_val, loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score \
            = val(test_loader, model, seg_loss, evaluator_loc, evaluator_cls)
    # Print summary
    print(f"\nEpoch {epoch}: Details")
    print(
        f"\nEpoch No. {epoch}:\tTrain Loss = {loss_train:.4f}\t"
        f"Val Loss = {loss_val:.4f}\tloc_f1_score = {loc_f1_score:.4f}\t"
        f"harmonic_mean_f1 = {harmonic_mean_f1:.4f}\t"
        f"oaf1 = {oaf1:.4f}\t"
        f"damage_f1_score = {damage_f1_score}"
    )
    
    # Log validation results
    damage_scores = "\t\t".join(["%.4f"]*len(damage_f1_score))
    logger.write(
        ("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t" + damage_scores) % (
            epoch,
            loss_val,
            loc_f1_score,
            harmonic_mean_f1,
            oaf1,
            *damage_f1_score,
        )
    )
    logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--dataset',
        default="xBD",
        help='Dataset selection | xBD |'
    )
    parser.add_argument(
        '--file_root',
        default="path/to/xBD",
        help='path to the dataset directory'
    )
    parser.add_argument(
        '--in_height',
        type=int,
        default=256,
        help='Height of RGB image'
    )
    parser.add_argument(
        '--in_width',
        type=int,
        default=256,
        help='Width of RGB image'
    )
    parser.add_argument(
        '--num_perception_frame',
        type=int,
        default=2,
        help='Number of perception frames'
    )
    parser.add_argument(
        '--num_class',
        type=int,
        default=5,
        help='Number of classes'
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=200000,
        help='Max number of iterations'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=12,
        help='Batch size'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of parallel threads'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=2e-4,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--lr_mode',
        default='poly',
        help='Learning rate policy: step or poly'
    )
    parser.add_argument(
        '--step_loss',
        type=int,
        default=100,
        help='Decrease learning rate after how many epochs'
    )
    parser.add_argument(
        '--pretrained',
        default='model/X3D_L.pyth',
        type=str,
        help='Path to pretrained weight'
    )
    parser.add_argument(
        '--save_dir',
        default='./exp',
        help='Directory to save the experiment results'
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='Checkpoint to resume training'
    )
    parser.add_argument(
        '--log_file',
        default='train_val_log.txt',
        help='File that stores the training and validation logs'
    )
    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int,
        help='GPU ID number'
    )

    args = parser.parse_args()
    trainValidate(args)