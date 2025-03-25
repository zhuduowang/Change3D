import sys
import time
import json
import argparse
import os
from os.path import join as osp
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import StepLR

# Insert current path for local module imports
sys.path.insert(0, '.')

from model.trainer import Trainer
from data.dataset import CaptionDataset
from model.utils import AverageMeter, clip_gradient, adjust_learning_rate,\
                        caption_accuracy, eval_caption_score

from einops import rearrange
from tqdm import tqdm


# Find keys in dictionary by value
def get_key(dict_, value):
    return [k for k, v in dict_.items() if v == value]

# Save generated captions and references to JSON files
def save_captions(args, word_map, hypotheses, references):
    result_json_file = {}
    reference_json_file = {}
    m = -1
    for item in hypotheses:
        m += 1
        line_hypo = ""

        for word_idx in item:
            word = get_key(word_map, word_idx)
            # print(word)
            line_hypo += word[0] + " "

        result_json_file[str(m)] = []
        result_json_file[str(m)].append(line_hypo)

        line_hypo += "\r\n"

    n = -1
    for item in tqdm(references):
        n += 1

        reference_json_file[str(n)] = []

        for sentence in item:
            line_repo = ""
            for word_idx in sentence:
                word = get_key(word_map, word_idx)
                line_repo += word[0] + " "
            reference_json_file[str(n)].append(line_repo)
            line_repo += "\r\n"

    save_path = osp(args.save_path, 'eval_results')
    os.makedirs(save_path, exist_ok=True)

    with open(osp(save_path, 'res.json'), 'w') as f:
        json.dump(result_json_file, f)

    with open(osp(save_path, 'gts.json'), 'w') as f:
        json.dump(reference_json_file, f)

def train(args, train_loader, encoder, decoder, criterion,
          encoder_optimizer, encoder_lr_scheduler,
          decoder_optimizer, decoder_lr_scheduler, epoch):
    """
    Performs one epoch's training.
    
    Args:
        args: Command line arguments.
        train_loader (DataLoader): DataLoader for training data.
        encoder: Encoder model for image features.
        decoder: Decoder model for caption generation.
        criterion: Loss function.
        encoder_optimizer: Optimizer to update encoder's weights (if fine-tuning).
        encoder_lr_scheduler: Learning rate scheduler for encoder.
        decoder_optimizer: Optimizer to update decoder's weights.
        decoder_lr_scheduler: Learning rate scheduler for decoder.
        epoch (int): Current epoch number.
    """
    encoder.train()
    decoder.train()  # train mode (dropout and batchnorm is used)

    # Tracking metrics
    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Iterate through batches
    for i, (img_pairs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        img_pairs = img_pairs.cuda()
        caps = caps.cuda()
        caplens = caplens.cuda()

        # Split image pairs and encode
        imgs_A = img_pairs[:, 0, :, :, :]
        imgs_B = img_pairs[:, 1, :, :, :]
        
        # Get fused features from encoder
        percep_feat = encoder(imgs_A, imgs_B, output_final=True)
        percep_feat = rearrange(percep_feat, 'b c h w -> (h w) b c')

        # Decode captions
        scores, caps_sorted, decode_lengths, sort_ind = decoder(percep_feat, caps, caplens)

        # Set targets (all words after <start>, up to <end>)
        targets = caps_sorted[:, 1:]

        # Remove timesteps that weren't decoded or are pads
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(scores, targets)

        # Backpropagation
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients if needed
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)

        # Update weights
        encoder_optimizer.step()
        encoder_lr_scheduler.step()
        decoder_optimizer.step()
        decoder_lr_scheduler.step()

        # Update metrics
        top5 = caption_accuracy(scores, targets, 1)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        
        # Print status
        if i % args.print_freq == 0:
            print(
                f"Epoch: {epoch}/{args.epochs} step: {i}/{len(train_loader)} "
                f"Loss: {losses.val:.4f} AVG_Loss: {losses.avg:.4f} "
                f"Top-5 Accuracy: {top5accs.val:.4f} Batch_time: {batch_time.val:.4f}s"
            )


# Evaluate the model using transformer decoder
def evaluate(args, encoder, decoder, normalize):
    """
    Evaluation for decoder: transformer
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # Load model to GPU and set to evaluation mode
    encoder = encoder.cuda()
    encoder.eval()
    decoder = decoder.cuda()
    decoder.eval()

    # Load word map (word2idx)
    word_map_file = osp(args.file_root, 'WORDMAP_' + args.dataset + '.json')
    with open(word_map_file, 'r') as f:
        word_map = json.load(f)

    rev_word_map = {v: k for k, v in word_map.items()}
    vocab_size = len(word_map)

    # Set beam search size
    beam_size = args.beam_size
    Caption_End = False
    
    # Create data loader
    loader = torch.utils.data.DataLoader(
        CaptionDataset(args.file_root, args.dataset, args.Split, transform=transforms.Compose([normalize])),
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Lists to store references and hypotheses
    references = list()
    hypotheses = list()
    change_references = list()
    change_hypotheses = list()
    nochange_references = list()
    nochange_hypotheses = list()
    change_acc = 0
    nochange_acc = 0

    with torch.no_grad():
        for i, (image_pairs, caps, caplens, allcaps) in enumerate(
                tqdm(loader, desc=args.Split + " EVALUATING AT BEAM SIZE " + str(beam_size))):
            # Process only 1 of every 5 images since they're the same with "shuffle=False"
            if (i + 1) % 5 != 0:
                continue
            
            k = beam_size

            # Move image pairs to GPU
            image_pairs = image_pairs.cuda()  # [1, 2, 3, 256, 256]

            # Encode image pairs
            imgs_A = image_pairs[:, 0, :, :, :]
            imgs_B = image_pairs[:, 1, :, :, :]
            encoder_out = encoder(imgs_A, imgs_B, output_final=True)
            encoder_out = rearrange(encoder_out, 'b c h w -> (h w) b c')

            # Initialize decoder input and attention mask
            tgt = torch.zeros(52, k).cuda().to(torch.int64)
            tgt_length = tgt.size(0)
            mask = (torch.triu(torch.ones(tgt_length, tgt_length)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            mask = mask.cuda()

            # Set start token
            tgt[0, :] = torch.LongTensor([word_map['<start>']]*k).cuda()  # k_prev_words:[52,k]
            
            # Tensor to store top k sequences; initially just <start>
            seqs = torch.LongTensor([[word_map['<start>']]*1] * k).cuda()  # [1,k]
            
            # Tensor to store top k sequences' scores; initially 0
            top_k_scores = torch.zeros(k, 1).cuda()
            
            # Lists to store completed sequences and scores
            complete_seqs = []
            complete_seqs_scores = []
            step = 1

            k_prev_words = tgt.permute(1, 0)
            S = encoder_out.size(0)
            encoder_dim = encoder_out.size(-1)

            # Expand encoder_out for beam search
            encoder_out = encoder_out.expand(S, k, encoder_dim)  # [S,k, encoder_dim]
            encoder_out = encoder_out.permute(1, 0, 2)

            # Start decoding
            # Note: s is a number less than or equal to k, as sequences are removed once they hit <end>
            while True:
                tgt = k_prev_words.permute(1, 0)
                tgt_embedding = decoder.vocab_embedding(tgt)
                tgt_embedding = decoder.position_encoding(tgt_embedding)  # (length, batch, feature_dim)

                encoder_out = encoder_out.permute(1, 0, 2)
                pred = decoder.transformer(tgt_embedding, encoder_out, tgt_mask=mask)  # (length, batch, feature_dim)
                encoder_out = encoder_out.permute(1, 0, 2)
                pred = decoder.wdc(pred)  # (length, batch, vocab_size)
                scores = pred.permute(1, 0, 2)  # (batch, length, vocab_size)
                scores = scores[:, step - 1, :].squeeze(1)  # [s, 1, vocab_size] -> [s, vocab_size]
                scores = F.log_softmax(scores, dim=1)
                
                # top_k_scores: [s, 1]
                scores = top_k_scores.expand_as(scores) + scores  # [s, vocab_size]
                
                # For the first step, all k points have the same scores (since same k previous words)
                if step == 1:
                    top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
                else:
                    # Unroll and find top scores, and their unrolled indices
                    top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

                # Convert unrolled indices to actual indices of scores
                prev_word_inds = top_k_words // vocab_size  # (s)
                # if max(top_k_words)>vocab_size:
                #     print(">>>>>>>>>>>>>>>>>>")
                # prev_word_inds = torch.div(top_k_words, vocab_size, rounding_mode='floor')
                next_word_inds = top_k_words % vocab_size  # (s)

                # Add new words to sequences
                seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
                
                # Which sequences are incomplete (didn't reach <end>)?
                incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                                  next_word != word_map['<end>']]
                complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))
                
                # Store completed sequences
                if len(complete_inds) > 0:
                    Caption_End = True
                    complete_seqs.extend(seqs[complete_inds].tolist())
                    complete_seqs_scores.extend(top_k_scores[complete_inds])
                
                k -= len(complete_inds)  # Reduce beam length accordingly
                
                # Exit if no incomplete sequences remain
                if k == 0:
                    break
                
                # Continue with incomplete sequences
                seqs = seqs[incomplete_inds]
                encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
                top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
                
                # Important: this won't work since decoder has self-attention
                # k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1).repeat(k, 52)
                k_prev_words = k_prev_words[incomplete_inds]
                k_prev_words[:, :step + 1] = seqs  # [s, 52]
                # k_prev_words[:, step] = next_word_inds[incomplete_inds]  # [s, 52]
                
                # Break if process has been going on too long
                if step > 50:
                    break
                step += 1

            # Select caption with best score
            if len(complete_seqs_scores) == 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            
            if len(complete_seqs_scores) > 0:
                assert Caption_End
                indices = complete_seqs_scores.index(max(complete_seqs_scores))
                seq = complete_seqs[indices]
                
                # References
                img_caps = allcaps[0].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                        img_caps))  # Remove <start> and padding

                references.append(img_captions)
                
                # Hypotheses
                new_sent = [w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
                hypotheses.append(new_sent)
                assert len(references) == len(hypotheses)

                # Check if there's a change in the image pair
                nochange_list = ["the scene is the same as before ", "there is no difference ",
                                "the two scenes seem identical ", "no change has occurred ",
                                "almost nothing has changed "]
                
                # Get reference caption text
                ref_sentence = img_captions[1]
                ref_line_repo = ""
                for ref_word_idx in ref_sentence:
                    ref_word = get_key(word_map, ref_word_idx)
                    ref_line_repo += ref_word[0] + " "

                # Get generated caption text
                hyp_sentence = new_sent
                hyp_line_repo = ""
                for hyp_word_idx in hyp_sentence:
                    hyp_word = get_key(word_map, hyp_word_idx)
                    hyp_line_repo += hyp_word[0] + " "
                
                # For image pairs with changes
                if ref_line_repo not in nochange_list:
                    change_references.append(img_captions)
                    change_hypotheses.append(new_sent)
                    if hyp_line_repo not in nochange_list:
                        change_acc = change_acc + 1
                else:
                    nochange_references.append(img_captions)
                    nochange_hypotheses.append(new_sent)
                    if hyp_line_repo in nochange_list:
                        nochange_acc = nochange_acc + 1

        # Save generated captions
        save_captions(args, word_map, hypotheses, references)

    # Output evaluation results
    print('len(nochange_references):', len(nochange_references))
    print('len(change_references):', len(change_references))
    
    # Calculate BLEU1~4, METEOR, ROUGE_L, CIDEr scores
    if len(nochange_references) > 0:
        print('nochange_metric:')
        nochange_metric = eval_caption_score(nochange_references, nochange_hypotheses)
        print("nochange_acc:", nochange_acc / len(nochange_references))
    
    if len(change_references) > 0:
        print('change_metric:')
        change_metric = eval_caption_score(change_references, change_hypotheses)
        print("change_acc:", change_acc / len(change_references))
    
    print(".......................................................")
    metrics = eval_caption_score(references, hypotheses)

    return metrics

def trainValidate(args):
    """
    Main training and validation routine.
    
    Args:
        args: Command line arguments.
    """
    # Set GPU device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    # Initialize training variables
    start_epoch = 0
    best_bleu4 = 0.  # BLEU-4 score right now
    
    # Set CUDA benchmark for performance
    cudnn.benchmark = True  # good for fixed size inputs

    # Load word map
    word_map_file = os.path.join(args.file_root, f'WORDMAP_{args.dataset}.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    args.vocab_size = len(word_map)

    # create save path
    args.save_path = osp(
        args.save_dir,
        f"{args.dataset}_iter_{args.epochs}_lr_{args.encoder_lr}"
    )
    os.makedirs(args.save_path, exist_ok=True)

    # Initialize models
    trainer = Trainer(args).cuda()

    # Initialize optimizers and schedulers
    encoder_optimizer = None
    encoder_lr_scheduler = None
    
    if args.fine_tune_encoder:
        encoder_optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, trainer.encoder.parameters()),
            lr=args.encoder_lr, 
            weight_decay=1e-5
        )
        encoder_lr_scheduler = StepLR(
            encoder_optimizer, 
            step_size=900, 
            gamma=1
        )

    decoder_optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, trainer.decoder.parameters()),
        lr=args.decoder_lr, 
        weight_decay=1e-5
    )
    decoder_lr_scheduler = StepLR(
        decoder_optimizer, 
        step_size=900, 
        gamma=1
    )

    # Loss function
    criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()

    # Configure data loader
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(
            args.file_root, 
            args.dataset, 
            'TRAIN', 
            transform=transforms.Compose([normalize])
        ),
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True
    )

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        print(f'Current Epoch: {epoch}')

        # Learning rate adjustment at specific epochs
        if epoch > 0 and epoch % 10 == 0:
            adjust_learning_rate(args=None, optimizer=encoder_optimizer, shrink_factor=0.5)
            adjust_learning_rate(args=None, optimizer=encoder_optimizer, shrink_factor=0.5)

        # One epoch's training
        print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
        train(
            args,
            train_loader=train_loader,
            encoder=trainer.encoder,
            decoder=trainer.decoder,
            criterion=criterion,
            encoder_optimizer=encoder_optimizer,
            encoder_lr_scheduler=encoder_lr_scheduler,
            decoder_optimizer=decoder_optimizer,
            decoder_lr_scheduler=decoder_lr_scheduler,
            epoch=epoch
        )

        # Validation
        metrics = evaluate(
            args,
            encoder=trainer.encoder,
            decoder=trainer.decoder,
            normalize=normalize
        )

        # Check if there was an improvement
        recent_bleu4 = metrics["Bleu_4"]
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)

        # save checkpoints
        state = {'epoch': epoch,
                'bleu-4': recent_bleu4,
                'encoder_image': trainer.encoder.state_dict(),
                'decoder': trainer.decoder.state_dict(),
                'encoder_image_optimizer': encoder_optimizer,
                'decoder_optimizer': decoder_optimizer,
                }
        filename = 'checkpoint_' + args.dataset + '.pth.tar'
        
        if is_best:
            torch.save(state, osp(args.save_path, 'BEST_' + filename))

        torch.save(state, osp(args.save_path, 'checkpoint_' + args.dataset + '_epoch_' + str(epoch) + '.pth.tar'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change_Captioning')

    # Data parameters
    parser.add_argument(
        '--file_root', 
        default="../levir_cc_dataset",
        help='Folder with data files saved by create_input_files.py.'
    )
    parser.add_argument(
        '--dataset', 
        default="LEVIR_CC_5_cap_per_img_5_min_word_freq",
        help='Base name shared by data files.'
    )

    # Model parameters
    parser.add_argument(
        '--n_head', 
        type=int, 
        default=8, 
        help='Multi-head attention in Transformer.'
    )
    parser.add_argument(
        '--n_layer', 
        type=int, 
        default=3
    )
    parser.add_argument(
        '--decoder_n_layers', 
        type=int, 
        default=1
    )
    parser.add_argument(
        '--embed_dim', 
        type=int, 
        default=192
    )
    parser.add_argument(
        '--dropout', 
        type=float, 
        default=0.1, 
        help='Dropout rate'
    )
    parser.add_argument(
        '--num_perception_frame',
        type=int,
        default=1,
        help='Number of perception frames'
    )

    # Training parameters
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
        '--epochs', 
        type=int, 
        default=200, 
        help='Number of epochs to train for (if early stopping is not triggered).'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--print_freq', 
        type=int, 
        default=100, 
        help='Print training/validation stats every __ batches.'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=1, 
        help='For data-loading; right now, only 0 works with h5pys in Windows.'
    )
    parser.add_argument(
        '--encoder_lr', 
        type=float, 
        default=1e-4, 
        help='Learning rate for encoder if fine-tuning.'
    )
    parser.add_argument(
        '--decoder_lr', 
        type=float, 
        default=1e-4, 
        help='Learning rate for decoder.'
    )
    parser.add_argument(
        '--grad_clip', 
        type=float, 
        default=5., 
        help='Clip gradients at an absolute value.'
    )
    parser.add_argument(
        '--fine_tune_encoder', 
        type=bool, 
        default=True, 
        help='Whether to fine-tune encoder or not'
    )
    parser.add_argument(
        '--checkpoint', 
        default=None, 
        help='Path to checkpoint, None if none.'
    )
    parser.add_argument(
        '--pretrained',
        default='model/X3D_L.pyth',
        type=str,
        help='Path to pretrained weight'
    )
    parser.add_argument(
        '--gpu_id',
        default=0,
        type=int,
        help='GPU ID number'
    )

    # Validation parameters
    parser.add_argument(
        '--Split', 
        default="TEST", 
        help='Validation split'
    )
    parser.add_argument(
        '--beam_size', 
        type=int, 
        default=1, 
        help='Beam size for beam search.'
    )
    parser.add_argument(
        '--save_dir',
        default='./exp',
        help='Directory to save the experiment results'
    )
    
    args = parser.parse_args()
    print(args)
    
    trainValidate(args)