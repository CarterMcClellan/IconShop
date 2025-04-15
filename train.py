import os
import torch
import argparse
from dataset import SketchData
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import numpy as np 
import sys
sys.path.insert(0, 'utils')

from transformers import get_linear_schedule_with_warmup, set_seed
from accelerate import Accelerator
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from PIL import Image
import io

from model.decoder import SketchDecoder


def train(args, cfg):
    accum_step = cfg['gradient_accumulation_steps']
    accelerator = Accelerator(gradient_accumulation_steps=accum_step)
    
    # Initialize dataset loader
    tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_name'])
    train_dataset = SketchData(args.train_meta_file, args.svg_folder, args.maxlen, cfg['text_len'], tokenizer, require_aug=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                             shuffle=True, 
                                             batch_size=args.batchsize,
                                             num_workers=8,
                                             pin_memory=True)

    val_dataset = SketchData(args.val_meta_file, args.svg_folder, args.maxlen, cfg['text_len'], tokenizer, require_aug=False)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                 shuffle=False, 
                                                 batch_size=args.batchsize,
                                                 num_workers=8)

    set_seed(2023)

    model = SketchDecoder(
        config={
            'hidden_dim': cfg['hidden_dim'],
            'embed_dim': cfg['embed_dim'], 
            'num_layers': cfg['num_layers'], 
            'num_heads': cfg['num_heads'],
            'dropout_rate': cfg['dropout_rate'],
        },
        pix_len=train_dataset.maxlen_pix,
        text_len=cfg['text_len'],
        num_text_token=tokenizer.vocab_size,
        word_emb_path=cfg['word_emb_path'],
        pos_emb_path=cfg['pos_emb_path'],
    )
   
    # Compile the model with dynamic shape handling
    model = torch.compile(model)
   
    lr = cfg['lr'] * accelerator.num_processes
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = cfg['warm_up_steps'],
        num_training_steps = len(train_dataloader) * cfg['epoch']
    )

    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, val_dataloader
    )

    num_update_steps_per_epoch = len(train_dataloader) // accum_step

    # logging 
    if accelerator.is_local_main_process:
        writer = SummaryWriter(log_dir=os.path.join(args.output_dir, args.project_name))

    # We need to keep track of how many total steps we have iterated over
    overall_step = 0
    # We also need to keep track of the stating epoch so files are named properly
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
            accelerator.load_state(args.resume_from_checkpoint)
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", ""))
        else:
            raise ValueError("Only support resuming from epoch checkpoints")

    accelerator.print('Start training...')
    
    # Sample texts for visualization
    sample_texts = [
        'calendar',
        'emotion,face,sad',
        'bug,spider',
        'trash,basket,garbage',
        'car',
        'star',
        'signal,wifi',
        'book,agenda',
    ]
    
    for epoch in range(starting_epoch, cfg['epoch']):
        model = model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch + 1}")
        total_loss, total_pix_loss, total_text_loss = 0., 0., 0.

        for pix, xy, mask, text in train_dataloader:
            with accelerator.accumulate(model):
                loss, pix_loss, text_loss = model(pix, xy, mask, text, return_loss=True)
                total_loss += loss.item() / accum_step
                total_pix_loss += pix_loss.item() / accum_step
                total_text_loss += text_loss.item() / accum_step
            
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip gradient
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()


            if accelerator.sync_gradients and accelerator.is_local_main_process:
                if overall_step % cfg['log_every'] == 0:
                    writer.add_scalar("loss/total_loss", total_loss, overall_step)
                    writer.add_scalar("loss/pix_loss", total_pix_loss, overall_step)
                    writer.add_scalar("loss/text_loss", total_text_loss, overall_step)
                    writer.add_scalar("lr", lr_scheduler.get_last_lr()[0], overall_step)
                total_loss, total_pix_loss, total_text_loss = 0., 0., 0.
                progress_bar.update(1)
                overall_step += 1

                # TODO remove, breaking early just for testing 
                break

        progress_bar.close()
        accelerator.wait_for_everyone()
        
        # Generate and log samples every epoch
        if accelerator.is_local_main_process:
            model.eval()
            with torch.no_grad():
                for sample_text in sample_texts:
                    # Tokenize text
                    encoded_dict = tokenizer(
                        sample_text,
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=cfg['text_len'],
                        add_special_tokens=True,
                        return_token_type_ids=False,
                    )
                    tokenized_text = encoded_dict["input_ids"][0]  # Get first sequence
                    tokenized_text = tokenized_text.unsqueeze(0).to(accelerator.device)
                    
                    # Generate samples - access the underlying model if in distributed mode
                    if accelerator.distributed_type != "NO":
                        model_for_sampling = accelerator.unwrap_model(model)
                    else:
                        model_for_sampling = model
                    samples = model_for_sampling.sample(n_samples=1, text=tokenized_text)
                    
                    # Convert samples to images and log to TensorBoard
                    for i, sample in enumerate(samples):
                        # Create a figure
                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.plot(sample[:, 0], sample[:, 1], 'k-')
                        ax.set_xlim(0, 200)
                        ax.set_ylim(0, 200)
                        ax.axis('off')
                        
                        # Convert plot to image
                        buf = io.BytesIO()
                        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                        buf.seek(0)
                        img = Image.open(buf)
                        
                        # Log to TensorBoard
                        writer.add_image(f'samples/{sample_text}', 
                                      np.array(img).transpose(2, 0, 1), 
                                      epoch + 1)
                        
                        plt.close()
            
            model.train()
            writer.flush()

        # save model after n epoch
        if (epoch+1) % cfg['save_every'] == 0:
            if accelerator.is_local_main_process:
                ckpt_path = os.path.join(args.output_dir, args.project_name, f'epoch_{epoch+1}')
                accelerator.save_state(ckpt_path)

        # Validation loss 
        if (epoch+1) % cfg['val_every'] == 0:
            model.eval()
            accelerator.print('Testing...')
            all_losses = []
            with tqdm(val_dataloader, unit="batch", disable=not accelerator.is_local_main_process) as batch_data:
                for pix, xy, mask, text in batch_data:
                    with torch.no_grad():
                        loss, pix_loss, text_loss = model(pix, xy, mask, text, return_loss=True)
                        all_targets = accelerator.gather_for_metrics(loss)
                        # Take the mean of gathered losses if it's a tensor with multiple elements
                        if isinstance(all_targets, torch.Tensor) and all_targets.numel() > 1:
                            all_losses.append(float(all_targets.mean()))
                        else:
                            all_losses.append(float(all_targets))
            valid_loss = np.array(all_losses).mean()
            accelerator.print(f'Epoch {epoch + 1}: validation loss is {valid_loss}')

    if accelerator.is_local_main_process:
        writer.close()


if __name__ == "__main__":
    set_seed(2023)

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_meta_file", type=str, required=True)
    parser.add_argument("--val_meta_file", type=str, required=True)
    parser.add_argument("--svg_folder", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--batchsize", type=int, required=True)
    parser.add_argument("--maxlen", type=int, required=True)
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    config = {
        'tokenizer_name': 'google/bert_uncased_L-12_H-512_A-8',
        'text_len': 50,

        'hidden_dim': 1024,
        'embed_dim': 512, 
        'num_layers': 16, 
        'num_heads': 8,
        'dropout_rate': 0.1,
        'word_emb_path': 'ckpts/word_embedding_512.pt',
        'pos_emb_path': None,
        'gradient_accumulation_steps': 2,

        'lr': 3e-4,  # need scaling for different batch size
        'warm_up_steps': 16000,
        'epoch': 100,

        'log_every': 25,   # step
        'save_every': 25,  # epoch
        'val_every': 5,    # epoch

        'batch_size': args.batchsize,
        'max_len': args.maxlen,
    }

    # Create training folder
    result_folder = os.path.join(args.output_dir, args.project_name)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    with open(os.path.join(result_folder, 'config.json'), 'w') as f:
        import json
        json.dump(config, f, indent=4)
        
    train(args, config)

