#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm

import sys
sys.path.append('..')

import common.train_utils
from common.train_utils import (
    init_train_basics,
    save_model,
    get_optimizer,
    get_dataset,
    more_init
)
from common.dataset import visualize_maze, get_movements_from_path, get_path_from_movements
from types import SimpleNamespace
import diffusers
import wandb
from pathlib import Path
from PIL import Image
from transformer_move.models import Transformer

default_arguments = dict(
    model_path="runwayml/stable-diffusion-v1-5",
    output_dir="maze-output",
    seed=None,
    maze_size=13,
    train_batch_size=64,
    max_train_steps=40_000,
    validation_steps=1000,
    checkpointing_steps=1000,
    resume_from_checkpoint="/home/ubuntu/MazeSolver/transformer_move/maze-output/checkpoint-15000",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    learning_rate=5.0e-5,
    lr_scheduler="linear",
    lr_warmup_steps=50,
    lr_num_cycles=1,
    lr_power=1.0,
    dataloader_num_workers=4,
    use_8bit_adam=False,
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_weight_decay=1e-2,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    report_to="wandb",
    mixed_precision="bf16",
    allow_tf32=True,
    logging_dir="logs",
    local_rank=-1,
    num_processes=1,

    encoder_layers=6,
    decoder_layers=8,
    dim=512,
    heads=8,
    ff_mult=3,
)



@torch.no_grad()
def gen_samples(model, dataloader, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    batch = next(iter(dataloader))
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype

    mazes = batch["maze_labeled"].to(device).long()[:32]
    start_token = torch.tensor([4], device=device).long()[None,:].repeat(mazes.shape[0], 1)
    sequence = start_token.clone()

    for i in tqdm(range(model.total_pixs//2)):
        preds = model(sequence, mazes, attn_mask=None,)
        if i == 20:
            with torch.no_grad():
                print(torch.nn.functional.softmax(preds[0, :20], dim=-1))
                print(sequence[0, :20])

        # keep going until all paths have a stop token in them or we reach max_len
        preds = preds.argmax(dim=-1)
        sequence = torch.cat([sequence, preds[:, -1:]], dim=1)

        if len(torch.where(preds == 5)[0]) == mazes.shape[0]:
            break

    paths = []
    # truncate each path to the first stop token
    for i in range(sequence.shape[0]):
        moves = sequence[i]
        end_pos = torch.where(moves == 5)[0]
        if len(end_pos) > 0:
            moves = moves[:end_pos[0] + 1]
        path = get_path_from_movements(moves, mazes[i,0])
        paths.append(path)

    images = []
    for i in range(len(paths)):
        maze = mazes[i,0]
        solved_maze = visualize_maze(maze.float().cpu().numpy(), paths[i])
        filename = f"{f'{i}'.zfill(3)}.png"
        solved_maze.save(Path(out_dir) / filename)
        images.append(wandb.Image(solved_maze))

    wandb.log({"validation_images": images})


def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    accelerator, weight_dtype = init_train_basics(args, logger)

    model = Transformer(num_layers_encoder=args.encoder_layers,
                                num_layers_decoder=args.decoder_layers,
                                dim=args.dim,
                                heads=args.heads,
                                ff_mult=args.ff_mult,
                                maze_size=args.maze_size,
                                )

    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()

    optimizer, lr_scheduler = get_optimizer(args, model.parameters(), accelerator)
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step, first_epoch, progress_bar = more_init(model, accelerator, args, train_dataloader, 
                                                        train_dataset, logger, num_update_steps_per_epoch, wandb_name="transformer_maze")

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                maze = batch["maze_labeled"].to(accelerator.device).long()
                path = batch["path"]

                # turn path into series of movements
                new_paths = []
                for p in path:
                    movement_ids, _ = get_movements_from_path(p)
                    # add 4 to the start and 5 to the end to indicate start and end
                    movement_ids = [4] + movement_ids + [5]
                    movement_ids = torch.tensor(movement_ids, device=accelerator.device).long()
                    new_paths.append(movement_ids)

                def pad_to_size(x):
                    return F.pad(torch.tensor(x), (0, model.total_pixs // 2 - len(x)), value=5)

                path = [pad_to_size(x) for x in new_paths]
                path = torch.stack(path, dim=0).to(accelerator.device).long()

                # trim to longest path
                longest_pos = path.argmax(dim=-1).max()
                path = path[:, :longest_pos+1]

                input_path = path[:, :-1]
                target_path = path[:, 1:]

                preds = model(input_path, maze, attn_mask=None)
                loss = F.cross_entropy(preds.reshape(-1, preds.shape[-1]).float(), target_path.reshape(-1), reduction="none")
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_model(model, save_path, logger)
                        

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if global_step % args.validation_steps == 0 and global_step > 0:
                    save_path = os.path.join(args.output_dir, f"samples/checkpoint-{global_step}")
                    gen_samples(model, train_dataloader, save_path)

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-final-{global_step}")
        save_model(model, save_path, logger)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)