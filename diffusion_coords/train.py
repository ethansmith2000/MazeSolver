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
from common.dataset import visualize_maze
from types import SimpleNamespace
import diffusers
import wandb
from pathlib import Path
from PIL import Image
from diffusion_coords.models import DiffusionTransformer

default_arguments = dict(
    model_path="runwayml/stable-diffusion-v1-5",
    output_dir="maze-output",
    seed=None,
    maze_size=13,
    train_batch_size=64,
    max_train_steps=60_000,
    validation_steps=1000,
    checkpointing_steps=1000,
    resume_from_checkpoint="/home/ubuntu/MazeSolver/diffusion_coords/maze-output/checkpoint-4000",
    gradient_accumulation_steps=1,
    gradient_checkpointing=True,
    learning_rate=1.0e-4,
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
    guidance_scale=1.2,
    num_timesteps=35,

    dropout=0.05,

    encoder_layers=6,
    decoder_layers=8,
    dim=512,
    heads=8,
    ff_mult=3,
)


@torch.no_grad()
def gen_samples(model, noise_scheduler, dataloader, out_dir, guidance_scale=1.0, num_timesteps=25):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    batch = next(iter(dataloader))
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype

    mazes = batch["maze_labeled"].to(device).long()

    latents = torch.randn(mazes.shape[0], mazes.shape[2] * mazes.shape[3], 3, device=device).to(dtype)
    noise_scheduler.set_timesteps(num_timesteps, device=device)
    all_mazes = torch.cat([mazes, mazes], dim=0).to(device) # will use mask to dropout neg
    mask = torch.ones(all_mazes.shape[0], device=device).to(dtype)
    mask[:mask.shape[0] // 2] = 0

    for i, t in tqdm(enumerate(noise_scheduler.timesteps), total=num_timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1.0 else latents
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)

        noise_pred = model(
            latent_model_input,
            all_mazes,
            t,
            dropout_mask=mask,
        )

        if guidance_scale > 1.0:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # unnormalize coordinates
    masks = latents[:, :, 2:3]
    coords = latents[:, :, :2]
    coords = coords / 3
    coords = coords / 2 + 0.5
    coords = coords * mazes.shape[-1] # scale relative back to actual maze dim
    # round to nearest int
    coords = coords.round().long()

    images = []
    for i in range(coords.shape[0]):
        maze = mazes[i].squeeze()
        for j in range(coords.shape[1]):
            try:
                if masks[i, j, 0] > 0.5:
                    maze[coords[i, j, 1], coords[i, j, 0]] = 5 # mark path
            except:
                pass

        solved_maze = visualize_maze(maze.float().cpu().numpy())
        images.append(solved_maze)

    wandb_images = []
    for i in range(len(images)):
        filename = f"{f'{i}'.zfill(3)}.png"
        images[i].save(Path(out_dir) / filename)
        wandb_images.append(wandb.Image(images[i]))

    wandb.log({"validation_images": wandb_images})


def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    accelerator, weight_dtype = init_train_basics(args, logger)
    noise_scheduler = diffusers.DDIMScheduler.from_config(args.model_path, subfolder="scheduler")

    model = DiffusionTransformer(num_layers_encoder=args.encoder_layers,
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
                                                        train_dataset, logger, num_update_steps_per_epoch, wandb_name="diffusion_coords_maze")

    for epoch in range(first_epoch, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                maze = batch["maze_labeled"].to(accelerator.device).long()
                path = batch["path"]

                def pad_to_size(x):
                    return F.pad(torch.tensor(x), (0, 0, 0, model.total_pixs - len(x)), value=-10) # placeholder number
                path = [pad_to_size(x) for x in path] # pad to maximum possible path length
                path = torch.stack(path, dim=0).to(accelerator.device).to(weight_dtype)
                mask = (path[:,:,0:1] != -10).float() # mask out areas after path has finished
                path = torch.where(path == -10, torch.zeros_like(path), path) # replace placeholder with 0s
                path = (path / model.maze_one_side) * 2 - 1 # [-1, 1]
                path = path * 3 # path std naturally tends to be around 0.36
                path = torch.cat([path, mask], dim=-1) # b, s, 3, -> [x, y, mask]

                noise = torch.randn_like(path)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (path.shape[0],), device=accelerator.device
                ).long()
                noisy_model_input = noise_scheduler.add_noise(path, noise, timesteps)

                # Predict actual coordinates and mask will tell us where to stop
                mask = torch.rand(noise.shape[0], device=accelerator.device) > args.dropout
                noise_pred = model(noisy_model_input, maze, timesteps, attn_mask=None, dropout_mask=mask)
                noise_pred_coords = noise_pred[:, :, :2]
                noise_pred_masks = noise_pred[:, :, 2:3]
                noise_coords = noise[:, :, :2]
                noise_masks = noise[:, :, 2:3]

                loss_coords = F.mse_loss(noise_pred_coords, noise_coords, reduction="mean")
                loss_masks = F.mse_loss(noise_pred_masks, noise_masks, reduction="mean")
                loss = loss_coords + loss_masks

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
                    gen_samples(model, noise_scheduler, train_dataloader, save_path, guidance_scale=args.guidance_scale, num_timesteps=args.num_timesteps)

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"checkpoint-final-{global_step}")
        save_model(model, save_path, logger)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)