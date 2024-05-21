# MazeSolver

A couple of fun trainers using diffusion and autoregressive models to solve mazes. Explainer blog post [here](https://sweet-hall-e72.notion.site/Diffusion-and-Autoregressive-Models-for-Learning-to-Solve-Mazes-c3bc4bcdfa304ecd9531ee5445a4da66?pvs=4) 


## Requirements
torch, diffusers, transformers, accelerate should do the trick


## How To Use
There are 3 different trainers provided, namely diffusion, diffusion_coords, and transformer_move. Each can be run by editing the arguments within the script and calling python train.py