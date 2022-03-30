from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import torch

torch.cuda.set_device(0)

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    timesteps = 1000,   # number of steps
    loss_type = 'l1'    # L1 or L2
).cuda()

trainer = Trainer(
    diffusion,
    '/mnt/store/psemchyshyn/screw/train/good',
    image_size = 64,
    train_batch_size = 32,
    train_lr = 2e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    fp16 = False                       # turn on mixed precision training with apex
)

trainer.train()
