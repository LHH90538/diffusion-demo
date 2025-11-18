from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
model = Unet(
    dim=64,                       #
    dim_mults=(1, 2, 4, 8),       #
    channels=3)                   # 彩色图片用3通道

diffusion = GaussianDiffusion(
    model,
    image_size=48,                # 图片尺寸
    timesteps=800 )               # 扩散步数，可调

trainer = Trainer(
    diffusion_model=diffusion,
    folder="./cifar_png",
    train_batch_size=64,          # 批次大小，可调
    train_lr=1e-4,
    train_num_steps=20000,        #  训练20000步，可调
    ema_decay=0.995,
    calculate_fid=False,
    save_and_sample_every=2000,  # 每2000步采样一次，输出图片。可调
    results_folder="./results_cifar")

trainer.train()