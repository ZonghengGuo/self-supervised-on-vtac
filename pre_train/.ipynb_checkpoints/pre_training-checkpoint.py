import torch
from model import SimSiam, FCN
import os
from dataset import SiamDataset
from torch.utils.data import random_split, DataLoader
import losses
import numpy as np
from tqdm import tqdm
import json

import matplotlib.pyplot as plt

# 保存 loss 曲线图
def plot_losses(train_losses, val_losses, save_path='loss_curve.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


device = 'cuda'

# Setting
batch_size = 256
backbone = "FCN"
pair_data_path = "data/mimic/pair_segments"
lr = 1e-4
epochs = 50
ratio_train_val = 0.9
model_save_path = "model_saved"


# model = SimSiam(encoder = backbone,projector=True)
# based on FCN
# 初始化 FCN 编码器
fcn_encoder = FCN()

# 初始化 SimSiam
model = SimSiam(
    dim=64,  # FCN projector 输出的维度
    pred_dim=32,
    predictor=True,
    single_source_mode=False,
    encoder=fcn_encoder
)

model = model.to(device)

dataset = SiamDataset(pair_data_path)

train_size = int(ratio_train_val * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print("Total numbers of training pairs:", len(train_dataset))
print("Total numbers of validation pairs:", len(val_dataset))
print("Using the model of:", backbone)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

criterion = losses.NegativeCosineSimLoss().to(device)

param_groups = [
    {'params': list(set(model.parameters()))}
]

opt = torch.optim.Adam(param_groups, lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)


losses_list = []
val_losses_list = []

for epoch in range(0, epochs):
    model.train()
    losses_per_epoch = []

    pbar = tqdm(enumerate(train_dataloader))

    for batch_idx, (x1, x2) in tqdm(enumerate(train_dataloader)):
        x1, x2 = x1.to("cuda", dtype=torch.float32), x2.to("cuda", dtype=torch.float32)
        p1, p2, z1, z2 = model(x1, x2)
        loss = criterion(p1, p2, z1, z2)

        opt.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(model.parameters(), 10)

        losses_per_epoch.append(loss.cpu().data.numpy())
        opt.step()

        pbar.set_description(
            'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx + 1, len(train_dataloader),
                100. * batch_idx / len(train_dataloader),
                loss.item()))
    print(f"Training loss {np.mean(losses_per_epoch)}")
    losses_list.append(np.mean(losses_per_epoch))
    # wandb.log({'loss': losses_list[-1]}, step=epoch)
    scheduler.step()

    # Validation
    model_is_training = model.training
    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():
        losses_val = []
        pbar = tqdm(enumerate(val_dataloader))
        for batch_idx, (x1, x2) in pbar:
            x1, x2 = x1.to(device, dtype=torch.float32), x2.to(device, dtype=torch.float32)
            p1, p2, z1, z2 = model(x1, x2)
            loss = criterion(p1, p2, z1, z2)
            losses_val.append(loss.cpu().data.numpy())
        print(f"Validation loss {np.mean(losses_val)}")
        val_loss_epoch = np.mean(losses_val)
        val_losses_list.append(val_loss_epoch)

    model.train()
    model.train(model_is_training)

    if losses_list[-1] == min(losses_list):
        print("Model is going to save")
        print(f"last loss: {losses_list[-1]} | min loss: {min(losses_list)}")
        # if not os.path.exists('{}'.format(LOG_DIR)):
        #     os.makedirs('{}'.format(LOG_DIR))
        torch.save({'model_state_dict':model.state_dict()}, '{}/{}_.pth'.format(model_save_path, backbone))
plot_losses(losses_list, val_losses_list, save_path='train_val_loss_curve.png')