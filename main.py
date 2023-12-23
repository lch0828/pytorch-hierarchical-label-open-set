import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, BatchSampler, random_split

import torchvision
from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import resnet18, ResNet18_Weights

from model import HMCN
from dataloader import MultiClassImageDataset, MultiClassImageTestDataset
from trainer import HMCNTrainer
from config import conf, print_config

args = conf()
device = 'cuda'

print('Loading data...')

train_ann_df = pd.read_csv('../train_data.csv')
super_map_df = pd.read_csv('../superclass_mapping.csv')
sub_map_df = pd.read_csv('../subclass_mapping.csv')

train_img_dir = args.train_div
test_img_dir = args.test_div

image_augmentation = transforms.Compose([
#    transforms.RandomVerticalFlip(),
#    transforms.RandomRotation(20, interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0), std=(1))
])

image_preprocessing = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0), std=(1)),
])

# Create train and val split
train_dataset = MultiClassImageDataset(train_ann_df, super_map_df, sub_map_df, train_img_dir, transform=image_augmentation)
train_dataset, val_dataset = random_split(train_dataset, [0.9, 0.1])

# Create test dataset
test_dataset = MultiClassImageTestDataset(super_map_df, sub_map_df, test_img_dir, transform=image_preprocessing)

# Create dataloaders
batch_size = args.batch
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=True)

test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=False)

print('Loading model...')

if args.encoder == 'vit_b_16':
    weights = ViT_B_16_Weights.DEFAULT
    encoder_model = vit_b_16(weights=weights)
elif args.encoder == 'vit_b_32':
    weights = ViT_B_32_Weights.DEFAULT
    encoder_model = vit_b_32(weights=weights)
elif args.encoder == 'cnn':
    cnn_weights = ResNet18_Weights.DEFAULT
    encoder_model = resnet18(weights=cnn_weights)
    
encoder_model.to(device)
encoder_model.eval()

if args.deep_residual:
    print("deep_residual used")

model = HMCN(args).to(device)
model_name = f'{args.encoder}/{"deep" if args.deep_residual else ""}_E{args.epoch}_B{args.batch}_{args.global_weight_dim}_{args.encoder}.pt'
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

trainer = HMCNTrainer(model, encode_mode, encoder_model, weights, criterion, optimizer, train_loader, val_loader, test_loader, device)

print('Start training...')

train_losses = []
val_losses = []

# Training loop
for epoch in range(args.epoch):
    print(f'Epoch {epoch+1}')
    train_loss = trainer.train_epoch()
    val_loss = trainer.validate_epoch(B=0.5)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch == args.epoch - 1:
        torch.save(model.state_dict(), './models/' + model_name)

    print('')

dict = {'train_loss': train_losses, 'val_loss': val_losses}
df = pd.DataFrame(dict)
df.to_csv(f'./losses/{args.encoder}/{"deep" if args.deep_residual else ""}_E{epoch+1}_B{args.batch}_{args.global_weight_dim}_{args.encoder}_loss.csv')

print('Finished training')

print('Start testing...')
# Test
trainer.test(model_name, B=0.5, save_to_csv=True, return_predictions=False)

print('Finished testing')


