# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 03:10:25 2023

@author: vic_s
"""

import logging
import os
import sys
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import monai
from monai.data import ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity
import pickle
import argparse
from monai.data.utils import pad_list_data_collate
import matplotlib.pyplot as plt
import config_file as cfg
from utils import get_model
from torchsummary import summary

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # use parser if running from bash script
    parser = argparse.ArgumentParser()
    parser.add_argument('--expname', type=str, help='experiment name', required=True)
    parser.add_argument('--model_name', type=str, default='densenet', help='Name of the model to use: densenet, resnet, efficientnet, etc.')
    parser.add_argument('--seed', type=int, help='seed for reproducibility', required=True)

    args = parser.parse_args()
    exp_name = args.expname
    model_name = args.model_name


    # exp_name = 'exp_mini_test'
    # model_name = 'cnn3d'
    # seed = 1

    seed = args.seed  # You can use any integer as the seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(seed)


    home_dir = './'
    working_dir = home_dir + exp_name + '/'

    df_train = pd.read_csv(os.path.join(working_dir, "train.csv"))
    df_val = pd.read_csv(os.path.join(working_dir, "val.csv"))

    train_fpaths = df_train['filepath'].to_numpy()
    train_class_label = df_train['class_label'].to_numpy()

    val_fpaths = df_val['filepath'].to_numpy()
    val_class_label = df_val['class_label'].to_numpy()

    # Define transforms
    train_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((cfg.params['imagex'], cfg.params['imagey'], cfg.params['imagez']))])
    val_transforms = Compose([ScaleIntensity(), EnsureChannelFirst(), Resize((cfg.params['imagex'], cfg.params['imagey'], cfg.params['imagez']))])

    # create a training data loader - include padding
    train_ds = ImageDataset(image_files=train_fpaths, labels=train_class_label, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=cfg.params['batch_size'], shuffle=True, num_workers=2,worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda.is_available(), collate_fn=pad_list_data_collate)

    # create a validation data loader - include padding
    val_ds = ImageDataset(image_files=val_fpaths, labels=val_class_label, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=cfg.params['batch_size'], shuffle=False, worker_init_fn=seed_worker, generator=g,num_workers=2, pin_memory=torch.cuda.is_available(), collate_fn=pad_list_data_collate)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, spatial_dims=3, in_channels=1, out_channels=2).to(device)
    summary(model, (1, cfg.params['imagex'], cfg.params['imagey'], cfg.params['imagez']))

    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), cfg.params['lr'])

    # start a typical PyTorch training
    patience_counter = 0  # to keep track of the number of epochs without improvement
    val_interval = 1
    best_val_loss = float('inf')
    epoch_loss_values = list()
    # writer = SummaryWriter()

    # Initialize empty lists for metric values and epoch loss
    epoch_loss_values = []
    val_epoch_loss_values = []

    for epoch in range(cfg.params['epochs']):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{cfg.params['epochs']}")
        model.train()
        epoch_loss = 0
        val_epoch_loss = 0
        step = 0
        val_step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            optimizer.zero_grad()
            # print(inputs)
            outputs = model(inputs)
            # print(outputs)

            if model_name == 'cnn3d':
                labels = labels.long()  # labels.long()
            else:
                labels = labels

            # print(labels)

            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
            # writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_step += 1

                    # Convert val_data items to tensor if they are not
                    val_images, val_labels = val_data[0], val_data[1]
                    if not isinstance(val_images, torch.Tensor):
                        val_images = torch.tensor(val_images)
                    if not isinstance(val_labels, torch.Tensor):
                        val_labels = torch.tensor(val_labels)

                    val_labels = val_labels.to(device)
                    val_images = val_images.to(device)

                    if model_name == 'cnn3d':
                        val_labels = val_labels.long()  # labels.long()
                    else:
                        val_labels = val_labels

                    #early stopping based on val loss
                    val_outputs = model(val_images) #fwd pass
                    val_loss = loss_function(val_outputs, val_labels) # calculate the loss
                    val_epoch_loss += val_loss.item()# update running validation loss
                    val_epoch_len = len(val_ds) // val_loader.batch_size
                    print(f"{val_step}/{val_epoch_len}, val_loss: {val_loss.item():.4f}")

                val_epoch_loss /= val_step
                val_epoch_loss_values.append(val_epoch_loss)
                print(f"epoch {epoch + 1} average val loss: {val_epoch_loss:.4f}")

                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), working_dir + "best_model_" + exp_name + ".pth")
                    print("saved new best metric model")

                    patience_counter = 0  # reset the counter when a better metric is found

                else:
                    patience_counter += 1  # increment the counter when no improvement is found

                if patience_counter >= cfg.params['patience']:
                    print(f"Early stopping at epoch {epoch + 1}, best val loss: {best_val_loss:.4f} at epoch: {best_metric_epoch}")
                    break  # exit the loop when the patience limit is reached

                print(
                    "current epoch: {} current val loss: {:.4f} best val loss: {:.4f} at epoch {}".format(
                        epoch + 1, val_epoch_loss, best_val_loss, best_metric_epoch
                    )
                )
                # writer.add_scalar("val_loss", val_epoch_loss, epoch + 1)


    # Save the last model
    torch.save(model.state_dict(), working_dir + model_name + r"_last_model.pth")
    print(f"train completed, best val loss: {best_val_loss:.4f} at epoch: {best_metric_epoch}")
    # writer.close()

    # Plotting section
    plt.figure(figsize=(12, 12))  # Adjust the size to fit both plots

    # Subplot for Epoch Loss vs. Epoch
    plt.plot(range(1, len(epoch_loss_values) + 1), epoch_loss_values, '-o', color='red', label="Epoch Loss")
    plt.plot(range(1, len(val_epoch_loss_values) + 1), val_epoch_loss_values, '-o', color='blue', label="Validation Epoch Loss")
    plt.title("Epoch Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Epoch Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(working_dir + "epoch_loss_plot.png")

    #plt.show()

if __name__ == "__main__":
    main()
