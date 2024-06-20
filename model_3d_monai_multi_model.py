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
import torchvision
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import monai
from monai.data import ImageDataset, DataLoader, ITKReader
from monai.transforms import EnsureChannelFirst, Compose, RandRotate90, Resize, ScaleIntensity, NormalizeIntensity, ToTensor
import pickle
import argparse
from monai.data.utils import pad_list_data_collate
import matplotlib.pyplot as plt
# import config_file as cfg
# from utils import get_model
# from torchsummary import summary

from SFCN import SFCNModel

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # use parser if running from bash script
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='far_bias', help='experiment name')
    parser.add_argument('--model_name', type=str, default='resnet', help='Name of the model to use: densenet, resnet, efficientnet, etc.')
    parser.add_argument('--seed', type=int, help='seed for reproducibility', default=1)

    args = parser.parse_args()
    exp_name = args.exp_name
    model_name = args.model_name

    BATCH_SIZE = 16
    N_WORKERS = 0
    N_EPOCHS = 20
    MAX_IMAGES = -1
    LR = 0.0001
    PATIENCE = 5


    # exp_name = 'exp_mini_test'
    # model_name = 'cnn3d'
    # seed = 1

    seed = 1  # You can use any integer as the seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)


    home_dir = './'
    working_dir = home_dir + exp_name + '/'

    df_train = pd.read_csv(os.path.join(home_dir, "splits/train.csv"))
    df_val = pd.read_csv(os.path.join(home_dir, "splits/val.csv"))

    train_fpaths = [os.path.join(working_dir, "train", filename) for filename in df_train['filename']]
    train_class_label = df_train['bias_label']

    val_fpaths = [os.path.join(working_dir, "val", filename) for filename in df_val['filename']]
    val_class_label = df_val['bias_label']

    # Define transforms
    transforms = Compose([torchvision.transforms.CenterCrop(180), EnsureChannelFirst(), NormalizeIntensity(), ToTensor()])

    # create a training data loader - include padding
    train_ds = ImageDataset(image_files=train_fpaths, labels=train_class_label, transform=transforms, reader="PILReader")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2,worker_init_fn=seed_worker, generator=g, pin_memory=torch.cuda.is_available(), collate_fn=pad_list_data_collate)

    # create a validation data loader - include padding
    val_ds = ImageDataset(image_files=val_fpaths, labels=val_class_label, transform=transforms, reader="PILReader")
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, worker_init_fn=seed_worker, generator=g,num_workers=N_WORKERS, pin_memory=torch.cuda.is_available(), collate_fn=pad_list_data_collate)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SFCNModel().to(device)
    # summary(model, (1, cfg.params['imagex'], cfg.params['imagey'], cfg.params['imagez']))

    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), LR)

    # start a typical PyTorch training
    patience_counter = 0  # to keep track of the number of epochs without improvement
    val_interval = 1
    best_val_loss = float('inf')
    epoch_loss_values = list()
    # writer = SummaryWriter()

    # Initialize empty lists for metric values and epoch loss
    epoch_loss_values = []
    val_epoch_loss_values = []

    val_epoch_accuracy = []
    val_epoch_accuracy_values = []

    for epoch in range(N_EPOCHS):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{N_EPOCHS}")
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

            loss = loss_function(outputs, labels.float())

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

                total_correct_predictions_val = 0
                total_predictions_val = 0

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
                    val_loss = loss_function(val_outputs, val_labels.float()) # calculate the loss
                    val_epoch_loss += val_loss.item()# update running validation loss
                    val_epoch_len = len(val_ds) // val_loader.batch_size

                    # Compute accuracy for the current batch
                    val_predictions = (val_outputs > 0.5).float()  # Assuming sigmoid activation function for binary classification
                    val_correct_predictions = (val_predictions == val_labels).sum().item()
                    total_correct_predictions_val += val_correct_predictions
                    total_predictions_val += val_labels.numel()
                    val_batch_accuracy = val_correct_predictions / val_labels.numel()

                    print(f"{val_step}/{val_epoch_len}, val_loss: {val_loss.item():.4f}, batch_accuracy: {val_batch_accuracy:.4f}")

                val_epoch_loss /= val_step
                val_epoch_loss_values.append(val_epoch_loss)

                val_epoch_accuracy = total_correct_predictions_val / total_predictions_val
                val_epoch_accuracy_values.append(val_epoch_accuracy)

                print(f"epoch {epoch + 1} average val loss: {val_epoch_loss:.4f}, epoch_accuracy: {val_epoch_accuracy:.4f}")

                if val_epoch_loss < best_val_loss:
                    best_val_loss = val_epoch_loss
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), working_dir + "best_model_" + exp_name + ".pth")
                    print("saved new best metric model")

                    patience_counter = 0  # reset the counter when a better metric is found

                else:
                    patience_counter += 1  # increment the counter when no improvement is found

                if patience_counter >= PATIENCE:
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
