# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 12:49:54 2023

@author: vic_s
"""

import logging
import os
import sys
import numpy as np
import pandas as pd
import torch
import random
import monai
import torchvision
from matplotlib import pyplot as plt
from monai.data import  ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, Resize, ScaleIntensity, NormalizeIntensity, ToTensor
import argparse

from monai.visualize import SmoothGrad, GradCAM
from tqdm import tqdm

from SFCN import SFCNModel
from utils.customTransforms import ToFloatUKBB
# import config_file as cfg
from utils.utils import model_eval, compute_metrics, plot_roc_curves

GT_CONFIG = False       # True if getting ground truth baseline, False if evaluating counterfactuals
EXP_NAME = "moin_bias"
LABEL = "intensity_bias"
CSV_DIR = "splits2/exp199/"

BATCH_SIZE = 16
N_WORKERS = 0
N_EPOCHS = 20
MAX_IMAGES = -1
LR = 0.0001
PATIENCE = 5

from sklearn.preprocessing import OneHotEncoder

# Function to convert labels to one-hot encoding
def one_hot_encode(labels):
    encoder = OneHotEncoder(sparse=False, categories='auto')
    labels = labels.reshape(-1, 1)
    return encoder.fit_transform(labels)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level = logging.INFO)

    # use parser if running from bash script
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default=EXP_NAME, help='experiment name')
    parser.add_argument('--model_name', type=str, default='densenet', help='Name of the model to use: densenet, resnet, efficientnet, etc.')

    args = parser.parse_args()
    exp_name = args.exp_name
    model_name = args.model_name

    seed = 1
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    g = torch.Generator()
    g.manual_seed(seed)


    home_dir = './'
    working_dir = home_dir + exp_name
    if GT_CONFIG:
        working_dir += '/SFCN/'


    df_test = pd.read_csv(os.path.join(home_dir, CSV_DIR, "test.csv"))
    test_fpaths = [os.path.join(home_dir,exp_name, "test", filename.replace("nii.gz", "tiff")) for filename in df_test['filename']]
    # test_fpaths = [os.path.join("./data/cfs", filename.replace("nii.gz", "tiff")) for filename in
    #                df_test['filename'][:248]]

    test_class_label = one_hot_encode(df_test[LABEL].values)
    # test_class_label = np.zeros(len(test_fpaths))

    # Define transforms for image
    transforms = Compose([torchvision.transforms.CenterCrop(180), EnsureChannelFirst(), ToFloatUKBB(), ToTensor()])

    # Define image dataset
    test_ds = ImageDataset(image_files=test_fpaths, labels=test_class_label, transform=transforms, image_only=True, reader="PILReader")

    # create a validation data loader
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4, generator=g, pin_memory=torch.cuda.is_available())

    # Create DenseNet121
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SFCNModel().to(device)
    model.load_state_dict(torch.load(working_dir + "best_model_" + exp_name + ".pth"))

    model.eval()

    all_preds = []
    with torch.no_grad():
        # saver = CSVSaver(output_dir="./output")

        for idx, test_data in tqdm(enumerate(test_loader), total=len(test_loader)):
            test_images = test_data[0].to(device)

            # print(test_data[0][0].max(), test_data[0][0].min())

            # Get model's probability outputs
            outputs = model(test_images)
            # probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(outputs.cpu().numpy())

            # saver.save_batch(torch.tensor(test_outputs).to(device), test_images.meta)

    print(len(all_preds))

    # concat predictions to test dataframe
    df = model_eval(df_test, all_preds)
    df.to_csv(working_dir + 'preds_' + exp_name + '.csv')  # save file with predictions

    smooth_grad = SmoothGrad(model)

    for idx, batch_data in enumerate(test_loader):
        inputs, labels = batch_data[0].to(device), batch_data[1].to(device)

        for i in range(inputs.size(0)):
            input_image = inputs[i].unsqueeze(0)  # Add batch dimension
            input_image.requires_grad_()

            # Apply SmoothGrad
            # print(input_image)
            print(input_image.shape)
            print(model(input_image))
            print(model(input_image).shape)
            # print(model(inputs))

            saliency_map = smooth_grad(input_image, class_idx=0)

            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(input_image[0].cpu().permute(1, 2, 0), cmap='gray')
            ax[0].set_title('Input Image')
            ax[0].axis('off')

            ax[1].imshow(saliency_map[0].cpu().numpy(), cmap='hot')
            ax[1].set_title('SmoothGrad Saliency Map')
            ax[1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join("./data/saliency/", f'saliency_map_{idx}_{i}.png'))
            plt.close()

        if idx == 4:  # Visualize saliency maps for the first 5 batches
            break

    #create one-hot encoded columns for TP, TN, FP, FN
    df['TP'] = df.apply(lambda row: 1 if ((row['bias_label'] == 1) & (row['preds'] ==1)) else 0, axis=1)
    df['TN'] = df.apply(lambda row: 1 if ((row['bias_label'] == 0) & (row['preds'] ==0)) else 0, axis=1)
    df['FP'] = df.apply(lambda row: 1 if ((row['bias_label'] == 0) & (row['preds'] ==1)) else 0, axis=1)
    df['FN'] = df.apply(lambda row: 1 if ((row['bias_label'] == 1) & (row['preds'] ==0)) else 0, axis=1)

    # df.to_csv(working_dir + 'preds_' + exp_name + '.csv') #save file with predictions


    # Compute metrics
    df_B1 = df.loc[df['ground_truth']==1]
    df_B0 = df.loc[df['ground_truth']==0]

    #generate file with performance metrics
    metrics = compute_metrics(df, save_dir=working_dir, label='Agg')
    metrics_B1 = compute_metrics(df_B1, save_dir=working_dir, label='B1')
    metrics_B0 = compute_metrics(df_B0, save_dir=working_dir, label='B0')

    metrics_df = pd.DataFrame(['Acc', 'Sens', 'Spec', 'FPR', 'AUROC'], columns=['metrics'])
    metrics_df = metrics_df.set_index('metrics')

    metrics_df['Aggregate'] = metrics
    metrics_df['disease_1'] = metrics_B1
    metrics_df['disease_0'] = metrics_B0
    metrics_df.to_csv(working_dir + 'metrics_' + exp_name + '.csv')


    plot_roc_curves(df, df_B1, df_B0, save_dir=working_dir)

if __name__ == "__main__":
    main()
