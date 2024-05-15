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
from monai.data import  ImageDataset, DataLoader
from monai.transforms import EnsureChannelFirst, Compose, Resize, ScaleIntensity, NormalizeIntensity, ToTensor
import argparse
from tqdm import tqdm

from SFCN import SFCNModel
# import config_file as cfg
from utils import model_eval, compute_metrics, plot_roc_curves

BATCH_SIZE = 16
N_WORKERS = 0
N_EPOCHS = 20
MAX_IMAGES = -1
LR = 0.0001
PATIENCE = 5

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level = logging.INFO)

    # use parser if running from bash script
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='no_bias', help='experiment name')
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
    working_dir = home_dir + exp_name + '/'


    df_test = pd.read_csv(os.path.join(home_dir, "splits/test.csv"))

    test_fpaths = [os.path.join(working_dir, "test", filename.replace(".nii.gz", ".tiff")) for filename in df_test['filename']]
    test_class_label = df_test['class_label']

    # Define transforms for image
    transforms = Compose([torchvision.transforms.CenterCrop(180), EnsureChannelFirst(), NormalizeIntensity(), ToTensor()])

    # Define image dataset
    test_ds = ImageDataset(image_files=test_fpaths, labels=test_class_label, transform=transforms, image_only=True, reader="ITKReader")

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

            # Get model's probability outputs
            outputs = model(test_images)
            # probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(outputs.cpu().numpy())

            # saver.save_batch(torch.tensor(test_outputs).to(device), test_images.meta)

    # concat predictions to test dataframe
    df = model_eval(df_test, all_preds)

    #create one-hot encoded columns for TP, TN, FP, FN
    df['TP'] = df.apply(lambda row: 1 if ((row['ground_truth'] == 1) & (row['preds']==1)) else 0, axis=1)
    df['TN'] = df.apply(lambda row: 1 if ((row['ground_truth']== 0) & (row['preds'] ==0)) else 0, axis=1)
    df['FP'] = df.apply(lambda row: 1 if ((row['ground_truth'] == 0) & (row['preds'] ==1)) else 0, axis=1)
    df['FN'] = df.apply(lambda row: 1 if ((row['ground_truth'] == 1) & (row['preds'] ==0)) else 0, axis=1)

    df.to_csv(working_dir + 'preds_' + exp_name + '.csv') #save file with predictions


    # Compute metrics
    df_B1 = df.loc[df['bias_label']==1]
    df_B0 = df.loc[df['bias_label']==0]

    #generate file with performance metrics
    metrics = compute_metrics(df, save_dir=working_dir, label='Agg')
    metrics_B1 = compute_metrics(df_B1, save_dir=working_dir, label='B1')
    metrics_B0 = compute_metrics(df_B0, save_dir=working_dir, label='B0')

    metrics_df = pd.DataFrame(['Acc', 'Sens', 'Spec', 'FPR', 'AUROC'], columns=['metrics'])
    metrics_df = metrics_df.set_index('metrics')

    metrics_df['Aggregate'] = metrics
    metrics_df['bias_label_1'] = metrics_B1
    metrics_df['bias_label_0'] = metrics_B0
    metrics_df.to_csv(working_dir + 'metrics_' + exp_name + '.csv')


    plot_roc_curves(df, df_B1, df_B0, save_dir=working_dir)

if __name__ == "__main__":
    main()
