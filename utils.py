#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:36:13 2023

@author: emma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import monai
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc
# import config_file as cfg
# from SFCN_Class import SFCNModelMONAIClassification, SFCN
# from CNN3D_Class import CNN3D

# def get_model(model_name, spatial_dims=3, in_channels=1, out_channels=2):
#     if model_name == 'densenet':
#         return monai.networks.nets.DenseNet(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             out_channels=out_channels
#             )
#     elif model_name == 'resnet':
#         return monai.networks.nets.ResNet(
#             block = 'basic',
#             layers = [2, 2, 2, 2],
#             block_inplanes = [64, 128, 256, 512],
#             spatial_dims=spatial_dims,
#             n_input_channels=in_channels # ,
#             # out_channels=out_channels
#             )
#
#     elif model_name == 'efficientnet':
#         return monai.networks.nets.EfficientNetBN(
#             model_name="efficientnet-b0",
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             num_classes=out_channels
#             )
#     elif model_name == 'highresnet':
#         return monai.networks.nets.HighResNet(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             out_channels=out_channels
#             )
#     elif model_name == 'senet':
#         return monai.networks.nets.SENet(
#             spatial_dims=spatial_dims,
#             in_channels=in_channels,
#             block = 'se_bottleneck',
#             layers = [3, 4, 6, 3],
#             groups = 64,
#             reduction = 16,
#             num_classes=out_channels
#             )
#     elif model_name == 'vit':
#         return monai.networks.nets.ViT(
#             spatial_dims=spatial_dims,
#             img_size=(cfg.params['imagex'], cfg.params['imagey'], cfg.params['imagez']),
#             in_channels=in_channels,
#             num_classes=out_channels,
#             pos_embed='conv',
#             patch_size=(16, 16, 16)
#             )
#     # elif model_name == 'sfcn_monai':
#     #     return SFCNModelMONAIClassification(
#     #         spatial_dims = spatial_dims,
#     #         in_channels = in_channels,
#     #         out_channels = out_channels
#     #         )
#     # elif model_name == 'sfcn':
#     #     return SFCN()
#     # elif model_name == 'cnn3d':
#     #     return CNN3D(
#     #         cfg.params,
#     #         spatial_dims = spatial_dims,
#     #         in_channels = in_channels,
#     #         out_channels = out_channels
#     #         )
#     else:
#         raise ValueError(f"Unsupported model_name: {model_name}. Supported models are: densenet, resnet, efficientnet.")



def model_eval(df_test, all_preds):
    '''
    append softmax model outputs to corresponding data in test dataframe
    '''
    df = df_test.copy()
    y_pred_raw = np.vstack(all_preds) #turn list of arrays into a stacked array
    y_pred = (y_pred_raw > 0.5).astype(int).reshape(-1, 1)

    print("Shape of y_pred_raw:", y_pred_raw.shape)

    df = df.rename(columns={'class_label': 'ground_truth'})
    df['preds'] = y_pred
    df['preds_raw_1'] = y_pred_raw[:,0].astype(float)
    return df


def compute_metrics(df, save_dir, label, plot=True):
    y_test = df['ground_truth'].values
    y_pred = df['preds'].values
    y_pred_raw = df['preds_raw_1'].values

    cm = confusion_matrix(y_test, y_pred)
    cm = cm / np.sum(cm, axis=1, keepdims=True)
    print(cm)
    cm_df = pd.DataFrame(cm,
            index = ['ND','D'],
            columns = ['ND','D'])
    if plot:
        #Plotting the confusion matrix
        plt.figure(figsize=(10,8))
        sns.heatmap(cm_df, cmap="Blues", annot=True,fmt='.2f', vmin=0, vmax=1.0, center=0.5,cbar=True)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Values')
        plt.xlabel('Predicted Values')
        plt.savefig(save_dir + label + '_cm.png', bbox_inches='tight', dpi = 800)
        plt.clf()

    ac=accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)

    fpr = 1-spec

    # print(f'y test:{y_test}')
    # print(f'y preds:{y_pred_raw}')
    auc = roc_auc_score(y_test, y_pred_raw)

    return [ac, sens, spec, fpr, auc]



def plot_roc_curves(df, df_b1, df_b0, save_dir):
    fpr, tpr, _ = roc_curve(df['ground_truth'], df['preds_raw_1'])
    roc_auc = auc(fpr, tpr)

    fpr_b1, tpr_b1, _ = roc_curve(df_b1['ground_truth'], df_b1['preds_raw_1'])
    roc_auc_b1 = auc(fpr_b1, tpr_b1)

    fpr_b0, tpr_b0, _ = roc_curve(df_b0['ground_truth'], df_b0['preds_raw_1'])
    roc_auc_b0 = auc(fpr_b0, tpr_b0)


    # Plot the ROC curve
    print('plotting roc curve :-)')
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, linestyle='-', label='Aggregate ROC curve (area = %0.2f)' % roc_auc)
    plt.plot(fpr_b1, tpr_b1, color='darkmagenta', lw=2, linestyle='-.',label='Bias=1 ROC curve (area = %0.2f)' % roc_auc_b1)
    plt.plot(fpr_b0, tpr_b0, color='darkturquoise', lw=2, linestyle='--',label='Bias=0 ROC curve (area = %0.2f)' % roc_auc_b0)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle=':')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(save_dir + 'ROC.png', bbox_inches='tight', dpi = 800)
    # plt.show()
    plt.clf()
