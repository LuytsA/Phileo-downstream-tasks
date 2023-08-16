import torch
import torch.nn as nn
import torchmetrics

import buteo as beo
import numpy as np
from datetime import date

import sys; sys.path.append("../")
from models.model_SimpleUNet import SimpleUnet
from models.model_ViT import vit_mse_losses, ViT
from models.model_ConvNext import ConvNextV2Unet_tiny, ConvNextV2Unet_atto, ConvNextV2Unet_pico, ConvNextV2Unet_base
from models.model_Diamond import DiamondNet
from models.model_CoreCNN import CoreUnet_tiny
from models.model_MixerMLP import MLPMixer
from models.model_VisionTransformer import ViT

import os
from glob import glob
from tqdm import tqdm
import json
import config_lc
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt 

from utils import (
    load_data,
    data_protocol_bd,
    visualise
)

def precision_recall(y_pred_classification,y_test_classification):
    diff = y_pred_classification.astype(int)-y_test_classification.astype(int)
    fp = np.count_nonzero(diff == 1) # false positives
    fn = np.count_nonzero(diff == -1) # false negatives
    tp = np.sum(y_test_classification) - fn # true positives = all positives - false negatives

    return np.array([tp, fp, fn])



def calculate_metrics(y_pred,y_test, chunks=10, binary_classification_threshold=0.1):
    meta_data = {}

    y_test = np.squeeze(y_test)
    y_pred = np.squeeze(y_pred)

    num_samples = y_test.shape[0]
    total_pixels = num_samples*y_test.shape[1]*y_test.shape[2]
    chunk_size = num_samples // chunks if num_samples % chunks == 0 else num_samples // chunks + 1
    se = 0 # squared error
    ae = 0 # average error
    ve = 0 # volume error
    tp_fp_fn = np.array([0,0,0])
    acc = 0

    for i in range(chunks):
        y_test_chunk = y_test[i*chunk_size:(i+1)*chunk_size]
        y_pred_chunk = y_pred[i*chunk_size:(i+1)*chunk_size]

        error = y_test_chunk - y_pred_chunk
        squared_error = error**2
        sum_squared_error = np.sum(squared_error)

        se += sum_squared_error
        ae += np.sum(np.abs(error))
        ve += np.sum(np.abs(np.sum(y_test_chunk, axis=(1,2))-np.sum(y_pred_chunk, axis=(1,2))))

        # If 'binary_classification_threshold' (between 0-1) of the pixel is covered by buildings, we say that buildings are present
        y_test_classification = (y_test_chunk > binary_classification_threshold)
        y_pred_classification = (y_pred_chunk > binary_classification_threshold)
        tp_fp_fn = tp_fp_fn + precision_recall(y_pred_classification,y_test_classification)

        acc = acc + np.sum(np.equal(y_test_classification,y_pred_classification))

    tp,fp,fn = tuple(tp_fp_fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)

    meta_data['test_set'] = y_test.shape
    meta_data['test_rmse'] = float(np.sqrt(se/total_pixels))
    meta_data['test_mae'] = float(ae/total_pixels)
    meta_data['test_mse'] = float(se/total_pixels)
    meta_data['test_volume_error'] = float(ve/total_pixels)
    meta_data['test_accuracy'] = float(acc/total_pixels)
    meta_data['test_precision'] = float(precision)
    meta_data['test_recall'] = float(recall)
    meta_data['threshold_binary_classification'] = binary_classification_threshold

    # Baseline for model that always predicts zero
    meta_data['baseline_0_mse'] = float(np.mean((y_test) ** 2))
    meta_data['baseline_0_acc'] = float(np.mean((y_test <= binary_classification_threshold)))

    return meta_data


def evaluate_model_landcover(model, dataloader_test, device, save_path_visualisations, num_visualisations = 20):
    num_classes= len(config_lc.lc_model_map)
    confmat = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to('cpu')
    running_conf = np.zeros((num_classes,num_classes))

    torch.set_default_device(device)
    model.to(device)
    model.eval()

    y_pred = []
    x_true = []
    y_true = []
    if num_visualisations > len(dataloader_test):
        num_visualisations = len(dataloader_test)
    with torch.no_grad():
        for inputs,targets in tqdm(dataloader_test):
            batch_pred = model(inputs.to(device)).detach().cpu()
            if len(x_true)<num_visualisations:
                x_true.append(inputs[:1, 0:3, :, :].detach().cpu().numpy()) # only a few per batch to avoid memory issues
                y_pred.append(batch_pred[:1].numpy().argmax(axis=1))
                y_true.append(targets[:1].detach().cpu().numpy())
            running_conf += confmat(batch_pred.argmax(axis=1),torch.squeeze(targets.detach().cpu())).numpy()

    y_pred = np.concatenate(y_pred,axis=0)
    x_true = np.concatenate(x_true,axis=0)
    y_true = np.concatenate(y_true,axis=0)

    total = np.sum(running_conf)
    tp = running_conf.trace()

    s = np.sum(running_conf, axis=1, keepdims=True)
    s[s==0]=1
    running_conf = running_conf/s
    
    plt.figure(figsize = (12,9))
    ax = sn.heatmap(running_conf, annot=True, fmt='.2f')
    ax.xaxis.set_ticklabels(config_lc.lc_raw_classes.values(),rotation = 90)
    ax.yaxis.set_ticklabels(config_lc.lc_raw_classes.values(),rotation = 0)
    plt.savefig(save_path_visualisations.replace('vis','cm'))
    
    print(x_true[0].shape, y_true[0].shape, y_pred[0].shape)

    batch_size = dataloader_test.batch_size
    y_vis = [i*batch_size for i in range(0,num_visualisations)]
    visualise(x_true, np.squeeze(y_true), np.squeeze(y_pred), images=num_visualisations, channel_first=True, vmin=0, vmax=0.5, save_path=save_path_visualisations, for_landcover=True)

    metrics = {'acc':tp/total, 'total_pixels':total}
    return metrics


def evaluate_model(model, dataloader_test, device, save_path_visualisations, num_visualisations = 20):
    torch.set_default_device(device)
    model.to(device)
    model.eval()

    y_pred = []
    x_true = []
    y_true = []
    if num_visualisations > len(dataloader_test):
        num_visualisations = len(dataloader_test)
    with torch.no_grad():
        for inputs,targets in tqdm(dataloader_test):
            batch_pred = model(inputs.to(device))
            y_pred.append(batch_pred.detach().cpu().numpy())
            y_true.append(targets.detach().cpu().numpy())
            if len(x_true)<num_visualisations:
                x_true.append(inputs[:1, 0:3, :, :].detach().cpu().numpy()) # only a few per batch to avoid memory issues

    y_pred = np.concatenate(y_pred,axis=0)
    x_true = np.concatenate(x_true,axis=0)
    y_true = np.concatenate(y_true,axis=0)

    metrics = calculate_metrics(y_pred=y_pred,y_test = y_true, binary_classification_threshold=0.1)

    batch_size = dataloader_test.batch_size
    y_vis = [i*batch_size for i in range(0,num_visualisations)]
    visualise(x_true, np.squeeze(y_true[y_vis]), np.squeeze(y_pred[y_vis]), images=num_visualisations, channel_first=True, vmin=0, vmax=0.5, save_path=save_path_visualisations)

    return metrics
        


if __name__ == "__main__":

    DATA_FOLDER = '/home/andreas/vscode/GeoSpatial/Phileo-downstream-tasks/data_landcover'
    REGIONS = ['eq-guinea','east-africa','north-america', 'europe', 'japan','south-america', 'nigeria', 'senegal'] #['north-america']#, ['east-africa', 'europe','eq-guinea', 'japan','south-america', 'nigeria', 'senegal'] # 'north-america',

    model_dir = 'trained_models/16082023_MLPMixer_split0.1' #'trained_models/08082023_CoreUnet'
    model_name = 'MLPMixer'#'CoreUnet'
    model = MLPMixer(
        chw=(10, 64, 64),
        output_dim=11,
        patch_size=8,
        embed_dim=1024,
        dim=512,
        depth=6,
    )
    # model = DiamondNet(
    #     input_dim=10,
    #     output_dim=11,
    #     input_size=64,
    #     depths=[3, 3, 3, 3],
    #     dims=[40, 80, 160, 320],

    # )

    # load model
    best_sd = torch.load(os.path.join(model_dir, f"{model_name}_best.pt"))
    model.load_state_dict(best_sd)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # make folder to store results
    results_dir = f'{model_dir}/results'
    os.makedirs(results_dir, exist_ok=True)
    results = {}

    for region in REGIONS:
        print('Calculating metrics for ',region)
        x_train, y_train, x_val, y_val, x_test, y_test = data_protocol_bd.protocol_regions(folder=DATA_FOLDER, regions=[region], y='lc')
        
        _, _, dl_test = load_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                            with_augmentations=False,
                                            num_workers=8,
                                            batch_size=64,
                                            encoder_only=False,
                                            )
        
        save_path_visualisations = f"{model_dir}/results/vis_{region}.png"
        metrics = evaluate_model_landcover(model, dl_test, device,save_path_visualisations, num_visualisations=16)
        results[region] = metrics


    with open(f'{results_dir}/{date.today().strftime("%d%m%Y")}_metrics.json', 'w') as fp:
        json.dump(results, fp)