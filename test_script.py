import torch
import torch.nn as nn
import torchmetrics

import buteo as beo
import numpy as np
from datetime import date

import sys; sys.path.append("../")
from models.model_SimpleUNet import SimpleUnet
from models.model_ViT import vit_mse_losses, unpatchify

import os
from glob import glob
from tqdm import tqdm
import json

from utils import (
    load_data,
    data_protocol_bd,
    visualise
)

from utils.load_data import callback_decoder
from torch.utils.data import DataLoader
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



def evaluate_model(model, dataloader_test, device, save_path_visualisations, num_visualisations = 20, n_patches=None):
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

            if targets.shape != batch_pred.shape:
                batch_pred = unpatchify(targets.shape[0], targets.shape[1], targets.shape[2], targets.shape[3],
                                     n_patches=n_patches, tensors=batch_pred)

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

def test():
    DATA_FOLDER = '/home/lcamilleri/data/s12_buildings/data_patches/'

    model_dir = '/home/lcamilleri/git_repos/Phileo-downstream-tasks/trained_models/06072023_SimpleUnet_reduce_on_plateau'
    model_name = 'SimpleUnet'
    model = SimpleUnet(input_dim=10, output_dim=1)

    # load model
    best_sd = torch.load(os.path.join(model_dir, f"{model_name}_best.pt"))
    model.load_state_dict(best_sd)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    REGIONS_BUILDINGS = ['DNK', 'EGY', 'GHA', 'ISR', 'TZA', 'UGA']
    results = {}
    for region in REGIONS_BUILDINGS:
        print('Calculating metrics for ', region)
        _, _, _, _, x_test, y_test = data_protocol_bd.protocol_regions(folder=DATA_FOLDER, regions=[region], y='y')
        ds_test = beo.Dataset(x_test, y_test, callback=callback_decoder)
        dl_test = DataLoader(ds_test, batch_size=6, shuffle=False, pin_memory=True, num_workers=0,
                             drop_last=True, generator=torch.Generator(device='cuda'))

        save_path_visualisations = f"test_{region}.png"
        metrics = evaluate_model(model, dl_test, device, save_path_visualisations, num_visualisations=5)
        results[region] = metrics

    print(results)


if __name__ == "__main__":
    test()

    # DATA_FOLDER = '/home/andreas/vscode/GeoSpatial/Phileo-downstream-tasks/data'
    # REGIONS = ['north-america','east-africa', 'europe','eq-guinea', 'japan','south-america', 'nigeria', 'senegal']
    #
    # model_dir = 'trained_models/07072023_SimpleUnet_NA_reduce_on_plateau'
    # model_name = 'SimpleUnet'
    # model = SimpleUnet(input_dim=10, output_dim=1)
    #
    # # load model
    # best_sd = torch.load(os.path.join(model_dir, f"{model_name}_best.pt"))
    # model.load_state_dict(best_sd)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #
    # # make folder to store results
    # results_dir = f'{model_dir}/results'
    # os.makedirs(results_dir, exist_ok=True)
    # results = {}
    #
    # for region in REGIONS:
    #     print('Calculating metrics for ',region)
    #     x_train, y_train, x_val, y_val, x_test, y_test = data_protocol_bd.protocol_regions(folder=DATA_FOLDER, regions=[region], y='roads')
    #
    #     _, _, dl_test = load_data(x_train, y_train, x_val, y_val, x_test, y_test,
    #                                         with_augmentations=False,
    #                                         num_workers=0,
    #                                         batch_size=64,
    #                                         encoder_only=False,
    #                                         )
    #
    #     save_path_visualisations = f"{model_dir}/results/vis_{region}.png"
    #     metrics = evaluate_model(model, dl_test, device,save_path_visualisations, num_visualisations=16)
    #     results[region] = metrics
    #
    #
    # with open(f'{results_dir}/{date.today().strftime("%d%m%Y")}_metrics.json', 'w') as fp:
    #     json.dump(results, fp)