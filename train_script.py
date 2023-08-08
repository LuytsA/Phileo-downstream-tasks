import os

import torch
import torchmetrics
from functools import partial

import torch.nn as nn
from datetime import date

import sys; sys.path.append("../")
from models.model_ViT import vit_mse_losses
from models.models_vit_timm import Vit_basic


from utils import (
    load_data,
    data_protocol_bd,
    training_loop
)

from test_script import evaluate_model
from utils.load_data import callback_decoder
from torch.utils.data import DataLoader

import buteo as beo
import pandas as pd


if __name__ == "__main__":


    # data_percentages = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1]
    data_percentages = [1]
    for data_percentage in data_percentages:
        LEARNING_RATE = 0.0001
        NUM_EPOCHS = 500
        BATCH_SIZE = 64
        num_workers = 1
        n_patches = 16 # (64/4)
        REGIONS =['']#, 'east-africa', 'europe','eq-guinea', 'japan','south-america', 'nigeria', 'senegal']
        DATA_FOLDER = '/home/lcamilleri/data/s12_buildings/data_patches/'
        model = Vit_basic(chw=(10, 64, 64), out_chans=1, patch_size=4,
                 embed_dim=768, depth=12, num_heads=16)
        criterion = vit_mse_losses(n_patches=n_patches) # nn.MSELoss() # vit_mse_losses(n_patches=4)
        lr_scheduler = 'reduce_on_plateau' # None, 'reduce_on_plateau', 'cosine_annealing'

        NAME = model.__class__.__name__
        OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_{data_percentage}'
        if lr_scheduler is not None:
            OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_{data_percentage}_{lr_scheduler}'
            if lr_scheduler == 'reduce_on_plateau':
                LEARNING_RATE = LEARNING_RATE / 100000 # for warmup start

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        x_train, y_train, x_val, y_val, x_test, y_test = data_protocol_bd.protocol_split(folder=DATA_FOLDER,
                                                                                         split_percentage=data_percentage)
        dl_train, dl_test, dl_val = load_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                              with_augmentations=False,
                                              num_workers=num_workers,
                                              batch_size=BATCH_SIZE,
                                              encoder_only=False,
                                              )


        wmape = torchmetrics.WeightedMeanAbsolutePercentageError(); wmape.__name__ = "wmape"
        mae = torchmetrics.MeanAbsoluteError(); mae.__name__ = "mae"
        mse = torchmetrics.MeanSquaredError(); mse.__name__ = "mse"

        training_loop(
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            model=model,
            criterion=criterion,
            device=device,
            # metrics=[
            #     mse.to(device),
            #     wmape.to(device),
            #     mae.to(device),
            # ],
            metrics=[],
            lr_scheduler=lr_scheduler,
            train_loader=dl_train,
            val_loader=dl_val,
            test_loader=dl_test,
            name=NAME,
            out_folder=OUTPUT_FOLDER,
            predict_func=None,
            n_patches=n_patches,
        )
        # test script
        sd = torch.load(os.path.join(OUTPUT_FOLDER, f"{NAME}_best.pt"))
        model.load_state_dict(sd)
        model.eval()

        REGIONS_BUILDINGS = ['DNK', 'EGY', 'GHA', 'ISR', 'TZA', 'UGA']
        results = {}
        for region in REGIONS_BUILDINGS:
            print('Calculating metrics for ', region)
            _, _, _, _, x_test, y_test = data_protocol_bd.protocol_regions(folder=DATA_FOLDER, regions=[region], y='y')
            ds_test = beo.Dataset(x_test, y_test, callback=callback_decoder)
            dl_test = DataLoader(ds_test, batch_size=6, shuffle=False, pin_memory=True, num_workers=0,
                                 drop_last=True, generator=torch.Generator(device='cuda'))

            save_path_visualisations = f"{OUTPUT_FOLDER}/test_{region}.png"
            metrics = evaluate_model(model, dl_test, device, save_path_visualisations, num_visualisations=5,n_patches=n_patches)
            results[region] = metrics

        print(results)
        df = pd.DataFrame.from_dict(results, orient="index")
        df.to_csv(f'{OUTPUT_FOLDER}/results.csv')