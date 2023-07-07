import torch
import torch.nn as nn
import torchmetrics

import buteo as beo
import numpy as np
from datetime import date

import sys; sys.path.append("../")
from models.model_SimpleUNet import SimpleUnet
from models.model_ViT import vit_mse_losses, ViT

from utils import (
    load_data,
    training_loop,
    TiledMSE,
    data_protocol_bd
)


if __name__ == "__main__":

    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 250
    BATCH_SIZE = 64
    REGIONS =['']#, 'east-africa', 'europe','eq-guinea', 'japan','south-america', 'nigeria', 'senegal']
    DATA_FOLDER = '/home/lcamilleri/data/s12_buildings/data_patches/'
    model = SimpleUnet() #ViT(chw=(10, 64, 64),  n_patches=4, n_blocks=2, hidden_d=768, n_heads=12)# SimpleUnet(input_dim=10, output_dim=1)
    criterion = nn.MSELoss() # vit_mse_losses(n_patches=4)
    lr_scheduler = 'reduce_on_plateau' # None, 'reduce_on_plateau', 'cosine_annealing'
    
    NAME = model.__class__.__name__
    OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}'
    if lr_scheduler is not None:
        OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_{lr_scheduler}'
        if lr_scheduler == 'reduce_on_plateau':
            LEARNING_RATE = LEARNING_RATE / 100000 # for warmup start

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    x_train, y_train, x_val, y_val, x_test, y_test = data_protocol_bd.protocol_split(folder=DATA_FOLDER,
                                                                                     split_percentage=1)
    dl_train, dl_val, dl_test = load_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                          with_augmentations=False,
                                          num_workers=0,
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
    )