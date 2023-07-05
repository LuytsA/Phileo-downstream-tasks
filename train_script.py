import torch
import torch.nn as nn
import torchmetrics

import buteo as beo
import numpy as np
from datetime import date

import sys; sys.path.append("../")
from models.model_SimpleUNet import SimpleUnet
from utils import (
    load_data,
    training_loop,
    TiledMSE,
)



if __name__ == "__main__":

    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 250
    BATCH_SIZE = 64
    REGIONS =['north-america']#, 'east-africa', 'europe','eq-guinea', 'japan','south-america', 'nigeria', 'senegal']
    DATA_FOLDER = '/home/andreas/vscode/GeoSpatial/phi-lab-rd/data/road_segmentation'
    model = SimpleUnet(input_dim=10, output_dim=1)
    criterion = nn.MSELoss()
    
    NAME = model.__class__.__name__
    OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dl_train, dl_val, dl_test = load_data(
        x="s2",
        y="area",
        with_augmentations=False,
        num_workers=6,
        batch_size=BATCH_SIZE,
        encoder_only=False,
        folder=DATA_FOLDER,
        regions=REGIONS
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
        metrics=[
            mse.to(device),
            wmape.to(device),
            mae.to(device),
        ],
        train_loader=dl_train,
        val_loader=dl_val,
        test_loader=dl_test,
        name=NAME,
        out_folder=OUTPUT_FOLDER,
        predict_func=None,
    )