import torch
import torch.nn as nn
import torchmetrics
from datetime import date
import time

import sys; sys.path.append("../")
from models.model_SimpleUNet import SimpleUnet
from models.model_ViT import vit_mse_losses, ViT
from models.model_ConvNext import ConvNextV2Unet_tiny, ConvNextV2Unet_atto, ConvNextV2Unet_pico, ConvNextV2Unet_base
from models.model_Diamond import DiamondNet
from models.model_CoreCNN import CoreUnet_tiny, CoreUnet
from models.model_MixerMLP import MLPMixer
from models.model_VisionTransformer import ViT
from models.model_MetaFormer import MetaFormer

from utils import (
    load_data,
    training_loop,
    TiledMSE,
    data_protocol_bd,
)

torch.set_float32_matmul_precision('high')

if __name__ == "__main__":

    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20
    BATCH_SIZE = 32
    num_workers = 6
    REGIONS = ['north-america','japan', 'east-africa', 'europe','eq-guinea','south-america', 'nigeria', 'senegal']
    DATA_FOLDER = '/phileo_data/downstream/downstream_dataset_patches_np' #'data_geography'#
    augmentations = False
    split_percentage =0.02

    # model = MetaFormer(
    #     chw=(10, 64, 64),
    #     output_dim=11,
    #     patch_size=4,
    #     embed_dim=512,
    # )
    # model = MLPMixer(
    #     chw=(10, 64, 64),
    #     output_dim=11,
    #     patch_size=8,
    #     embed_dim=1024,
    #     dim=512,
    #     depth=6,
    #     clamp_output = False
    # )

    # model = DiamondNet(
    #     input_dim=10,
    #     output_dim=11,
    #     input_size=64,
    #     depths=[3, 3, 3, 3],
    #     dims=[40, 80, 160, 320],)

    criterion = nn.MSELoss() #nn.CrossEntropyLoss() #SoftSpatialCrossEntropyLoss(classes=[i for i in range(11)]) 
    lr_scheduler = None #'reduce_on_plateau' # None, 'reduce_on_plateau', 'cosine_annealing'
    model = CoreUnet_tiny(input_dim=10, output_dim=1)


    NAME = model.__class__.__name__
    OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_allregions_split{split_percentage}'
    OUTPUT_FOLDER = OUTPUT_FOLDER + '_augm' if augmentations else OUTPUT_FOLDER
    if lr_scheduler is not None:
        OUTPUT_FOLDER = f'trained_models/{date.today().strftime("%d%m%Y")}_{NAME}_{lr_scheduler}'
        if lr_scheduler == 'reduce_on_plateau':
            LEARNING_RATE = LEARNING_RATE / 100000 # for warmup start

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #model = torch.compile(model)#, mode='reduce-overhead')
    x_train, y_train, x_val, y_val, x_test, y_test = data_protocol_bd.protocol_split(folder=DATA_FOLDER,
                                                                                    split_percentage=split_percentage,
                                                                                    regions=REGIONS,
                                                                                    y='roads')
    
    dl_train, dl_val, dl_test = load_data(x_train, y_train, x_val, y_val, x_test, y_test,
                                        with_augmentations=augmentations,
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
    )