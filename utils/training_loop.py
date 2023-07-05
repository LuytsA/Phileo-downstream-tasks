# Standard Library
import os
from tqdm import tqdm

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from .training_utils import patience_calculator, visualise

# utils
from utils import visualise

def training_loop(
    num_epochs: int,
    learning_rate: float,
    model: nn.Module,
    device: torch.device,
    criterion: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    metrics: list = None,
    name="model",
    out_folder="trained_models/",
    predict_func=None,
    visualise_validation=True
) -> None:
        
    torch.set_default_device(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
        print("No CUDA device available.")

    print("Starting training...")
    print("")

    model.to(device)
    os.makedirs(out_folder, exist_ok=True)

    # Loss and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()

    # Save the initial learning rate in optimizer's param_groups
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = learning_rate


    best_epoch = 0
    best_loss = None
    best_model_state = model.state_dict().copy()
    epochs_no_improve = 0


    # Training loop
    for epoch in range(num_epochs):
        model.train()

        # Initialize the running loss
        train_loss = 0.0
        train_metrics_values = { metric.__name__: 0.0 for metric in metrics }

        # Initialize the progress bar for training
        train_pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, (images, labels) in enumerate(train_pbar):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(device), labels.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Cast to bfloat16
            with autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            train_loss += loss.item()

            for metric in metrics:
                train_metrics_values[metric.__name__] += metric(outputs, labels)

            train_pbar.set_postfix({
                "loss": f"{train_loss / (i + 1):.4f}",
                **{name: f"{value / (i + 1):.4f}" for name, value in train_metrics_values.items()}
            })

            # Validate at the end of each epoch
            # This is done in the same scope to keep tqdm happy.
            if i == len(train_loader) - 1:

                val_metrics_values = { metric.__name__: 0.0 for metric in metrics }
                # Validate every epoch
                with torch.no_grad():
                    model.eval()

                    val_loss = 0

                    # visualise some validation results
                    num_visualisations = 16
                    vis_batches = [i*len(val_loader)//(num_visualisations+1) for i in range(num_visualisations)]
                    print(vis_batches, len(val_loader))
                    vis_images = []
                    vis_labels = []
                    vis_preds = []
                    save_dir = f'{out_folder}/visualisations/'
                    os.makedirs(save_dir, exist_ok=True)

                    for j, (images, labels) in enumerate(val_loader):
                        images = images.to(device)
                        labels = labels.to(device)

                        outputs = model(images)

                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        if j in vis_batches:
                            vis_images.append(images.detach().cpu().numpy()[0])
                            vis_labels.append(labels.detach().cpu().numpy()[0])
                            vis_preds.append(outputs.detach().cpu().numpy()[0])
                        
                        for metric in metrics:
                            val_metrics_values[metric.__name__] += metric(outputs, labels)
                        
                    if visualise_validation:
                        print(len(vis_images),len(vis_labels),len(vis_preds))
                        visualise(vis_images,np.squeeze(vis_labels),np.squeeze(vis_preds), images=num_visualisations, channel_first=True, vmin=0,vmax=1, save_path=os.path.join(save_dir, f"val_pred_{epoch}.png"))

                # Append val_loss to the train_pbar
                train_pbar.set_postfix({
                    "loss": f"{train_loss / (i + 1):.4f}",
                    **{name: f"{value / (i + 1):.4f}" for name, value in train_metrics_values.items()},
                    "val_loss": f"{val_loss / (j + 1):.4f}",
                    **{f"val_{name}": f"{value / (j + 1):.4f}" for name, value in val_metrics_values.items()},
                }, refresh=True)

                if best_loss is None:
                    best_epoch = epoch
                    best_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    torch.save(best_model_state, os.path.join(out_folder, f"{name}_best.pt"))
                    
                    if predict_func is not None:
                        predict_func(model, epoch + 1)

                elif best_loss > val_loss:
                    best_epoch = epoch
                    best_loss = val_loss
                    best_model_state = model.state_dict().copy()
                    torch.save(best_model_state, os.path.join(out_folder, f"{name}_best.pt"))

                    if predict_func is not None:
                        predict_func(model, epoch + 1)

                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                torch.save(best_model_state, os.path.join(out_folder, f"{name}_last.pt"))


        # # Early stopping
        # if epochs_no_improve == patience_calculator(epoch, t_0, t_mult, max_patience):
        #     print(f'Early stopping triggered after {epoch + 1} epochs.')
        #     break

    # Load the best weights
    model.load_state_dict(best_model_state)

    print("Finished Training. Best epoch: ", best_epoch + 1)
    print("")
    print("Starting Testing...")
    model.eval()

    # Test the model
    with torch.no_grad():
        test_loss = 0
        for k, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

        print(f"Test Accuracy: {test_loss / (k + 1):.4f}")

    # Save the model
    torch.save(best_model_state, os.path.join(out_folder, f"{name}.pt"))