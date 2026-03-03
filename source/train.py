# Entry Point (Optuna + MLflow)

import optuna
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

from model import CNN
from data import get_dataloaders, get_dataset_metadata, get_test_dataloader
from engine import train_one_epoch, evaluate
from config import DEVICE, TUNING_EPOCHS, FINAL_EPOCHS, CLASS_NAMES,NUM_STARTUP_TRIALS,NUM_WARMUP_STEPS,INTERVAL_STEPS, NUM_OPTUNA_TRIALS


mlflow.set_experiment("cifar10_cnn_optuna")

def objective(trial):
    
    # Optional: measure trial duration
    trial_start_time = time.time()
    
    # --- Hyperparameter Search Space Definition using Optuna ---
    
    # Stochastic Gradient Descent (SGD) Hyperparameters to be tuned
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    momentum = trial.suggest_float("momentum", 0.8, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    
    # --- Trial tracking via MLflow --- 
    
    # Activate nested mlflow runs (each trial will be a "child" run)
    with mlflow.start_run(nested=True):
        
        # Log the specific hyperparameters chosen for this trial
        mlflow.log_params({"lr":lr, "momentum":momentum, "weight_decay":weight_decay,"batch_size":batch_size})

        # Log dataset metadata
        mlflow.log_params(get_dataset_metadata())

        # Instantiate model and move it to the appropriate device
        model = CNN().to(DEVICE)        

        # Initialize SGD Optimizer with the chosen hyperparameter values
        optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
        
        # Initialize Loss Criterion (Cross-Entropy Loss)
        criterion = nn.CrossEntropyLoss()
        
        # Load the data using the selected batch size()
        train_loader, val_loader = get_dataloaders(batch_size)
        
        # Best/highest validation accuracy achieved during training
        best_val_acc = 0
        
        # Training Loop
        for epoch in range(TUNING_EPOCHS):
            
            # Calculate train loss over one epoch (for one batch)
            train_loss, train_acc, train_top5, train_time = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE
            )
            
            # Calculate validation loss oover eval data
            val_loss, val_acc, val_top5, _, _, val_time = evaluate(
                model, val_loader, criterion, DEVICE
            )
        
            # Log metrics per epoch for both Train and Eval
            mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_top5_accuracy": train_top5,           
                "train_epoch_time": train_time,             
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_top5_accuracy": val_top5,              
                "val_epoch_time": val_time                  
             },
            step=epoch
            )

            # Report intermediate objective value to Optuna for pruning
            trial.report(val_acc, step=epoch)

            # Check whether the trial should be pruned
            if trial.should_prune():
                mlflow.set_tag("pruned", True)
                raise optuna.TrialPruned()
            
            # Keep validation accuracy achieved (max)
            best_val_acc = max(best_val_acc, val_acc)
        
        # Log trial duration
        trial_duration = time.time() - trial_start_time
        mlflow.log_metric("trial_duration_sec", trial_duration)
        
    # Return - Optuna uses this value to evaluate how good the hyperparameter combo used actually was
    return best_val_acc



if __name__ == "__main__":
    
    # Median Pruner for early stopping of underperforming trials
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=NUM_STARTUP_TRIALS, 
        n_warmup_steps=NUM_WARMUP_STEPS,   
        interval_steps=INTERVAL_STEPS    
    )

    # Initialize Optuna study - maximization of objective function which returns validation accuracy (metric)
    study = optuna.create_study(
        direction="maximize",
        study_name="cifar10_cnn_study",
        pruner=pruner
    )
    
    # Start optimization process - Random Search
    study.optimize(objective, n_trials=NUM_OPTUNA_TRIALS)

    best_params = study.best_trial.params
    print("Best hyperparameters:")
    print(best_params)

    # --- Final training with best hyperparameters ---
    with mlflow.start_run(run_name="final_model"):

        mlflow.log_params(best_params)
        mlflow.log_params(get_dataset_metadata())
        mlflow.log_dict(best_params, "best_params.json") 


        model = CNN().to(DEVICE)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=best_params["lr"],
            momentum=best_params["momentum"],
            weight_decay=best_params["weight_decay"]
        )
        criterion = nn.CrossEntropyLoss()

        train_loader, val_loader = get_dataloaders(best_params["batch_size"])
        test_loader = get_test_dataloader(best_params["batch_size"])

        for epoch in range(FINAL_EPOCHS):
            train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)

        # Final evaluation on test set
        test_loss, test_acc, test_top5, test_preds, test_targets, _ = evaluate(
            model, test_loader, criterion, DEVICE
        )

        mlflow.log_metrics({
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_top5_accuracy": test_top5
        })

        # Log confusion matrix
        cm = confusion_matrix(test_targets.numpy(), test_preds.numpy())
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Test Confusion Matrix")
        plt.tight_layout()
        plt.savefig("test_confusion_matrix.png")
        mlflow.log_artifact("test_confusion_matrix.png")
        plt.close()

        # Save final model
        mlflow.pytorch.log_model(model, artifact_path="final_model")
