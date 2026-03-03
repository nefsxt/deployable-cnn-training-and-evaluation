# Traning and Evaluation Loops

import torch
import time


# Return top-k accuracy for classification outputs
def top_k_accuracy(outputs, targets, k=5):

    # With disabled gradients
    with torch.no_grad():
        _, top_k_preds = outputs.topk(k, dim=1)
        correct = top_k_preds.eq(targets.view(-1, 1).expand_as(top_k_preds))
        return correct.any(dim=1).float().mean().item()



def train_one_epoch(model, loader, optimizer, criterion, device):
        
        start_time = time.time()

        # Set model to train mode
        model.train()

        running_loss = 0
        correct_classifications = 0
        total_inputs = 0
        top5_correct = 0

        #Iterate over train data
        for inputs, targets in loader:
            
            # Move to GPU, if available
            inputs, targets = inputs.to(device), targets.to(device)
            
            #Clear gradients
            optimizer.zero_grad()
            
            # Forward pass - compute the predicted outputs 
            pred_outputs = model(inputs)
            
            #Compute loss
            loss = criterion(pred_outputs, targets)
            
            #Backpropagate - compute gradient of the loss w.r.t model params
            loss.backward()
            
            #Update parameters
            optimizer.step()

            #Update runnning metrics
            running_loss += loss.item() * inputs.size(0)
            _, preds = pred_outputs.max(1)
            correct_classifications += preds.eq(targets).sum().item()
            total_inputs += targets.size(0)
            
            _, top5_preds = pred_outputs.topk(5, dim=1)
            top5_correct += top5_preds.eq(targets.view(-1, 1)).any(dim=1).sum().item()

        epoch_time = time.time() - start_time

        return (running_loss / total_inputs, correct_classifications / total_inputs, top5_correct / total_inputs, epoch_time)
    
    
def evaluate(model, loader, criterion, device):
    
    start_time = time.time()


    # Set model to train mode
    model.eval()
    
    loss_sum = 0
    correct_classifications = 0
    total_inputs = 0

    
    all_preds = []
    all_targets = []
    top5_correct = 0



    # With disabled gradients
    with torch.no_grad():
        
        #Iterate over eval data
        for inputs, targets in loader:
            
            # Move to GPU, if available
            inputs, targets = inputs.to(device), targets.to(device)
            
            pred_outputs = model(inputs)
            loss = criterion(pred_outputs, targets)

            loss_sum += loss.item() * inputs.size(0)
            _, preds = pred_outputs.max(1)
            correct_classifications += preds.eq(targets).sum().item()
            total_inputs += targets.size(0)

            # Store predictions and targets for confusion matrix
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

            # Top-5 accuracy
            _, top5_preds = pred_outputs.topk(5, dim=1)
            top5_correct += top5_preds.eq(targets.view(-1, 1)).any(dim=1).sum().item()

    epoch_time = time.time() - start_time

    return (loss_sum / total_inputs, correct_classifications / total_inputs, top5_correct / total_inputs, torch.cat(all_preds), torch.cat(all_targets), epoch_time)
 