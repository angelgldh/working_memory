import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import numpy as np

def average_loss_multiclass(outputs, labels, criterion) : 
    """
    Determines the average loss function in task of determining the class of every image in the sequence. 
    """
    loss = 0.0
    valid_sequences = 0 

    for b in range(outputs.size(0)):  # Iterate over the batch dimension
        # Skip sequences with all noise images (all labels are -1)
        if (labels[b, :] == -1).all():
            continue
        
        # Calculate the loss for the current sequence
        loss_t = criterion(outputs[b, :, :], labels[b, :])
                
        # Check for NaN or Inf in the loss
        if not torch.isnan(loss_t) and not torch.isinf(loss_t):
            loss += loss_t
            valid_sequences += 1  # Count valid sequences that contributed to the loss

    # Average the loss by the number of valid sequences
    if valid_sequences > 0:
        loss /= valid_sequences
    else:
        raise ValueError("All sequences were noise, resulting in an invalid loss computation.")

    return loss



def accuracy_of_class_labels(model,valid_loader, device, noise_image_class = -1):
    # Move the model to evaluation mode
    model.eval()

    # Metrics initialization
    individual_accuracies = []
    sequence_accuracies = []

    # Disable gradient calculations
    with torch.no_grad():
        for sequence, model_classes, repeats in valid_loader:
            images = sequence.to(device)
            labels = model_classes.to(device)
            
            # Perform forward pass and get predictions
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=2)  
            
            # As we aim for accuracy at sequence-level, let us iterate through every sequence in the batch
            for seq_idx in range(outputs.shape[0]):
                sequence_predictions = predictions[seq_idx]
                sequence_labels = labels[seq_idx]

                # Ignore those positions where labels == noise_image_class
                mask = sequence_labels != noise_image_class

                filtered_predictions = sequence_predictions[mask]
                filtered_labels = sequence_labels[mask]

                # Calculate individual image accuracy
                correct_images = (filtered_predictions == filtered_labels).cpu().numpy()
                individual_accuracy = np.mean(correct_images)
                individual_accuracies.append(individual_accuracy)

                # Calculate whole sequence accuracy
                correct_sequence = (correct_images == True).all()
                sequence_accuracies.append(correct_sequence)
                

    # Calculate overall metrics
    individual_accuracy = np.mean(individual_accuracies)
    sequence_accuracy = np.mean(sequence_accuracies)

    return individual_accuracy, sequence_accuracy
        
