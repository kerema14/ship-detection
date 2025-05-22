import torch
def calculate_accuracy(labels:torch.Tensor, predictions:torch.Tensor) -> float:
    """
    Calculate the accuracy of predictions.
    
    Args:
        labels (torch.Tensor): True labels.
        predictions (torch.Tensor): Predicted labels.
        
    Returns:
        float: Accuracy score.
    """
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy

