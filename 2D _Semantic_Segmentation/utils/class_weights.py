import torch

def get_class_weights(device):
    """
    Returns class weights to handle class imbalance in the Cityscapes dataset.
    
    Weights are calculated based on inverse class frequency (Reference: ENet implementation).
    Rare classes (e.g., poles, traffic signs) are assigned higher weights to penalize
    misclassification more severely during training.
    """
    weights = [
        2.8149201869965,   # 0: road
        6.9850029945374,   # 1: sidewalk
        3.7890393733978,   # 2: building
        9.9428062438965,   # 3: wall
        9.7702074050903,   # 4: fence
        9.5110931396484,   # 5: pole
        10.311357498169,   # 6: traffic light
        10.026463508606,   # 7: traffic sign
        4.6323022842407,   # 8: vegetation
        9.5608062744141,   # 9: terrain
        7.8698215484619,   # 10: sky
        9.5168733596802,   # 11: person
        10.373730659485,   # 12: rider
        6.6616044044495,   # 13: car
        10.260489463806,   # 14: truck
        10.287888526917,   # 15: bus
        10.289801597595,   # 16: train
        10.475443840027,   # 17: motorcycle
        10.155650138855    # 18: bicycle
    ]
    
    # Convert list to Tensor and move to the active device
    return torch.FloatTensor(weights).to(device)