import torch

class EarlyStopper():
        
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_monitor = torch.inf

    def early_stopping(self, monitor, model, ceckpoint_folder):
        
        if monitor < self.min_monitor - self.min_delta:
            self.min_monitor = monitor
            self.counter = 0
            model.save_model(ceckpoint_folder)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            
        return False
