class Callback:
    def on_epoch_end(self, epoch, batch_idx, logs=None):
        pass
    
    def on_batch_end(self, batch_idx, loader, logs=None):
        pass
    
    def on_train_end(self, logs=None):
        pass

class LossLogger(Callback):
    def on_epoch_end(self, epoch, batch_idx, logs=None):
        print(f"Epoch {epoch+1}, Mapping_Loss: {logs['Mapping_loss']/(batch_idx+1):.4f}, Confidence_Loss: {logs['Confidence_loss']/(batch_idx+1):.4f}")

    def on_batch_end(self, batch_idx, train_loader, logs=None):
        print(f"Step [{batch_idx + 1}/{len(train_loader)}], Mapping_Loss: {logs['Mapping_loss']/(batch_idx+1):.4f}, Confidence_Loss: {logs['Confidence_loss']/(batch_idx+1):.4f}")

class EarlyStopping(Callback):
    def __init__(self, patience=20, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = None

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs['Mapping_loss']
        if self.best_loss is None or current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("Early stopping")
                self.stopped_epoch = epoch
                self.model.stop_training = True
