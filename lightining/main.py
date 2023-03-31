
import torch
import time
import torch.nn.functional as F
from torch.utils.data.dataset import random_split
from torchvision import datasets
from torchvision import transforms
import pytorch_lightning as pl
import torchmetrics
import pandas as pd
import matplotlib.pyplot as plt
    

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger




HIDDEN_UNITS = (128, 256)
BATCH_SIZE = 256
NUM_EPOCHS = 10
LEARNING_RATE = 0.005
NUM_WORKERS = 4
NUM_CLASSES = 10

# create a timer decorator
def timer(func):
    

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result

    return wrapper




class PyTrochModel(torch.nn.Module):
    def __init__(self, input_size, hidden_units, num_classes):
        super(PyTrochModel, self).__init__()
        
        all_layers = []
        for hidden_unit in hidden_units:
            all_layers.append(torch.nn.Linear(input_size, hidden_unit))
            all_layers.append(torch.nn.ReLU())
            input_size = hidden_unit
        output_layer = torch.nn.Linear(hidden_units[-1], num_classes)

        all_layers.append(output_layer)

        self.layers = torch.nn.Sequential(*all_layers)
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.layers(x)
        
        return x
        



class LightiningModule(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        
        self.learning_rate = lr
        self.model = model
        
        if hasattr(model, "dropout_proba"):
            self.dropout_proba = model.dropout_proba
        
        self.save_hyperparameters(ignore=["model"])

        self.train_acc = torchmetrics.Accuracy(task ="multiclass", num_classes=NUM_CLASSES, average="weighted")
        self.valid_acc = torchmetrics.Accuracy(task ="multiclass", num_classes=NUM_CLASSES, average="weighted")
        self.test_acc = torchmetrics.Accuracy(task ="multiclass", num_classes=NUM_CLASSES, average="weighted")
        
    def forward(self, x):
        return self.model(x)
    
    
    def _shared_step(self, batch):
        features, true_labels = batch
        logits = self.model(features)
        
        loss = torch.nn.functional.cross_entropy(logits, true_labels)
        predicted_labels = torch.argmax(logits, dim=1)
        
        return loss, true_labels, predicted_labels
    
    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        
        self.log("train_loss", loss)
        
        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._shared_step(batch)
            
        self.train_acc(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.model.train()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        
        self.log("valid_loss", loss)
        
        self.valid_acc(predicted_labels, true_labels)
        self.log("valid_acc", self.valid_acc, on_epoch=True, on_step=False, prog_bar=True)
    
    def testing_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_step(batch)
        
        self.log("test_loss", loss)
        
        self.test_acc(predicted_labels, true_labels)
        self.log("test_acc", self.test_acc, on_epoch=True, on_step=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class DataModule(pl.LightningModule):
    def __init__(self, data_path = "/.data"):
        
        super().__init__()
        self.data_path = data_path
        
    def prepare_data(self):
        datasets.MNIST(root=self.data_path, download=True)
        
    def setup(self, stage=None):
        
        train = datasets.MNIST(root=self.data_path, train=True, download=False, transform=transforms.ToTensor())
        
        self.test = datasets.MNIST(root=self.data_path, train=False, download=False, transform=transforms.ToTensor())
        
        self.train, self.valid = random_split(train, [55000, 5000])
        
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=False, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, drop_last=False, shuffle=False)
    


def main():
    torch.manual_seed(1)
    
    data_module = DataModule(data_path="./data")

    pytorch_model = PyTrochModel(input_size=28*28, hidden_units=HIDDEN_UNITS, num_classes=NUM_CLASSES)
    
    lightining_module = LightiningModule(model=pytorch_model, lr=LEARNING_RATE)
    
    callbacks = [ModelCheckpoint(save_top_k=1, mode = "max", monitor="valid_acc")]
    
    logger = CSVLogger("logs/", name="mnist")
    
    
    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,
        callbacks=callbacks,
        accelerator="auto",
        devices = "auto",
        logger= logger,
        deterministic=True,
        log_every_n_steps=10,
        )
    
    start = time.time()
    
    trainer.fit(model = lightining_module, datamodule=data_module)
    
    runtime = time.time() - start
    
    print(f"Runtime: {runtime:.2f} seconds")
        
    metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")
    
    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)

    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "valid_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )
    df_metrics[["train_acc", "valid_acc"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )

    plt.show()
    
# python main block
if __name__ == '__main__':
    main()