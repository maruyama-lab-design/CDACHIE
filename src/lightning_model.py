import torch
import torch.nn as nn
import torch.nn.functional as F
from lion_pytorch import Lion
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from src.models.BERT import BERT1kb
from src.models.MLP import MLP


class ClusteringModel(pl.LightningModule):
    def __init__(self, signal_resolution, hic_embedding_dim, d_model=64, num_layers=6, num_heads=8, dropout=0.1, dim_feedforward=64*4, share_attn=False, share_ffn=False, cls_init='random',
                  func_lr=1e-4, struc_lr=1e-4, func_head_lr=1e-4, struc_head_lr=1e-4, weight_decay=0., optimizer='Adam'):
        super().__init__()
        self.save_hyperparameters()
        if signal_resolution=='1kb':
            self.functional_encoder = BERT1kb(d_model, num_layers, num_heads, dropout, dim_feedforward, input_size=100, output_size=8, share_attn=share_attn, share_ffn=share_ffn, cls_init=cls_init)
        #self.structural_encoder = MLP(input_size=hic_embedding_dim, output_size=d_model, hidden_size=128, depth=4)
        self.structural_encoder = MLP(d_in=128, d=128, d_hidden_factor=1, n_layers=4, hidden_dropout=dropout, residual_dropout=dropout, d_out=64)

        self.functional_head = nn.Linear(d_model, 16)
        self.structural_head = nn.Linear(d_model, 16)
        self.test_outputs = {'func_emb': [], 'struc_emb': []}

        self.func_lr = func_lr
        self.struc_lr = struc_lr
        self.func_head_lr = func_head_lr
        self.struc_head_lr = struc_head_lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer

    def forward(self, x, y):
        _, _, func_emb = self.functional_encoder(x)
        struc_emb = self.structural_encoder(y)
        func_emb = self.functional_head(func_emb)
        struc_emb = self.structural_head(struc_emb)
        return func_emb, struc_emb

    def configure_optimizers(self):
        parameters = [
            {'params': self.functional_encoder.parameters(), 'lr': self.func_lr},
            {'params': self.structural_encoder.parameters(), 'lr': self.struc_lr},
            {'params': self.functional_head.parameters(), 'lr': self.func_head_lr},
            {'params': self.structural_head.parameters(), 'lr': self.struc_head_lr}
            ]

        if self.optimizer=='Adam':
            opt = torch.optim.Adam(parameters, weight_decay=self.weight_decay)
        elif self.optimizer=='Lion':
            opt = Lion(parameters, weight_decay=self.weight_decay)
            print('Lion optimizer is used.')
        else:
            raise ValueError(f'optimizer must be one of Adam or Lion, but got {self.optimizer}')
        return opt

    def compute_loss(self, func_emb, struc_emb, temperture=0.5):
        
        # func_emb: chip-seq, struc_emb: hic
        func_emb = F.normalize(func_emb, dim=1)
        struc_emb = F.normalize(struc_emb, dim=1)

        sim = func_emb @ struc_emb.t() / temperture
        labels = torch.arange(len(struc_emb)).long().to(sim.device)

        func_loss = F.cross_entropy(sim, labels)
        struc_loss = F.cross_entropy(sim.t(), labels)

        return func_loss, struc_loss
    
    def training_step(self, batch, batch_idx):
        X, y, chr, pos, idx = batch
        func_emb, struc_emb = self(X, y)
        func_loss, struc_loss = self.compute_loss(func_emb, struc_emb)
        data_dict = {"loss": (func_loss + struc_loss) / 2}
        log_dict = {"train/func_loss": func_loss, "train/struc_loss": struc_loss, "train/loss": data_dict["loss"]}
        self.log_dict(log_dict, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return data_dict
    
    def validation_step(self, batch, batch_idx):
        X, y, chr, pos, idx = batch
        func_emb, struc_emb = self(X, y)
        func_loss, struc_loss = self.compute_loss(func_emb, struc_emb)
        data_dict = {"loss": (func_loss + struc_loss) / 2}
        log_dict = {"val/func_loss": func_loss, "val/struc_loss": struc_loss, "val/loss": data_dict["loss"]}
        self.log_dict(log_dict, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        return data_dict
    
    def test_step(self, batch, batch_idx):
        X, y, chr, pos, idx = batch
        func_emb, struc_emb = self(X, y)
        func_loss, struc_loss = self.compute_loss(func_emb, struc_emb)
        data_dict = {"loss": (func_loss + struc_loss) / 2}
        log_dict = {"test/func_loss": func_loss, "test/struc_loss": struc_loss, "test/loss": data_dict["loss"]}
        self.log_dict(log_dict, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        self.test_outputs['func_emb'].append(func_emb)
        self.test_outputs['struc_emb'].append(struc_emb)

        return data_dict
    
    def on_test_epoch_end(self):
        self.test_outputs['func_emb'] = torch.cat(self.test_outputs['func_emb']).cpu().numpy()
        self.test_outputs['struc_emb'] = torch.cat(self.test_outputs['struc_emb']).cpu().numpy()


class OverrideEpochStepCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self._log_step_as_current_epoch(trainer, pl_module)

    def _log_step_as_current_epoch(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        pl_module.log("step", trainer.current_epoch)
