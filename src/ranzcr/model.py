from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from .utils import competition_metric, CosineAnnealingWarmupRestarts
import timm

class RanzcrClassifier(LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

        self.classifier = timm.create_model(self.hparams.base_model, pretrained=True)
        n_features = self.classifier._modules[self.hparams.base_model_classifier].in_features
        self.classifier._modules[self.hparams.base_model_classifier] = nn.Linear(n_features, self.hparams.classes)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss(logits, labels)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = self.loss(logits, labels)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        probs = torch.sigmoid(logits)
        return {'preds': probs, 'labels': labels}

    def validation_epoch_end(self, validation_step_outputs):
        preds_list = []
        labels_list = []
        for output in validation_step_outputs:
            preds_list.append(output['preds'])
            labels_list.append(output['labels'])
        preds = torch.cat(preds_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        score = competition_metric(labels, preds)
        self.log('val_score', score, on_epoch=True, prog_bar=True)

    # do something with a pred
    def configure_optimizers(self):
        optimizer = Adam(self.classifier.parameters(),
                         lr=self.hparams.lr,
                         weight_decay=self.hparams.weight_decay)
        if self.hparams.scheduler == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max = self.hparams.t_max, eta_min=self.hparams.min_lr, last_epoch=-1)
            return {'optimizer': optimizer, 'scheduler': scheduler}
        elif self.hparams.scheduler == 'CosineAnnealingWarmupRestarts':
            scheduler = {'scheduler': CosineAnnealingWarmupRestarts(optimizer,
                                                      max_lr=self.hparams.lr,
                                                      first_cycle_steps=self.hparams.first_cycle_steps,
                                                      min_lr=self.hparams.min_lr,
                                                      warmup_steps=self.hparams.warmup_steps,
                                                      gamma=self.hparams.gamma),
                        'name':'learning_rate',
                        'interval':'step'}
            return {'optimizer': optimizer, 'scheduler': scheduler, 'interval':'step'}
        else:
            return {'optimizer': optimizer}