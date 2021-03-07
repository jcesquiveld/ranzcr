from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, CyclicLR, ReduceLROnPlateau
from .utils import competition_metric, CosineAnnealingWarmupRestarts
import timm


def t_sigmoid(self, x, T=1):
    return 1 / (1 + torch.exp(-x / T))

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
            return {'optimizer':optimizer, 'scheduler': scheduler}
        elif self.hparams.scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                      T_0=self.hparams.T_0,
                                                      eta_min=self.hparams.min_lr,
                                                      last_epoch=-1),
            return {'optimizer': optimizer, 'scheduler': scheduler}
        elif self.hparams.scheduler == 'CyclicLR':
            scheduler = CyclicLR(optimizer,
                                 base_lr = self.hparams.min_lr,
                                 max_lr = self.hparams.lr,
                                 step_size_up=100, step_size_down=1000,
                                 scale_mode='iteration',
                                 mode='triangular2',
                                 cycle_momentum=False)
            return [optimizer], {'scheduler':scheduler, 'interval':'step'}
        elif self.hparams.scheduler == 'CosineAnnealingWarmupRestarts':
            scheduler = CosineAnnealingWarmupRestarts(optimizer=optimizer,
                                                      first_cycle_steps=self.hparams.first_cycle_steps,
                                                      warmup_steps=self.hparams.warmup_steps,
                                                      min_lr=self.hparams.min_lr,
                                                      max_lr=self.hparams.lr,
                                                      gamma=self.hparams.gamma)
            return [optimizer], {'scheduler': scheduler, 'interval': 'step'}
        elif self.hparams.scheduler == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer=optimizer, mode='max',
                                          factor=self.hparams.factor,
                                          patience=self.hparams.patience,
                                          min_lr=self.hparams.min_lr)
            return [optimizer], {'scheduler':scheduler, 'monitor':'val_score'}
            return {'optimizer': optimizer}

class RanzcrStudentClassifier(RanzcrClassifier):
    def __init__(self, config, teacher):
        super().__init__(config)
        self.teacher = teacher
        self.loss_teacher = nn.MSELoss()

    def forward(self, x):
        x = self.classifier(x)

        return x

    def t_sigmoid(self, x, T=1):
        return 1 / (1 + torch.exp(-x/T))

    def training_step(self, batch, batch_idx):
        images, labels = batch
        teacher_images = images[:,0,:,:,:]
        student_images = images[:,1,:,:,:]

        student_logits = self(student_images)
        teacher_logits = self.teacher(teacher_images)

        student_probas = self.t_sigmoid(student_logits, T=2)
        teacher_probas = self.t_sigmoid(teacher_logits, T=2)


        loss_bce = self.loss(student_logits, labels)
        loss_mse = self.loss_teacher(teacher_probas, student_probas)
        loss = loss_bce + 5 * loss_mse
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss