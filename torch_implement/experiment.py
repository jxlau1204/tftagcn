import json
import logging
import pytorch_lightning as pl
import gc
import math
from abc import ABC
import seaborn as sns
import matplotlib.pyplot as plt
import sys 
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule
from torch.optim import Adamax, Adadelta, Adam, RMSprop
from torch.optim.lr_scheduler import ExponentialLR
# from torch.optim.rmsprop import RMSprop
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from modules.metricsTop import MetricsTop
from units import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import datasets
import models
from pytorch_lightning import Trainer, seed_everything
import os
import numpy as np
import random

class Experiment(LightningModule, ABC):
    def __init__(self, cfg):
        super().__init__()
        # self.hparams.update(OmegaConf.to_container(cfg, resolve=True))
        self.save_hyperparameters()
        self.cfg = cfg
        self.model = getattr(models, cfg.model.name)(cfg.model)
        self.my_logger = None
        self.metric = self.model.metric
        self.loss = self.model.loss
    def forward(self, data):
        return self.model(**data)
    def on_train_start(self):
        self.my_logger = logging.getLogger(self.trainer.logger.log_dir+"/mylog.txt")
        self.my_logger.handlers = []
        self.my_logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(filename=self.trainer.logger.log_dir+"/mylog.txt",
                                     encoding="utf8")
        self.my_logger.addHandler(fh)
        torch.cuda.empty_cache()
        # if self.cfg.lr_scheduler.active:
        #     current_lr = self.get_lr(self.trainer.optimizers[0])
            # self.logger.experiment.add_scalar(
            #     'learning_rate', current_lr, self.current_epoch)
        super().on_train_start()
    def prepare_data(self):
        Datasets = getattr(datasets, self.cfg.dataset.name)
        self.train_dataset = Datasets("train", **self.cfg.dataset)
        self.val_dataset = Datasets("valid", **self.cfg.dataset)
        self.test_dataset = Datasets("test", **self.cfg.dataset)
        

    def configure_optimizers(self):
        lr = self.cfg.optimizer.learning_rate
        weight_decay = self.cfg.optimizer.weight_decay
        # eps = 1e-2 / float(self.cfg.data_loader.batch_size if self.cfg.data_loader.batch_size else  * 40) ** 2
        params = self.model.parameters()
        if self.cfg.optimizer.type.lower() == "rmsprop":
            optimizer = RMSprop(params,
                                lr=lr,
                                momentum=self.cfg.optimizer.momentum,
                                # eps=eps,
                                weight_decay=weight_decay)
        elif self.cfg.optimizer.type.lower() == "adam":
            optimizer = Adam(params,
                             lr=lr,
                             eps=1e-7,
                             weight_decay=weight_decay)
        elif self.cfg.optimizer.type.lower() == "adamax":
            optimizer = Adamax(params,
                             lr=lr,
                             eps=1e-7,
                             weight_decay=weight_decay)
        else:
            raise ValueError("Unknown optimizer type.")


        if not self.cfg.lr_scheduler.active:
            return optimizer
        scheduler = ExponentialLR(optimizer=optimizer,
                                  gamma=self.cfg.lr_scheduler.decay_rate)

        return [optimizer], [scheduler]

    def train_dataloader(self):

        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.data_loader.batch_size if not hasattr(self.train_dataset, "batch_sampler") else 1,
                          collate_fn=self.train_dataset.collate_fn if hasattr(self.train_dataset, "collate_fn") else None,
                        #   shuffle=True,
                          batch_sampler = self.train_dataset.batch_sampler() if hasattr(self.train_dataset, "batch_sampler") else None,
                          num_workers=self.cfg.data_loader.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.cfg.data_loader.batch_size if not hasattr(self.val_dataset, "batch_sampler") else 1 ,
                          collate_fn=self.val_dataset.collate_fn if hasattr(self.val_dataset, "collate_fn") else None,
                        #   shuffle=True,
                          batch_sampler = self.val_dataset.batch_sampler() if hasattr(self.val_dataset, "batch_sampler") else None ,
                          num_workers=self.cfg.data_loader.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.cfg.data_loader.batch_size if not hasattr(self.test_dataset, "batch_sampler") else 1,
                          collate_fn= self.test_dataset.collate_fn if hasattr(self.test_dataset, "collate_fn") else None,
                        #   shuffle=True,
                          batch_sampler = self.test_dataset.batch_sampler() if hasattr(self.test_dataset, "batch_sampler") else None,
                          num_workers=self.cfg.data_loader.num_workers)

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']


    def on_batch_end(self) -> None:
        torch.cuda.empty_cache()
        gc.collect()

    def training_step(self, batch, batch_idx):
        out, res = self(batch)
        labels = batch["labels"]
        loss = self.loss(
            out,
            labels,
            self.cfg.dataset.get("train_mode", "classification")
        )
        log = self.metric(out, labels)
        # log["train_loss"] = loss.detach()
        self.log("train_accuracy", log["accuracy"])
        self.log("train_loss", loss.detach())
        return loss

    def validation_step(self, batch, batch_idx):
        out, res = self(batch)
        labels = batch["labels"]
        log = {}
        log["out"] = out
        log["labels"] = labels
        # log["loss"] = loss
        # self.log
        return log

    def validation_epoch_end(self, outputs):
        aggra_out = torch.cat([x["out"] for x in outputs])
        aggra_labels = torch.cat([x["labels"] for x in outputs])
        val_loss = self.loss(aggra_out, aggra_labels, self.cfg.dataset.get("train_mode", "classification"))
        log = self.metric(aggra_out, aggra_labels)
        self.log("val_loss", float(val_loss), on_epoch=True, on_step =False)
        # self.log("val_accuracy", float(list(log.values())[0]), on_epoch=True, on_step =False)
        c_m = confusion_matrix(aggra_labels.data.cpu().numpy(), aggra_out.data.cpu().argmax(-1).numpy())
        c_m_nor = confusion_matrix(aggra_labels.data.cpu().numpy(), aggra_out.data.cpu().argmax(-1).numpy(), normalize="true")
        c_m_nor_str = str(c_m_nor.round(4).tolist()).replace("], [", "; ")
        if self.my_logger != None:
            self.my_logger.info(f"epoch{self.current_epoch}:{c_m_nor_str}")
        fig = plot_confusion_matrix(c_m, normalize= True, classes=self.cfg.model.classes_name)
        self.logger.experiment.add_figure("conf/valid_nor", fig, self.current_epoch)
        self.log("val_loss", float(val_loss), on_epoch=True, on_step =False)
        self.log("val_accuracy", float(log["accuracy"]), on_epoch=True, on_step =False)
        if self.my_logger != None:
            log.update(cur_epoch=self.current_epoch)
            self.my_logger.info("epoch%d:val_wacc=%.4f, val_uacc=%.4f, val_single_acc=%s val_f1-macro=%.4f, val_f1-micro=%.4f,val_f1-weighted=%.4f, val_loss=%.4f;"%(self.current_epoch,log["accuracy"],  log["unweight_accuracy"],log["single_accuracy"],log["f1_macro"], log["f1_micro"], log["f1_weighted"], float(val_loss)))
        # del(log["single_accuracy"])
        # self.log_dict(log)

    def test_step(self, batch, batch_idx):
        out, res = self(batch)
        labels = batch["labels"]
        log = {}
        log["out"] = out
        log["labels"] = labels
        return log
    def test_epoch_end(self, outputs):
        aggra_out = torch.cat([x["out"] for x in outputs])
        aggra_labels = torch.cat([x["labels"] for x in outputs])
        test_loss = self.loss(aggra_out, aggra_labels, self.cfg.dataset.get("train_mode", "classification"))
        log = self.metric(aggra_out, aggra_labels)
        self.log("test_loss", float(test_loss), on_epoch=True, on_step =False)
        self.log("test_accuracy", float(list(log.values())[0]), on_epoch=True, on_step =False)
        c_m = confusion_matrix(aggra_labels.data.cpu().numpy(), aggra_out.data.cpu().argmax(-1).numpy())
        fig = plot_confusion_matrix(c_m, normalize= True, classes=self.cfg.model.classes_name)
        self.logger.experiment.add_figure("conf/test_nor", fig, self.current_epoch)
        if self.my_logger != None:
            log.update(cur_epoch=self.current_epoch)
            self.my_logger.info("epoch%d:val_wacc=%.4f, val_uacc=%.4f, val_single_acc=%s val_f1-macro=%.4f, val_f1-micro=%.4f,val_f1-weighted=%.4f, val_loss=%.4f;"%(self.current_epoch,log["accuracy"],  log["unweight_accuracy"],log["single_accuracy"],log["f1_macro"], log["f1_micro"], log["f1_weighted"], float(val_loss)))
            # self.my_logger.info(log)

def train(cfg):
    experiment = Experiment(cfg)
    checkpoint_callback = ModelCheckpoint(**cfg.checkpoints)
    earlystop_callback = EarlyStopping(**cfg.earlystop)
    trainer = Trainer(**cfg.trainer, callbacks=[checkpoint_callback, earlystop_callback])
    # trainer = Trainer(**cfg.trainer, callbacks=[checkpoint_callback])
    if cfg.exp_mode == "train":
        trainer.fit(experiment)
        # trainer.test(ckpt_path="best")
    # else:
        # trainer.test(experiment, ckpt_path="best")
        # trainer.test(experiment, ckpt_path=cfg.test_ckpt)


def torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.set_float32_matmul_precision("highest")


if __name__=="__main__":
    config_path = os.path.split(__file__)[0]+"/config.yaml"
    cfg = OmegaConf.load(config_path)
    sub_config_path = os.path.split(__file__)[0]+"/model_configs/%s.yaml"%(cfg.model.name)
    if os.path.isfile(sub_config_path):
        sub_config = OmegaConf.load(sub_config_path)
        cfg = OmegaConf.merge(cfg, sub_config)

    sub_config_path = os.path.split(__file__)[0]+"/dataset_configs/%s.yaml"%(cfg.dataset.name)
    if os.path.isfile(sub_config_path):
        sub_config = OmegaConf.load(sub_config_path)
        cfg = OmegaConf.merge(cfg, sub_config)
    sub_config_path = os.path.split(__file__)[0]+"/configs/%s_%s.yaml"%(cfg.model.name, cfg.dataset.name)
    if os.path.isfile(sub_config_path):
        sub_config = OmegaConf.load(sub_config_path)
        cfg = OmegaConf.merge(cfg, sub_config)
    print(f"model:{cfg.model.name}")
    print(f"dataset:{cfg.dataset.name}")
    seed_everything(cfg.seed)
    torch_seed(cfg.seed)
    train(cfg)