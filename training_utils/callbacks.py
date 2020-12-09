import copy
import torch
import pandas as pd 
from torchvision import transforms
from collections import OrderedDict
from IPython.display import clear_output, display
from training_utils.base import BaseCallback

class DFCallback(BaseCallback):

  def start(self, trainer, *args):
    self.df = self.create_df(trainer)

  def before_epoch(self, trainer, *args):
    self.display_df()
    trainer.epoch += 1 
    epoch_string = '{}/{}'.format(trainer.epoch, trainer.num_epochs)
    self.row = [epoch_string]

  def after_phase(self, trainer, *args):
    self.row.append(trainer.epoch_loss)
    self.row += ['{:.4f}'.format(metric.value) for metric in trainer.metrics]

  def after_epoch(self, trainer, *args):
    self.update_df(self.row)
    self.display_df()

  def end(self, trainer, *args):
    self.display_df()

  def create_df(self, trainer):
    columns = OrderedDict({'Epoch':[]})
    columns.update({'train_loss': []})
    columns.update({f'train_{metric.name}': [] for metric in trainer.metrics})
    columns.update({'val_loss':[]})
    columns.update({f'val_{metric.name}': [] for metric in trainer.metrics})
    return pd.DataFrame(columns)

  def update_df(self, row):
    epoch_data = pd.Series(row, index=self.df.columns)
    self.df = self.df.append(epoch_data, ignore_index=True) 

  def display_df(self):
    clear_output(wait=True)
    display(self.df) 


class SaveWeightsCallback(BaseCallback):

  def __init__(self, path):
    self.path = path
    self.best_loss = 100

  def start(self, trainer, *args):
    self.best_model_wts = copy.deepcopy(trainer.model.state_dict())

  def after_epoch(self, trainer, *args):
    if trainer.phase == 'val' and trainer.epoch_loss < self.best_loss:
      self.best_loss = trainer.epoch_loss
      self.best_model_wts = copy.deepcopy(trainer.model.state_dict())
      torch.save(trainer.model.state_dict(), self.path)

  def end(self, trainer, *args):
    trainer.model.load_state_dict(torch.load(self.path))


class FiveCropTTA(BaseCallback):

  def __init__(self, size):
    self.tfms = transforms.Compose([
      transforms.FiveCrop(size),
      transforms.Lambda(lambda crops: torch.stack([crop for crop in crops]))
    ])

  def before_batch(self, trainer, *args):
    if trainer.phase == 'val':
      trainer.inputs = self.tfms(trainer.inputs)
      self.ncrops, self.bs, self.c, self.h, self.w = trainer.inputs.size()
      trainer.inputs = trainer.inputs.view(-1, self.c, self.h, self.w)
  
  def before_loss(self, trainer, *args):
    if trainer.phase == 'val':
      trainer.outputs = trainer.outputs.view(self.bs, self.ncrops, -1).mean(1)
