import torch
from training_utils.base import BaseCallback

class Accuracy(Callback):

  def __init__(self):
    self.name = 'accuracy'

  def before_training(self, trainer, *args):
    self.target_corrects = 0
  
  def after_batch(self, trainer, *args):
    self.target_corrects += torch.sum(torch.max(trainer.outputs, 1)[1] == torch.max(trainer.targets.view(trainer.outputs.size()[0], -1), 1)[0])

  def after_phase(self, trainer, *args):
    self.calc_accuracy(trainer.dataset_sizes[trainer.phase])

  def calc_accuracy(self, dataset_size, *args):
    self.value = self.target_corrects.double() / dataset_size