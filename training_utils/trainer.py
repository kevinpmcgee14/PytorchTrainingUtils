import time
import math
import torch
from torch import optim
from tqdm.auto import tqdm
from functools import partial
from base import BaseCallback 
from torch_lr_finder import LRFinder
from torch_lr_finder.lr_finder import TrainDataLoaderIter

class Trainer(BaseCallback):

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  def __init__(self, model, data, loss_func, metrics=[], cbs=[]):
    self.model = model
    self.data = data
    self.metrics = metrics
    self.cbs = cbs
    self.loss_func = loss_func
    self.initial_state = self.model.state_dict()

  def start(self, *args):
    self.model.to(device)
    self.epoch = 0
    self.best_loss = 100
    self.dataset_sizes =  {
      'train': len(self.data['train'].dataset),
      'val': len(self.data['val'].dataset)
    }
    partial(self.calc_metrics, 'start', self)(*args)
    partial(self.run_cbs, 'start', self)(*args)
    
  
  def before_epoch(self, *args):
    partial(self.calc_metrics, 'before_epoch', self)(*args)
    partial(self.run_cbs, 'before_epoch', self)(*args)
    

  def before_training(self, *args):
    if self.phase == 'train':
      self.model.train()  
    else:
      self.model.eval()  

    self.running_loss = 0.0
    partial(self.calc_metrics, 'before_training', self)(*args) 
    partial(self.run_cbs, 'before_training', self)(*args)
    

  def before_batch(self, *args):
    self.inputs = self.batch['x']
    self.targets = self.batch['y']
    self.inputs = self.inputs.to(device)
    self.targets = self.targets.to(device)
    self.opt.zero_grad()
    partial(self.calc_metrics, 'before_batch', self)(*args)
    partial(self.run_cbs, 'before_batch', self)(*args)
    

  def before_loss(self, *args):
    partial(self.calc_metrics, 'before_loss', self)(*args)
    partial(self.run_cbs, 'before_loss', self)(*args)
    

  def after_loss(self, *args):
    partial(self.calc_metrics, 'after_loss', self)(*args)
    partial(self.run_cbs, 'after_loss', self)(*args)
    

  def after_backward(self, *args):
    self.opt.step()
    partial(self.calc_metrics, 'after_backward', self)(*args)
    partial(self.run_cbs, 'after_backward', self)(*args)
    

  def after_batch(self, *args):
    self.running_loss += self.loss.item() * self.inputs.size(0)
    if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR) and self.phase=='train':
      self.scheduler.step()
    partial(self.calc_metrics, 'after_batch', self)(*args)
    partial(self.run_cbs, 'after_batch', self)(*args)
    
 
  def after_phase(self, *args):
    self.epoch_loss = self.running_loss / self.dataset_sizes[self.phase]
    partial(self.calc_metrics, 'after_phase', self)(*args)
    partial(self.run_cbs, 'after_phase', self)(*args)

  def after_epoch(self, *args):
    if not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
      self.scheduler.step()
    partial(self.calc_metrics, 'after_epoch', self)(*args)
    partial(self.run_cbs, 'after_epoch', self)(*args)
    

  def end(self, *args):
    partial(self.calc_metrics, 'end', self)(*args)
    partial(self.run_cbs, 'end', self)(*args)

  class LRDataloader(TrainDataLoaderIter):

    def __init__(self, dataloader, auto_reset=True):
      super().__init__(dataloader)
      self.auto_reset = auto_reset

    def inputs_labels_from_batch(self, batch):
      inputs, labels, = batch['x'], batch['y']

      return inputs, labels

  def find_lr(self, start_lr=1e-6, end_lr=1e2, accum_steps=1, opt='AdamW', wd=0):
    
    self.set_optimizer(opt=opt, lr=start_lr, wd=wd)
    dl = self.LRDataloader(self.data['train'])
    lr_finder = LRFinder(self.model, self.opt, self.loss_func, device="cuda")
    lr_finder.range_test(dl, end_lr=end_lr, num_iter=100, accumulation_steps=accum_steps)
    lr_finder.plot() 
    lr_finder.reset()

  def fit(self, lr, num_epochs, optim='AdamW', schedule_type='cos_onecycle', wd=0):
  
    self.num_epochs = num_epochs
    self.set_optimizer(opt=optim, lr=lr, wd=wd)
    self.set_scheduler(schedule_type, lr)
    self.start()

    since = time.time()

    for epoch in range(self.num_epochs):
      
      self.before_epoch()
      
      for phase in ['train', 'val']:
        
        self.phase = phase
        self.before_training()

        for batch in tqdm(self.data[phase]):

          self.batch = batch
          self.before_batch()

          self.outputs = self.model(self.inputs)
          self.before_loss()

          self.loss = self.loss_func(self.outputs, self.targets)
          self.after_loss()
          
          if phase == 'train':
              self.loss.backward()
              self.after_backward()
          
          self.after_batch()
          
        self.after_phase()

      self.after_epoch()
    
    self.end()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))


  def set_optimizer(self, opt, lr, wd):
    if opt == 'AdamW':
      self.opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)
    elif hasattr(optim, opt):
      self.opt = getattr(optim, opt)(self.model.parameters(), lr=lr)
    else:
      print(f'{opt} does not exit in PyTorch optim library. Defaulting to AdamW.')
      self.opt = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5, amsgrad=True)
      
  def set_scheduler(self, schedule_type, lr, gamma=None):
    if schedule_type =='standard':
      self.scheduler = optim.lr_scheduler.LambdaLR(self.opt, lambda epoch: lr)
    elif schedule_type =='linear_onecycle':
      self.scheduler = optim.lr_scheduler.OneCycleLR(self.opt, lr, anneal_strategy='linear', steps_per_epoch=len(self.data['train']), epochs=self.num_epochs)
    elif schedule_type =='cos_onecycle':
      self.scheduler = optim.lr_scheduler.OneCycleLR(self.opt, lr, anneal_strategy='cos', steps_per_epoch=len(self.data['train']), epochs=self.num_epochs)
    elif schedule_type == 'cyclic':
      if any([pg.get('momentum') for pg in self.opt.param_groups]):
        cycle_momentum = True
      else:
        cycle_momentum = False
      self.scheduler = optim.lr_scheduler.CyclicLR(self.opt, lr, lr*100, step_size_up=len(self.data['train'])/2, cycle_momentum=cycle_momentum)
    elif schedule_type == 'exponential_finder':
      self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, 1/10)

  def calc_metrics(self, time, *args):
    for metric in self.metrics:
      if hasattr(metric, time):
        getattr(metric, time)(*args)

  def run_cbs(self, time, *args):
    for cb in self.cbs:
      if hasattr(cb, time):
        getattr(cb, time)(*args)
