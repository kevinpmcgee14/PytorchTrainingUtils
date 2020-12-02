""" 
    A BaseClass which all Classes will derive from.
"""
class BaseCallback(object):

  def __init__(self):
    pass

  def start(self, *args):
    pass
  
  def before_epoch(self, *args):
    pass

  def before_training(self, *args):
    pass

  def before_validation(self, *args):
    pass

  def after_training(self, *args):
    pass

  def after_loss(self, *args):
    pass

  def after_backward(self, *args):
    pass

  def after_validation(self, *args):
    pass
  
  def after_epoch(self, *args):
    pass

  def end(self, *args):
    pass