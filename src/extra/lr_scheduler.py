import math
from typing import List
from tinygrad.nn.optim import Optimizer
from tinygrad import dtypes, Tensor
from typing import Callable

class LR_Scheduler:
  def __init__(self, optimizer: Optimizer):
    self.optimizer = optimizer
    self.epoch_counter = Tensor([0], requires_grad=False, device=self.optimizer.device)

  def get_lr(self): pass

  def schedule_step(self) -> list[Tensor]: return [self.epoch_counter.assign(self.epoch_counter + 1), self.optimizer.lr.assign(self.get_lr())]
  def step(self) -> None: Tensor.realize(*self.schedule_step())

class LRSchedulerGroup:
  def __init__(self, *schedulers: LR_Scheduler): self.schedulers = schedulers
  def step(self) -> None:
    for s in self.schedulers: s.step()

class MultiStepLR(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, milestones: List[int], gamma=0.1):
    super().__init__(optimizer)
    self.milestones = milestones
    self.gamma = gamma

  def get_lr(self) -> Tensor:
    if self.epoch_counter.numpy()[0] not in self.milestones:
      return self.optimizer.lr
    return self.optimizer.lr * self.gamma

class ReduceLROnPlateau(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, mode="min", factor=0.1, patience=10, threshold=1e-4, threshold_mode="rel"):
    assert mode in ["min", "max"] and threshold_mode in ["rel", "abs"]
    super().__init__(optimizer)
    self.mode, self.factor, self.patience, self.threshold, self.threshold_mode = mode, factor, patience, threshold, threshold_mode
    self.best = float('inf') if mode == "min" else float('-inf')
    self.bad_epoch = 0

    if mode == "min": self.threshold *= -1

  def is_better(self, current: float) -> bool:
    dynamic_threshold = self.best*(1+self.threshold) if self.threshold_mode == "rel" else self.best+self.threshold
    if self.mode == "min":
      return current < dynamic_threshold
    return current > dynamic_threshold

  def step(self, current: float) -> None:
    self.epoch_counter.assign(self.epoch_counter + 1).realize()
    if self.is_better(current):
      self.bad_epoch = 0
      self.best = current
    else:
      self.bad_epoch += 1

    if self.bad_epoch > self.patience:
      self.optimizer.lr *= self.factor
      self.bad_epoch = 0

class CosineAnnealingLR(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, T_max: int, eta_min=0):
    super().__init__(optimizer)
    self.T_max = T_max
    self.eta_min = eta_min
    self.eta_max = optimizer.lr.numpy()[0]

  def get_lr(self) -> Tensor:
    lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos((self.epoch_counter.numpy()[0]/self.T_max) * math.pi))
    return Tensor([lr], device=self.optimizer.device, dtype=self.optimizer.lr.dtype)

class OneCycleLR(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, max_lr: float, div_factor: float, final_div_factor: float, total_steps: int, pct_start: float,
               anneal_strategy: str = 'linear', cycle_momentum: bool = False):
    super().__init__(optimizer)
    self.initial_lr = max_lr / div_factor
    self.max_lr = max_lr
    self.min_lr = self.initial_lr / final_div_factor
    self.total_steps = total_steps
    self.pct_start = pct_start
    assert anneal_strategy == 'linear', 'only linear annealing supported'
    assert not cycle_momentum, 'cycle momentum not supported'
    self.optimizer.lr.assign(self.get_lr()).realize() # update the initial LR

  @staticmethod
  def _annealing_linear(start: float, end: float, pct: Tensor) -> Tensor: return (pct*(end-start)+start)

  def get_lr(self) -> Tensor:
    return (self.epoch_counter < self.total_steps*self.pct_start).where(
      self._annealing_linear(self.initial_lr, self.max_lr, self.epoch_counter/(self.total_steps*self.pct_start)),
      self._annealing_linear(self.max_lr, self.min_lr, (self.epoch_counter-(self.total_steps*self.pct_start))/(self.total_steps*(1-self.pct_start)))
    ).cast(self.optimizer.lr.dtype)

# https://github.com/mlcommons/training/blob/e237206991d10449d9675d95606459a3cb6c21ad/image_classification/tensorflow2/lars_util.py
class PolynomialDecayWithWarmup(LR_Scheduler):
  def __init__(self, optimizer: Optimizer, initial_lr, end_lr, train_steps, warmup, power=2):
    super().__init__(optimizer)
    self.epoch_counter = self.epoch_counter.cast(dtypes.float32)
    assert train_steps > 0 and warmup > 0
    self.warmup = min(warmup, train_steps)
    self.initial_lr, self.end_lr, self.epochs, self.power = initial_lr, end_lr, train_steps, power

    # set lr for first warmup step
    self.optimizer.lr.assign(self.get_lr()).realize()

  def get_lr(self):
    # LR is 0 on the first step, matching the reference.
    warmup_lr = (self.epoch_counter * (1.0 / self.warmup)) * self.initial_lr
    x = (1 - (self.epoch_counter - self.warmup) / (self.epochs - self.warmup + 1))
    return (self.epoch_counter <= self.warmup).where(warmup_lr, (self.initial_lr - self.end_lr) * x ** self.power + self.end_lr).cast(self.optimizer.lr.dtype)

class CosineAnnealingLRWithWarmup(LR_Scheduler):
  def __init__(self, optimizer:Optimizer, base_lr, end_lr, warmup_steps:int, decay_steps:int):
    assert warmup_steps > 0 and decay_steps > 0
    super().__init__(optimizer)
    self.base_lr = base_lr
    self.end_lr = end_lr
    self.warmup_steps = warmup_steps
    self.decay_steps = decay_steps
    # set lr for first warmup step
    self.optimizer.lr.assign(self.get_lr()).realize()

  def get_lr(self):
    warmup_lr = ((self.epoch_counter+1) / self.warmup_steps) * self.base_lr
    decay_lr = self.end_lr + 0.5 * (self.base_lr-self.end_lr) * (1 + (((self.epoch_counter+1-self.warmup_steps)/self.decay_steps) * math.pi).cos())
    return (self.epoch_counter < self.warmup_steps).where(warmup_lr, decay_lr).cast(self.optimizer.lr.dtype)

# Reference: https://github.com/mlcommons/training/blob/64b14a9abc74e08779a175abca7d291f8c957632/stable_diffusion/ldm/lr_scheduler.py, Lines 36-97
class LambdaLinearScheduler:
  def __init__(self, warm_up_steps:int, f_min:float, f_max:float, f_start:float, cycle_lengths:int):
    self.lr_warm_up_steps, self.f_min, self.f_max, self.f_start, self.cycle_lengths = warm_up_steps, f_min, f_max, f_start, cycle_lengths

  def schedule(self, n:Tensor) -> Tensor:
    warm_up = (n < self.lr_warm_up_steps)
    f_warm_up = (self.f_max - self.f_start) / self.lr_warm_up_steps * n + self.f_start
    return warm_up.where(f_warm_up, self.f_min + (self.f_max - self.f_min) * (self.cycle_lengths - n) / (self.cycle_lengths))

# based on torch.optim.lr_scheduler.LambdaLR
class LambdaLR(LR_Scheduler):
  def __init__(self, optimizer:Optimizer, base_lr:Tensor, lr_lambda:Callable):
    super().__init__(optimizer)
    self.base_lr, self.lr_lambda = base_lr, lr_lambda
    self.step()

  def get_lr(self):
    return self.base_lr * self.lr_lambda(self.epoch_counter - 1)