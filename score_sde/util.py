from .sde_lib import VPSDE, subVPSDE, VESDE
import torch
import numpy as np
import random
import os


def get_noise_fn(sde, model, continuous=False):
  """Wraps `noise_fn` so that the model output corresponds to a real time-dependent noise function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A noise prediction function.
  """
  # model_fn = get_model_fn(model, train=train)

  if isinstance(sde, VPSDE) and continuous:
    def noise_fn(x, t, cond):
      # For VP-trained models, t=0 corresponds to the lowest noise level
      # The maximum value of time embedding is assumed to 999 for
      # continuously-trained models.
      labels = t * (sde.N -  1)
      noise = model(x, labels, cond)
      return noise

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return noise_fn


def get_score_fn(sde, model, continuous=False):
  """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    model: A score model.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A score function.
  """

  if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
    def score_fn(x, t, cond):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * (sde.N - 1)
        score = model(x, labels, cond)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model(x, labels, cond)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None]
      return score

  elif isinstance(sde, VESDE):
    def score_fn(x, t, cond):
      if continuous:
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model(x, labels, cond=cond)
      return score

  else:
    raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

  return score_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))

def seed_torch(seed=1029):
  random.seed(seed)
  # os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
  # torch.backends.cudnn.benchmark = False
  # torch.backends.cudnn.deterministic = True