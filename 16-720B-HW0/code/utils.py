# some numpy functions for fucking pytorch
# Du 2018.9
import torch


# np.roll
def roll(a, shift, axis, mode='cyclic'):
  if shift == 0:
    return a
  if axis < 0:
    axis += a.dim()
  dim_size = a.size(axis)

  if mode == 'cyclic':
    if shift >= 0:
      after_start = dim_size - shift
    else: #shift < 0:
      after_start = -shift
      shift = dim_size - abs(shift)

    before = a.narrow(axis, 0, dim_size - shift)
    after = a.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)
  elif mode == 'pad':
    if shift >= 0:
      after_start = dim_size - shift
      before = a.narrow(axis, 0, dim_size - shift)
      after = torch.zeros_like(a.narrow(axis, after_start, shift))
    else: #shift < 0:
      after_start = -shift
      shift = dim_size - abs(shift)  
      before = torch.zeros_like(a.narrow(axis, 0, dim_size - shift))
      after = a.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)
  elif mode == 'valid':
    if shift >= 0:
      after_start = dim_size - shift
      return a.narrow(axis, 0, dim_size - shift)
    else: #shift < 0:
      after_start = -shift
      shift = dim_size - abs(shift)  
      return a.narrow(axis, after_start, shift)

  