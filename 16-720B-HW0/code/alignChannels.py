import cv2
import numpy as np
import scipy
import torch
from utils import roll


def alignChannels(red, green, blue):
  """Given 3 images corresponding to different channels of a color image,
  compute the best aligned result with minimum abberations

  Args:
    red, green, blue - each is a HxW matrix corresponding to an HxW image

  Returns:
    rgb_output - HxWx3 color image output, aligned as desired"""
  b,g,r = map(torch.from_numpy, [blue, red, green])
  
  b,g,r = map(
    lambda x:x.float().cuda()/255., 
    [b,g,r]
  )
  
  # use b as base
  
  def calib(base, dest, m=20):
    i0, j0, min_ssd = 0, 0, float('inf')
    h, w = base.shape
    for i in range(-m, m):
      for j in range(-m, m):
        l = torch.zeros(h+2*m+2,w+2*m+2)
        r = torch.zeros(h+2*m+2,w+2*m+2)
        r[m+1+i:m+1+i+h,m+1+j:m+1+j+w] = dest
        l[m+1:m+1+h,m+1:m+1+w] = base
        ssd = float(torch.mean((r-l)**2))
        print(i, j, ssd)
        if ssd<min_ssd:
          min_ssd = ssd
          i0, j0= i,j
          
    return i0, j0
    
  ib, jb = 0, 0  
  ig, jg = calib(b, g)
  ir, jr = calib(b, r)
  h, w = b.shape
  _mini = min(ib, ig, ir)
  _maxi = max(ib, ig, ir)+h
  _minj = min(jb, jg, jr)
  _maxj = max(jb, jg, jr)+w
    
    
  return _mini, _maxi, _minj, _maxj
