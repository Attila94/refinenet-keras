# -*- coding: utf-8 -*-

import math

# learning rate schedule
def step_decay(epoch):
  initial_lrate = 1e-4
  drop = 0.5
  epochs_drop = 8.0
  lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
  return lrate