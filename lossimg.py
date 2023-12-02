import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1 import host_subplot


import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch.nn.functional as F
import torch.nn as nn
from math import exp






data1_loss = np.load('losses.npy', allow_pickle = True)
host = host_subplot(111)
host.set_xlabel("iterations")
host.set_ylabel("loss")
plt.plot(data1_loss)
plt.draw()
plt.show()

