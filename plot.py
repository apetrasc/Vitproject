import os
#import scipy.io as sio

import matplotlib
#matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import numpy as np
import math
import time
from matplotlib import ticker
import matplotlib.colors as colors
import sys
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
from matplotlib.lines import Line2D
file ='.epoch_log/NN_WallReconfluct1TF2_3NormIn-3Out_1-15_432x432_Ret180_lr0.001_decay20drop0.5_relu-1732884982_log.npz'
data = np.load(file)
tLoss = data['tLoss']
vLoss = data['vLoss']
epoch = np.array(range(1,len(tLoss)+1),dtype='int')
fig = plt.figure(0)
axu = fig.add_subplot(111)
p1_uu = axu.plot(epoch,tLoss,'-', linewidth=2,markersize=5,color='C1')
p1_uu = axu.plot(epoch, vLoss, 'o', linewidth=2, markersize=5, color='C1', linestyle='dotted')

#plt.ylim(vLoss[-1]/1.5,vLoss[0])
#plt.ylim(None,None)
#plt.xlim(0,len(tLoss)+1)

legend_elements = [Line2D([0], [0], color='C1', label='Training loss'),Line2D([4],[0],color='C1',marker='o',markersize=3.5,linestyle='dotted',label='Validation loss')]
legend1=plt.legend(handles=legend_elements,loc='best',fontsize=12)
axu.add_artist(legend1)
plt.xlabel('Epoch')
plt.ylabel(r'$\mathcal{L}(\mathbf{u}_\mathrm{FCN};\mathbf{u}_\mathrm{DNS})$')
fig.savefig('./epoch.pdf')