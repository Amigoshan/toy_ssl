import matplotlib.pyplot as plt
import numpy as np
# from utils import groupPlot
from os.path import join

def groupPlot(datax, datay, group=10):
    datax, datay = np.array(datax), np.array(datay)
    if len(datax)%group>0:
        datax = datax[0:len(datax)/group*group]
        datay = datay[0:len(datay)/group*group]
    datax, datay = datax.reshape((-1,group)), datay.reshape((-1,group))
    datax, datay = datax.mean(axis=1), datay.mean(axis=1)
    return (datax, datay)


datadir = 'data200_label_%d_hidden_500_lr0.01000_batch20_alpha20_lamb0.10000_thresh0.05000_iros_%d'
numlist = [5,10,20,50,100]
filename = 'loss_gt.npy'
maxy = [1,5,1]
title = ['Labeled Loss', 'Unlabeled Loss', 'Loss for all the data']
# filelist = ['loss_label.npy','loss_unlabel.npy', 'loss_gt.npy']
imgoutdir = 'resimg_facing'
AvgNum = 100
casenum = 10

# semi-supervised
lossplot = []
for lnum in numlist:
	lossavg = 0
	lossnum = 0
	for k in range(casenum):
		dirname = datadir % (lnum, k)

		loss = np.load(join(dirname,filename))
		lossmean = np.mean(loss[-AvgNum:])
		if lossmean<0.5:
			lossavg += lossmean
			lossnum += 1
		print '  ',lossmean
	print '--',lossavg/casenum
	lossplot.append(lossavg/lossnum)

# supervised
datadir = 'data200_label_%d_hidden_500_lr0.01000_batch20_alpha20_lamb0.00000_thresh0.05000_iros_%d'
casenum = 3
lossplot2 = []
for lnum in numlist:
	lossavg = 0
	for k in range(casenum):
		dirname = datadir % (lnum, k)

		loss = np.load(join(dirname,filename))
		lossavg += np.mean(loss[-AvgNum:])
		print '  ',np.mean(loss[-AvgNum:])
	print lossavg/casenum
	lossplot2.append(lossavg/casenum)


plt.plot(np.array(numlist),np.array(lossplot2),'o-', label='Supervised')
plt.plot(np.array(numlist),np.array(lossplot),'o-', label='Semi-supervised')
plt.grid()
plt.legend()
plt.xlabel('Number of labeled data')
plt.ylabel('Loss after 500 epoch training')
# plt.title(title[k])

plt.show()