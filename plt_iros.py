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


datadir = 'data200_label_10_hidden_500_lr0.01000_batch20_alpha20_lamb%.5f_thresh0.05000_iros'
lambdalist = [0.02,0.1,0.5]#,0.05
maxy = [1,5,1]
title = ['Labeled Loss', 'Unlabeled Loss', 'Loss for all the data']
filelist = ['loss_label.npy','loss_unlabel.npy', 'loss_gt.npy']
imgoutdir = 'resimg_facing'
AvgNum = 100

# lines = []
for k,filename in enumerate(filelist):

	for lamb in lambdalist:
		dirname = datadir % (lamb)

		loss = np.load(join(dirname,filename))
		if(len(loss.shape)==1):
			datax, datay = groupPlot(range(1,loss.shape[0]+1),loss)
			plt.plot(range(1,501), datay, label='lambda=%.2f' % (lamb))
		else:
			plt.plot(range(1,501),loss[1], label='lambda=%.2f' % (lamb))
		# lines.append(line)
	plt.grid()
	plt.legend()
	plt.ylim(0,maxy[k])
	plt.xlabel('number of epoch')
	plt.ylabel('loss')
	plt.title(title[k])
	plt.show()
# 	trainloss = np.load(join(datadir,exp_pref+'lossplot.npy'))
# 	valloss = np.load(join(datadir,exp_pref+'vallossplot.npy'))
# 	unlabelloss = np.load(join(datadir,exp_pref+'unlabellossplot.npy'))

# trainloss = np.array(trainloss[0:plotnum])
# valloss = np.array(valloss[0:plotnum])
# unlabelloss = np.array(unlabelloss[0:plotnum])

# print 'train: %.5f, val: %.5f, unlabel: %.5f' % (np.mean(trainloss[-AvgNum:]), np.mean(valloss[-AvgNum:]), np.mean(unlabelloss[-AvgNum:]))
# print '%.2f, %.2f, %.2f' % (np.mean(trainloss[-AvgNum:]), np.mean(valloss[-AvgNum:]), np.mean(unlabelloss[-AvgNum:]))

# ax1 = plt.subplot(121)
# ax1.plot(trainloss)
# ax1.plot(valloss)
# ax1.grid()
# ax1.set_ylim(0,1)

# ax2 = plt.subplot(122)
# gpunlabelx, gpunlabely = groupPlot(range(len(unlabelloss)),unlabelloss,group=100)
# ax2.plot(unlabelloss)
# ax2.plot(gpunlabelx, gpunlabely, color='y')
# ax2.grid()
# ax2.set_ylim(0,10)

# plt.savefig(join(imgoutdir, logname+'.png'))

# plt.show()