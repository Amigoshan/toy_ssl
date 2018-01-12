from os.path import join, isdir, isfile
from os import listdir
import numpy as np

dirname = 'ex5'
lossfile = 'loss_label.npy'

minloss = 1000
minexp = ''
losslist = []
explist = []
for subdir in listdir(dirname):
	expdir = join(dirname, subdir)
	if isdir(expdir):
		loss_label = np.load(join(expdir,lossfile))
		finalloss = np.mean(loss_label[1,-5:])
		losslist.append(finalloss)
		explist.append(subdir)

		if minloss>finalloss:
			minloss = finalloss
			minexp = subdir

dtype = [('exp', 'S100'),('loss',float)]
values = [(explist[k], losslist[k]) for k in range(len(losslist))]

values_np = np.array(values, dtype=dtype)
values_sort = np.sort(values_np, order='loss')

print values_sort

# print losslist
# print minloss
# print minexp


