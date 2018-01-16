# 1. Input variable should be nomalized
# 2. Indexing a variable with another variable requires same size
# 3. 

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np
from math import pi, sqrt, exp
import matplotlib.pyplot as plt
import time

# record and debug
from os import mkdir
from os.path import isdir, join

class RegNet(nn.Module):
    """docstring for RegNet"""
    def __init__(self, hiddennum = 1000 ):
        super(RegNet, self).__init__()
        self.func = nn.Sequential(
            nn.Linear(1,hiddennum), 
            # nn.SELU(True),
            nn.ReLU(True),
#            nn.LeakyReLU(0.2, True),
#            nn.ELU(1, True),
            nn.Linear(hiddennum,hiddennum),
            # nn.SELU(True),
            nn.ReLU(True),
#            nn.LeakyReLU(0.2, True),
#            nn.ELU(1, True),
            nn.Linear(hiddennum,1)

        )

        self._initialize_weights()

    def forward(self, x):
        
        return self.func(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 1/sqrt(n))
                m.bias.data.zero_()

def dataPrepare(datanum, labelnum, vis = False, seed = 37, noisestd = 0.1):
    # data preparation
    np.random.seed(seed)
    randInd = np.random.permutation(datanum)
    labelInd = randInd[0:labelnum]
    unlabelInd = randInd[labelnum:]
    labelFlag = np.zeros(datanum)
    labelFlag[labelInd] = 1

    RegDataX =  np.linspace(0,4*pi,datanum)
    RegDataY = np.sin(RegDataX)
    RegDataX_norm = (RegDataX-np.mean(RegDataX))/np.std(RegDataX)

    # add noise to the x data
    noise = np.random.normal(0,noisestd,datanum)
    RegDataX_noise = RegDataX_norm + noise
    
    if vis:
        plt.plot(RegDataX_noise, RegDataY, '.b')
        plt.plot(RegDataX_noise[labelInd], RegDataY[labelInd],'xr')
        plt.ylim(-1.5,1.5)
        plt.grid()
        plt.show()

    data_dist = RegDataY
    return (RegDataX_noise, RegDataY, labelFlag, data_dist)
    # return (RegDataX_noise, RegDataY, labelFlag, RegDataX_norm)

def test(regnet, datanum, cmpx, cmpy, logdir, epoch):
    (dataX, dataY, _, _) = dataPrepare(datanum, datanum)
    regnet.eval()

    predicty = []
    for (datax, datay) in zip(dataX, dataY):

        inputTensor = torch.Tensor([datax])
        inputVariable = Variable(inputTensor.unsqueeze(0), volatile = True)
        targetTensor = torch.Tensor([datay])
        targetVariable = Variable(targetTensor.unsqueeze(0))

        # forward + backward + optimize
        output = regnet(inputVariable.cuda())
        predicty.append(output.cpu().data[0][0])

    plt.plot(dataX, np.array(predicty), 'r.')
    plt.hold(True)
    plt.plot(cmpx, cmpy, 'bx')
    plt.ylim(-1.5,1.5)
    plt.grid()
    plt.savefig(join(logdir,'epoch'+str(epoch)+'.jpg'))
    plt.hold(False)
    # plt.show()
#    raw_input()
    regnet.train()

def groupPlot(datax, datay, group=10):
    datax, datay = np.array(datax), np.array(datay)
    if len(datax)%group>0:
        datax = datax[0:len(datax)/group*group]
        datay = datay[0:len(datay)/group*group]
    datax, datay = datax.reshape((-1,group)), datay.reshape((-1,group))
    datax, datay = datax.mean(axis=1), datay.mean(axis=1)
    return (datax, datay)


def train(datanum, labelnum, epochnum, hiddennum = 100, lr=0.1, showiter=10, batch = 10, alpha=10, lamb=0.01, thresh = 0.1, slepoch=0):

    logdir = 'data%d_label_%d_hidden_%d_lr%.5f_batch%d_alpha%d_lamb%.5f_thresh%.5f_bugfree' % (datanum, labelnum, 
        hiddennum, lr, batch, alpha, lamb, thresh )
    if not isdir(logdir):
        mkdir(logdir)

    regnet = RegNet(hiddennum = hiddennum)
    regnet.cuda()
    regnet.train()
    
    (dataX_all, dataY_all, labelFlag_all, dataDist_all) = dataPrepare(datanum, labelnum, vis=False, noisestd=0)
    labeled_selector = (labelFlag_all==1)
    dataX_labeled, dataY_labeled, labelFlag_labeled, dataDist_labeled = \
        dataX_all[labeled_selector], dataY_all[labeled_selector], labelFlag_all[labeled_selector], dataDist_all[labeled_selector]

    # TODO:
    dataX, dataY, labelFlag, dataDist = dataX_all, dataY_all, labelFlag_all, dataDist_all


    # training
    criterion = nn.L1Loss().cuda()
    optimizer = optim.Adam(regnet.parameters(),lr=lr)
#    optimizer = optim.SGD(regnet.parameters(), lr = lr)

#    test(regnet, datanum = 937, cmpx = dataX[labelFlag==1], cmpy = dataY[labelFlag==1])
    lossplot = []
    loss_label_plot = []
    loss_unlabel_plot = []
    for epoch in range(epochnum+slepoch):

        if epoch <slepoch:
            lamb_use = 0
            # dataX, dataY, labelFlag, dataDist = dataX_labeled, dataY_labeled, labelFlag_labeled, dataDist_labeled
        else:
            lamb_use = lamb

        running_loss = 0
        running_loss_label = 0
        running_loss_unlabel = 0
        inputOrder = np.random.permutation(datanum) # random order
        # inputOrder = np.arange(datanum) # sequencial order

        for ind in range(0, datanum, batch):

            dataind = inputOrder[ind:ind+batch]
#            if labelFlag[dataind]==0:
#                continue

            (datax, datay, labelflag, datadist) = (dataX[dataind], dataY[dataind], labelFlag[dataind], dataDist[dataind])

            if batch==1:
                inputTensor = torch.Tensor([datax])
                inputVariable = Variable(inputTensor.unsqueeze(0), requires_grad = True)
                targetTensor = torch.Tensor([datay])
                targetVariable = Variable(targetTensor.unsqueeze(0), requires_grad=False)
            else:
                inputTensor = torch.Tensor(datax)
                inputVariable = Variable(inputTensor.unsqueeze(1), requires_grad = True)
                targetTensor = torch.Tensor(datay)
                targetVariable = Variable(targetTensor.unsqueeze(1), requires_grad=False)

            optimizer.zero_grad()

            # forward + backward + optimize
            output = regnet(inputVariable.cuda())

            # calculate the MSE loss for labeled samples
            if labelflag.sum()>0:
                labelflag = torch.Tensor(labelflag).unsqueeze(1)
                labelInds = Variable((labelflag == 1), requires_grad=False).cuda()
                unlabelInds = Variable((labelflag == 0), requires_grad=False).cuda()
                targetVariable = targetVariable.cuda()
                output_label = output[labelInds]
                loss_label = criterion(output_label, targetVariable[labelInds])
            else: # no labeled data in this batch
                loss_label = Variable(torch.Tensor([0])).cuda()

            # clear the loss of those unlabeled samples
            # this needs batch>1
            loss_unlabel = Variable(torch.Tensor([0])).cuda()
            for ind1 in range(batch-1):
                for ind2 in range(ind1+1, batch):
                    w = abs(datadist[ind1] - datadist[ind2])
                    loss_unlabel = loss_unlabel + ((output[ind1]-output[ind2]).abs()-thresh).clamp(0) * exp(-alpha*w)
#                    print (output[ind1]-output[ind2]).abs().data

            loss = loss_label + lamb_use * loss_unlabel


            running_loss += loss.data[0]
            running_loss_label += loss_label.data[0]
            running_loss_unlabel += loss_unlabel.data[0]
            lossplot.append(loss.data[0])
            loss_label_plot.append(loss_label.data[0])
            loss_unlabel_plot.append(loss_unlabel.data[0])
            # print loss.data[0]

            loss.backward()
            optimizer.step()

#        if (ind+batch) % (showiter*batch) == 0:    # print every 20 mini-batches
        timestr = time.strftime('%m/%d %H:%M:%S',time.localtime())
        print(' [epoch-%d %d %s] loss: %.5f, loss_label: %.5f, loss_unlabel: %.5f ' %
        (epoch, ind+1, timestr, running_loss / showiter, 
         running_loss_label / showiter, running_loss_unlabel / showiter))
        # add to tensorboard
        # logger.scalar_summary('loss',running_loss/showiter,ind)

        running_loss = 0.0
        # if (ind)%snapshot==0:
        #   torch.save(posenet.state_dict(), paramName+'_'+str(ind)+'.pkl')
        #   # savemat(lossfilename+'.mat',{'loss':np.array(lossplot)})
        test(regnet, datanum = 937, cmpx = dataX[labelFlag==1], cmpy = dataY[labelFlag==1], logdir= logdir,epoch=epoch)

    plt.hold(False)
    lossplot = groupPlot(range(len(lossplot)),lossplot)
    plt.plot(lossplot[0], lossplot[1])
    plt.ylim(0,1)
    plt.grid()
    plt.savefig(join(logdir,'loss.jpg'))
    # plt.show()
    loss_label_plot = groupPlot(range(len(loss_label_plot)),loss_label_plot)
    plt.plot(loss_label_plot[0], loss_label_plot[1])
    plt.ylim(0,1)
    plt.grid()
    plt.savefig(join(logdir,'loss_label.jpg'))
    # plt.show()
    loss_unlabel_plot = groupPlot(range(len(loss_unlabel_plot)),loss_unlabel_plot)
    plt.plot(loss_unlabel_plot[0], loss_unlabel_plot[1])
    plt.ylim(0,5)
    plt.grid()
    plt.savefig(join(logdir,'loss_unlabel.jpg'))
    # plt.show()
    np.save(join(logdir,'loss'),lossplot)
    np.save(join(logdir,'loss_label'),loss_label_plot)
    np.save(join(logdir,'loss_unlabel'),loss_unlabel_plot)

if __name__ == "__main__":
    # for alpha in [5,10,20]:
    #     for thresh in [0.1,0.05,0.01,0.0]:
    #         for lamb in [0,0.01,0.1,0.5]:
    #             train(datanum = 100,labelnum = 20,epochnum = 500,hiddennum = 500, lr = 0.01, batch = 20, alpha = alpha, lamb = lamb, thresh=thresh)    

    # (dataX, dataY, labelFlag, dataDist) = dataPrepare(100, 10, vis=True, noisestd=0)
    train(datanum = 200,labelnum = 10,epochnum = 500,hiddennum = 500, lr = 0.01, batch = 20, alpha = 20, lamb = 0.1, thresh=0.05)