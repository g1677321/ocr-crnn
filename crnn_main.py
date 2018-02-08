# -*- coding: utf-8 -*-
from __future__ import print_function
import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from warpctc_pytorch import CTCLoss
import os
import utility.utils as utils
import utility.dataset as dataset
import utility.keys as keys
import models.crnn as crnn
import time
import json

parser = argparse.ArgumentParser()
parser.add_argument('--trainroot', help='path to dataset', default='./data/lmdb/train')
parser.add_argument('--valroot', help='path to dataset', default='./data/lmdb/val')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=256, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=0, help='number of GPUs to use')
parser.add_argument('--crnn', default='', help="path to pretrained-crnn (to continue training)")
parser.add_argument('--alphabet', help='alphabet string', type=str)
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
#parser.add_argument('--displayInterval', type=int, default=50, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=1000, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=50, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=1000, help='Interval to be displayed')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
opt = parser.parse_args()
print(opt)
ifUnicode=True

if opt.experiment is None:
    #print (time.strftime("%y%m%d_%H_%M_%S", time.localtime()))
    opt.experiment = 'expr/' + time.strftime("%y%m%d_%H_%M_%S", time.localtime())
os.system('mkdir -p expr')
os.system('mkdir -p {}'.format(opt.experiment))
os.system('mkdir -p {}/batch'.format(opt.experiment))

#計時
t = time.time()

opt.manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

train_dataset = dataset.lmdbDataset(root=opt.trainroot)
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None

print ("Traing set:{} samples".format(train_dataset.nSamples))

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio))
test_dataset = dataset.lmdbDataset(
    root=opt.valroot, transform=dataset.resizeNormalize((256, 32)))

ngpu = int(opt.ngpu)
nh = int(opt.nh)
if opt.alphabet is None:
    alphabet = keys.alphabet
else:
    alphabet = opt.alphabet
nclass = len(alphabet) + 1
nc = 1

converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

crnn = crnn.CRNN(opt.imgH, nc, nclass, nh, ngpu)
crnn.apply(weights_init)
if opt.crnn != '':
    print('loading pretrained model from %s' % opt.crnn)
    crnn.load_state_dict(torch.load(opt.crnn))
    # Change dims
    #pretrained_dict = torch.load(opt.crnn)
    #model_dict = crnn.state_dict()
    #pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    #model_dict.update(pretrained_dict)
    #crnn.load_state_dict(model_dict)
print(crnn)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

if opt.cuda:
    crnn.cuda()
    #crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
length = Variable(length)

# loss averager
loss_avg_step = utils.averager()
loss_avg_epoch = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999))
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters(), lr=opt.lr)
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=2):
    # VAL計時
    t_val = time.time()
    #print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    #max_iter = len(data_loader)
    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        if ifUnicode:
             cpu_texts = [ clean_txt(tx.decode('utf-8'))  for tx in cpu_texts]
        t, l = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, l)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        # pytorch < v0.1.2 bug used
        # preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        #print('pred={}, target={}'.format(sim_preds, text))
	for pred, target in zip(sim_preds, cpu_texts):
            #if pred == target.lower():
	    if pred.strip() == target.strip():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    '''for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))
    '''
    #print('n_correct={} max_iter={} opt.batchSize={}'.format(n_correct, max_iter, opt.batchSize))
    accuracy = float(n_correct) / float(max_iter * opt.batchSize)
    testLoss = loss_avg.val()
    t_val = time.time() - t_val
    print('VAL_LOG:Test loss: %f, accuray: %f, time: %f' % (testLoss, accuracy, t_val))
    return testLoss,accuracy

def clean_txt(txt):
    """
    filter char where not in alphabet with ' '
    """
    newTxt = u''
    for t in txt:
        if t in alphabet:
            newTxt+=t
        else:
            newTxt+=u' '
    return newTxt

def trainBatch(net, criterion, optimizer, flage=False):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    #decode utf-8 to unicode
    if ifUnicode:
        cpu_texts = [ clean_txt(tx.decode('utf-8'))  for tx in cpu_texts]
        
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, l)

    preds = crnn(image)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    cost = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    cost.backward()
    if flage:
        lr = 0.0001
        optimizer = optim.Adadelta(crnn.parameters(), lr=lr)
    optimizer.step()
    return cost

def delete(path):
    import os
    import glob
    paths = glob.glob(path+'/*.pth')
    for p in paths:
        os.remove(p)

#numLoss = 0##判断训练参数是否下降
#print("Batch count = {}".format(len(train_loader)))

for epoch in range(opt.niter):
    # Epoch計時
    t_epoch = time.time()
    train_iter = iter(train_loader)

    for step in range(len(train_loader)):
        # Step計時
        t_step = time.time()
        #print('The step{} ........\n'.format(step))
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg_step.add(cost)
        loss_avg_epoch.add(cost)

        #if step % opt.displayInterval == 0:
        #    print('[%d/%d][%d/%d] Loss: %f' %
        #          (epoch, opt.niter, step, len(train_loader), loss_avg_step.val()))
        #    loss_avg_step.reset()

        #if step % opt.displayInterval == 0:

        t_step = time.time() - t_step
        
        if step % opt.valInterval == 0 and step != 0:
            testLoss, accuracy = val(crnn, test_dataset, criterion)
            print("STEP_LOG: Epoch:{},step:{:>10},Test loss:{:>15},Accuracy:{:>15},Train loss:{:>15},Time:{:>10}".format(epoch,step,testLoss,accuracy,loss_avg_step.val(),t_step))
            logfile = open('{}/batch/model.log'.format(opt.experiment), 'a')
            logfile.write(json.dumps({'epoch': epoch, 'step': step, 'test loss': testLoss, 'accuracy': accuracy, 'train loss': loss_avg_step.val(), 'time': t_step})+'\n')
            logfile.close()

            loss_avg_step.reset()
            torch.save(crnn.state_dict(), '{}/batch/model_e{}_b{}.pth'.format(opt.experiment, epoch, step))

    t_epoch = time.time() - t_epoch

    # Save models each epochs
    testLoss, accuracy = val(crnn, test_dataset, criterion)
    print("Epoch_LOG: Epoch:{},Test loss:{:>15},Accuracy:{:>15},Train loss:{:>15},Time:{:>10}".format(epoch, testLoss, accuracy, loss_avg_epoch.val(), t_epoch))
    logfile = open('{}/model.log'.format(opt.experiment), 'a')
    logfile.write(json.dumps({'epoch': epoch, 'test loss': testLoss, 'accuracy': accuracy, 'train loss': loss_avg_epoch.val(), 'time': t_epoch})+'\n')
    logfile.close()
    loss_avg_epoch.reset()
    torch.save(crnn.state_dict(), '{}/model_e{}.pth'.format(opt.experiment, epoch))


print ("It takes time:{}s".format(time.time()-t))
