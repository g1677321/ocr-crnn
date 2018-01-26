#coding:utf-8
import torch
from torch.autograd import Variable
import utility.utils as utils
import utility.dataset as dataset
import utility.keys as keys
from PIL import Image
import argparse

import models.crnn as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--img', help='path to the image', default='./data/demo.png')
parser.add_argument('--model', help='path to pretrained model(You should check alphabet in keys applied to your model)', default='./models/pretrain/model_acc97.pth')
parser.add_argument('--alphabet', help='alphabet string', type=str)
parser.add_argument('--gpu', action='store_true', help='Use GPU or CPU only')
opt = parser.parse_args()
#print(opt)

def crnnSource():
    # Choose alphabet from arg or key.py(default)
    if opt.alphabet is None:
        alphabet = keys.alphabet
    else:
        alphabet = opt.alphabet

    converter = utils.strLabelConverter(alphabet)

    if torch.cuda.is_available() and opt.gpu:
        model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1)
        model = model.cuda()
    else:
        model = crnn.CRNN(32, 1, len(alphabet)+1, 256, 1)
        model = model.cpu()
    path = opt.model
    print('loading pretrained model from %s' % path)
    model.eval()
    model.load_state_dict(torch.load(path))
    return model, converter

def recognize(image):
   """
   crnn模型，ocr识别
   @@model,
   @@converter,
   @@im
   @@text_recs:text box

   """
   scale = image.size[1]*1.0 / 32
   w = image.size[0] / scale
   w = int(w)
   #print "im size:{},{}".format(image.size,w)
   transformer = dataset.resizeNormalize((w, 32))
   if torch.cuda.is_available() and opt.gpu:
       image = transformer(image).cuda()
   else:
       image = transformer(image).cpu()
        
   image = image.view(1, *image.size())
   image = Variable(image)
   model.eval()
   preds = model(image)
   _, preds = preds.max(2)
   preds = preds.transpose(1, 0).contiguous().view(-1)
   preds_size = Variable(torch.IntTensor([preds.size(0)]))
   raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
   sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
   if len(sim_pred)>0:
      if sim_pred[0]==u'-':
         sim_pred=sim_pred[1:]

   return raw_pred, sim_pred

model,converter = crnnSource()

image = Image.open(opt.img).convert('L')

raw_pred, sim_pred = recognize(image)
print('%-20s => %-20s' % (raw_pred, sim_pred))
