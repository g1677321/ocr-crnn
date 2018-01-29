Convolutional Recurrent Neural Network
======================================

Convolutional Recurrent Neural Network (CRNN) for OCR in pytorch.
Origin software could be found in [crnn](https://github.com/bgshih/crnn)

Setup
-----
Check the commands applied to your system environment in ``setup,sh``.
([virtualenv](https://virtualenv.pypa.io/en/stable/installation/) and [conda](https://conda.io/docs/user-guide/install/index.html) are both your good options)

    sh setup.sh

Dependence
----------
* [warp_ctc_pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
* lmdb 

Run demo
--------
A demo program can be found in ``./demo.py``. Before running the demo, download a pretrained model
from [Baidu Netdisk](https://pan.baidu.com/s/1pLbeCND) or [Dropbox](https://www.dropbox.com/s/dboqjk20qjkpta3/crnn.pth?dl=0). 
This pretrained model is from [meijieru](https://github.com/meijieru/crnn.pytorch).
Put the downloaded model file ``crnn.pth`` into directory ``model/pretrain/``. Then launch the demo by:

    python demo.py

The demo reads an example image and recognizes its text content.

Example image:
![Example Image](./data/demo.png)

Expected output:
    loading pretrained model from ./models/pretrain/crnn.pth
    a-----v--a-i-l-a-bb-l-ee-- => available



Train a new model
-----------------
1. Construct dataset following ``utility/create_dataset.py``. (For training with variable length, please sort the image according to the text length.)
2. ``python crnn_main.py [--param val]``. Explore ``crnn_main.py`` for details.
