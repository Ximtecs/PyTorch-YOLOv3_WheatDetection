# PyTorch-YOLOv3
A minimal PyTorch implementation of YOLOv3, with support for training, inference and evaluation.

This is a fork of https://github.com/eriklindernoren/PyTorch-YOLOv3 customized for the https://www.kaggle.com/c/global-wheat-detection kaggle competition.

Note that the README files have been cut down to only include decription for setup, training and testing on the kaggle competitions dataset.
For full description of the code, I refer the the original https://github.com/eriklindernoren/PyTorch-YOLOv3
## Installation
##### Clone and install requirements
    $ git clone https://github.com/Ximtecs/PyTorch-YOLOv3_WheatDetection
    $ cd PyTorch-YOLOv3_WheatDetection/
    $ conda install -c conda-forge --file requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh
    (Note that the default implementation only uses the tiny-weights)
    
#### Download Global_Wheat_Competition
    If you want to use the kaggle API you need to setup username and key.
    This can be done by creating a json file in:    $Home/.kaggle/kaggle.json
    Afterwards, run the comand:

    $ chmod 600 $Home/.kagggle/kaggle.json
    
    Once the username and key is set up, run the following command:

    $ cd data/WheatDetection/
    $ bash get_Wheat.sh

## Train/test/detect
    Training, testing and detection, has all been setup to use the data as setup in the previous step. 
    To run these simply run:

    $ python train.py
    $ python testing.py
    $ python detection.py

    Note that the train.py as default takes the 50th checkpoint as input. 
    As such, it must run for at least 50 iterations or you should change the weights.
    The detection.py takes as default the target images that requiere object detection for the competitions



### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```
