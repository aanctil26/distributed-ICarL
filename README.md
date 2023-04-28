# iCaRL-Updated-Distributed

This code has been refined and enhanced from the initial implementation, which can be found at the following link: https://github.com/srebuffi/iCaRL

As the original implementation has become outdated, this repository presents a modernized version that utilizes TensorFlow 2.0 and incorporates multi-GPU training capabilities.

For additional information, kindly refer to the original iCaRL code and accompanying research paper.

Code: https://github.com/srebuffi/iCaRL
Paper: https://arxiv.org/abs/1611.07725





To run the code on ImageNet1000, the data must be downloaded from either the ImageNet website or Kaggle:
https://www.kaggle.com/c/imagenet-object-localization-challenge





### To run iCaRL on ImageNet where hyperparameters are hard coded into main_resnet.py:
```
python iCaRL-Updated-Distributed/iCaRL-Tensorflow/main_resnet_tf.py
```
