Question Answering with Noisy-Contrastive Estimation with Deep Neural Networks.

Introduction
-------------
Given a pair of question and answer candidate, this tool can be used to predict how likely the candidate is the correct answer. It can also be extended to other semantic search tasks (i.e., Microblog Search, Duplicate Detection).

It's the open-source implementation of our CIKM'16 paper [1], in which we implement our noisy-contrastive estimation approach on an existing convolution neural network based approach [2]. Our model was evaluated on two standard QA datasets: TrecQA and WikiQA, achieving competitve or even state-of-the-art performance compared with previous work. We also cleaned the TrecQA dataset to two versions: raw and clean. For their difference and model details, please refer to our paper:
- ``Noisy-Contrastive Estimation for Answer Selection with Deep Neural Networks.``
- Jinfeng Rao, Hua He, Jimmy Lin, CIKM 2016

Getting Started
-----------
``1.`` Please install the Torch library by following instructions here: https://github.com/torch/distro

``2.`` Checkout our repo:
```
git clone https://github.com/marquis-wu/pairwise-nn.git
```

``3.`` Using our script to download and preprocess the Glove word embedding:
```
$ sh fetch_and_preprocess.sh
``` 
Please make sure your python version >= 2.7, otherwise you will encounter an exception when unzip the downloaded embedding file.

``4.`` Currently our tool only supports running on CPUs. 

``5.`` Before you run our model, please set the number of threads >= 5 for parallel processing. This is because our model need a large number of computation resource for training. In our cases, it usually takes 2-3 days to get some good results with the number of threads set to 5.

Running
--------
``1.`` There are several command line paramters to specify for running our models:
```
-dataset, the dataset you want to evaluate, which can be TrecQA and WikiQA. 
-version, the version of TrecQA dataset, which can be raw and clean. 
-neg_mode, the negative sampling strategy, 1 is for Random sampling, 2 is for MAX sampling, 3 is for MIX sampling 
-num_pairs, the number of negative samples, can be any reasonable value, by default it's set to 8
```

``2.`` To evaluate on the TrecQA raw dataset, with MAX sampling and number of negative pairs as 8, run:
```
$ th PairwiseTrainQA.lua -dataset TrecQA -version raw -neg_mode 2 -num_pairs 8
```
To evaluate on the TrecQA clean dataset, simply change -version to clean.
Similarly, if you want to evaluate on the WikiQA dataset, change -dataset to WikiQA.
You can also change the -neg_mode and -num_pairs to select different sampling strategies or negative pairs.

``3.`` To run the base convolutional neural network model in [2], please follow the same parameter setting:
```
$ th trainQA.lua -dataset TrecQA -version raw
```

Results
-------
You should be able to reproduce some scores close to the numbers in below tables (-num_pairs is set to 8 by default):

``1. TrecQA raw`` 

TrecQA raw       |  MAP   |  MRR
-----------------|--------|------
BaseConvNet [2]  | 0.762  | 0.830
Pairwise(Random) | 0.765  | 0.810
Pairwise(MAX)    | 0.780  | 0.835
Pairwise(MIX)    | 0.763  | 0.813

``2. TrecQA clean`` 

TrecQA clean     |  MAP   |  MRR
-----------------|--------|------
BaseConvNet [2]  | 0.777  | 0.836
Pairwise(Random) | 0.768  | 0.831
Pairwise(MAX)    | 0.801  | 0.877
Pairwise(MIX)    | 0.798  | 0.872

``3. WikiQA`` 

WikiQA           |  MAP   |  MRR
-----------------|--------|------
BaseConvNet [2]  | 0.693  | 0.709
Pairwise(Random) | 0.677  | 0.697
Pairwise(MAX)    | 0.681  | 0.705
Pairwise(MIX)    | 0.685  | 0.706
Though the numbers above don't outperform the base ConvNet model [2] in WikiQA dataset, still they are close. Our best score in WikiQA dataset are 0.701(MAP), 0.718(MRR), which is obtained in the setting of MAX sampling and num_pairs as 10.

Reference
--------
``[1]. Noisy-Contrastive Estimation for Answer Selection with Deep Neural Networks, Jinfeng Rao, Hua He, Jimmy Lin, CIKM 2016`` 

``[2]. Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks, Hua He, Kevin Gimpel, and Jimmy Lin, EMNLP 2014`` 
