Question Answering with Noisy-Contrastive Estimation with Deep Neural Networks.

Introduction
-------------
Given a question and a pair of answer candidates, this tool can be used to predict which answer is more likely to be the correct answer. It can also be extended to other semantic search tasks (i.e., Microblog Search, Duplicate Detection).

It's the open-source implementation of our CIKM'16 paper [1], in which we implement our noisy-contrastive estimation approach on an existing convolution neural network based approach [2]. Our model was evaluated on two standard QA datasets: TrecQA and WikiQA, achieving competitve or even state-of-the-art performance compared with previous work. We also cleaned the TrecQA dataset to two versions: raw and clean. For their difference, please refer to our paper:
- [1] ``Noisy-Contrastive Estimation for Answer Selection with Deep Neural Networks.``
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

``4.`` Currently our tool only supports running on CPUs. We recommend you to install the INTEL MKL library so that Torch can run much faster on CPUs. 

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

``2.`` To run our approach on the TrecQA raw dataset, with MAX sampling and number of negative pairs as 8:
```
$ th PairwiseTrainQA.lua -dataset TrecQA -version raw -neg_mode 2 -num_pairs 8
```
If we want to evaluate on the TrecQA clean dataset, simply change -version to clean.
Similarly, if you want to evaluate on the WikiQA dataset, change -dataset to WikiQA.
You can also change the -neg_mode and -num_pairs to select different sampling strategies or negative pairs.

``3.`` To run the base convolutional neural network model in [2], please follow the same parameter setting:
```
$ th trainQA.lua -dataset TrecQA -version raw
```

Results
-------


Reference
--------
[1] ``Noisy-Contrastive Estimation for Answer Selection with Deep Neural Networks,`` Jinfeng Rao, Hua He, Jimmy Lin, CIKM 2016

[2] ``Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks,`` Hua He, Kevin Gimpel, and Jimmy Lin, EMNLP 2014
