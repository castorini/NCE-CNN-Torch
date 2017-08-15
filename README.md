# Noise-Contrastive Estimation for Answer Selection with Convolutional Neural Networks

This repo contains the Torch implementation of noise-contrastive estimation approach for answer selection in question answering with Convolutional Neural Networks, described in the following paper:

+ Jinfeng Rao, Hua He, and Jimmy Lin. [Noise-Contrastive Estimation for Answer Selection with Deep Neural Networks.](http://dl.acm.org/citation.cfm?id=2983872) *Proceedings of the 25th ACM International on Conference on Information and Knowledge Management (CIKM 2016)*, pages 1913-1916.

Our model was evaluated on two standard QA datasets: TrecQA and WikiQA. On TrecQA, we achieved the [best reported results at that time](https://aclweb.org/aclwiki/Question_Answering_(State_of_the_art)). Another contribution of this paper is to clarify the distinction between the *raw* and *clean* versions of the TrecQA test set.

Getting Started
-----------
``1.`` Please install the Torch library by following instructions here: https://github.com/torch/distro

``2.`` Checkout our repo:
```
git clone https://github.com/Jeffyrao/pairwise-neural-network.git
```

``3.`` Using following script to download and preprocess the Glove word embedding:
```
$ sh fetch_and_preprocess.sh
``` 
Please make sure your python version >= 2.7, otherwise you will encounter an exception when unzip the downloaded embedding file.

``4.`` Currently our tool only supports running on CPUs. 

``5.`` Before you run our model, please set the number of threads >= 5 for parallel processing. This is because our model need a large number of computation resource for training. 
```
$ export OMP_NUM_THREADS=5
```

Running
--------
``1.`` There are several command line paramters to specify for running our model:
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
Similarly, if you want to evaluate on the WikiQA dataset, change -dataset to WikiQA (don't need to set the -version).
You can also change the -neg_mode and -num_pairs to select different sampling strategies or negative pairs.

``3.`` To run the base convolutional neural network model in [2], please follow the same parameter setting:
```
$ th trainQA.lua -dataset TrecQA -version raw
```

Results
-------
You should be able to reproduce some scores close to the numbers in tables below (-num_pairs is set to 8 by default):

``1. TrecQA raw`` 

TrecQA raw       |  MAP   |  MRR
-----------------|--------|------
SentLevel [2]    | 0.762  | 0.830
Pairwise(Random) | 0.765  | 0.810
Pairwise(MAX)    | 0.780  | 0.835
Pairwise(MIX)    | 0.763  | 0.813

``2. TrecQA clean`` 

TrecQA clean     |  MAP   |  MRR
-----------------|--------|------
SentLevel [2]    | 0.777  | 0.836
Pairwise(Random) | 0.768  | 0.831
Pairwise(MAX)    | 0.801  | 0.877
Pairwise(MIX)    | 0.798  | 0.872

``3. WikiQA`` 

WikiQA           |  MAP   |  MRR
-----------------|--------|------
SentLevel [2]    | 0.693  | 0.709
Pairwise(Random) | 0.677  | 0.697
Pairwise(MAX)    | 0.681  | 0.705
Pairwise(MIX)    | 0.682  | 0.697

Though the numbers above don't outperform the base ConvNet model [2] in WikiQA dataset, still they are close. Our best scores in WikiQA dataset are 0.701(MAP), 0.718(MRR), which were obtained in the setting of MAX sampling and num_pairs as 10.

Reference
--------
``[1]. Noisy-Contrastive Estimation for Answer Selection with Deep Neural Networks, Jinfeng Rao, Hua He, Jimmy Lin, CIKM 2016`` 

``[2]. Multi-Perspective Sentence Similarity Modeling with Convolutional Neural Networks, Hua He, Kevin Gimpel, and Jimmy Lin, EMNLP 2014`` 
