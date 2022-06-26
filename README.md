# Speech2Vec

## Overview
Reenactment of Speech2Vec from [1]. Unfortunately, model architecture for the paper is not fully explained in detail, and many of others' implementation of _Speech2Vec_ varied. This is a best of my attempt to accurately model the architecture.


_Note on Model Architecture_
1. _Conditional GRU introduced from  is used from [2]._
2. _Single decoder for each corresponding window._
3. _Little teacher forcing learning._
4. _Feed zeros-vector as first input of decoder._

Throughout training and analyzing, decoder seemed to output a value from previous step and give a trivial result. I believe this is due to having MSE loss function and taking a mean of data disregarding its variance. If not, please help to debug.


## Installation
Clone the repository and install the required packages.

```bash
git clone https://github.com/yjang43/Speech2Vec.git
cd ./Speech2Vec
pip install -r requirements.txt
```

## Data Preparation
The paper uses LibriSpeech dataset, which can be downloaded from [here](https://www.openslr.org/12).
Then, each audio file from the dataset needs to be forced aligned. 
The mapping for force alignment with Montreal Forced Aligner can be obtained from [here](https://github.com/CorentinJ/librispeech-alignments).
Make sure the directories are merged as directed by [here](https://github.com/CorentinJ/librispeech-alignments/blob/master/README.md).

Then, you need to extract vocabularies given the dataset and its mapping, which will be saved in the following structure:

```bash
[DATA_DIR]
    +- [READER_ID]
    +- ...
        +- [CHAPTER_ID]
        +- ...
            +- [SENTENCE_ID]
            +- ...
                +- [WORD_ID].wav
                +- ...

    +- words.txt
    +- index.txt

```

_words.txt_ is a list of words and _index.txt_ is a map to the location of corresponding file.


To extract vocabularies, run the script.
```bash
python ./word_alignment.py --data_dir ./data \
                           --librispeech_path ./LibriSpeech/test-clean
```


## Train

Start training the model with the script.
```bash
python ./train.py --epochs 200 \      # the number of epoch for paper is set to 200
                  --device cuda:0 \   # default is cpu
                  ...                 # set other arguments

```
Train time takes about 8 hours for 10 epochs in NVIDIA GTX 1080 Ti, but note that I experienced bottleneck from loading data due to inferior CPU.


## Analysis

TBA



## Reference
[1] C. Yu-An and J. Glass, "Speech2vec: A sequence-to-sequence framework for learning word embeddings from speech", INTERSPEECH, pp. 811-815, 2018.

[2] S. Subramanian, A. Trischler, Y. Bengio, and C. Pal, “Learning general purpose distributed sentence representations via large scale multi-task learning,” in ICLR, 2018.
