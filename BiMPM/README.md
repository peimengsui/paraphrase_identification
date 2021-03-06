# BiMPM: Bilateral Multi-Perspective Matching for Natural Language Sentences

## Most content of this directory is forked from this [repo](https://arxiv.org/pdf/1702.03814.pdf).

## Description
This directory includes the source code for natural language sentence matching. 
Basically, the program will take two sentences as input, and predict a label for the two input sentences.


## Requirements
* python 2.7
* tensorflow 0.12


## Training
You can find the training script at BiMPM/src/SentenceMatchTrainer.py

To see all the **optional arguments**, just run
> python BiMPM/src/SentenceMatchTrainer.py --help

Here is an example of how to train a very simple model:
> python  BiMPM/src/SentenceMatchTrainer.py --train\_path train.tsv --dev\_path dev.tsv --test\_path test.tsv --word\_vec_path wordvec.txt --suffix sample --fix\_word\_vec --model\_dir models --MP\_dim 20 

To get a better performance on your own datasets, you need to play with other arguments. Here is one example of the command line [configuration](https://drive.google.com/file/d/0B0PlTAo--BnaQ3N4cXR1b0Z0YU0/view?usp=sharing) I used in my experiments.

## Testing
You can find the testing script at BiMPM/src/SentenceMatchDecoder.py


To see all the **optional arguments**, just run
> python BiMPM/src/SentenceMatchDecoder.py --help

Here is an example of how to test your model:
> python  BiMPM/src/SentenceMatchDecoder.py --in\_path test.tsv --word\_vec\_path wordvec.txt --mode prediction --model\_prefix models/SentenceMatch.sample --out\_path test.prediction

The SentenceMatchDecoder.py can run in two modes:
* prediction: predicting the label for each sentence pair
* probs: outputting probabilities of all labels for each sentence pair

