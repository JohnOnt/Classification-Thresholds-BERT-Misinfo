#!/bin/bash

#------------------------------------------------
# Three Cats
#------------------------------------------------

# Run basic BERT
python model_trainer.py --num_labels 3 --epochs 10

# Run BERT Large
python model_trainer.py --num_labels 3 --tokenizer 'bert-large-uncased' --model 'bert-large-uncased' --epochs 10

# Run BERTweet Large
python model_trainer.py --num_labels 3 --tokenizer 'vinai/bertweet-large' --model 'vinai/bertweet-large' --epochs 10


#------------------------------------------------
# Two Cats
#------------------------------------------------

# Run basic BERT
python model_trainer.py --num_labels 2 --epochs 10

# Run BERT Large
python model_trainer.py --num_labels 2 --tokenizer 'bert-large-uncased' --model 'bert-large-uncased' --epochs 10

# Run BERTweet Large
python model_trainer.py --num_labels 2 --tokenizer 'vinai/bertweet-large' --model 'vinai/bertweet-large' --epochs 10