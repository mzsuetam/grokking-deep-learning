#!/bin/bash

DIRNAME="data"

if [ ! -d ".git" ]; then
  echo "Please run this script from the root of the repository."
else
  if [ ! -d $DIRNAME ]; then
    mkdir $DIRNAME
  fi


  # For chapter 11
  wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz -O $DIRNAME/aclImdb_v1.tar.gz
  tar -xvf $DIRNAME/aclImdb_v1.tar.gz -C $DIRNAME
  rm $DIRNAME/aclImdb_v1.tar.gz

  # For chapter 12
  wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-1.tar.gz -O $DIRNAME/tasks_1-20_v1-1.tar.gz
  tar -xvf $DIRNAME/tasks_1-20_v1-1.tar.gz -C $DIRNAME
  rm $DIRNAME/tasks_1-20_v1-1.tar.gz

  # For chapter 14
  mkdir $DIRNAME/shakespeare -p
  wget https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt -O $DIRNAME/shakespeare/shakespeare.txt

fi
