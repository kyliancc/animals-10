# animals-10

An AI experiment of Animals-10 dataset from kaggle.

# Introduction

We use ResNet-34 model for classification.

Paper link: https://arxiv.org/pdf/1512.03385 (Deep Residual Learning for Image Recognition)

# Download and Setup Dataset

Make sure you have installed kaggle previously.

Run `data/download.sh`, then the dataset directory and index JSON files will appear in the `data` directory.
They're required for training.

# Start Training

Run the command below:
~~~
python train.py
~~~

There's some arguments you can set:
- `-e`: (_int_) How many epochs to train
- `-b`: (_int_) Batch size
- `-w`: (_int_) Number of workers of DataLoader
- `-l`: (_float_) Learning rate
- `-f`: (_str_) Load a model to continue training
- `-v`: (_int_) How many iterations per validation
- `-s`: (_int_) How many iterations per save

# Continue Training

Run the command below:
~~~
python train.py -f [your_checkpoint]
~~~

# Validate
~~~
python val.py -f [your_checkpoint]
~~~

# Result

I got accuracy 72.13%, where batch size is 64 and trained 6400 iterations.
