# Natural language processing - Spam filter

This project was a proof of achievement as part of my studies in artificial intelligence. We should implement a spam
filter based on Naive Bayes without using any library.

## Table of Contents

- [Task](#task)
- [Install](#install)
- [Result](#result)

## Task

Implement a spam filter that is based on naive Bayes:

- Create your own data set of messages, extract messages from your email box or find a data set online
- Implement naive Bayes, do not use a library such as scikit-learn
- Report accuracy, precision, recall and F1 on the test set


## Install

```sh
$ pip install -r requirements.txt
```

## Result

Basic natural language processing methods were used for the model such as:

- case intensity 
- stop words
- stemming
- laplacian/laplace smoothing


The output quality of the naive bayes classifier has the following values:

- Accuracy: 0.92
- Precision: 0.86
- Recall: 0.87
- F1: 0.86
