import os
import time
import datetime
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.datasets import cifar10

from sklearn.utils import check_random_state
#from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from get_dataset import get_dataset, Normalize
from get_k_random_samples import get_k_random_samples
from model1 import train_model1
from model2 import train_model2
from sampling_methods import QBC, RandomSelection, uncertainty_sampling
import plot_function
import sampling_methods

labels= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#define no of samples (to select from pool or seed set) per round
samples = 2000
trainset_size = 50000

#store the accuray and losses in a dataframe
recordmodel1 = pd.DataFrame(columns=('TrainDS','Seedset', 'Train_Accuracy', 'Train_Loss', 'Val_Accuracy', 'Val_Loss', 'Test_Accuracy', 'Test_Loss' ))
recordmodel2 = pd.DataFrame(columns=('TrainDS','Seedset', 'Train_Accuracy', 'Train_Loss', 'Val_Accuracy', 'Val_Loss', 'Test_Accuracy', 'Test_Loss' ))

#get cifar dataset
X_train_full, y_train_full, X_test, y_test = get_dataset()

#get initial 2000 samples
permutation, X_train, y_train = get_k_random_samples(X_train_full.shape[0],samples, X_train_full, y_train_full)

print("Train set size X :",X_train.shape)
print(y_train.shape)

#deine the seedset or pool
X_seedset = np.array([])
y_seedset = np.array([])
X_seedset = np.copy(X_train_full)
X_seedset = np.delete(X_seedset, permutation, axis=0)
y_seedset = np.copy(y_train_full)
y_seedset = np.delete(y_seedset, permutation, axis=0)
print ('Seed set (Pool) :', X_seedset.shape, y_seedset.shape)

#Normalize
normalizer = Normalize()
X_train, y_train = normalizer.normalize( X_train, y_train)
X_test, y_test = normalizer.normalize(X_test, y_test)
X_seedset, y_seedset = normalizer.normalize(X_seedset, y_seedset)

#train model
#if u r using QBC approach, use both models.
#if  r using uncertainty sampling, use only one model.
iterations = 0
probas_val1 = train_model1(X_train, y_train, X_seedset,X_test, y_test, labels, iterations)
probas_val2 = train_model2(X_train, y_train, X_seedset, X_test, y_test, labels, iterations)

#choose uncertain samples
#QBC
#qbc = QBC()
#selection_ranking = qbc.vote_entropy(probas_val1, probas_val2,X_train, y_train, X_seedset, y_seedset, samples)

#uncertainty sampling 
us = uncertainty_sampling()
#pass the predict values of the desired model as first parameter 
selection_ranking = us.least_confident(probas_val1, samples)

#random selection
#rs = RandomSelection()
#random_selection = rs.select(probas_val, samples)

#get how many uncertain samples are selected from which classes
selected_samples = y_seedset[selection_ranking]
selected_samples = selected_samples.reshape((selected_samples.shape[0],))
bin_count = np.bincount(selected_samples)
unique = np.unique(selected_samples)
print ('Selected Uncertainty samples :', bin_count, "from each classes ",unique )

#add the selected samples to the training set
X_train = np.concatenate((X_train, X_seedset[selection_ranking, :]))
y_train = np.concatenate((y_train, y_seedset[selection_ranking]))

#delete selected sampling from the seeed set or pool
X_seedset = np.delete(X_seedset, selection_ranking, axis=0)
y_seedset = np.delete(y_seedset, selection_ranking, axis=0)

print('\nAfter selecting samples based on Samping methods on round',iterations+1,' :')
print('Train : ', X_train.shape, y_train.shape)
print('Seedset  : ', X_seedset.shape, y_seedset.shape)

#print the total number of samples in each class in training set
y_train_bin = y_train.reshape((y_train.shape[0],))
bin_count = np.bincount(y_train_bin.astype('int64'))
unique = np.unique(y_train_bin.astype('int64'))
print ('Total number of samples :', bin_count, ' in each classes',unique )

iterations = 1
while len(X_seedset) > 1 :
    print('\n-------Round ',iterations+1,'----------------')
    #normalize
    #   normalizer = Normalize()
    #   X_train, y_train = normalizer.normalize( X_train, y_train)
    #   X_test, y_test = normalizer.normalize(X_test, y_test)
    y_train_bin = y_train.reshape((y_train.shape[0],))
    bin_count = np.bincount(y_train_bin.astype('int64'))
    unique = np.unique(y_train_bin.astype('int64'))
    print ('Unique(labels):', bin_count, unique )
    #   X_seedset, y_seedset = normalizer.normalize(X_seedset, y_seedset)

    #train
    probas_val1 = train_model1(X_train, y_train, X_seedset,X_test, y_test, labels, iterations)
    probas_val2 = train_model2(X_train, y_train, X_seedset, X_test, y_test, labels, iterations)


    #get uncertain examples
    selection_ranking = us.least_confident(probas_val1, samples)

    # normalization needs to be inversed and recalculated based on the new train and test set.
    #   X_train, y_train = normalizer.inverse(X_train, y_train)
    #   X_test, y_test = normalizer.inverse(X_test, y_test)
    #   X_seedset, y_seedset = normalizer.inverse(X_seedset, y_seedset)
    X_train = np.concatenate((X_train, X_seedset[selection_ranking, :]))
    y_train = np.concatenate((y_train, y_seedset[selection_ranking]))

    X_seedset = np.delete(X_seedset, selection_ranking, axis=0)
    y_seedset = np.delete(y_seedset, selection_ranking, axis=0)

    print('After selecting samples based on Samping methods on round',iterations+1,' :')
    print('Train : ', X_train.shape, y_train.shape)
    print('Seedset  : ', X_seedset.shape, y_seedset.shape)

    y_train_bin = y_train.reshape((y_train.shape[0],))
    bin_count = np.bincount(y_train_bin.astype('int64'))
    unique = np.unique(y_train_bin.astype('int64'))
    print ('Total number of samples :', bin_count, ' in each classes',unique )
    
    print('-------Finish training anthor 1000 samples on round ',iterations+1,'----------------')
    iterations += 1
    