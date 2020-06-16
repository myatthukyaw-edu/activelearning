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
from sampling_methods import KL_QBC, vote_entropy_QBC, RandomSelection
#from sampling_methods import UncertaintySampling
#import get_k_random_samples
import plot_function
import sampling_methods
#from model1 import train_resnet50

labels= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#get cifer dataset
X_train_full, y_train_full, X_test, y_test = get_dataset()

#get initial 1000 samples
permutation, X_train, y_train = get_k_random_samples(X_train_full.shape[0],1000, X_train_full, y_train_full)

print("Train set size X :",X_train.shape)
print(y_train.shape)

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
iterations = 0
probas_val1 = train_model1(X_train, y_train, X_seedset,X_test, y_test, labels)

probas_val2 = train_model2(X_train, y_train, X_seedset, X_test, y_test, labels)

#choose uncertain samples
#uncertain_samples = UncertaintySampling.uncertainty_sampling(probas_val)
# uncertain_samples = 1 - np.amax(probas_val, axis=1)
# np.amax()
# print('Uncertain_samples :', uncertain_samples.shape, uncertain_samples )

selection_ranking = KL_QBC.average_KL_divergence(probas_val1, probas_val2,X_train, y_train, X_seedset, y_seedset)

X_train = np.concatenate((X_train, X_seedset[selection_ranking, :]))
y_train = np.concatenate((y_train, y_seedset[selection_ranking]))

X_seedset = np.delete(X_seedset, selection_ranking, axis=0)
y_seedset = np.delete(y_seedset, selection_ranking, axis=0)

print('After selecting samples based on Samping methods on round',iterations+1,' :')
print('Train : ', X_train.shape, y_train.shape)
print('Seedset  : ', X_test.shape, y_test.shape)

y_train_bin = y_train.reshape((y_train.shape[0],))
bin_count = np.bincount(y_train_bin.astype('int64'))
unique = np.unique(y_train_bin.astype('int64'))
print ('Unique(labels):', bin_count, unique )

iterations = 1
while len(X_seedset) > 1 :
    print('Round ', iterations+1)
    #normalize
    #   normalizer = Normalize()
    #   X_train, y_train = normalizer.normalize( X_train, y_train)
    #   X_test, y_test = normalizer.normalize(X_test, y_test)
    #   X_seedset, y_seedset = normalizer.normalize(X_seedset, y_seedset)

    #train
    print("Model 1 training")
    probas_val = train_model1(X_train, y_train, X_seedset,X_test, y_test, labels)
    print("Model 2 training")
    probas_val2 = train_model2(X_train, y_train, X_seedset, X_test, y_test, labels)


    #get uncertain examples
    X_train, y_train, X_seedset, y_seedset = KL_QBC.average_KL_divergence(probas_val, probas_val2,X_train, y_train, X_seedset, y_seedset)
    #print('Uncertain_samples :', uncertain_samples)

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
    print('Seedset  : ', X_test.shape, y_test.shape)

    y_train_bin = y_train.reshape((y_train.shape[0],))
    bin_count = np.bincount(y_train_bin.astype('int64'))
    unique = np.unique(y_train_bin.astype('int64'))
    print ('Unique(labels):', bin_count, unique )

  
    iterations += 1
    print('-------Finish training anthor 1000 samples on round ',iterations+1,'----------------')