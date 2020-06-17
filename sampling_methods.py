import numpy as np
from scipy.stats import entropy
from sklearn.utils import check_random_state

labels= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class RandomSelection:
    def __init__(self):
      pass

    #@staticmethod
    def select(self, probas_val, initial_labeled_samples):
        random_state = check_random_state(0)
        selection = np.random.choice(probas_val.shape[0], initial_labeled_samples, replace=False)
        return selection

class QBC:

    def __init__(self):
        pass
    def average_KL_divergence(self, probas_val, probas_val2, X_train, y_train, X_seedset, y_seedset, samples):
      preds = []
      preds.append(probas_val)
      preds.append(probas_val2)

      #change normal arry to numpy array with stack
      consensus = np.mean(np.stack(preds), axis=0)
      print('consensus :',consensus)
      divergence = []
      for y_out in preds:
        divergence.append(entropy(consensus.T, y_out.T))

      result = np.apply_along_axis(np.mean, 0, np.stack(divergence))

      #argsort Returns the indices that would sort an array.
      #argsort sort in increasing order but if use (-result) then it sort in decreasing order.
      rankings = np.argsort(-result)[:samples]

      return rankings

    def vote_entropy(self, probas_val, probas_val2, X_train, y_train, X_seedset, y_seedset, samples):
      #vote entropy
      preds = []

      probas_val_not_cat = np.argmax(probas_val,axis=1)
      probas_val2_not_cat = np.argmax(probas_val2, axis=-1)

      preds.append(np.eye(len(labels))[probas_val_not_cat])
      preds.append(np.eye(len(labels))[probas_val2_not_cat])

      # C = no of models
      C = 2
      votes = np.apply_along_axis(np.sum, 0, np.stack(preds)) / C
      results = np.apply_along_axis(entropy, 1, votes)

      rankings = np.argsort(-results)[:samples]

      return rankings

class uncertainty_sampling:
    def init(self):
      pass

    def least_confident(self, probs, samples):
        #get the least uncertain value by subtracting the most uncertain value from 100% or 1
        scores = 1 - np.amax(probs, axis=1)
        #get the index of the least uncertain 
        rankings = np.argsort(-scores)[:samples]
        return rankings
    
    def max_margin(self, probs, samples):
        margin = np.partition(-probs, 1, axis=1)
        scores = -np.abs(margin[:,0] - margin[:, 1])
        rankings = np.argsort(-scores)[:samples]
        return rankings

    def entropy(self, probs, samples):
        scores = np.apply_along_axis(entropy,1, probs)
        rankings = np.argsort(-scores)[:samples]
        return rankings