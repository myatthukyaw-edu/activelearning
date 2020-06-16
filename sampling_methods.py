import numpy as np
from scipy.stats import entropy
from sklearn.utils import check_random_state

labels= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

class BaseSelectionFunction(object):

    def __init__(self):
        pass

    def select(self):
        pass


class RandomSelection(BaseSelectionFunction):

    @staticmethod
    def select(probas_val, initial_labeled_samples):
        random_state = check_random_state(0)
        selection = np.random.choice(probas_val.shape[0], initial_labeled_samples, replace=False)
        return selection

class KL_QBC(BaseSelectionFunction):
    def average_KL_divergence(probas_val, probas_val2, X_train, y_train, X_seedset, y_seedset):
      preds = []
      preds.append(probas_val)
      preds.append(probas_val2)
      print(probas_val.shape)
      print(probas_val2.shape)

      #stack so tar ka array list ko numpy arrary change tar
      consensus = np.mean(np.stack(preds), axis=0)
      print('consensus :',consensus)
      divergence = []
      for y_out in preds:
        divergence.append(entropy(consensus.T, y_out.T))
      #print('divergence values :', len(divergence),divergence)

      result = np.apply_along_axis(np.mean, 0, np.stack(divergence))
      #print('result after np along axis the divergence :', result.shape, result)
      # np.apply_along_axis(np.mean, 0, np.stack(divergence))

      #argsort Returns the indices that would sort an array.
      #argsort sort in increasing order but if -result then it sort in decreasing order.
      rankings = np.argsort(-result)[:1000]
      rankings
      #print(result[28324], result[13958], result[44722])

    #   X_train = np.concatenate((X_train, X_seedset[rankings, :]))
    #   y_train = np.concatenate((y_train, y_seedset[rankings]))

    #   X_seedset = np.delete(X_seedset, rankings, axis=0)
    #   y_seedset = np.delete(y_seedset, rankings, axis=0)

      #print("New train and seedset after ",X_train.shape, X_seedset.shape)
      return rankings

class vote_entropy_QBC():
    def vote_entropy(probas_val, probas_val2, X_train, y_train, X_seedset, y_seedset):
      #vote entropy
      preds = []
      probas_val_not_cat = np.argmax(probas_val,axis=1)
      print(probas_val_not_cat[0:20])
      probas_val2_not_cat = np.argmax(probas_val2, axis=-1)
      print(probas_val2_not_cat[0:20])

      preds.append(np.eye(len(labels))[probas_val_not_cat])
      preds.append(np.eye(len(labels))[probas_val2_not_cat])
      print(preds)
      #np.round(probas_val)

      # C = no of models
      votes = np.apply_along_axis(np.sum, 0, np.stack(preds)) / 2
      #print('votes :', votes.shape, votes)
      results = np.apply_along_axis(entropy, 1, votes)

      rankings = np.argsort(-results)[:1000]
      rankings

      #print(result[28324], result[13958], result[44722])
      #print(X_train.shape, X_seedset.shape)

    #   X_train = np.concatenate((X_train, X_seedset[rankings, :]))
    #   y_train = np.concatenate((y_train, y_seedset[rankings]))
    #   X_seedset = np.delete(X_seedset, rankings, axis=0)
    #   y_seedset = np.delete(y_seedset, rankings, axis=0)

      #print(X_augmented.shape, X_seedset.shape)
      return rankings

# class BaseSelectionFunction(object):

#     def __init__(self):
#         pass

#     def select(self):
#         pass


# class RandomSelection(BaseSelectionFunction):

#     @staticmethod
#     def select(probas_val, initial_labeled_samples):
#         random_state = check_random_state(0)
#         selection = np.random.choice(probas_val.shape[0], initial_labeled_samples, replace=False)

# #     print('uniques chosen:',np.unique(selection).shape[0],'<= should be equal to:',initial_labeled_samples)

#         return selection



# class UncertaintySampling(BaseSelectionFunction):
#     # def __init__(self):
#     #     super().__init__()
#     #     self.clf = clf
#     #     self.X_unlabeled = X_unlabeled

#     def uncertainty_sampling(probs):
#         #probs = clf.predict_proba(X_unlabeled)

#         if strategy == 'least_confident':
#             return 1 - np.amax(probs, axis=1)

#         elif strategy == 'max_margin':
#             margin = np.partition(-probs, 1, axis=1)
#             return -np.abs(margin[:,0] - margin[:, 1])

#         elif strategy == 'entropy':
#             return np.apply_along_axis(entropy, 1, probs)
 
# class QBC(BaseSelectionFunction):
#     def __query_by_committee(self, clf, X_unlabeled, strategy):
#         num_classes = len(clf[0].classes_)
#         print('num classes : ',num_classes)
#         C = len(clf)
#         preds = []

#         if strategy == 'vote_entropy':
#             for model in clf:
#                 #y_out = map(int, model.predict(X_unlabeled))
#                 y_out = model.predict(X_unlabeled)
#                 print('y_out', y_out)
#                 preds.append(np.eye(num_classes)[y_out])

#             votes = np.apply_along_axis(np.sum, 0, np.stack(preds)) / C
#             return np.apply_along_axis(entropy, 1, votes)

#         elif strategy == 'average_kl_divergence':
#             for model in clf:
#                 preds.append(model.predict_proba(X_unlabeled))

#             print('len(preds[0]) :', len(preds[0]))
#             print('np.stack(preds) : ', np.stack(preds).shape)
#             consensus = np.mean(np.stack(preds), axis=0)
#             divergence = []
#             for y_out in preds:
#                 divergence.append(entropy(consensus.T, y_out.T))
            
#             return np.apply_along_axis(np.mean, 0, np.stack(divergence))