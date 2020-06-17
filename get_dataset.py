from tensorflow.keras.datasets import cifar10

trainset_size = 50000
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']

def get_dataset():
  (X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()
  # summarize loaded dataset
  print('Cifar Train: X=%s, y=%s' % (X_train_full.shape, y_train_full.shape))
  print('Cifar Test: X=%s, y=%s' % (X_test.shape, y_test.shape))
  return (X_train_full, y_train_full, X_test, y_test)

class Normalize(object):
    
    def normalize(self, X, y):
        X = X.astype('float32')/255.
        y = y.astype('int32')
        return (X, y) 
    
    def inverse(self, X, y):
        X = X.astype('float32')*255.
        y = y.astype('int32')
        return (X, y) 