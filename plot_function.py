import matplotlib.pyplot as plt
import numpy as np

def plot_fun(X, y):
  #X_train, X_test = X_train.astype('int32')*255. , X_test.astype('int8')*255.

  # Define the labels of the dataset
  labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']

  # Let's view more images in a grid format
  # Define the dimensions of the plot grid 
  W_grid = 10
  L_grid = 10

  # fig, axes = plt.subplots(L_grid, W_grid)
  # subplot return the figure object and axes object
  # we can use the axes object to plot specific figures at various locations

  fig, axes = plt.subplots(L_grid, W_grid, figsize = (17,17))

  axes = axes.ravel() # flaten the 15 x 15 matrix into 225 array

  n_train = len(X) # get the length of the train dataset

  # Select a random number from 0 to n_train
  for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

      # Select a random number
      index = np.random.randint(0, n_train)
      # read and display an image with the selected index    
      axes[i].imshow(X[index,1:])
      label_index = int(y[index])
      axes[i].set_title(labels[label_index], fontsize = 8)
      axes[i].axis('off')

  plt.subplots_adjust(hspace=0.4)