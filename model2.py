import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPool2D, Dropout, MaxPooling2D, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.keras.callbacks import ModelCheckpoint
#%load_ext tensorboard

def train_model2( X_train_org, y_train_org,X_seedset, X_test, y_test, labels, iterations):

        X_train, X_val, y_train, y_val = train_test_split( X_train_org, y_train_org, test_size=0.3, random_state=42)
        print('X_train :', X_train.shape)
        print('y train :', y_train.shape)
        print('X val :', X_val.shape)
        print('y val:', y_val.shape)

        filepath = 'modelfiles/model_2.h5'
        cp = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')

        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model = tf.keras.Sequential([
            
            Conv2D(48, kernel_size=(3,3), activation='relu', padding='same', input_shape=(32,32,3)),
            MaxPooling2D((2,2),(2,2)),
            
            Conv2D(96, kernel_size=(3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2),(2,2)),
            
            Conv2D(192,kernel_size=(3,3), activation='relu', padding='same'),
            Conv2D(192,kernel_size=(3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2),(2,2)),
            
            Conv2D(256,kernel_size=(3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2),(2,2)),

            Flatten(),
            Dense(512, activation='tanh'),
            Dense(256, activation='tanh'),
            Dense(10, name='logits'),
            Activation('softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stop = EarlyStopping(monitor='val_loss', patience=2)

        r = model.fit(X_train, y_train,  epochs=50, validation_data=(X_val, y_val), verbose=0, callbacks=[cp, tensorboard_callback])

        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(r.history['loss'], label='Loss')
        plt.plot(r.history['val_loss'], label='val_Loss')
        plt.legend()
        plt.title('Loss evolution')

        plt.subplot(2, 2, 2)
        plt.plot(r.history['accuracy'], label='accuracy')
        plt.plot(r.history['val_accuracy'], label='val_accuracy')
        plt.legend()
        plt.title('Accuracy evolution')

        # Evaluate the model on the test data using `evaluate`
        print('\n# Evaluate on test data')
        results = model.evaluate(X_test, y_test, batch_size=128)
        print('test loss, test acc:', results)
        
        test_y_predicted = model.predict(X_test)
        test_y_predicted = np.argmax(test_y_predicted, axis=1)
        #test_y_predicted = (model.predict(X_test) > 0.5).astype("int32")
        val_y_predicted  = model.predict(X_val)
        val_y_predicted = np.argmax(val_y_predicted, axis=1)
        #val_y_predicted = (model.predict(X_val) > 0.5).astype("int32")
        # print(y_val.shape)
        # print(val_y_predicted.shape)
        # print(y_val)
        # print(val_y_predicted)
        cm = confusion_matrix(y_test, test_y_predicted)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(10, 10))
        disp = disp.plot(xticks_rotation='vertical', ax=ax,cmap='summer')
        plt.show()

        probas_val = model.predict(X_seedset)
        print ('probabilities:', probas_val.shape, '\n', np.argmax(probas_val, axis=1))
        #print('----------------')
        #print(probas_val)
        #print('-----------------')
        #record.loc[iterations] = [X_train_org.shape[0], X_seedset.shape[0], r.history['accuracy'][-1], r.history['loss'][-1],r.history['val_accuracy'][-1],r.history['val_loss'][1], results[1], results[0]]

        #uncertain_samples = self.sample_selection_function.select(probas_val, self.initial_labeled_samples)
        #print('Uncertain_samples :', uncertain_samples)
        return probas_val