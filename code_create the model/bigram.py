import time
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers.legacy import Adam
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from scipy.sparse import save_npz, load_npz
from nltk.stem import WordNetLemmatizer
import nltk
import re
nltk.download('wordnet')




'''So far
I have prepared three different text representations (bag of words, bigrams, and unigram-bigrams) as input features 
(X_bow, X_bigram, and X_bigram_bow), and encoded the class labels (y) using LabelEncoder for the classification task.'''


'''
I will apply 10 fold cross validation but to do task d I do not want evaluating a model on the same data it was trained 
on as can lead to overly optimistic results and doesn't give a true measure of the model's performance on unseen data. 
To get a better estimate of the model's performance and do part d of the task, I will  evaluate it on a separate test dataset that was not used 
during training.
I want to point out I know when I performed 10-fold cross-validation in my code, I already splited the dataset into training and validation sets. 
 If I want to evaluate my model on a completely new test dataset, I should preprocess this new data using the same steps that is used 
 for original data (e.g., stopword removal, and creating the BoW representation) and then use this new test data for 
 evaluation, to do so I prefer to split into train and test after BOW representation and I will save the `X_bow_20` and `y_20` to be able 
 to load them latter.

'''

from preprocessing_ngram import X_bigram, y

X_bigram_80, X_bigram_20, y_80, y_20 = train_test_split(X_bigram, y, test_size=0.2, random_state=11)

save_npz("X_bigram_20.npz", X_bigram_20)  # Save the X_bow_20 matrix
np.save("y_20.npy", y_20)  # Save the y_20 array



# Define the model creation function
def create_model(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(256, activation='relu', kernel_initializer='he_normal', input_dim=input_dim))
    model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(output_dim, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

# Create a directory to save the models
folder_name = "models_bigram_10_folds"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

input_dim = X_bigram.shape[1]
output_dim = np.unique(y).shape[0]

batch_sizes = [100] # 100, 500, 1000
histories = {}
fold_number = 1
t_0 = time.time()

best_models = {}
best_val_losses = {batch_size: float('inf') for batch_size in batch_sizes}  # Initialize with large values

for batch_size in batch_sizes:
    histories[batch_size] = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for train_index, test_index in skf.split(X_bigram_80, y_80):
        X_train_bigram_bow, X_test_bigram_bow = X_bigram_80[train_index], X_bigram_80[test_index]
        y_train, y_test = y_80[train_index], y_80[test_index]

        model = create_model(input_dim=input_dim, output_dim=output_dim)
        history = model.fit(X_train_bigram_bow, y_train, epochs=10, batch_size=batch_size, verbose=2,
                            validation_data=(X_test_bigram_bow, y_test))

        for key in history.history.keys():
            histories[batch_size][key].append(history.history[key])

        current_val_loss = history.history['val_loss'][-1]
        if current_val_loss < best_val_losses[batch_size]:  # Check if the current val_loss is better
            best_val_losses[batch_size] = current_val_loss   # Update the best_val_losses dict
            model.save(f'{folder_name}/model_bigram_batch_size_{batch_size}_best.h5') # Save the best model
            best_models[batch_size] = model

        fold_number += 1
    fold_number = 1

for batch_size in batch_sizes:
    for key in histories[batch_size].keys():
        histories[batch_size][key] = np.mean(histories[batch_size][key], axis=0)

t_1 = time.time()
print("Processing time with bigram representation in minute is :", (t_1 - t_0)/60)



fig, axes = plt.subplots(1, 2, figsize=(10, 4))


batch_size = 100

# Plot accuracy
axes[0].plot(histories[batch_size]['accuracy'], label='Training', marker='o')
axes[0].plot(histories[batch_size]['val_accuracy'], label='Validation', marker='o')
axes[0].set_title(f'Bigram representation_Batch size {batch_size} - Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()

# Plot loss
axes[1].plot(histories[batch_size]['loss'], label='Training', marker='o')
axes[1].plot(histories[batch_size]['val_loss'], label='Validation', marker='o')
axes[1].set_title(f'Bigram representation_Batch size {batch_size} - Loss')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].legend()
plt.show()






