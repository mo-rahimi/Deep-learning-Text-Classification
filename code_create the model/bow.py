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


def load_data_from_folder(folder_path):
    texts = []
    labels = []
    for class_name in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, class_name)
        if os.path.isdir(class_folder):
            for file_name in os.listdir(class_folder):
                file_path = os.path.join(class_folder, file_name)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    texts.append(content)
                    labels.append(class_name)
    return texts, labels


train_folder = '20news-bydate-train'  # path to my train folder
test_folder = '20news-bydate-test'
train_texts, train_labels = load_data_from_folder(train_folder)
test_texts, test_labels = load_data_from_folder(test_folder)

all_texts = train_texts + test_texts
all_labels = train_labels + test_labels

print("\n")
print("The frequency of each class in training data set as below:")

from collections import Counter

class_counts = Counter(all_labels)
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")
print("\n")

'''
the class distribution is not severely imbalanced, but it's still not perfectly balanced either. As the frequency of
 most of the classes in over 970, I want to increase the number of documents in each class to 1000.

  To balance my text dataset using oversampling, we can use Random Oversampling.
   This method involves duplicating instances from the minority classes randomly until the desired class 
   distribution is reached. So I have to determine the maximum number of instances in any class (in our case,
    it's 999 for 'rec.sport.hockey'), but I prefer to set 1000 as it would be easier to work with.
'''



def oversample(texts, labels):
    class_counts = {}
    for label in set(labels):
        class_counts[label] = labels.count(label)

    max_count = 1000

    balanced_texts = []
    balanced_labels = []

    for label in class_counts:
        label_texts = [text for text, l in zip(texts, labels) if l == label]
        oversampled_texts = random.choices(label_texts, k=max_count)
        balanced_texts.extend(oversampled_texts)
        balanced_labels.extend([label] * max_count)

    return balanced_texts, balanced_labels


balanced_texts, balanced_labels = oversample(all_texts, all_labels)

print("\n")
print("The frequency of each class in data set after balancing:")

class_counts = Counter(balanced_labels)
for class_name, count in class_counts.items():
    print(f"{class_name}: {count}")
print("\n")

print("Type of balanced_texts is ", type(balanced_texts))
print("Shape of balanced_texts is ", len(balanced_texts))

print("Type of balanced_labels is ", type(balanced_labels))
print("Shape of balanced_labels is ", len(balanced_labels))


''' Preprocessing:
I will remove stop words, special characters and apply lemmatization, but not convert to lowercase.

remove_stopwords:
Removing stop words before creating the Bag of Words (BoW) and bigram representations can reduce the dimensionality 
of the feature space and make the text representation more focused on meaningful words. This can lead to better 
performance in text classification tasks, as the model can better capture the significant words and phrases in the text.

 Lemmatization: 
This helps to treat words with the same base meaning, further reducing the vocabulary size and improving the 
model's ability to generalize.

Removing special characters and digits: 
Remove non-alphabetic characters, such as punctuation marks and digits, to reduce noise in the data.

NOT convert to lowecase:
lowercasing is not a good choice, when proper nouns and brand names are important for text classification.

'''
def remove_stopwords(text):
    return ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)


balanced_texts_no_stopwords = [remove_stopwords(text) for text in balanced_texts]
balanced_texts_lemmatized = [lemmatize_text(text) for text in balanced_texts_no_stopwords]
balanced_texts_cleaned = [remove_special_characters(text) for text in balanced_texts_lemmatized]




# Create bag of words representation
bow_vectorizer = CountVectorizer()   # Create a CountVectorizer for bag of words
bow = bow_vectorizer.fit_transform(balanced_texts_cleaned)   # Fit the vectorizer on the training data and transform it

# Create bigram representation
bigram_vectorizer = CountVectorizer(ngram_range=(2, 2))  # Create a CountVectorizer with bigram setting
bigram = bigram_vectorizer.fit_transform(balanced_texts_cleaned)

# Create unigram_bigram representation
bigram_bow_vectorizer = CountVectorizer(ngram_range=(1, 2))
bigram_bow = bigram_bow_vectorizer .fit_transform(balanced_texts_cleaned)


print("The type of `bow`         is :", type(bow))
print("The shape of `bow`        is :", bow.shape)
print("The dimension of `bow`    is :", bow.ndim)
print("\n")

print("The type of `bigram`      is :", type(bigram))
print("The shape of `bigram`     is :", bigram.shape)
print("\n")

print("The type of `bigram_bow`  is :", type(bigram_bow))
print("The shape of `bigram_bow` is :", bigram_bow.shape)
print("\n")


labels = np.array(balanced_labels)
print("The type of lables is      :", type(labels))
print("The shape of lables is     :", labels.shape)
print("The dimension of lables is :", labels.ndim)

# Define X and y for each representation
X_bow = bow.astype('float32')
X_bigram = bigram.astype('float32')
X_bigram_bow =bigram_bow.astype('float32')

y = LabelEncoder().fit_transform(labels)

print("\n")
print("The type of `X_bow`         is :", type(X_bow))
print("The shape of `X_bow`        is :", X_bow.shape)
print("The dimension of `X_bow`    is :", X_bow.ndim)
print("\n")

print("The type of `X_bigram`      is :", type(X_bigram))
print("The shape of `X_bigram`     is :", X_bigram.shape)
print("\n")

print("The type of `X_bigram_bow`  is :", type(X_bigram_bow))
print("The shape of `X_bigram_bow` is :", X_bigram_bow.shape)
print("\n")

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
X_bow_80, X_bow_20, y_80, y_20 = train_test_split(X_bow, y, test_size=0.2, random_state=10)

save_npz("X_bow_20.npz", X_bow_20)  # Save the X_bow_20 matrix
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
folder_name = "models_bow_10_folds"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

n_splits = 10
skf = StratifiedKFold(n_splits=n_splits, shuffle=True)

input_dim = X_bow.shape[1]
output_dim = np.unique(y).shape[0]

batch_sizes = [100, 500, 1000, 3000]
histories = {}
fold_number = 1 # start with fold 1
t_0 = time.time()

best_models = {}
best_val_losses = {batch_size: float('inf') for batch_size in batch_sizes}  # Initialize with large values

for batch_size in batch_sizes:
    histories[batch_size] = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    for train_index, test_index in skf.split(X_bow_80, y_80):
        X_train_bow, X_test_bow = X_bow_80[train_index], X_bow_80[test_index]
        y_train, y_test = y_80[train_index], y_80[test_index]

        model = create_model(input_dim=input_dim, output_dim=output_dim)
        history = model.fit(X_train_bow, y_train, epochs=10, batch_size=batch_size, verbose=2,
                            validation_data=(X_test_bow, y_test))

        for key in history.history.keys():
            histories[batch_size][key].append(history.history[key])

        current_val_loss = history.history['val_loss'][-1]
        if current_val_loss < best_val_losses[batch_size]:  # Check if the current val_loss is better
            best_val_losses[batch_size] = current_val_loss   # Update the best_val_losses dict
            model.save(f'{folder_name}/model_bow_batch_size_{batch_size}_best.h5') # Save the best model
            best_models[batch_size] = model

        fold_number += 1
    fold_number = 1

for batch_size in batch_sizes:
    for key in histories[batch_size].keys():
        histories[batch_size][key] = np.mean(histories[batch_size][key], axis=0)

t_1 = time.time()
print("Processing time for BOW representation in minute is :", (t_1 - t_0)/60)





n_rows = len(batch_sizes)
n_cols = 2

fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 4 * n_rows))

for i, batch_size in enumerate(batch_sizes):
    # Plot accuracy
    axes[i][0].plot(histories[batch_size]['accuracy'], label='Training', marker='o')
    axes[i][0].plot(histories[batch_size]['val_accuracy'], label='Validation', marker='o')
    axes[i][0].set_title(f'Batch size {batch_size} - Accuracy')
    axes[i][0].set_xlabel('Epoch')
    axes[i][0].set_ylabel('Accuracy')
    axes[i][0].legend()

    # Plot loss
    axes[i][1].plot(histories[batch_size]['loss'], label='Training', marker='o')
    axes[i][1].plot(histories[batch_size]['val_loss'], label='Validation', marker='o')
    axes[i][1].set_title(f'Batch size {batch_size} - Loss')
    axes[i][1].set_xlabel('Epoch')
    axes[i][1].set_ylabel('Loss')
    axes[i][1].legend()

plt.tight_layout()
plt.show()






