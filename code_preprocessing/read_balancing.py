import os
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
import random

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