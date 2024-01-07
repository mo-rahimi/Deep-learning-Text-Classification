from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score, accuracy_score, log_loss
import numpy as np
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import pandas as pd

# Load the saved models
model_100 = load_model("models_bow_10_folds/model_bow_batch_size_100_best.h5")   # model based on batch_size = 100
model_500 = load_model("models_bow_10_folds/model_bow_batch_size_500_best.h5")   # model based on batch_size = 500
model_1000 = load_model("models_bow_10_folds/model_bow_batch_size_1000_best.h5") # model based on batch_size = 1000
model_3000 = load_model("models_bow_10_folds/model_bow_batch_size_3000_best.h5") # model based on batch_size = 3000



X_bow_20 = load_npz("X_bow_20.npz")
y_20 = np.load("y_20.npy", allow_pickle=True)

# Evaluate the model on the loaded test data
loss_100, accuracy_100 = model_100.evaluate(X_bow_20, y_20)
print("Loss_100: ", loss_100)
print("Accuracy_100: ", accuracy_100)

y_pred_probs_100 = model_100.predict(X_bow_20)    # Get the predicted probabilities
y_pred_100 = y_pred_probs_100.argmax(axis=1)   # Convert the predicted probabilities to class labels
f1_100 = f1_score(y_20, y_pred_100, average='weighted')  # Calculate the F1-score
print("F1-score_100: ", f1_100)
# Function to evaluate a model
def evaluate_model(model, X, y):
    loss, accuracy = model.evaluate(X, y)
    y_pred_probs = model.predict(X)
    y_pred = y_pred_probs.argmax(axis=1)
    f1 = f1_score(y, y_pred, average='weighted')
    return loss, accuracy, f1

# Evaluate the models
models = [model_100, model_500, model_1000, model_3000]
batch_sizes = [100, 500, 1000, 3000]
losses = []
accuracies = []
f1_scores = []

for model in models:
    loss, accuracy, f1 = evaluate_model(model, X_bow_20, y_20)
    losses.append(loss)
    accuracies.append(accuracy)
    f1_scores.append(f1)

# Print the results
for batch_size, loss, accuracy, f1 in zip(batch_sizes, losses, accuracies, f1_scores):
    print(f"Batch size {batch_size}: Loss={loss}, Accuracy={accuracy}, F1-score={f1}")

# Create a bar plot for comparison
x = np.arange(len(batch_sizes))
width = 0.3

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, accuracies, width, label='Accuracy')
rects2 = ax.bar(x, f1_scores, width, label='F1-score')
rects3 = ax.bar(x + width, losses, width, label='Loss')

ax.set_ylabel('Scores')
ax.set_title('Comparison of Models with Different Batch Sizes')
ax.set_xticks(x)
ax.set_xticklabels(batch_sizes)
ax.set_xlabel('Batch Size')
ax.legend()
plt.show()

# Create a dataframe with the results
data = {'Batch Size': batch_sizes, 'Loss': losses, 'Accuracy': accuracies, 'F1-score': f1_scores}
df = pd.DataFrame(data)

# Set the index to be the batch size
df.set_index('Batch Size', inplace=True)

# Display the results in a table
print(df)
