import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold

from data_preprocessing_1 import NoteDataset

# This is hardcoded to my file path, but will change later when pushed to github. 
input_array_train = NoteDataset(r"../data", split="train")

#Set up the input as tensor, and the labels as an array.
input_tensor = []
input_label = []

# Iterate through the input array and append the input and label to the input_tensor and input_label arrays.
for i in range(len(input_array_train)):
    input, label = input_array_train[i]
    input_tensor.append(input)
    input_label.append(label)

np_inputs = np.stack([x.detach().cpu().numpy().flatten() for x in input_tensor])
np_labels = np.array(input_label)

labels = np.unique(np_labels)

# Set up the k-fold cross-validation. We decided to use a k-fold cross-validation, 
# in order to get a more accurate estimate of the model's performance.
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Set up the C values. Using a range of C values, in order to find the best C value.
C_values = [0.1, 1.0, 10.0, 100.0]

# Set up the PCA values. Using a range of PCA values, in order to find the best PCA value.
PCA_values = [50,100,150]
# Initiate the results.
results = []

# Loop through n_components values.
# We've decided to use 0.9,0.8,0.7.
for n_components in PCA_values:
    print(f"N Components: {n_components}")

    fold_splits = []
    for train_idx, test_idx in kf.split(np_inputs):
        # Using a standard scaler to scale the data.
        scaler = StandardScaler()
        input_train = scaler.fit_transform(np_inputs[train_idx])
        input_test = scaler.transform(np_inputs[test_idx])

        # Using a PCA to reduce the dimensionality of the datax
        pca = PCA(n_components=n_components)
        X_train = pca.fit_transform(input_train)
        X_test = pca.transform(input_test)

        fold_splits.append((X_train, X_test, np_labels[train_idx], np_labels[test_idx]))

    for C in C_values:
        fold_loss = []
        fold_acc = []
        fold_prec = []
        fold_rec = []
        fold_f1 = []

        # Loop through the folds.
        for fold, (X_train, X_test, y_train, y_test) in enumerate(fold_splits):
            print(f"Fold {fold + 1}")

            # Initially used 'auto' for the gamma value, but changed to 'scale', 
            # in order to get a more accurate estimate of the model's performance, 
            # to avoid overfitting.
            # Note: Was given this idea by Claude Code, in order to improve the performance of the model.
            model = SVC(C=C, gamma='scale')
            model.fit(X_train, y_train)
            input_prediction = model.predict(X_test)
            input_score = model.decision_function(X_test)

            #calculates the hinge-loss, the common metric of SVM when applied to one vs. one classification method applied to
            #multiclass classification (i.e., SVC).
            loss = sklearn.metrics.hinge_loss(y_test, input_score, labels=labels)
            acc = np.mean(input_prediction == y_test)
            prec = precision_score(y_test, input_prediction, average="weighted", zero_division=0)
            rec = recall_score(y_test, input_prediction, average="weighted", zero_division=0)
            f1 = f1_score(y_test, input_prediction, average="weighted", zero_division=0)

            fold_loss.append(loss)
            fold_acc.append(acc)
            fold_prec.append(prec)
            fold_rec.append(rec)
            fold_f1.append(f1)

        results.append({
            "n_components": n_components,
            "C": C,
            "fold_splits": fold_splits,
            "avg_loss": np.mean(fold_loss),
            "avg_accuracy": np.mean(fold_acc),
            "avg_precision": np.mean(fold_prec),
            "avg_recall": np.mean(fold_rec),
            "avg_f1": np.mean(fold_f1)
        })

    # This saves the results to a csv file
    # Where the results are stored, and I will be able to plot them later.
with open("svm_results.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_components", "C", "avg_loss", "avg_accuracy", "avg_precision", "avg_recall", "avg_f1"])
    for r in results:
        writer.writerow([r["n_components"], r["C"], r["avg_loss"], r["avg_accuracy"],
                         r["avg_precision"], r["avg_recall"], r["avg_f1"]])

print("\nSaved results to svm_results.csv")
best = max(results, key=lambda x: x["avg_accuracy"])
print("\nBest model:", {k: v for k, v in best.items() if k != "fold_splits"})

# reuse the already-computed fold_splits from the best result, 
# to use for the confusion matrix.
X_train, X_test, y_train, y_test = best["fold_splits"][-1]
model = SVC(C=best["C"], gamma='scale')
model.fit(X_train, y_train)
preds = model.predict(X_test)

cm = confusion_matrix(y_test, preds)
np.savetxt("confusion_matrix.csv", cm, delimiter=",", fmt="%d")
print("Saved confusion_matrix.csv")