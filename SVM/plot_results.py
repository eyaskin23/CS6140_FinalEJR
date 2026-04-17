import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# File for plotting the results of the SVM implementation. 
df = pd.read_csv("svm_results.csv")
df = df.sort_values(by=["n_components", "C"])

# Group by unique n_components values.
n_values = df["n_components"].unique()

#### ACCURACY VS C ####
plt.figure()
for n in n_values:
    subset = df[df["n_components"] == n]
    plt.plot(subset["C"],subset["avg_accuracy"], marker = "o", label=f"n={n}")

plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.title("Accuracy vs C")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("accuracy_vs_c.png")

plt.figure()
for n in n_values:
    subset = df[df["n_components"] == n]
    plt.plot(subset["C"],subset["avg_loss"], marker = "o", label=f"n={n}")

#### LOSS VS C ####
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Loss")
plt.title("Loss vs C")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("loss_vs_c.png")

#### PRECISION VS C ####
plt.figure()
for n in n_values:
    subset = df[df["n_components"] == n]
    plt.plot(subset["C"],subset["avg_precision"], marker = "o", label=f"n={n}")

plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Precision")
plt.title("Precision vs C")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("precision_vs_c.png")

#### RECALL VS C ####
plt.figure()
for n in n_values:
    subset = df[df["n_components"] == n]
    plt.plot(subset["C"],subset["avg_recall"], marker = "o", label=f"n={n}")

plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Recall")
plt.title("Recall vs C")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("recall_vs_c.png")

#### F1 SCORE VS C ####
plt.figure()
for n in n_values:
    subset = df[df["n_components"] == n]
    plt.plot(subset["C"],subset["avg_f1"], marker = "o", label=f"n={n}")

plt.xscale("log")
plt.xlabel("C")
plt.ylabel("F1 Score")
plt.title("F1 Score vs C")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("f1_score_vs_c.png")

#### CONFUSION MATRIX ####
cm = np.loadtxt("confusion_matrix.csv", delimiter=",")
labels = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap="Blues", vmin=0, vmax=100)

plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(range(len(labels)), labels, rotation=60)
plt.yticks(range(len(labels)), labels)

# Add gridlines
plt.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
plt.gca().set_xticks(np.arange(-.5, len(labels), 1), minor=True)
plt.gca().set_yticks(np.arange(-.5, len(labels), 1), minor=True)

# Add numbers
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        color = "white" if cm[i, j] > cm.max()/2 else "black"
        plt.text(j, i, str(cm[i, j]),
                 ha="center", va="center", fontsize=12, color=color)

plt.colorbar()
plt.savefig("confusion_matrix.png")