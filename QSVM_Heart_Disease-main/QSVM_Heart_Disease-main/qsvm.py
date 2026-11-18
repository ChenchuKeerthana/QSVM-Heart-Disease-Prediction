import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


print("\nLoading dataset...")

data = pd.read_csv("heart-disease-dataset.csv")
data = shuffle(data, random_state=42)

X = data.drop("target", axis=1)
y = data["target"]

print("Original dataset shape:", data.shape)


# ============================
# Classical SVM
# ============================

print("\nRunning Classical SVM...")

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y, test_size=0.2, random_state=42
)

svm = SVC(kernel="rbf")
svm.fit(X_train_c, y_train_c)

pred_c = svm.predict(X_test_c)

classical_acc = accuracy_score(y_test_c, pred_c)

print("Classical Accuracy:", round(classical_acc, 4))


# ============================
# QSVM PREPROCESSING
# ============================

print("\nPreparing data for QSVM...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensions for quantum circuit
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

# Normalize values between 0 and 1 for quantum encoding
minmax = MinMaxScaler()
X_quantum = minmax.fit_transform(X_pca)

X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(
    X_quantum, y, test_size=0.2, random_state=42
)

# Balanced dataset for quantum training
X_train_q = X_train_q[:120]
y_train_q = y_train_q[:120]

X_test_q = X_test_q[:40]
y_test_q = y_test_q[:40]

print("QSVM training samples:", len(X_train_q))
print("QSVM test samples:", len(X_test_q))


# ============================
# Quantum Feature Map
# ============================

print("\nBuilding Quantum Kernel...")

feature_map = ZZFeatureMap(
    feature_dimension=4,
    reps=3,
    entanglement="full"
)

quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)


# ============================
# Train QSVM
# ============================

print("\nTraining QSVM...")

start = time.time()

qsvc = QSVC(quantum_kernel=quantum_kernel)

qsvc.fit(X_train_q, y_train_q)

pred_q = qsvc.predict(X_test_q)

qsvm_acc = accuracy_score(y_test_q, pred_q)

quantum_time = time.time() - start


# ============================
# RESULTS
# ============================

print("\n===========================")
print("FINAL RESULTS")
print("===========================")

print("Classical Accuracy :", round(classical_acc, 4))
print("QSVM Accuracy      :", round(qsvm_acc, 4))
print("Quantum Time       :", round(quantum_time, 2), "seconds")


# ============================
# Graph
# ============================

labels = ["Classical SVM", "QSVM"]
values = [classical_acc, qsvm_acc]

plt.bar(labels, values)
plt.ylabel("Accuracy")
plt.title("Classical vs Quantum SVM")

plt.savefig("results/comparison.png")

print("\nGraph saved in results folder")


# ============================
# Patient Prediction
# ============================

print("\n===========================")
print("PATIENT PREDICTION SYSTEM")
print("===========================")

print("\nEnter patient details:\n")

age = float(input("Age: "))
sex = float(input("Sex (0=female,1=male): "))
cp = float(input("Chest pain type (0-3): "))
trestbps = float(input("Resting blood pressure: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting blood sugar (0/1): "))
restecg = float(input("Rest ECG (0-2): "))
thalach = float(input("Max heart rate: "))
exang = float(input("Exercise angina (0/1): "))
oldpeak = float(input("Oldpeak: "))
slope = float(input("ST slope (0-2): "))

patient = [[
    age, sex, cp, trestbps, chol,
    fbs, restecg, thalach,
    exang, oldpeak, slope
]]

patient_scaled = scaler.transform(patient)
patient_pca = pca.transform(patient_scaled)
patient_quantum = minmax.transform(patient_pca)

prediction = qsvc.predict(patient_quantum)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("⚠ Heart Disease Detected")
else:
    print("✔ No Heart Disease")