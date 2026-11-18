# Quantum Support Vector Machine for Heart Disease Prediction

## Overview

This project implements a Quantum Support Vector Machine (QSVM) to predict heart disease using a medical dataset. The goal is to compare the performance of classical machine learning models with quantum machine learning techniques.

The project explores how quantum computing can be applied to healthcare prediction problems and evaluates whether quantum kernels can improve classification accuracy compared to classical methods.

## Features

* Heart disease prediction using machine learning
* Implementation of Quantum Support Vector Machine (QSVM)
* Comparison between classical SVM and QSVM
* Performance evaluation using accuracy and confusion matrix
* Visualization of model results

## Technologies Used

* Python
* Qiskit (Quantum Machine Learning)
* NumPy
* Pandas
* Scikit-learn
* Matplotlib

## Dataset

The project uses a heart disease dataset containing medical attributes such as:

* Age
* Sex
* Chest pain type
* Blood pressure
* Cholesterol level
* Maximum heart rate
* Exercise induced angina

These features are used to classify whether a patient is likely to have heart disease.

## Project Structure

QSVM_Heart_Disease
│
├── data/                        # Heart disease datasets
├── src/                         # Source code for QSVM implementation
├── results/                     # Experimental results and graphs
│   └── comparison.png           # QSVM vs Classical SVM comparison
├── qsvm.py                      # Main QSVM implementation script
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation

## Methodology

1. Data preprocessing and feature selection
2. Train classical Support Vector Machine (SVM)
3. Implement Quantum Kernel using Qiskit
4. Train Quantum Support Vector Machine (QSVM)
5. Compare classification performance

## Results

The results show a comparison between classical SVM and Quantum SVM in predicting heart disease. Visualization graphs are included in the results folder.

## Applications

* AI in healthcare diagnostics
* Medical decision support systems
* Quantum machine learning research
* Early disease prediction systems

## Future Improvements

* Use larger healthcare datasets
* Implement hybrid quantum-classical models
* Evaluate models on real quantum hardware
* Deploy the system as a healthcare prediction web application

## Author

Balumuru Chaithanya
B.Tech Computer Science and Engineering

Research Interests:

* Artificial Intelligence
* Machine Learning
* Quantum Machine Learning
* Healthcare AI

## License

This project is intended for academic and research purposes.
