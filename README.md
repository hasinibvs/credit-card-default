# Credit Card Default Prediction

## Project Overview
This project predicts the probability of a customer defaulting on their credit card payments. Using the UCI Credit Card Clients dataset, I implemented a Machine Learning pipeline to classify customers as "Default" or "Non-Default."

**Key Tech:** Python, Scikit-Learn, Pandas, SMOTE (Imbalanced Learning).

## Key Challenges Solved
* **Class Imbalance:** The dataset was highly imbalanced (fewer defaulters than payers). I utilized **SMOTE** (Synthetic Minority Over-sampling Technique) to balance the training data, improving the model's recall for detecting defaults.
* **Data Scaling:** Applied Standard Scaling to normalize feature distributions for better model convergence.

## Results
* **ROC-AUC Score:** [Insert your score here after running code]
* **Recall:** [Insert Recall score here] - High recall was prioritized to minimize financial risk by catching as many potential defaulters as possible.

## Dataset
The data used for this analysis is the "Default of Credit Card Clients" dataset from the UCI Machine Learning Repository.

**[Download the Dataset Here](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)**

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the analysis: `python analysis.py`
