#  AI-Powered Credit Card Fraud Detection
An intelligent fraud detection system that analyzes transaction data and identifies fraudulent activities using a Random Forest Classifier. Built with Python and scikit-learn.

## Dataset
The model uses the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle.

To use it:

1. Download `creditcard.csv` from the link above.
2. Place it in the root directory of this project.

##Output:
1. Console output: Accuracy, classification report, confusion matrix
2. detected_fraud_transactions.csv: Stores predicted fraud transactions
3. Plots: ROC curve and scatter plot of fraud over time

## 🛠️ Requirements
Install the required Python packages:
```bash
pip install -r requirements.txt
