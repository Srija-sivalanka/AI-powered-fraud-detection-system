import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc


os.environ["LOKY_MAX_CPU_COUNT"] = "4"


df = pd.read_csv("creditcard.csv")


X = df.drop(columns=['Class'])  # Features
y = df['Class']  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f}")


conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", conf_matrix)

print("\nClassification Report:\n", classification_report(y_test, y_pred))


fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()


fraud_predictions = X_test[y_pred == 1].copy()
fraud_predictions['Actual_Class'] = y_test[y_pred == 1]
fraud_predictions['Predicted_Class'] = y_pred[y_pred == 1]

fraud_predictions.to_csv("detected_fraud_transactions.csv", index=False)
print("\n⚠️ Fraudulent transactions saved to detected_fraud_transactions.csv")

plt.figure(figsize=(10, 6))
plt.scatter(fraud_predictions["Time"], fraud_predictions["Amount"], color='red', label="Fraud", alpha=0.6)
plt.xlabel("Time")
plt.ylabel("Transaction Amount")
plt.title("Fraudulent Transactions Over Time")
plt.legend()
plt.show()
