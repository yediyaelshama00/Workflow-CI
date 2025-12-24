import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Autologging MLflow 
mlflow.sklearn.autolog()

# 2. Membaca Data
df = pd.read_csv('diabetes_preprocessed.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Menjalankan Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
    
# 4. Verifikasi di log GitHub Actions
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

print("-" * 30)
print("RE-TRAINING CI SELESAI")
print("-" * 30)
print("Laporan Klasifikasi:\n", report)