import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Membaca Data 
df = pd.read_csv('diabetes_preprocessed.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Menjalankan Training
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 3. Simpan Model Secara Manual
# Simpan dengan nama 'model'. MLflow akan menempatkannya di artifacts/model
mlflow.sklearn.log_model(model, "model")

# 4. Verifikasi di log GitHub Actions 
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)

print("\n" + "="*30)
print("HASIL TRAINING WORKFLOW CI")
print("="*30)
print(report)
print("="*30)
print("Model berhasil disimpan ke folder: artifacts/model")