import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# 1. Autologging MLflow 
mlflow.sklearn.autolog()

# 2. Membaca Data
df = pd.read_csv('diabetes_preprocessed.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Menjalankan Training
# set_experiment dihapus agar MLflow Project mengaturnya secara otomatis di Runner
with mlflow.start_run(run_name="Logistic_CI_Retraining"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Verifikasi di log GitHub Actions
    accuracy = model.score(X_test, y_test)
    print(f"Modelling CI Selesai. Akurasi: {accuracy:.4f}")