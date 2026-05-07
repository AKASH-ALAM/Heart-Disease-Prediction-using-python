import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

def main():
    print("--- Loading Data ---")
    df = pd.read_csv('data/heart.csv')
    
    # Drop duplicates to prevent data leakage
    df_cleaned = df.drop_duplicates()
    print(f"Data shape after dropping duplicates: {df_cleaned.shape}")

    # Separate Features (X) and Target (y)
    X = df_cleaned.drop('target', axis=1)
    y = df_cleaned['target']

    # Continuous numerical features that need scaling
    numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    # Feature scaling (Standardization)
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    print("\n--- Training Best Support Vector Machine (SVM) ---")
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    }

    svm = SVC(probability=True, random_state=42)
    svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='accuracy', n_jobs=-1)
    svm_grid.fit(X_train, y_train)
    best_svm = svm_grid.best_estimator_

    # Evaluate the chosen model
    y_pred_svm = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_svm)
    print(f"Model trained successfully. Accuracy on test set: {accuracy:.4f}")
    
    # Save the models
    os.makedirs('models', exist_ok=True)
    
    scaler_path = 'models/scaler.joblib'
    model_path = 'models/svm_model.joblib'
    
    print("\n--- Saving Models ---")
    joblib.dump(scaler, scaler_path)
    print(f"Saved StandardScaler to '{scaler_path}'")
    
    joblib.dump(best_svm, model_path)
    print(f"Saved trained SVM model to '{model_path}'")

if __name__ == "__main__":
    main()
