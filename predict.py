import pandas as pd
import joblib
import sys
import os

def get_user_input():
    print("\n--- Please enter the patient's clinical data ---")
    try:
        age = float(input("Age: "))
        sex = float(input("Sex (1 = male; 0 = female): "))
        cp = float(input("Chest Pain Type (0-3): "))
        trestbps = float(input("Resting Blood Pressure (e.g., 120): "))
        chol = float(input("Serum Cholestoral in mg/dl (e.g., 200): "))
        fbs = float(input("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false): "))
        restecg = float(input("Resting Electrocardiographic results (0-2): "))
        thalach = float(input("Maximum Heart Rate achieved (e.g., 150): "))
        exang = float(input("Exercise Induced Angina (1 = yes; 0 = no): "))
        oldpeak = float(input("ST depression induced by exercise relative to rest (e.g., 1.5): "))
        slope = float(input("Slope of the peak exercise ST segment (0-2): "))
        ca = float(input("Number of major vessels (0-4) colored by flourosopy: "))
        thal = float(input("Thal (0-3): "))
        
        feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        df = pd.DataFrame(data, columns=feature_names)
        return df
    except ValueError:
        print("Invalid input! Please enter numerical values.")
        return None

def main():
    scaler_path = 'models/scaler.joblib'
    model_path = 'models/svm_model.joblib'
    
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print("Error: Model files not found! Please run train.py first to generate the models.")
        sys.exit(1)
        
    print("Loading models from 'models/' directory...")
    scaler = joblib.load(scaler_path)
    best_svm = joblib.load(model_path)
    
    while True:
        patient_df = get_user_input()
        if patient_df is not None:
            # Scale numerical features
            numerical_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            patient_df[numerical_features] = scaler.transform(patient_df[numerical_features])
            
            # Predict
            predicted_class = best_svm.predict(patient_df)[0]
            probabilities = best_svm.predict_proba(patient_df)[0]
            
            prob_no_disease = probabilities[0] * 100
            prob_disease = probabilities[1] * 100
            
            # Output report
            print("\n" + "=" * 50)
            print("          DIAGNOSTIC PREDICTION REPORT")
            print("=" * 50)
            if predicted_class == 1:
                print("Diagnosis      : HIGH RISK OF HEART DISEASE DETECTED")
                print("Recommendation : Please consult a cardiologist immediately.")
            else:
                print("Diagnosis      : LOW RISK OF HEART DISEASE")
                print("Recommendation : The patient is likely healthy.")
            print("-" * 50)
            print(f"Probability of Heart Disease    : {prob_disease:.2f}%")
            print(f"Probability of No Heart Disease : {prob_no_disease:.2f}%")
            print("=" * 50)
        
        cont = input("\nWould you like to test another patient? (y/n): ")
        if cont.lower() != 'y':
            print("Exiting predictive diagnostic system.")
            break

if __name__ == "__main__":
    main()
