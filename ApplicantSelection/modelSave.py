import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
from createDataBase import generate_hiring_data

def fit_and_save_svm_from_df(df, model_path="model.pkl"):
    X = df[["experience_years", "technical_score"]]
    y = df["hire_label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVC(kernel="linear", C=1.0)
    model.fit(X_scaled, y)

    joblib.dump(model, model_path)
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, scaler

def predict(experience_years, technical_score):
    input_df = pd.DataFrame([{
        "experience_years": experience_years,
        "technical_score": technical_score
    }])

    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    return prediction


if __name__ == "__main__":

    df = generate_hiring_data()

    model, scaler = fit_and_save_svm_from_df(df)

    experience_years = 8
    technical_score = 80

    prediction = predict(experience_years, technical_score)
    print(f"Tahmin edilen miktar: {prediction}")