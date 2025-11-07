from flask import Flask, render_template, request
import pandas as pd
import pickle
import warnings
from sklearn.preprocessing import OrdinalEncoder

warnings.filterwarnings("ignore")

app = Flask(__name__)

MODEL_PATH = r"./output/Gradient.pkl"
SCALER_PATH = r"./output/scaler.pkl"

model = pickle.load(open(MODEL_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))

model_features = list(getattr(model, "feature_names_in_", []))

nominal_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'PaperlessBilling', 'PaymentMethod'
]

contract_order = [['Month-to-month', 'One year', 'Two year']]
sim_map = {'Jio': 0, 'Airtel': 1, 'Vi': 2, 'BSNL': 3}
ordinal_encoder = OrdinalEncoder(categories=contract_order, handle_unknown='use_encoded_value', unknown_value=-1)

def preprocess_input(form_dict):
    df = pd.DataFrame([form_dict])
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except:
            pass

    if 'sim' in df.columns:
        df['sim'] = df['sim'].map(sim_map).fillna(0)

    if 'Contract' in df.columns:
        df['Contract'] = ordinal_encoder.fit_transform(df[['Contract']])

    df = pd.get_dummies(df, columns=[col for col in nominal_cols if col in df.columns], drop_first=False)

    df['MonthlyCharges_qt_outl'] = df.get('MonthlyCharges', df.get('MonthlyCharges_itimp', 0))
    df['TotalCharges_qt_outl'] = df.get('TotalCharges', df.get('TotalCharges_itimp', 0))

    scale_cols = ['MonthlyCharges_qt_outl', 'TotalCharges_qt_outl']
    df[scale_cols] = scaler.transform(df[scale_cols])

    for col in model_features:
        if col not in df.columns:
            df[col] = 0

    df_final = df.reindex(columns=model_features, fill_value=0)
    return df_final


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/developer')
def developer():
    return render_template('developer.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        X_final = preprocess_input(form_data)
        prediction = model.predict(X_final)[0]
        try:
            probability = model.predict_proba(X_final)[0][1]
        except:
            probability = None

        if prediction == 1:
            result = f"❌ Customer likely to CHURN (prob={probability:.2f})" if probability is not None else "❌ Customer likely to CHURN"
        else:
            result = f"✅ Customer likely to STAY (prob={probability:.2f})" if probability is not None else "✅ Customer likely to STAY"

        return render_template('result.html', prediction_text=result)
    except Exception as e:
        return render_template('result.html', prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run()

