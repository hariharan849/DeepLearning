import pandas as pd

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Check dataset
print(df.head())
print(df.info())

from pycaret.classification import setup, compare_models

# Initialize PyCaret experiment
clf = setup(data=df, target='Churn', session_id=123, normalize=True)

# Compare multiple models and select the best one
best_model = compare_models()

from pycaret.classification import tune_model

# Hyperparameter tuning
tuned_model = tune_model(best_model, optimize='AUC')

from pycaret.classification import save_model, load_model

# Save the tuned model
save_model(tuned_model, 'churn_model')

# Load the model later
model = load_model('churn_model')


from flask import Flask, request, jsonify
from pycaret.classification import load_model, predict_model
import pandas as pd

app = Flask(__name__)

# Load trained model
model = load_model("churn_model")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    predictions = predict_model(model, data=df)
    return jsonify(predictions.to_dict(orient="records"))

if __name__ == '__main__':
    app.run(debug=True)
