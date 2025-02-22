import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, delimiter=';')

# Prepare data
X = df.drop("quality", axis=1)
y = df["quality"]
y = np.where(y >= 6, 1, 0)  # Convert to binary classification (Good vs. Bad)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# Save model
with open("wine_quality_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Flask API
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        with open("wine_quality_model.pkl", "rb") as f:
            model = pickle.load(f)
        prediction = model.predict(features)[0]
        return jsonify({"quality": "Good" if prediction == 1 else "Bad"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)