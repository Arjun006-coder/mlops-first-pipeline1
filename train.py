from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the Random Forest model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# Create the models directory
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)

# Define the model file path
model_path = model_dir / "iris_random_forest_model.joblib"

# Save the trained model
joblib.dump(model, model_path)

print("Model saved successfully at:", model_path)