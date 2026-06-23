import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("featured_covid_data.csv")

# Features and target
X = df[["cases", "recovered", "active"]]
y = df["deaths"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate model
score = r2_score(y_test, predictions)

print("Model trained successfully!")
print("R2 Score:", score)