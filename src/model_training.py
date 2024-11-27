import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from data_preprocessing import load_and_preprocess_data

print("Model training is started")
# Load and preprocess the data
X, y = load_and_preprocess_data('../data/Housing.csv')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model training is finished")
# Save the trained model
joblib.dump(model, '../house_price_model.joblib')

# Evaluate the model
score = model.score(X_test, y_test)
print(f'Model R^2 score: {score}')
