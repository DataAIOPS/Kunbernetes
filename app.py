from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("house_price_model.joblib")

@app.route('/predict', methods=['POST'])
def predict():
    # Get input features from request
    data = request.get_json()
    print(data)
    # Extract features
    area = data['area']
    bedrooms = data['bedrooms']
    stories = data['stories']
    mainroad = 1 if data['mainroad'] == 'yes' else 0
    basement = 1 if data['basement'] == 'yes' else 0

    # Make prediction
    prediction = model.predict([[area, bedrooms, stories, mainroad, basement]])

    # Return prediction as JSON
    return jsonify({'predicted_price': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
