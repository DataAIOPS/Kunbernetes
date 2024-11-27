import joblib
import numpy as np
import pandas as pd

# Load the pre-trained model and preprocessor
model = joblib.load('../house_price_model.joblib')
preprocessor = joblib.load('preprocessor.joblib')

def predict_price(area, bedrooms, stories, mainroad, basement):
    # Create a dataframe with the input data
    input_data = pd.DataFrame([[area, bedrooms, stories, mainroad, basement]],
                              columns=['area', 'bedrooms', 'stories', 'mainroad', 'basement'])

    # Preprocess the input data
    input_data_preprocessed = preprocessor.transform(input_data)

    # Make prediction
    predicted_price = model.predict(input_data_preprocessed)
    return predicted_price[0]

if __name__ == "__main__":
    # Example usage
    area = 3000
    bedrooms = 3
    stories = 2
    mainroad = 'yes'
    basement = 'no'
    print("Given inputs are area=3000, bedrooms=3, stories=2, mainroad='yes', basement='no',")
    price = predict_price(area, bedrooms, stories, mainroad, basement)
    print(f'Predicted house price: {price}')
