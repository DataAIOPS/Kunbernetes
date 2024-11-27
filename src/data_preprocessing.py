import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath)

    # Separate features and target
    X = df[['area', 'bedrooms', 'stories', 'mainroad', 'basement']]
    y = df['price']

    # Define column transformer
    numeric_features = ['area', 'bedrooms', 'stories']
    categorical_features = ['mainroad', 'basement']

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Preprocess data
    X_preprocessed = preprocessor.fit_transform(X)

    # Save the preprocessor for future use
    joblib.dump(preprocessor, 'preprocessor.joblib')

    return X_preprocessed, y

if __name__ == "__main__":
    print("Execution started for data preprocessing")
    X, y = load_and_preprocess_data('../data/Housing.csv')
    print("Shape of data",X.shape, y.shape)
    print("Execution is finished for data preprocessing")
