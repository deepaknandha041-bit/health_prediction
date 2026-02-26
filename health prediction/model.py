import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

def train_model():
    # Load dataset
    if not os.path.exists('dataset.csv'):
        print("Dataset not found!")
        return

    df = pd.read_csv('dataset.csv')
    
    # Split features and target
    X = df.drop('Disease', axis=1)
    y = df['Disease']
    
    # Train Random Forest Classifier
    # Using a small number of estimators for this small dataset
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved as model.pkl")
    print(f"Features: {list(X.columns)}")
    print(f"Classes: {list(model.classes_)}")

if __name__ == "__main__":
    train_model()
