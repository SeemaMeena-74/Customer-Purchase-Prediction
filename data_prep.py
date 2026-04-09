import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import joblib
import os

def prepare_purchase_model():
    # 1. Generate Synthetic Purchase Data
    np.random.seed(42)
    n_samples = 400
    age = np.random.randint(18, 65, n_samples)
    salary = np.random.randint(20000, 150000, n_samples)
    
    # Simple logic: Older + Higher salary = more likely to buy
    # Buy = 1 if (Age/65 * 0.5 + Salary/150000 * 0.5) + noise > 0.6 else 0
    noise = np.random.normal(0, 0.1, n_samples)
    buy_prob = (age/65 * 0.5 + salary/150000 * 0.5) + noise
    buy = (buy_prob > 0.6).astype(int)
    
    df = pd.DataFrame({'Age': age, 'Salary': salary, 'Buy': buy})
    df.to_csv('purchase_data.csv', index=False)
    
    # Train Logistic Regression
    X = df[['Age', 'Salary']]
    y = df['Buy']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    # Save model
    joblib.dump(model, 'purchase_model.pkl')
    print("Purchase model and data saved.")

def prepare_movie_model():
    # 2. Generate Synthetic Movie Rating Data
    # Users: User1 to User10
    # Movies: Inception, Titanic, Matrix, Toy Story, Pulp Fiction, Avatar
    movies = ['Inception', 'Titanic', 'The Matrix', 'Toy Story', 'Pulp Fiction', 'Avatar', 'Interstellar', 'The Godfather']
    users = [f'User {i}' for i in range(1, 11)]
    
    # Random ratings 1-5, with some 0s for unrated
    data = np.random.randint(0, 6, size=(len(users), len(movies)))
    df = pd.DataFrame(data, index=users, columns=movies)
    df.to_csv('movie_ratings.csv')
    
    # Train KNN
    # We use cosine similarity for recommendation
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(df.values)
    
    joblib.dump(model, 'movie_knn_model.pkl')
    print("Movie model and data saved.")

if __name__ == "__main__":
    prepare_purchase_model()
    prepare_movie_model()
