# NexGen AI Dashboard: Multi-ML System

This project is a high-performance, modern Streamlit dashboard that integrates two distinctive Machine Learning models: a **Logistic Regression** based Customer Purchase Predictor and a **KNN-based** Movie Recommendation System.

## 📁 Project Structure

```text
customer purchase prediction/
├── app.py              # Main Streamlit Application
├── data_prep.py        # Dataset generation & Model training script
├── purchase_data.csv   # Synthetic purchase dataset
├── movie_ratings.csv   # Synthetic movie rating matrix
├── purchase_model.pkl  # Trained Logistic Regression model
├── movie_knn_model.pkl # Trained KNN model
├── requirements.txt    # Project dependencies
└── README.md           # Instructions
```

## ✨ Features

### 1. Customer Purchase Prediction
- **Algorithm**: Logistic Regression.
- **Goal**: Predicts if a customer will buy a product based on Age and Salary.
- **UI**: Interactive sliders for input and a custom gauge chart for probability scores.

### 2. Movie Recommendation System
- **Algorithm**: K-Nearest Neighbors (KNN) with Cosine Similarity.
- **Goal**: Recommends movies by finding users with similar rating patterns.
- **UI**: Profile selection and dynamic recommendation cards.

### 3. Analytics Dashboard
- **Visuals**: Plotly charts (Scatter, Pie, Heatmaps).
- **Goal**: Provides insights into data distributions and model performance.

### 4. Modern UI/UX
- Custom CSS injected with Google Fonts (Inter).
- SaaS-style cards with hover effects and shadows.
- Sidebar navigation for seamless page transitions.

## 🚀 How to Run

### Step 1: Install Dependencies
Open your terminal and run:
```bash
pip install -r requirements.txt
```

### Step 2: Prepare Models (Optional)
The models and data have been pre-generated. If you want to retrain them, run:
```bash
python data_prep.py
```

### Step 3: Launch the Dashboard
Run the Streamlit application:
```bash
streamlit run app.py
```

## 💡 Tips for Deployment & Improvement
- **UI**: You can add more Lottie animations for a truly premium feel.
- **Data**: Replace synthetic CSVs with real datasets like the IBM Purchase Dataset or MovieLens 100k.
- **Optimization**: For larger datasets, use `parquet` format instead of `csv` to speed up loading.
