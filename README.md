# Robust Model Evaluation on Diabetes Dataset Using Bootstrapped Classifier Ensembles

This project explores multiple classification algorithms to predict diabetes outcomes based on medical data. It introduces bootstrapping as a statistical technique to estimate uncertainty in model performance and provides a comparative evaluation using accuracy, mean absolute error (MAE), and root mean squared error (RMSE).

## 📊 Dataset

The dataset used is the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), which contains diagnostic measurements for female patients of Pima Indian heritage.

- **Samples**: 768
- **Features**: 8 medical predictor variables (e.g., glucose level, BMI)
- **Target**: Binary outcome (`0`: No Diabetes, `1`: Diabetes)

## 🛠️ Technologies & Libraries

- Python 3
- Pandas, NumPy, Matplotlib
- Scikit-learn
- Keras
- XGBoost, LightGBM (for extensibility)
- Colab for development

## 🧪 Models Compared

1. Logistic Regression
2. Support Vector Machine (SVM)
3. Decision Tree Classifier
4. Naive Bayes
5. Multi-layer Perceptron (MLP)

## 📈 Methodology

1. **Data Preprocessing**:
   - Missing values handled using `SimpleImputer`.
   - Target variable separated from predictors.
   - Train/test split (80/20).

2. **Bootstrapping**:
   - Each model is evaluated using 100 bootstrap resampling iterations.
   - Standard deviation is used to compute uncertainty (± margin) of accuracy.

3. **Performance Metrics**:
   - Accuracy (mean + standard deviation)
   - Mean Absolute Error (MAE)
   - Root Mean Squared Error (RMSE)
   - Correlation matrix of features
   - Visualization of bootstrap accuracy convergence

## 📊 Results Summary

| Model               | Accuracy (± Uncertainty) | MAE      | RMSE    |
|--------------------|--------------------------|----------|---------|
| Logistic Regression| 0.7512 ± 0.0198          | 0.2532   | 0.5032  |
| SVM                | 0.7575 ± 0.0127          | 0.2337   | 0.4835  |
| Decision Tree      | 0.6820 ± 0.0299          | 0.2402   | 0.4902  |
| Naive Bayes        | 0.7603 ± 0.0227          | 0.2337   | 0.4835  |
| MLP                | 0.6771 ± 0.0341          | 0.3247   | 0.5698  |

Naive Bayes showed the best average performance with lowest MAE and RMSE among tested models.

## 📉 Visualizations

- Correlation heatmap of features
- Model accuracy distribution over 100 bootstrap iterations
- Bar plots for model performance and error comparison

## 📂 Folder Structure

