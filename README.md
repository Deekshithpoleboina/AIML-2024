# Diabetes Model Comparison Using Bootstrap Evaluation

*This project compares different machine learning classifiers to predict diabetes outcomes using the Pima Indians Diabetes dataset. It applies bootstrapping to evaluate model performance with statistical reliability.*
---
## 📊 Dataset

- **Source**: [Pima Indians Diabetes Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records**: 768
- **Features**: 8 predictor variables (e.g., Glucose, BMI, Age)
- **Target**: `Outcome` (0 = No Diabetes, 1 = Diabetes)
---
## 🛠️ Tools & Libraries

- Python, Pandas, NumPy
- Scikit-learn, Matplotlib
- Keras (MLP only)
- XGBoost, LightGBM (prepared for future extensions)
---
## 📌 Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Naive Bayes
- Multi-layer Perceptron (MLP)
---
## ⚙️ Methodology

1. **Data Preprocessing**
   - Handle missing values with `SimpleImputer`.
   - Split into features and target.
   - Train/test split (80/20).

2. **Bootstrapping Evaluation**
   - Resample training data with replacement (100 iterations).
   - Train each model repeatedly.
   - Evaluate on test set for each iteration.
   - Collect mean accuracy and standard deviation (uncertainty).

3. **Metrics**
   - Accuracy ± Standard Deviation
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)
---
## 📈 Results Summary

| Model             | Accuracy ± SD        | MAE      | RMSE    |
|------------------|----------------------|----------|---------|
| Logistic Reg.     | 0.7512 ± 0.0198      | 0.2532   | 0.5032  |
| SVM               | 0.7575 ± 0.0127      | 0.2338   | 0.4835  |
| Decision Tree     | 0.6820 ± 0.0299      | 0.2403   | 0.4902  |
| Naive Bayes       | 0.7603 ± 0.0227      | 0.2338   | 0.4835  |
| MLP               | 0.6771 ± 0.0341      | 0.3247   | 0.5698  |

✅ **Best Accuracy**: Naive Bayes  
❌ **Least Reliable**: MLP (highest error and variance)
---
## 📁 Project Structure
```bash
📦 aiml-diabetes-bootstrap
├── AIML_Project.ipynb # Jupyter notebook with full code
├── diabetes.csv # Dataset used for training and evaluation
├── AIML_Project.pdf # PDF version of the project report
├── AIML_Project-Colab.pdf # Colab-exported PDF notebook
├── README.md # Project overview and documentation
```
---

## 📉 Visualizations

- Feature Correlation Heatmap
- Accuracy vs. Iterations for each model
- Bar plots comparing Accuracy, MAE, and RMSE
---
## 🚀 Future Improvements

- Add XGBoost and LightGBM into the bootstrap loop
- Perform GridSearchCV for hyperparameter tuning
- Try ensemble methods (voting, stacking)
- Use k-fold cross-validation for better generalization
---
## 👨‍💻 Author

**Deekshith**  
- GitHub: [https://github.com/Deekshithpoleboina](https://github.com/Deekshithpoleboina)
---
## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.
