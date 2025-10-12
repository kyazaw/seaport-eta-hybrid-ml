1. Import necessary libraries

2. Load the dataset

3. Preprocess Data
   
   a. Convert datetime columns, and calculate target (e.g., travel time in seconds)
   b. Extract relevant features (numeric and categorical)
   c. Build preprocessing pipelines for numeric (imputation, scaling) and categorical (imputation, one-hot encoding) features
   d. Apply transformations (fit on train, transform train/test)

5. Train-Test Split
   
   a. Split data into training and test sets

6. Train Base Models (on training data)
   
   a. Extra Trees Regressor (with hyperparameter tuning)
   b. LightGBM Regressor (with hyperparameter tuning)
   c. AutoGluon Tabular Predictor (automated stacking/ensembling)

7. Generate Predictions (on the test data)
   
   a. Predict with Extra Trees, LightGBM, and AutoGluon on test set
   b. Form meta-features as a DataFrame of these predictions

8. Train Meta-Learner (Stacking)
   
   a. Define hyperparameter search space for Random Forest Regressor
   b. Use cross-validated Bayesian optimization to search best Random Forest parameters
   c. Train Random Forest meta-learner on meta-features (test predictions of base models) and true test targets

9. Make Final Ensemble Predictions

   a. Use trained meta-learner to predict on test set meta-features

10. Evaluate Model Performance
    
   a. Compute performance metrics (MAE, MSE, R², MAPE) for:
      - Stacking ensemble
      - Each base model (Extra Trees, LightGBM, AutoGluon)
   b. Optionally, perform and report k-fold cross-validation scores for ensemble

11. Visualization
    
    a. Plot Actual vs. Predicted values for ensemble and base models
