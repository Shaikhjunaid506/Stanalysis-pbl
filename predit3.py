import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Load the dataset
stud = pd.read_csv("C:/Stanalysis pbl/modified_student_data_india.csv")

# Step 2: Identify categorical columns
categorical_cols = stud.select_dtypes(include=['object']).columns
print("Categorical columns:", categorical_cols)

# Step 3: One-hot encode categorical columns
stud = pd.get_dummies(stud, columns=categorical_cols, drop_first=True)

# Step 4: Splitting data into features and target variable
X = stud.drop(columns='G3')  # Features
y = stud['G3']                # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 3: Function to calculate MAE and RMSE
def evaluate_predictions(predictions, true):
    mae = np.mean(abs(predictions - true))
    rmse = np.sqrt(np.mean((predictions - true) ** 2))
    return mae, rmse

# Create a baseline using the median of the training set
median_pred = y_train.median()
median_preds = [median_pred] * len(y_test)

# Display the naive baseline metrics
mb_mae, mb_rmse = evaluate_predictions(median_preds, y_test)
print('Median Baseline  MAE: {:.4f}'.format(mb_mae))
print('Median Baseline RMSE: {:.4f}'.format(mb_rmse))

# Step 4: Evaluate models
def evaluate(X_train, X_test, y_train, y_test):
    model_name_list = ['Linear Regression', 'ElasticNet Regression',
                       'Random Forest', 'Extra Trees', 'SVM', 'Gradient Boosted', 'Baseline']
    
    # Instantiate the models
    models = [
        LinearRegression(),
        ElasticNet(alpha=1.0, l1_ratio=0.5),
        RandomForestRegressor(n_estimators=100),
        ExtraTreesRegressor(n_estimators=100),
        SVR(kernel='rbf', degree=3, C=1.0, gamma='auto'),
        GradientBoostingRegressor(n_estimators=50)
    ]
    
    # Dataframe for results
    results = pd.DataFrame(columns=['mae', 'rmse'], index=model_name_list)
    
    # Train and predict with each model
    for i, model in enumerate(models):
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calculate metrics
        mae = np.mean(abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        
        # Insert results into the dataframe
        results.loc[model_name_list[i]] = [mae, rmse]
    
    # Median Value Baseline Metrics
    baseline = np.median(y_train)
    baseline_mae = np.mean(abs(baseline - y_test))
    baseline_rmse = np.sqrt(np.mean((baseline - y_test) ** 2))
    results.loc['Baseline'] = [baseline_mae, baseline_rmse]
    
    return results

# Call the evaluate function and display results
results = evaluate(X_train, X_test, y_train, y_test)
print("Model Evaluation Results:")
print(results)

# Step 5: Make predictions about students scoring above average
# Calculate the average score
average_score = stud['G3'].mean()

# Use the best-performing model to predict outcomes (for example, Random Forest)
best_model = RandomForestRegressor(n_estimators=100)
best_model.fit(X, y)  # Fit on the entire dataset

# Get predictions for all students
predictions = best_model.predict(X)

# Add predictions to the DataFrame
stud['Predicted_G3'] = predictions

# Step 6: Predict Pass/Fail Outcomes
# Assuming passing score is greater than 10
passing_score = 10
stud['Pass/Fail_Prediction'] = np.where(stud['Predicted_G3'] > passing_score, 'Pass', 'Fail')

# Step 7: Display predictions for each student with pass/fail prediction
print("\nPredictions for each student:")
print(stud[['G3', 'Predicted_G3', 'Pass/Fail_Prediction', 'school_KV']])

# Step 8: Analyze pass/fail ratio based on school
pass_fail_summary = stud.groupby('school_KV')['Pass/Fail_Prediction'].value_counts().unstack(fill_value=0)

# Calculate pass/fail ratios
pass_fail_summary['Pass/Fail_Ratio'] = pass_fail_summary['Pass'] / (pass_fail_summary['Fail'] + 1e-5)  # Added small value to avoid division by zero

# Display pass/fail ratios
print("\nPass/Fail Ratio by School:")
print(pass_fail_summary[['Pass', 'Fail', 'Pass/Fail_Ratio']])
