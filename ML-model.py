import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
usda_data = pd.read_csv(r"C:\Users\Aditya\Downloads\USDA.csv") 
classification_data = pd.read_excel(r"C:\Users\Aditya\OneDrive\Desktop\python\calorie_tracker\Classificationmodel.xlsx") 

# Prepare data
data = classification_data[['Calories (kcal)', 'Carbohydrates (g)', 'Protein (g)', 'Fat (g)', 'Fiber (g)', 'Sugar (g)', 'Sodium (mg)']]
def classify_food(row):
    if row['Protein (g)'] > 15:
        return 'High-Protein'
    elif row['Carbohydrates (g)'] > 20:
        return 'High-Carb'
    elif row['Fat (g)'] < 5:
        return 'Low-Fat'
    else:
        return 'Balanced'
data['Category'] = data.apply(classify_food, axis=1)

# Feature columns and target column
X = data.drop('Category', axis=1)
y = data['Category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter tuning
param_grid = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Training with cross-validation
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5)
print("Cross-Validation Accuracy:", cv_scores.mean())

# Train and evaluate model
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Feature importance
importances = best_model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance in Food Classification")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.show()

# Save model and scaler
joblib.dump(best_model, 'food_classification_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Example of using the model on new data
def classify_new_food(data_row):
    data_row_scaled = scaler.transform([data_row])
    return best_model.predict(data_row_scaled)

# Example usage
new_food_data = [200, 30, 5, 2, 4, 6, 50] 
category = classify_new_food(new_food_data)
print("Predicted Category:", category[0])
