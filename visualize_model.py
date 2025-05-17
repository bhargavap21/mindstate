"""
Script to visualize the model's accuracy using confusion matrix and classification report.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import FeatureSelection as fs
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def visualize_model_performance(training_path, model_path):
    """
    Visualize model performance using confusion matrix and classification report.
    
    Args:
        training_path (str): Path to the training data CSV file
        model_path (str): Path to the saved model file
    """
    # Load the data
    data = pd.read_csv(training_path)
    
    # Feature selection
    selected_features = fs.feature_selection(training_path)
    
    # Prepare the data
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()
    
    # Split the data
    X = selected_data.drop('Label', axis=1)
    y = selected_data['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the model
    model = joblib.load(model_path)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create a figure with two subplots
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Confusion Matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Plot 2: Feature Importance
    plt.subplot(1, 2, 2)
    feature_importance = pd.DataFrame({
        'feature': selected_features,
        'importance': model.feature_importances_
    })
    feature_importance = feature_importance.sort_values('importance', ascending=False).head(10)
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('model_performance.png')
    plt.close()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Calculate and print accuracy
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    print(f"\nOverall Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    training_file_path = "CSV files/ParticipantOne_Training_Matrix.csv"
    model_path = "Models/ParticipantOne_RF_Model_New.pkl"
    visualize_model_performance(training_file_path, model_path) 