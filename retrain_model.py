"""
Script to retrain the model with modern scikit-learn version.
"""

import pandas as pd
import FeatureSelection as fs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib

def retrain_model(training_path, test_size, clf_output_file):
    """
    Retrains and saves a random forest classifier using modern scikit-learn.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :param clf_output_file: String
        Name of file to save the classifier to.
    """
    # Load dataset into pandas.DataFrame
    data = pd.read_csv(training_path)

    # Feature selection - Note: feature selection is based on the entire dataset
    selected_features = fs.feature_selection(training_path)

    # Create new dataset containing only selected features
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()

    # Split dataset into train and test (x is data, y is labels)
    x = selected_data.drop('Label', axis=1)
    y = selected_data['Label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf = clf.fit(x_train, y_train)
    
    # Save the model
    joblib.dump(clf, clf_output_file)

    # Predict on the testing data
    y_predict = clf.predict(x_test)
    print("Accuracy of classifier = " + str(accuracy_score(y_test, y_predict)))

    return None

if __name__ == "__main__":
    # TODO: Update these paths to match your setup
    training_file_path = "CSV files/ParticipantOne_Training_Matrix.csv"
    clf_output_file = "Models/ParticipantOne_RF_Model_New.pkl"
    
    retrain_model(training_file_path, test_size=0.2, clf_output_file=clf_output_file) 