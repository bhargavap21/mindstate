import pandas as pd
import FeatureSelection as fs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib


def build_classifier(training_path, test_size, clf_output_file):
    """
    Builds and saves a trained random forest classifier.
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
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size)

    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(x_train, y_train)
    # Save the model
    joblib.dump(clf, clf_output_file)

    # Predict on the testing data
    y_predict = clf.predict(x_test)
    print("Accuracy of classifier = " + str(accuracy_score(y_test, y_predict)))

    return None


def split_dataset(dataset, test_size):
    """
    Split the data into train and test sets.
    :param dataset: pandas.DataFrame
        The dataset to split.
    :param test_size: float
        Proportion of data to use for testing.
    :return: Training and test sets.
    """
    x = dataset.drop('Label', axis=1)
    y = dataset['Label']
    # split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    return x_train, x_test, y_train, y_test


def classification_accuracy(clf_path, training_path, test_size):
    """
    Returns the accuracy of a trained classifier given unseen data.
    :param clf_path: String
        File path of the classifier to test.
    :param training_path: String
        File path for the training matrix.
    :param test_size: float
        Proportion of data to use for testing.
    :return: The accuracy of the given classifier
    """
    # Load dataset into pandas.DataFrame
    data = pd.read_csv(training_path)
    # Load classifier
    clf = joblib.load(clf_path)

    # Feature selection - Note: feature selection is based on the entire dataset
    selected_features = fs.feature_selection(training_path)
    feature_names_plus_label = selected_features.copy()
    feature_names_plus_label.append("Label")
    selected_data = data[feature_names_plus_label].copy()

    # Split the dataset into train and test data
    x_train, x_test, y_train, y_test = split_dataset(selected_data, test_size)
    # Predict on the testing data and calculate accuracy
    y_predict = clf.predict(x_test)
    print(accuracy_score(y_test, y_predict))

    return accuracy_score(y_test, y_predict)


# # Old methods for reference
# def build_classifier(training_path, testing_path, clf_output_file):
#     """
#     Builds and saves a trained random forest classifier.
#     :param training_path: File path for the training matrix.
#     :param testing_path: File path for the testing matrix.
#     :param clf_output_file: Name of file to save the classifier to.
#     """
#
#     data = pd.read_csv(training_path)
#
#     # Feature selection
#     selected_features = fs.feature_selection(training_path)
#
#     # Create new dataset containing only selected features
#     feature_names_plus_label = selected_features.copy()
#     feature_names_plus_label.append("Label")
#     selected_data = data[feature_names_plus_label].copy()
#     x_train = selected_data.drop('Label', axis=1)
#     y_train = selected_data['Label']
#
#     # Train Random Forest classifier
#     clf = RandomForestClassifier(n_estimators=100)
#     clf = clf.fit(x_train, y_train)
#     # Save the model
#     joblib.dump(clf, clf_output_file)
#
#     # Testing
#     testing_data = pd.read_csv(testing_path)
#     selected_testing_data = testing_data[feature_names_plus_label].copy()
#     x_test = selected_testing_data.drop('Label', axis=1)
#     y_true = selected_testing_data['Label']
#     y_predict = clf.predict(x_test)
#     print("Accuracy of classifier = " + str(accuracy_score(y_true, y_predict)))
#
#     return None
#
#
# def classification_accuracy(clf, training_path, testing_path):
#     """
#     Returns the accuracy of a trained classifier given unseen data.
#     :param clf: the classifier to test.
#     :param training_path: File path for the training matrix.
#     :param testing_path: File path for the testing matrix.
#     :return: The accuracy of the given classifier
#     """
#
#     # Feature selection
#     selected_features = fs.feature_selection(training_path)
#
#     testing_data = pd.read_csv(testing_path)
#
#     # Feature selection
#     feature_names_plus_label = selected_features.copy()
#     feature_names_plus_label.append("Label")
#     selected_testing_data = testing_data[feature_names_plus_label].copy()
#
#     y_true = selected_testing_data['Label']
#     x_test = selected_testing_data.drop('Label', axis=1)
#
#     y_predict = clf.predict(x_test)
#     return accuracy_score(y_true, y_predict)
