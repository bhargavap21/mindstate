"""
Real-time EEG Classification with Muse
Version: 2024.1
"""

import sys
import ModifiedRecord as MyRecord
import ModifiedStream as MyStream
from MockDataStream import MockStreamInlet
from EEGFeatureExtraction import generate_feature_vectors_from_samples
import FeatureSelection as fs
import numpy as np
import pandas as pd
import joblib
import time

"""
Note: for streaming to work the BlueMuse application must already be open and the headset connected/online 
"""

def connect_to_muse(muse_address=None):
    """
    Attempt to connect to Muse headset with proper error handling
    """
    print("\n=== Muse Connection Setup ===")
    print("1. Make sure your Muse headset is:")
    print("   - Turned on")
    print("   - In pairing mode (blinking light)")
    print("   - Fully charged")
    print("2. Ensure Bluetooth is enabled on your computer")
    print("3. Attempting to connect...\n")
    
    try:
        # If no address provided, try to find Muse
        if muse_address is None:
            print("Searching for available Muse devices...")
            # Add a small delay to allow for device discovery
            time.sleep(2)
            # You can add code here to list available devices if needed
            print("Please enter your Muse's Bluetooth address (e.g., 00:55:da:b3:9a:2c):")
            muse_address = input().strip()
        
        print(f"Attempting to connect to Muse at {muse_address}...")
        MyStream.stream(muse_address)
        print("Connection successful!")
        return True
    except Exception as e:
        print(f"\nError connecting to Muse: {str(e)}")
        print("\nTroubleshooting steps:")
        print("1. Check if Bluetooth is enabled")
        print("2. Make sure Muse is in pairing mode")
        print("3. Try moving closer to the Muse")
        print("4. Restart the Muse headset")
        print("5. Restart your computer's Bluetooth")
        return False

def main(clf_path, training_file_path):
    """
    :param clf_path: String
        File path of the classifier to use
    :param training_file_path: String
        File path of the training matrix to use
    """
    # Feature selection
    selected_features = fs.feature_selection(training_file_path)

    # Load classifier
    clf = joblib.load(clf_path)

    # Connect to Muse
    if not connect_to_muse():
        print("\nWould you like to:")
        print("1. Try connecting again")
        print("2. Use mock data for testing")
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            return main(clf_path, training_file_path)
        else:
            print("\nUsing mock data for testing...")
            inlet = MockStreamInlet()
    else:
        # Finds existing LSL stream & starts acquiring data
    inlet = MyRecord.start_stream()

    print("\n=== Starting EEG Classification ===")
    print("Press Ctrl+C to stop the program\n")

    try:
    while True:
        """
        Generates a 2D array containing features as columns and time windows as rows and a list containing all feature names
        cols_to_ignore: -1 to remove last column from csv (remove Right AUX column) 
        """
        # Feature extraction
        results, names = generate_feature_vectors_from_samples(MyRecord.record_numpy(2, inlet), 150, 1, cols_to_ignore=-1)
        # Code commented below used for testing as it runs the script without needing the Muse headset
        # results, names = generate_feature_vectors_from_samples(MyRecord.record_numpy(2, MockStreamInlet()), 150, 1, cols_to_ignore=-1)
        data = pd.DataFrame(data=results, columns=names)

        # Feature selection
        selected_data = data[selected_features].copy()

        # Classification
        probability = clf.predict_proba(selected_data)
        for sample in probability:
                print("\r", end="")  # Clear current line
                if sample[0] > 0.5:
                    print("State: Relaxed", end="")
                if sample[1] == 0:
                    odds = sys.maxsize
                else:
                    odds = round((sample[0] / sample[1]), 2)
                    print(f" (Confidence: {sample[0]:.2%}, Odds: {odds})", end="")
                elif sample[1] > 0.5:
                    print("State: Concentrating", end="")
                if sample[0] == 0:
                    odds = sys.maxsize
                else:
                    odds = round((sample[1] / sample[0]), 2)
                    print(f" (Confidence: {sample[1]:.2%}, Odds: {odds})", end="")
            else:
                    print("State: Unknown", end="")
                sys.stdout.flush()  # Ensure output is displayed immediately

    except KeyboardInterrupt:
        print("\n\nProgram stopped by user")
    except Exception as e:
        print(f"\n\nError during execution: {str(e)}")
    finally:
        print("\nCleaning up...")
        # Add any cleanup code here if needed

    return None


# TODO enter file path of the training matrix
training_file_path = r"CSV files/ParticipantOne_Training_Matrix.csv"

# TODO enter file path of the classifier
clf_path = r"Models/ParticipantOne_RF_Model_New.pkl"

if __name__ == "__main__":
    main(clf_path, training_file_path)
