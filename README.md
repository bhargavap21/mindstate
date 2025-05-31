# MindState

A Python-based system for real-time EEG classification using the Muse headset. This project enables real-time classification of mental states (relaxed vs. concentrating) using EEG data from the Muse headband.

## Author

**Bhargava Perumalla**

## Features

- Real-time EEG data acquisition from Muse headset
- Feature extraction and selection from EEG signals
- Machine learning-based classification of mental states
- Support for both real-time classification and model training
- Mock data streaming for testing without a Muse headset
- High accuracy classification (96.4% with the provided model)

## Requirements

- Python 3.9 or higher
- Muse headset (or mock data for testing)
- Bluetooth connectivity
- BlueMuse application (for Muse headset connection)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/bhargavap21/real-time-eeg-classification.git
cd real-time-eeg-classification
```

2. Create and activate a virtual environment:
```bash
python -m venv eeg-venv
source eeg-venv/bin/activate  # On Windows: eeg-venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

### Core Files
- `main.py` - Main program for real-time classification
- `Build&TestClassifier.py` - Model training and testing
- `EEGFeatureExtraction.py` - Feature extraction from EEG data
- `FeatureSelection.py` - Feature selection for model training
- `ModifiedStream.py` - Muse headset connection handling
- `ModifiedRecord.py` - EEG data recording
- `MockDataStream.py` - Mock data generation for testing

### Data and Models
- `CSV files/` - Directory containing training data
- `Models/` - Directory containing trained models

## Usage

### Real-Time Classification

1. Ensure BlueMuse is running and your Muse headset is connected
2. Run the main program:
```bash
python main.py
```

The program will:
- Connect to your Muse headset
- Stream EEG data in real-time
- Extract features from the data
- Classify the mental state (relaxed vs. concentrating)
- Display the results in real-time

### Model Training

1. Prepare your training data in CSV format
2. Run the training script:
```bash
python Build&TestClassifier.py
```

This will:
- Load the training data
- Perform feature selection
- Train a Random Forest classifier
- Save the trained model
- Display the model's accuracy

### Testing Without Muse Headset

If you don't have a Muse headset, the program can use mock data:
1. Run the main program
2. When prompted for connection, choose option 2 for mock data
3. The program will run with simulated EEG data

## Model Performance

The current model achieves:
- Overall accuracy: 96.4%
- Real-time classification of mental states
- Confidence scores for predictions

## Troubleshooting

### Muse Connection Issues
1. Ensure Bluetooth is enabled
2. Check if Muse is in pairing mode
3. Verify BlueMuse is running
4. Try moving closer to the Muse
5. Restart the Muse headset
6. Restart your computer's Bluetooth

### Program Errors
1. Verify all dependencies are installed
2. Check if the model file exists in the Models directory
3. Ensure training data is in the correct format
4. Check if BlueMuse is running (for real-time classification)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Acknowledgments

- Original Muse LSL implementation
- Contributors to the scikit-learn library
- The Muse headset development team
