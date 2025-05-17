# Real-Time EEG Classification with Muse

A real-time EEG classification system that uses a Muse headset to detect whether a person is in a relaxed or concentrated state.

## Features

- Real-time EEG data processing
- Machine learning-based state classification (Relaxed vs. Concentrated)
- Live confidence scores and odds ratios
- Support for both real Muse headset and mock data
- Modern scikit-learn compatibility
- Apple Silicon (M1/M2) support

## Requirements

- Python 3.9 or higher
- Muse headset (optional, can use mock data)
- Bluetooth connectivity (for real Muse headset)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/real-time-eeg-classification.git
cd real-time-eeg-classification
```

2. Create a virtual environment (recommended):
```bash
python -m venv eeg-venv
source eeg-venv/bin/activate  # On Windows: eeg-venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Connect your Muse headset:
   - Turn on the Muse headset
   - Put it in pairing mode (blinking light)
   - Ensure Bluetooth is enabled on your computer

2. Run the main script:
```bash
python main.py
```

3. When prompted, enter your Muse's Bluetooth address (e.g., 00:55:da:b3:9a:2c)

4. The program will start classifying your mental state in real-time:
   - "State: Relaxed" - When you're in a relaxed state
   - "State: Concentrating" - When you're in a concentrated state
   - "State: Unknown" - When the state is unclear

5. Press Ctrl+C to stop the program

## Using Mock Data

If you don't have a Muse headset, you can use mock data for testing:
1. Run the program
2. When connection fails, choose option 2 to use mock data

## Model Performance

The current model achieves:
- Overall accuracy: 96.40%
- Balanced performance for both relaxed and concentrated states
- High precision and recall for both classes

## Project Structure

- `main.py` - Main program for real-time classification
- `EEGFeatureExtraction.py` - Feature extraction from EEG data
- `FeatureSelection.py` - Feature selection for the model
- `ModifiedStream.py` - Muse headset streaming interface
- `ModifiedRecord.py` - Data recording functionality
- `MockDataStream.py` - Mock data generation for testing
- `Models/` - Directory containing trained models
- `CSV files/` - Directory containing training data

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 
