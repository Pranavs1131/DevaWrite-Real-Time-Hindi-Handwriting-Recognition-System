DevaWrite is a real-time handwriting recognition system built to recognize Hindi characters written in the Devanagari script. The system utilizes Convolutional Neural Networks (CNN) and OpenCV for detecting blue pen strokes and classifying them into predefined Devanagari characters.

Features
Real-time Recognition: Detects and classifies handwritten Devanagari characters in real-time using a webcam.

Stroke Tracking: Tracks pen strokes in real-time and processes the drawn character.

User-Friendly Interface: Displays the predicted character and confidence level as feedback.

Reset Functionality: Allows the user to clear the canvas and start drawing a new character.

Technology Stack
Python: The main programming language.

OpenCV: For image processing, capturing webcam feed, and stroke tracking.

Keras (TensorFlow): Used for building and training the Convolutional Neural Network (CNN) model.

NumPy: Used for handling arrays and mathematical operations.

Installation
To run the project locally, follow these steps:

Clone the repository:

bash
Copy
git clone https://github.com/Pranavs1131/DevaWrite-Real-Time-Hindi-Handwriting-Recognition-System.git
cd DevaWrite-Real-Time-Hindi-Handwriting-Recognition-System
Set up the virtual environment (Optional):
It is highly recommended to create a virtual environment to avoid conflicts with system dependencies.

bash
Copy
python -m venv venv
source venv/bin/activate  # On Linux/macOS
.\venv\Scripts\activate  # On Windows
Install dependencies:
Use pip to install the necessary libraries:

bash
Copy
pip install -r requirements.txt
Run the application:
Once dependencies are installed, you can run the application:

bash
Copy
python Application.py
This will open the webcam feed, allowing you to draw and get predictions for handwritten Devanagari characters.

Model Details
The handwriting recognition model is a Convolutional Neural Network (CNN) that has been trained to recognize 47 distinct Devanagari characters, including vowels, consonants, and other specific symbols used in the Hindi language.

Model File: devanagari_model.h5

This file contains the trained model weights and architecture.

Make sure to load the model using Keras for predictions.

File Structure
bash
Copy
DevaWrite-Real-Time-Hindi-Handwriting-Recognition-System/
│
├── Application.py              # Main application to run the recognition system
├── devanagari_model.h5         # Trained model for character recognition
├── data/
│   └── data.csv                # Sample dataset used to train the model
├── requirements.txt            # List of Python dependencies
└── README.md                   # Project documentation (this file)
Contributing
Feel free to fork the repository, create issues, and submit pull requests. Contributions to improve the recognition model or add features are always welcome.

Steps to Contribute:
Fork this repository.

Create a feature branch (git checkout -b feature-branch).

Commit your changes (git commit -m 'Add new feature').

Push to the branch (git push origin feature-branch).

Create a new Pull Request.

License
This project is licensed under the MIT License.

Acknowledgments
TensorFlow/Keras: For providing the deep learning framework.

OpenCV: For computer vision tasks, like webcam feed capture and image processing.

