# Sentiment Analysis Web App

This project is a simple sentiment analysis web application built with FastAPI, using a Naive Bayes model to classify text comments into multiple sentiment categories.

## Project Structure

- `app.py` — Main FastAPI application that serves the web interface and handles prediction requests.
- `train_nb_model.py` — Script to train the Naive Bayes sentiment classification model on the dataset.
- `sentiment_dataset.csv` — Dataset file used to train the model.
- `templates/index.html` — HTML template for the web front-end.
- `static/style.css` — CSS stylesheet for styling the web interface.

## Features

- Multi-label sentiment classification of input text.
- Supports labels such as toxic, severe toxic, obscene, threat, insult, and identity hate.
- Simple and responsive web interface for user interaction.

## Installation

1. Install Python 3.8 or higher.
2. Install required dependencies:

```bash
pip install fastapi uvicorn scikit-learn joblib pandas
```
## Usage

Train the model (optional, if you want to retrain it):

python train_nb_model.py

This will train the Naive Bayes classifier on the dataset and save the model pipeline.

Run the FastAPI application:

uvicorn app:app --reload

Open your browser and navigate to:

http://127.0.0.1:8000

Use the web interface to input text and receive sentiment classification results.

## How It Works
The model predicts probabilities for each sentiment label.

A threshold (default 0.5) determines whether a label is assigned.

The app supports multiple labels per input (multi-label classification).


## Notes
The train_nb_model.py script uses sentiment_dataset.csv as training data.

The model is saved and loaded using joblib.

The front-end HTML and CSS are located in the templates and static directories respectively.

Ensure the model file path in app.py matches where the trained model is saved.


## License
This project is provided as-is for educational and demonstration purposes. By Zhanibek
