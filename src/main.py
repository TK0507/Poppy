from poppy.models import Model
from poppy.models import ModelTrainer
from poppy.models import load_model, save_model
from poppy.datasets import load_samples, bertify, save_samples
from poppy.datasets import Sample
from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
from datetime import datetime
import torch
import pathlib

LOCATION_DATA_SAMPLE_USER = pathlib.Path('./data/sample/user/')
LOCATION_DATA_SAMPLE = pathlib.Path('./data/sample/')
LOCATION_DATA_CACHE = pathlib.Path('./data/cache/')
LOCATION_MODEL = pathlib.Path('./model/poppy-beta.pth')

model = Model()
model_trainer = ModelTrainer(model)

app = Flask(__name__)
app_cors = CORS(app)


@app.route('/api/beta/predict', methods=['POST'])
def __api_beta_predict():
    data = request.json

    text = data.get('text')
    if not text:
        return jsonify({'error': 'No sample provided!'}), 400
    if not isinstance(text, str):
        return jsonify({'error': 'Invalid sample type!'}), 400

    try:
        prediction = model(bertify(text)).tolist()
        return jsonify({'content': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/beta/upload', methods=['POST'])
def __api_beta_upload():
    data = request.json

    print('Received data:', data)

    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided!'}), 400
    if not isinstance(text, str):
        return jsonify({'error': 'Text must be a string!'}), 400

    scores = data.get('scores')
    if not scores:
        return jsonify({'error': 'No scores provided!'}), 400
    if not isinstance(scores, list):
        return jsonify({'error': 'Scores must be a list!'}), 400
    if len(scores) != 3:
        return jsonify(
            {'error': 'Scores list must contain exactly 3 elements!'}), 400

    try:
        scores = [float(score) for score in scores]
    except ValueError:
        return jsonify({'error': 'All scores must be valid numbers!'}), 400

    if not all(0.0 <= score <= 1.0 for score in scores):
        return jsonify({'error': 'Scores must be between 0.0 and 1.0!'}), 400

    todaylocation = LOCATION_DATA_SAMPLE_USER.joinpath(
        datetime.now().strftime('%Y-%m-%d'))

    if not todaylocation.exists():
        try:
            todaylocation.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print('Error creating directory:', str(e))
            return jsonify({'error': 'Failed to create directory!'}), 500

    todaystorage = todaylocation.joinpath('uploaded_samples.csv')
    sample = Sample(bertify(text), torch.tensor(scores))

    try:
        if todaystorage.exists():
            existing_samples = load_samples(todaystorage)
            existing_samples.append(sample)
            save_samples(todaystorage, existing_samples)
        else:
            save_samples(todaystorage, [sample])
    except Exception as e:
        print('Error saving samples:', str(e))
        return jsonify({'error': 'Failed to save sample data!'}), 500

    return jsonify({'message': 'Sample uploaded successfully!'})


def main():
    samples = []
    samples.extend(load_samples(LOCATION_DATA_CACHE.joinpath('momotaro.csv')))
    samples.extend(load_samples(LOCATION_DATA_CACHE.joinpath('urashima.csv')))

    if not LOCATION_MODEL.exists():
        print('Training new model...')
        print('loss:', model_trainer.train(samples, maxcount=100))
        save_model(LOCATION_MODEL, model)
    else:
        load_model(LOCATION_MODEL, model)

    app.run(host='localhost', port=8080)


if __name__ == '__main__':
    main()
