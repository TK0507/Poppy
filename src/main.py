from poppy.models import Model
from poppy.models import ModelTrainer
from poppy.models import load_model, save_model
from poppy.datasets import load_samples
from poppy.datasets import bertify
from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
import os


model = Model()
model_trainer = ModelTrainer(model)

app = Flask(__name__)
CORS(app)


@app.route('/api/beta/predict', methods=['POST'])
def _api_beta_predict():

    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'No sample provided!'})

    return jsonify({'score': model(bertify(text)).item()})


def main():

    samples = load_samples('./sample/sample-20240801.csv')

    if not os.path.exists('./model/poppy-beta.pth'):
        print('loss', model_trainer.train(samples, maxcount=1000))
        save_model('./model/poppy-beta.pth', model)
    else:
        load_model('./model/poppy-beta.pth', model)
        print('loss', model_trainer.train(samples, maxcount=100))
        save_model('./model/poppy-beta.pth', model)

    app.run(host='localhost', port=8080)


if __name__ == '__main__':
    main()
