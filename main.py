

import pickle
from flask import Flask, request, jsonify

app = Flask('medico_prediction')


@app.route('/predict', methods=['POST'])
def predict():
    medicine = request.get_json()
    print(medicine)
    with open('Chain_medico_care.pkl', 'rb') as f_in:
        model = pickle.load(f_in)
        f_in.close()
    predictions = ml_model.Pipe.predict(medicine, model)

    result = {
        'medico_prediction': list(predictions)
    }
    return jsonify(result)

#
# @app.route('/ping', methods=['GET'])
# def ping():
#     return "Pinging Model!!"
import ml_model