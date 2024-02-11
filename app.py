from flask import Flask, request, jsonify
import torch
from BART_model import BART  # Ensure this imports your model class
from transformers import BartTokenizer, BartForConditionalGeneration
import os
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

# Load your model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
bart_model = BART(tokenizer, pretrained_bart, )
bart_model.load_state_dict(torch.load('bart_best_model.pth', map_location=torch.device('cpu')))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['text']
    # You might need to adjust the following line according to your model's method for prediction
    output = bart_model.predict(input_text, K=1, max_T = 300)
    return jsonify({'sql_query': output})