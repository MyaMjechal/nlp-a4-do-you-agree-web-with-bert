from flask import Flask, request, render_template
import torch
from utils import *


app = Flask(__name__)

model = BERT(n_layers, n_heads, d_model, d_ff, d_k, n_segments, vocab_size, max_len, device)
model.load_state_dict(torch.load("models/sbert_model.pt", map_location=device))
model.to(device)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input from the form
        premise = request.form['premise']
        hypothesis = request.form['hypothesis']

        # Encode the sentences into embeddings
        # premise_embedding = model.encode(premise, convert_to_tensor=True)
        # hypothesis_embedding = model.encode(hypothesis, convert_to_tensor=True)

        # # Compute cosine similarity between the embeddings
        # similarity = util.cos_sim(premise_embedding, hypothesis_embedding).item()

        # # Determine the NLI label based on similarity
        # if similarity > 0.75:
        #     label = "Entailment"
        # elif similarity < 0.25:
        #     label = "Contradiction"
        # else:
        #     label = "Neutral"
        similarity = calculate_similarity(model, tokenizer, premise, hypothesis, device)
        similarity = round(similarity, 4)
        predicted_label = predict_nli(model, premise, hypothesis)

        return render_template('index.html', premise=premise, hypothesis=hypothesis, label=predicted_label, similarity=similarity)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)
