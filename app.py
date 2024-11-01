from flask import Flask, render_template, request, jsonify
import pickle
import attacut
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from safetensors.torch import load_file  # ใช้สำหรับโหลดโมเดลจาก safetensors
import torch.nn.functional as F

app = Flask(__name__)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Naive Bayes model and TF-IDF vectorizer
with open('/Users/nattawadeekwankao/Desktop/KMITL/ThirdYear/Practical Project/Demo/Model/model_tfidfNaive.pkl', 'rb') as model_file:
    best_mnb = pickle.load(model_file)

with open('/Users/nattawadeekwankao/Desktop/KMITL/ThirdYear/Practical Project/Demo/Model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

tokenizer_bert = AutoTokenizer.from_pretrained("KoichiYasuoka/bert-base-thai-upos")
# Load BERT model
bert_model = AutoModelForSequenceClassification.from_pretrained("KoichiYasuoka/bert-base-thai-upos", num_labels=10, ignore_mismatched_sizes=True)
bert_model.load_state_dict(torch.load('/Users/nattawadeekwankao/Desktop/KMITL/ThirdYear/Practical Project/Demo/Model/model.pt', map_location=device), strict=False)
bert_model.eval().to(device)


# Load WangchanBERTa model and tokenizer
tokenizer_wangchanberta = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased")
wangchanberta_model_directory = '/Users/nattawadeekwankao/Desktop/KMITL/ThirdYear/Practical Project/Demo/Model/model_wangchan.pt'
wangchanberta_model = AutoModelForSequenceClassification.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased", num_labels=10)
wangchanberta_model.load_state_dict(torch.load(wangchanberta_model_directory, map_location=device), strict=False)
wangchanberta_model.eval().to(device)  # Set model to evaluation mode

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    categories = ['การเมือง', 'ข่าวการเงิน', 'ข่าวกีฬา', 'ข่าวบันเทิง', 'ข่าวรถยนต์', 'ข่าวเกมส์', 'ข่าวไอที', 'ดูดวง', 'สุขภาพ', 'อาชญากรรม']
    data = request.get_json()
    text = data['text'] 

    # Preprocess text using attacut for tokenization
    tokenized_text = preprocess_text(text)

    # Naive Bayes prediction
    tfidf_vector = vectorizer.transform([tokenized_text])
    prediction_nb, probabilities_nb = predict_naive_bayes(tfidf_vector)

    # Convert probabilities to a list
    probabilities_nb = probabilities_nb.flatten().tolist()

    # Get the index of the prediction
    prediction_nb_index = categories.index(prediction_nb[0])  # Find the index of the predicted category

    # BERT prediction
    prediction_bert, probabilities_bert = predict(text, tokenizer_bert, bert_model)
    probabilities_bert = probabilities_bert.flatten().tolist()

    # WangchanBERTa prediction
    prediction_wangchanberta, probabilities_wangchanberta = predict(text, tokenizer_wangchanberta, wangchanberta_model)
    probabilities_wangchanberta = probabilities_wangchanberta.flatten().tolist()

    return jsonify({
        'tfidf': {
            'legend': prepare_response_data(best_mnb.classes_, [probabilities_nb]),
            'prediction': prediction_nb_index,  # Use the index of the prediction
            'probabilities': probabilities_nb
        },
        'bert': {
            'legend': prepare_response_data(categories, [probabilities_bert]),
            'prediction': prediction_bert,  # Directly use the integer prediction
            'probabilities': probabilities_bert
        },
        'wangchanberta': {
            'legend': prepare_response_data(categories, [probabilities_wangchanberta]),
            'prediction': prediction_wangchanberta,  # Directly use the integer prediction
            'probabilities': probabilities_wangchanberta
        }
    })


def preprocess_text(text):
    # Use attacut for tokenization
    return ' '.join(attacut.tokenize(text))

def predict_naive_bayes(tfidf_vector):
    prediction = best_mnb.predict(tfidf_vector)
    probabilities = best_mnb.predict_proba(tfidf_vector)
    return prediction, probabilities

def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    
    with torch.no_grad():  # Disable gradient tracking
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)  # Convert logits to probabilities
    predicted_class = torch.argmax(logits, dim=-1).item()  # Get the class with the highest score

    # Convert probabilities to numpy array for output
    probabilities = probabilities.squeeze().cpu().numpy()  # Reduce tensor dimension and move to CPU
    return predicted_class, probabilities

def prepare_response_data(class_names, probabilities):
    return [
        {
            'label': class_names[i],
            'value': float(probabilities[0][i]) * 100,
            'color': '#' + ''.join([hex(int(probabilities[0][i] * 255))[2:].zfill(2)] * 3)
        } for i in range(len(class_names))
    ]

if __name__ == "__main__":
    app.run(debug=True)
