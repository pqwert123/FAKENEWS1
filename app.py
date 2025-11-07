from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load ML model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news = request.form['news']
        data = [news]
        vect = vectorizer.transform(data).toarray()
        prediction = model.predict(vect)[0]
        return render_template('index.html', prediction=prediction, text=news)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
