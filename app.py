from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Load dataset
df = pd.read_csv("spam.csv", encoding='latin-1')

# Keep needed columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Inputs and outputs
X = df['message']
y = df['label']

# Text vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    data = vectorizer.transform([message])

    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Spam Message ❌"
    else:
        result = "Not Spam ✅"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)