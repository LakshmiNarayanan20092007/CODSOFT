from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load dataset
data = pd.read_csv(
    "Genre Classification Dataset/train_data.txt",
    sep=":::",
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"],
    engine="python"
)

# Keep needed columns
data = data[["GENRE", "DESCRIPTION"]]

# Remove empty values
data.dropna(inplace=True)

# Input and output
X = data["DESCRIPTION"]
y = data["GENRE"]

# Convert text into numbers
vectorizer = TfidfVectorizer(stop_words='english')

X_vectorized = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        movie_description = request.form["description"]

        transformed_text = vectorizer.transform([movie_description])

        prediction = model.predict(transformed_text)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)