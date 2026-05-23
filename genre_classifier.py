import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv(
    "Genre Classification Dataset/train_data.txt",
    sep=":::",
    names=["ID", "TITLE", "GENRE", "DESCRIPTION"],
    engine="python"
)

# Keep only needed columns
data = data[["GENRE", "DESCRIPTION"]]

# Remove empty values
data.dropna(inplace=True)

# Input and output
X = data["DESCRIPTION"]
y = data["GENRE"]

# Convert text into numbers
vectorizer = TfidfVectorizer(max_features=5000)

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

# Test accuracy
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Test custom prediction
sample = ["A group of astronauts travel through space to save humanity"]

sample_vector = vectorizer.transform(sample)

prediction = model.predict(sample_vector)

print("Predicted Genre:", prediction[0])