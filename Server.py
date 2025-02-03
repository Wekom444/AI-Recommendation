from flask import Flask, request, jsonify
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Sample product data
products = pd.DataFrame({
    "id": [1, 2, 3],
    "description": ["Red shoes", "Blue jeans", "White shirt"]
})

# Recommendation function
def recommend_products(query):
    vectorizer = TfidfVectorizer()
    product_vectors = vectorizer.fit_transform(products['description'])
    query_vector = vectorizer.transform([query])
    similarity = cosine_similarity(query_vector, product_vectors)
    recommended_index = similarity.argmax()
    return products.iloc[recommended_index].to_dict()

@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.json.get("query")
    recommendation = recommend_products(query)
    return jsonify(recommendation)

if __name__ == '__main__':
    app.run(debug=True)
