from flask import Flask, render_template, request, send_file, jsonify
import os
import torch
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from googlesearch import search

app = Flask(__name__)
REPORT_FOLDER = "reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
ai_model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector").to(device)
tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")

def check_ai_generated(text):
    """Detects AI-generated content using RoBERTa model."""
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = ai_model(**inputs)
    score = torch.softmax(outputs.logits, dim=1)[0][1].item()
    return round(score, 2)

def search_web(text):
    """Performs a Google search for similar content."""
    query = " ".join(text.split()[:10])  # Use first 10 words for a meaningful search
    results = []
    try:
        for url in search(query, num_results=5):  # Fetch top 5 results
            try:
                response = requests.get(url, timeout=5)
                soup = BeautifulSoup(response.content, "html.parser")
                page_text = " ".join([p.get_text() for p in soup.find_all("p")])
                results.append((url, page_text))
            except Exception as e:
                print(f"Skipping {url} due to error: {e}")
    except Exception as e:
        print(f"Google Search Error: {e}")
    return results

def compute_similarity(user_text, web_texts):
    """Computes similarity scores between user input and web results."""
    if len(user_text.split()) < 10:
        return []

    texts = [user_text] + [t[1] for t in web_texts]  # Combine texts for comparison
    vectorizer = TfidfVectorizer().fit_transform(texts)
    vectors = vectorizer.toarray()
    similarities = (vectors @ vectors.T)[0][1:]  # Compute cosine similarity

    return [(web_texts[i][0], round(similarities[i] * 100, 2)) for i in range(len(web_texts))]  # Convert to percentage

def generate_pdf(content):
    """Generates a PDF report of the content."""
    pdf_path = os.path.join(REPORT_FOLDER, "Content.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [Paragraph("", styles["Title"]), Spacer(1, 12)]

    for line in content.split("\n"):
        elements.append(Paragraph(line, styles["BodyText"]))
        elements.append(Spacer(1, 6))

    doc.build(elements)
    return pdf_path

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/check", methods=["POST"])
def check():
    """Handles plagiarism checking."""
    text = request.form.get("content")
    ai_score = check_ai_generated(text)
    web_results = search_web(text)
    similarity_scores = compute_similarity(text, web_results)

    return jsonify({"ai_score": ai_score, "plagiarism_results": similarity_scores})

@app.route("/download", methods=["POST"])
def download():
    """Handles PDF download."""
    content = request.form.get("content")
    pdf_path = generate_pdf(content)
    return send_file(pdf_path, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
