# CodeAlpha FAQ Chatbot

## Project Overview
This project is an NLP-based FAQ chatbot developed for the CodeAlpha internship. It answers frequently asked questions related to accounts, passwords, customer support, working hours, and orders.

## Technologies Used
- Python
- Natural Language Processing (NLP)
- Scikit-learn
- TF-IDF Vectorization
- Cosine Similarity
- Gradio

## How It Works
1. FAQ questions and answers are stored in a knowledge base.
2. User input is converted to lowercase and cleaned using basic text preprocessing.
3. FAQ questions are converted into TF-IDF vectors.
4. The chatbot uses cosine similarity to find the closest FAQ question.
5. Common FAQ categories are handled directly.
6. A similarity threshold of 0.75 is used before returning a matched FAQ answer.
7. If no suitable match is found, the chatbot returns a fallback response.
8. Gradio provides the interactive web interface.

## Main Features
- Account and registration questions
- Password reset questions
- Customer support questions
- Working-hours questions
- Order cancellation and tracking questions
- Similarity-based FAQ matching
- Fallback response for unsupported questions
- Gradio web interface

## Run in Google Colab
Open `CodeAlpha_FAQ_Chatbot.ipynb` in Google Colab.

The final **ONE-CLICK START** cell recreates `app.py`, installs the required packages, and launches the chatbot with a temporary public Gradio link.

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```

The application will start a Gradio interface and create a temporary public link.

## Example Questions
- How do I reset my password?
- How can I create an account?
- How do I cancel my order?
- What are your working hours?
- How can I contact support?

## Project Files
- `CodeAlpha_FAQ_Chatbot.ipynb` — Colab notebook
- `app.py` — standalone chatbot application
- `requirements.txt` — Python dependencies
- `README.md` — project documentation
