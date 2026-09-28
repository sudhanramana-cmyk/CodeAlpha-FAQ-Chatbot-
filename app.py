import re
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# FAQ KNOWLEDGE BASE
faqs = [
    {"question": "How do I create an account?", "answer": "Click the Sign Up button and enter your required details."},
    {"question": "How do I register? I want to register. How can I sign up? I want to create an account.",
     "answer": "Click the Sign Up button and enter your required details."},
    {"question": "I want to create a new account.", "answer": "Click the Sign Up button and enter your required details."},

    {"question": "How do I reset my password?", "answer": "Click on 'Forgot Password' on the login page and follow the instructions."},
    {"question": "I forgot my password.", "answer": "Click on 'Forgot Password' on the login page and follow the instructions."},
    {"question": "How can I change my password?", "answer": "You can change your password from the Account Settings section."},

    {"question": "How can I contact customer support?", "answer": "You can contact customer support through our support email."},
    {"question": "Where can I get help?", "answer": "You can contact customer support through our support email."},
    {"question": "How can I contact customer service?", "answer": "You can contact customer support through our support email."},

    {"question": "What are your working hours?", "answer": "Our working hours are 9 AM to 6 PM, Monday to Friday."},
    {"question": "When are you open?", "answer": "Our working hours are 9 AM to 6 PM, Monday to Friday."},
    {"question": "What time do you close?", "answer": "Our working hours are 9 AM to 6 PM, Monday to Friday."},

    {"question": "How can I cancel my order?", "answer": "You can cancel your order from the Orders section before it is shipped."},
    {"question": "I want to cancel my order.", "answer": "You can cancel your order from the Orders section before it is shipped."},
    {"question": "Can I cancel an order?", "answer": "Yes, you can cancel your order from the Orders section before it is shipped."},
    {"question": "How do I track my order?", "answer": "You can track your order from the Orders section of your account."},

    {"question": "What is this service?", "answer": "This service helps users manage their accounts and orders."},
    {"question": "How does this service work?", "answer": "You can create an account, manage your orders, and contact customer support."}
]

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

questions = [faq["question"] for faq in faqs]
clean_questions = [preprocess(question) for question in questions]

vectorizer = TfidfVectorizer()
faq_vectors = vectorizer.fit_transform(clean_questions)

def chatbot(user_question):
    if not user_question.strip():
        return "Please enter a question. 😊"

    question = user_question.lower().strip()

    # Direct handling for common FAQ categories
    if ("register" in question or "sign up" in question or
        "signup" in question or "create account" in question or
        "new account" in question):
        return "Click the Sign Up button and enter your required details."

    if ("forgot password" in question or "reset password" in question or
        "forgot my password" in question):
        return "Click on 'Forgot Password' on the login page and follow the instructions."

    if ("cancel" in question and ("order" in question or "purchase" in question)):
        return "You can cancel your order from the Orders section before it is shipped."

    if ("working hours" in question or "when are you open" in question):
        return "Our working hours are 9 AM to 6 PM, Monday to Friday."

    if ("contact support" in question or "customer support" in question):
        return "You can contact customer support through our support email."

    # TF-IDF + cosine similarity fallback
    clean_question = preprocess(user_question)
    user_vector = vectorizer.transform([clean_question])
    similarities = cosine_similarity(user_vector, faq_vectors)

    best_match = similarities.argmax()
    best_score = similarities[0][best_match]

    if best_score >= 0.75:
        return faqs[best_match]["answer"]

    return "Sorry, I don't have an answer for that."

demo = gr.Interface(
    fn=chatbot,
    inputs=gr.Textbox(
        label="Ask your question",
        placeholder="Type your question here..."
    ),
    outputs=gr.Textbox(label="Chatbot Response"),
    title="🤖 FAQ Chatbot",
    description="Ask questions about accounts, passwords, orders, support and working hours.",
    examples=[
        ["How do I reset my password?"],
        ["How can I create an account?"],
        ["How do I cancel my order?"],
        ["What are your working hours?"],
        ["How can I contact support?"]
    ]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
