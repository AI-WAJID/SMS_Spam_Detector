# 📨 SMS Spam Detector:

A robust SMS spam classification system built with **Streamlit**, **TF–IDF**, and **Multinomial Naive Bayes**. This repository includes everything needed to train, deploy, and run the model.

---

## 📁 Repository Structure

SMS_Spam_Detector/
├── app.py # Streamlit application
├── requirements.txt # Python dependencies
├── model/
│ ├── vectorizer.pkl # Serialized TF–IDF vectorizer
│ └── spam_model.pkl # Serialized Naive Bayes model
├── notebooks/
│ └── training_notebook.ipynb # Notebook for preprocessing & model training
├── data/
│ └── spam.csv # Raw SMS spam dataset
├── README.md # Project overview and instructions
└── .gitignore # Files to ignore in Git

---

## 🔍 Project Overview

**SMS Spam Detector** is an interactive web application that:

- Accepts user-entered SMS messages
- Preprocesses text (lowercasing, tokenization, stopword removal, stemming)
- Applies TF–IDF vectorization
- Classifies messages as **Spam** or **Ham** via a pre-trained Multinomial Naive Bayes model
- Displays results with confidence scores in a user-friendly interface

---

## ⚙️ Installation & Setup

1. **Clone the repository**  
   ```bash
   git clone https://github.com/AI-WAJID/SMS_Spam_Detector.git
   cd SMS_Spam_Detector
2. pip install -r requirements.txt
3. streamlit run app.py
4. Use the application
       Enter an SMS message
       Click Classify
       View whether it’s Spam or Ham, with confidence score


