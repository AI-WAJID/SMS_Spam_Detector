# 📨 SMS Spam Detector : 
A comprehensive SMS spam detection app built with Streamlit, powered by TF–IDF and Multinomial Naive Bayes. This repository contains everything needed to train, run, and deploy the model.

# 📁 Repository Structure
kotlin
Copy
Edit
SMS_Spam_Detector/
├── app.py
├── requirements.txt
├── model/
│   ├── vectorizer.pkl
│   └── spam_model.pkl
├── notebooks/
│   └── training_notebook.ipynb
├── data/
│   └── spam.csv
├── README.md
└── .gitignore
# 🗂️ Files & Folders Breakdown
app.py
Streamlit app: Contains the frontend logic for user interaction, input processing, model loading, and prediction display.

Features:

Custom CSS for styling background, buttons, and layout.

Caches vectorizer and model loading for speed.

Text preprocessing (transform_text) using re, stemming, and stopword removal.

Form-based input to avoid button firing issues.

Prediction results displayed with spam/ham status and confidence score.

requirements.txt
Lists all Python dependencies required for running and deploying the project:

nginx
Copy
Edit
streamlit
scikit-learn
nltk
pandas
numpy
gunicorn
model/
Contains serialized artifacts:

vectorizer.pkl: Trained TfidfVectorizer object (used to convert text to features).

spam_model.pkl: Trained MultinomialNB classifier (used to classify messages).

🧠 These files must match: the vectorizer’s vocabulary size and the classifier’s training data must align, or you’ll get feature mismatch errors.

notebooks/training_notebook.ipynb
A Jupyter Notebook covering:

Data Loading: Reads in spam.csv.

Cleaning: Removes nulls, renames columns.

Text Preprocessing: Lowercasing, tokenizing, cleaning, stemming.

Feature Extraction: TF–IDF vectorization with max_features=3000.

Model Training: Splits data, trains MultinomialNB.

Evaluation: Reports accuracy, confusion matrix, ROC AUC.

Model Saving: Serializes both vectorizer and classifier.

👉 Important: Make sure the TF–IDF vectorizer is fitted before saving to ensure feature alignment indicated by len(tfidf.vocabulary_).

data/spam.csv
Source SMS spam collection.

Contains columns like v1 (label), v2 (message), and filler columns.

Used to build the model; not required for deployment.

.gitignore
Excludes unnecessary and sensitive files:

bash
Copy
Edit
__pycache__/
*.pyc
.env
Add models or logs here if you want to exclude built artifacts—not required in this repo since models need to be tracked.

⚙️ Setup & Running
Clone this repo:

bash
Copy
Edit
git clone https://github.com/AI-WAJID/SMS_Spam_Detector.git
cd SMS_Spam_Detector
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Enter a message → click Classify → view result (Spam or Ham) with confidence score.
