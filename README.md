📊 Loan Default Prediction System

A machine learning classifier with a circular linked-list–based navigation UI

📌 Overview

This project predicts whether borrowers are likely to default on a loan using a Decision Tree Classification Model. It features an interactive command-line navigation system built using a Circular Doubly Linked List, allowing users to move through borrower profiles with recommendation results (Accept / Reject).

This system combines data science and core data structure concepts into a single practical finance-based application.

🚀 Features

🔍 Predictive Analytics using Decision Tree Classifier

🧼 Automated preprocessing (noise filtering, missing value handling, feature scaling)

📈 Visual reports (histograms & pie charts) showing borrower trends

🔁 Circular Doubly Linked List navigation for user-friendly record browsing

📝 Displays detailed borrower profiles with model-based recommendations

🧪 Performance evaluation with accuracy, classification report & confusion matrix

🛠️ Tech Stack
Category	Tools
Programming	Python
Machine Learning	Scikit-Learn
Data Handling	Pandas, NumPy
Visualization	Matplotlib
Data Structure	Circular Doubly Linked List
📂 Project Structure
📦 Loan-Default-Prediction
├─ 📁 data              # Dataset files
├─ 📁 src
│  ├─ model.py          # ML model training & evaluation
│  ├─ preprocess.py     # Data cleaning & feature engineering
│  ├─ linked_list.py    # Circular DLL implementation
│  ├─ ui.py             # CLI-based record viewer
│  └─ main.py           # Application entry point
├─ 📊 charts             # Generated graphs for insights
└─ README.md            # Project documentation

🧠 How It Works

The dataset is cleaned and filtered

The classifier is trained on borrower indicators

Users can cycle through borrower records using:

n → next borrower

p → previous borrower

q → quit

Prediction displayed as:

✔ Accepted → low risk of default

✖ Rejected → high risk of default

📊 Model Insights

Shows financial risk distribution using Matplotlib visualizations

Highlights key contributors to default decisions

📈 Performance Metrics

Accuracy Score

Confusion Matrix

Classification Report

(Values depend on dataset)

▶️ Run it Yourself
# Clone repo
git clone https://github.com/yourusername/loan-default-prediction.git
cd loan-default-prediction

# Install dependencies
pip install -r requirements.txt

# Run program
python main.py

🏁 Future Improvements

✔ GUI/Web interface
✔ Support for multiple ML models (Random Forest, XGBoost)
✔ Performance optimization on larger datasets

📜 License

This project is open-source and free to use for learning and research.
