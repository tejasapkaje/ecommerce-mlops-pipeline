# 🛒 Enterprise E-Commerce AI Predictor & MLOps Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)
![Flask](https://img.shields.io/badge/Backend-Flask-green.svg)
![MongoDB](https://img.shields.io/badge/Database-MongoDB-brightgreen.svg)
![PowerBI](https://img.shields.io/badge/Analytics-PowerBI-yellow.svg)

An end-to-end & production-ready Machine Learning system designed to predict customer purchase intent. This project evolves from a standard ML model into a complete **MLOps architecture**, featuring a continuous data feedback loop, automated model retraining & a dynamic real-time dashboard.

---

## 🌟 Key Features

* **🧠 Advanced ML Engine:** Utilizes a highly tuned `RandomForestClassifier` optimized via `RandomizedSearchCV`.
* **🌐 Interactive Web Application:** A futuristic, Glassmorphism-styled UI with CSS Grid, animations & asynchronous API communication.
* **⚙️ RESTful Flask API:** A robust backend inference engine that dynamically encodes, scales & evaluates live user inputs.
* **🗄️ Stateful Database Integration:** Logs every user interaction & AI prediction locally into **MongoDB** for historical auditing.
* **🔄 Closed-Loop Data Pipeline:** Automatically appends new predictive data to the master `.csv` dataset.
* **🤖 Automated MLOps (CT):** A background bot (`auto_scheduler.py`) that periodically triggers `retrain_pipeline.py` to retrain the model on new data, eliminating **Catastrophic Forgetting**.
* **📊 Live BI Dashboard:** A connected **Power BI** interface providing real-time analytical insights into customer demographics & revenue streams.

---

## 🏗️ System Architecture

1. **Frontend:** User inputs 17 specific demographic & behavioral variables.
2. **API Layer:** Flask processes the request, loads `.pkl` artifacts & computes the probability.
3. **Database Layer:** Prediction & metadata are saved to MongoDB (`ecommerce_db`).
4. **Data Engineering:** Data is appended to `cleaned_clothing_dataset.csv`.
5. **Analytics:** PowerBI reads the updated CSV to reflect live business metrics.
6. **MLOps:** Scheduled background tasks retrain the model iteratively.

---

## 📂 Project Structure

```text
ecommerce_ml_pipeline/
│
├── app/                  # Web Application Backend
│   ├── webapp.py         # Main Flask API server
│
├── data/                 # Data Layer 
│   └── cleaned_clothing_dataset.csv  # Auto-updating master dataset
│
├── frontend/             # Presentation Layer
│   └── index.html        # Glassmorphism UI
│
├── src/                  # ML & MLOps Source Code
│   ├── retrain_pipeline.py  # Automated retraining script
│   ├── random_forest_model_tuned.pkl # Serialized Model
│   ├── standard_scaler.pkl  # Serialized Scaler
│   └── label_encoders.pkl   # Serialized Encoders
│
├── notebooks/            # Jupyter Notebooks for EDA & initial training
│
├── .github/workflows/    # CI/CD Pipeline logic
├── auto_scheduler.py     # Background MLOps Daemon
└── requirements.txt      # Python dependencies
```
---
## 🚀 Installation & Setup

1. **Clone the Repository:**
   git clone [https://github.com/tejasapkaje/Enterprise-E-Commerce-AI-Predictor-MLOps-Pipeline.git](https://github.com/tejasapkaje/Enterprise-E-Commerce-AI-Predictor-MLOps-Pipeline.git)
   cd Enterprise E-Commerce AI Predictor MLOps Pipeline
3. **Set Up Virtual Environment:**
   python -m venv venv
   Windows:
   .\venv\Scripts\activate
4. **Install Dependencies:**
   pip install -r requirements.txt
5. **Database Setup:**
   Install MongoDB Community Server.
   Ensure it is running locally on port 27017.

6. **Run the System:**
   You need two terminals to run the complete architecture:

   * Terminal 1: Start the Flask API
   python `app/webapp.py` ``# Open frontend/index.html in your browser to access the UI.``

   * Terminal 2: Start the MLOps Auto-Scheduler
   python `auto_scheduler.py`

## Built with ❤️ by an AI/ML Enthusiast & Data Engineer.
