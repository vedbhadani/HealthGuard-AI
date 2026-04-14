# 🏥 HealthGuard-AI
### Agentic AI-Based Health Support System using ML + RAG

---

## 🚀 Overview

HealthGuard-AI is an intelligent healthcare analytics system that combines Machine Learning (ML), Agentic AI, and Retrieval-Augmented Generation (RAG) to provide structured and explainable health insights.

The system analyzes patient clinical data, predicts health risks, and generates evidence-based recommendations through an interactive interface.

---

## 🎯 Objective

The goal of this project is to:

- Predict patient health risks using structured clinical data
- Explain the reasoning behind predictions
- Retrieve relevant medical knowledge from trusted sources
- Generate structured, non-clinical health guidance
- Provide an interactive chat-based assistant for follow-up queries

---

## 🧠 Key Features

### ML-Based Risk Prediction
- Predicts disease risk using supervised learning models
- Supports models like Logistic Regression and Decision Trees

### Explainable Insights
- Highlights key contributing health factors
- Improves interpretability of predictions

### Agentic AI Workflow
- Built using LangGraph
- Uses modular tools:
  - Risk Analyzer
  - Knowledge Retriever
  - Report Generator

### Retrieval-Augmented Generation (RAG)
- Enhances responses using external medical knowledge
- Uses vector search with FAISS and embeddings

### Interactive Chat Interface
- Allows users to ask follow-up health questions
- Provides context-aware responses

### Structured Health Reports
Each generated report includes:
- Risk summary
- Key contributing factors
- Recommendations
- Sources
- Medical disclaimer

### PDF Export (Extension)
- Downloadable health reports for better usability

---

## 🏗️ System Architecture

User Input (CSV / Report / Chat)  
↓  
Data Processing Layer  
↓  
ML Risk Prediction Model  
↓  
Agentic AI Layer (LangGraph)  
↙        ↓        ↘  
Retriever  Reasoner  Report Generator  
↓                     ↓  
Vector DB (FAISS)   Structured Output  
↓  
Streamlit UI  

---

## 📥 Input

- Patient data (CSV or extracted report data)
  - Age
  - Vital signs
  - Lab values
  - Medical history

---

## 📤 Output

Example structured report:

{
  "risk_summary": "Moderate risk of cardiovascular disease",
  "key_contributing_factors": [
    "High cholesterol",
    "Elevated blood pressure"
  ],
  "recommendations": [
    "Increase physical activity",
    "Reduce salt intake"
  ],
  "sources": [
    "WHO guidelines",
    "CDC resources"
  ],
  "disclaimer": "This system provides non-clinical guidance only."
}

---

## 🛠️ Tech Stack

### Machine Learning
- scikit-learn
- pandas
- NumPy

### LLM & Agent Framework
- LangGraph
- Open-source LLMs (Hugging Face / Groq)

### RAG Components
- FAISS (vector database)
- SentenceTransformers (embeddings)

### Frontend
- Streamlit

### Deployment
- Hugging Face Spaces
- Streamlit Community Cloud

---

## 🔄 Workflow

1. Upload patient data  
2. Preprocess and clean data  
3. Predict risk using ML model  
4. Agent:
   - Analyzes risk
   - Retrieves relevant medical knowledge
   - Generates structured report  
5. User interacts via chat for further insights  

---

## 📊 Evaluation Metrics

### ML Model
- Accuracy
- Precision
- Recall
- F1 Score

### System-Level
- Explanation quality
- Retrieval relevance
- Hallucination reduction

---

## ⚠️ Disclaimer

This system is intended for educational and informational purposes only.  
It does NOT provide medical advice, diagnosis, or treatment.  
Always consult a qualified healthcare professional.

---

## 📌 Constraints Followed

- No paid APIs used  
- Only open-source or free-tier tools used  
- Includes a user interface  
- Deployable on public platforms  

---

## 🌐 Deployment

Add your deployed link here

---

## 🎥 Demo

Add demo video link here

---

## 📂 Repository Structure

HealthGuard-AI/
│
├── data/                  # Dataset  
├── models/                # Trained ML models  
├── rag/                   # RAG pipeline  
├── agent/                 # LangGraph workflows  
├── ui/                    # Streamlit app  
├── utils/                 # Helper functions  
├── app.py                 # Main entry point  
└── README.md  

---

## 👥 Team

- Member 1  
- Member 2  
- Member 3  

---

## 🏁 Conclusion

HealthGuard-AI demonstrates how Machine Learning, Agentic AI, and Retrieval-Augmented Generation can be combined to create a healthcare system that is:

- Predictive  
- Explainable  
- Interactive  
- Knowledge-driven  