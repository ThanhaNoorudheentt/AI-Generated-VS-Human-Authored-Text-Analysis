# AI-Generated vs. Human-Authored Text Analysis

## Live Demo
You can interact with the live AI Text Detector via the Streamlit dashboard:
*   **Access the App:** [Launch Live Detector](https://ai-generated-vs-human-appored-text-analysis-8dz44nzdozqechy2be.streamlit.app/)

## Project Overview
With the rise of LLMs (GPT-4, Claude, Gemini), distinguishing machine-generated text from human writing is a critical challenge for academic integrity and misinformation detection. This project implements a robust NLP pipeline to classify text origins using both traditional machine learning and deep learning architectures.

## 🛠️ How to Run Locally

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch the App
```bash
streamlit run streamlit_app/app.py
```

## Dataset Details: HC3
The project utilizes the **Human ChatGPT Comparison Corpus (HC3)**, a multi-domain dataset designed for the study of AI-generated text detection.
- **Source:** [Hello-SimpleAI/hc3](https://huggingface.co/datasets/Hello-SimpleAI/hc3)
- **Flattened Samples:** ~170,898 rows.

## Methodology & Models
1. **Baseline:** Logistic Regression with TF-IDF Vectorization (Accuracy: ~97%).
2. **Advanced Model:** Fine-tuned `distilroberta-base` (Accuracy: 98.6%).
3. **Linguistic Analysis:** AI signatures (formal transitions) vs. Human signatures (personal pronouns).

## Repository Structure
- `ai_human.ipynb`: The complete pipeline from data loading to evaluation.
- `streamlit_app/`: Application folder containing `app.py`, `style.css`, and model artifacts.
- `requirements.txt`: Dependencies for local and cloud deployment.
- `README.md`: Project documentation.

## Conclusion
This project demonstrates that while transformer-based models like RoBERTa can distinguish between AI-generated and human-authored text with high accuracy (98.6%), they remain highly susceptible to 'Style-Transfer' attacks. Our analysis revealed that AI models have a distinct linguistic signature characterized by formal transition words (e.g., 'however', 'overall') and a lack of personal pronouns. 

Detection is a "cat-and-mouse" game. While deep learning provides high accuracy on standard datasets, it is not foolproof against intentional stylistic manipulation. Future work should focus on cross-lingual detection and better generalization across different LLM families (Claude, Gemini).
