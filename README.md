# HateBertBN: A Hybrid Transformer-Based Model for Bangla Hate Speech Detection Across Various Social Contexts

## 📘 Abstract
The widespread use of online social media platforms has amplified the importance of efficient hate speech detection, especially in low-resource languages like Bengali. While traditional machine learning approaches show promise, deep learning is more effective in capturing the nuanced context of hate speech. Current challenges include a lack of diverse datasets and models capable of context-sensitive detection.  

To address these, we introduce **HateCorpBN-XL**, the largest labeled Bengali hate speech dataset to date, containing **65,251 comments** across five categories:  
- **Political (PoHS)**  
- **Religious (ReHS)**  
- **Misogynistic (MisoHS)**  
- **Slander (SlaHS)**  
- **Xenophobic (XenHS)**  

We also propose **HateBertBN**, a hybrid transformer-based model combining **BanglaBERT embeddings** with three neural network fusion strategies — **CNN**, **LSTM**, and **MLP**.  

We evaluate our approach on two tasks:  
- **Task 1:** Hate vs. Non-Hate classification  
- **Task 2:** Multi-class hate categorization (5 categories)

For **Task 1**, all HateBertBN variants outperform baseline transformer models, achieving an accuracy and weighted F1-score of **0.92**.  
For **Task 2**, the **HateBertBN-MLP** and **HateBertBN-CNN** variants achieve **0.90 accuracy** and **0.90 weighted F1**, surpassing M-BERT, Distil-M-BERT, BanglaBERT, and XLM-R-Base.  

---

## 🧩 Repository Structure
```
├── HateBertBN.ipynb           # Main Jupyter notebook (model training & evaluation)
├── README.md                  # Project documentation
├── requirements.txt           # List of dependencies (optional but recommended)
└── data/                      # Folder for dataset (if applicable)
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/HateBertBN.git
cd HateBertBN
```

### 2. Create and activate a virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\Scripts\activate      # For Windows
```

### 3. Install dependencies
If you have a `requirements.txt` file:
```bash
pip install -r requirements.txt
```

Otherwise, manually install the main libraries:
```bash
pip install torch torchvision torchaudio
pip install transformers
pip install scikit-learn
pip install pandas tqdm
```

### 4. Open the notebook
You can run the project in **Google Colab** or locally using Jupyter:

```bash
jupyter notebook HateBertBN.ipynb
```

---

## 🚀 Steps to Reproduce the Research

### **Step 1: Load Dataset**
Place the `HateCorpBN-XL` dataset in a local directory (e.g., `data/`).  
It should contain 65,251 labeled Bangla comments categorized into the five hate speech classes.

Each record should contain:
```text
text,label
"এই মন্তব্যটি রাজনৈতিক ঘৃণা প্রকাশ করে।",0
"এটি ধর্মীয় অপমান।",1
...
```

### **Step 2: Tokenization**
The notebook uses the **BanglaBERT tokenizer** from the `csebuetnlp/banglabert` model:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('csebuetnlp/banglabert')
```

### **Step 3: Model Selection**
You can choose one of the three variants:
- `HateBertBN-MLP`
- `HateBertBN-CNN`
- `HateBertBN-LSTM`

Each variant combines **BanglaBERT embeddings** with a different neural fusion head.

### **Step 4: Training**
Run the training cells in the notebook.  
Example configuration:
```python
epochs = 15
batch_size = 8
learning_rate = 2e-5
```

Training is done using `Adam` optimizer and `CrossEntropyLoss`.  
TQDM progress bars will display real-time training loss and accuracy.

### **Step 5: Evaluation**
After training, the notebook computes:
- Validation accuracy and loss
- Weighted F1-score
- Per-class metrics (precision, recall, F1)

Example output:
```
Task 1 → Accuracy: 0.92 | Weighted F1: 0.92
Task 2 → Accuracy: 0.90 | Weighted F1: 0.90
```

---


```

---

## 📊 Experimental Results (Summary)

| Task | Variant | Accuracy | Weighted F1 | Notable Performance |
|------|----------|-----------|--------------|---------------------|
| Task 1 | HateBertBN (Binary) | 0.92 | 0.92 | Hateful vs Non-Hateful |
| Task 2 | HateBertBN-MLP | 0.90 | 0.90 | Best overall balance |
| Task 2 | HateBertBN-CNN | 0.90 | 0.90 | Strong on Political & Misogynistic |
| Task 2 | HateBertBN-LSTM | 0.88 | 0.88 | Highest F1 on ReHS (0.93) and XenHS (1.00) |

---

## 🧾 Citation
---

## 🤝 Acknowledgements
- **BanglaBERT:** Pretrained model by [CSE BUET NLP Group](https://huggingface.co/csebuetnlp/banglabert)  
- **Dataset:** HateCorpBN-XL — Curated for this study  ( Will be avaialbe for download soon)
- **Libraries:** PyTorch, HuggingFace Transformers, scikit-learn, tqdm  

---

## 📬 Contact
For questions or collaborations:
**Tanvir Azhar**  
📧 [azhar.t@eastdelta.edu.bd] 
