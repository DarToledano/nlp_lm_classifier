
# NLP Course Final Project: IMDB Sentiment Classification

## Repo Intro 


### Task Overview:
1. **Task 1:** Train an **LSTM-based Language Model (LM)** for next-word prediction on IMDB movie reviews.
2. **Task 2:** Sentiment classification using two different approaches:
   - **Experiment A:** Use sentence embeddings from the pre-trained LM + train an **MLP classifier**.
   - **Experiment B:** Use **pre-trained Word2Vec embeddings + LSTM-based classifier** trained from scratch.

The goal was to compare the performance between these two feature extraction and classification strategies.

---

## Installation | Requirements ⚙️

### Python Version:
- Python 3.8 or above

### Required Libraries:
Install all dependencies via:

```bash
pip install -r requirements.txt
```

Key libraries include:
- `torch`
- `gensim`
- `nltk`
- `matplotlib`
- `sklearn`
- `numpy`

Also, make sure to download **NLTK tokenizers**:

```python
import nltk
nltk.download('punkt')
```

---

## Quickstart 

### Folder Setup:
Place the following files in the `data/` folder:

- IMDB dataset (from [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)) → Put under `data/aclImdb`
- Word2Vec pre-trained embeddings → (from [https://github.com/mmihaltz/word2vec-GoogleNews-vectors/blob/master/GoogleNews-vectors-negative300.bin.gz])

---

### Running the Full Pipeline:

```bash
python main.py
```

This will:

1. Run **EDA (Exploratory Data Analysis)** on the dataset.
2. Train the **LSTM Language Model** for Task 1.
3. Run **Experiment A** (LM + MLP).
4. Run **Experiment B** (Word2Vec + LSTM classifier).
5. Evaluate both experiments:
   - Generate training loss graphs
   - Confusion matrices
   - Error analysis examples

---

## Resources 

- IMDB dataset: [Stanford IMDB Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)
- Word2Vec pre-trained model: [GoogleNews Word2Vec](https://code.google.com/archive/p/word2vec/)
- PyTorch documentation: [https://pytorch.org/](https://pytorch.org/)
- NLTK: [https://www.nltk.org/](https://www.nltk.org/)

---

## Notes 

- Hyperparameters (like `SEQ_LEN`, `BATCH_SIZE`, `DROPOUT`, etc.) are configurable via `data/config.py`.
- The language model uses a **1-layer LSTM with dropout**, trained with **CrossEntropyLoss**.
- Word2Vec embeddings in Experiment B are **frozen (non-trainable)**.
- Results and error analysis for both experiments are output during runtime.

---

Feel free to extend, tune, or experiment with new architectures for better results!
