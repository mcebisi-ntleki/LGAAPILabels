# LDA Topic Labeller

A Streamlit app that uses OpenAI's GPT to automatically generate human-readable labels
and descriptions for topics produced by Latent Dirichlet Allocation (LDA) models.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Usage

1. Paste your **OpenAI API key** in the sidebar and pick a model.
2. Choose an input method:
   - **CSV** — upload your LDA output file (see formats below)
   - **PDF** — upload a PDF report containing LDA output
   - **Manual** — paste or type topic word lists directly
3. Click **Label Topics with OpenAI**.
4. Download the results as CSV.

## CSV Formats Supported

| Format | Description | Example columns |
|--------|-------------|-----------------|
| Long   | One row per word | `topic`, `word`, `weight` |
| Wide (gensim) | Paired word/weight columns | `topic_0`, `topic_0_weight`, … |
| Matrix | Rows = topics, cols = words | any word as column name |

## PDF Format

The app looks for lines like:
```
Topic 0: bank loan interest mortgage credit
Topic 1: 0.05*"election" + 0.03*"vote" + …
```

## Notes

- Only the top-N words per topic are sent to OpenAI — no document content is transmitted.
- API key is never stored; it lives only in your browser session.
