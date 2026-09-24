import streamlit as st
import pandas as pd
import openai
import json
import io
import re

# --- pdfplumber set up ------------------------------------------------------ #
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# --- Page configuration ----------------------------------------------------- #
st.set_page_config(
    page_title="LDA Topic Labeller",
    page_icon="🏷️",
    layout="wide",
)

# --- Title information ------------------------------------------------------ #
st.title("🏷️ LDA Topic Labeller")
st.markdown(
    "Upload your **LDA output** (CSV or PDF) or enter topic words manually. "
    "OpenAI will suggest a concise label and description for each topic."
)

# --- Sidebar: API key & model ----------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("OpenAI API Key", type="password", help="Your key is never stored.")
    model = st.selectbox("Model", ["gpt-4o-mini","gpt-4o", "gpt-3.5-turbo"], index=0)
    temperature = st.slider("🌡️ Temperature", 0.0, 1.0, 0.3, 0.01, help="Adjust the randomness of the model's output. Lower values produce more focussed and deterministic responses.")
    max_tokens = st.slider(
        "Max Tokens",
        1, 4000, 1000, 1,
        help="Maximum number of tokens to generate per response."
    )
    frequency_penalty = st.slider(
        "Frequency Penalty",
        -2.0, 2.0, 0.0, 0.01,
        help="Penalise new tokens based on their existing frequency in the text so far."
    )
    presence_penalty = st.slider(
        "Presence Penalty",
        -2.0, 2.0, 0.0, 0.01,
        help="Penalise new tokens based on whether they appear in the text so far."
    )
    top_n = st.slider("Top N words shown per topic", 5, 20, 10)
    st.markdown("---")
    st.subheader("Custom Prompt (Optional)")
    with st.expander("📜 Custom Prompt Guidelines"): # Added expander here
        st.markdown(
            "To improve label accuracy:\n"
            "- **Define the AI's Persona & Goal:** E.g., 'You are an expert in X.'\n"
            "- **Specify JSON Output:** Clearly state the required 'topic_id', 'label' (e.g., max 5 words), and 'description' (e.g., 1-2 sentences) keys, and that only JSON should be returned.\n"
            "- **Include Placeholder:** Crucially, use `{topic_lines}` where the topic words will be inserted.\n"
            "- **Be Specific:** Add constraints on length, tone, and format for labels and descriptions.\n"
            "- **Iterate:** Experiment with your prompt to find what works best."
        )
    custom_prompt = st.text_area(
        "Override default prompt (use {topic_lines} placeholder)",
        height=200,
        placeholder=(                "You are a highly skilled academic researcher tasked with categorizing research papers. "
            "For each topic provided, identify its core theme and summarize it in a scholarly tone. "
            "The output should be a JSON array where each object contains a 'topic_id' (string), "
            "a 'label' (a concise, academic label up to 7 words), and 'description' (a detailed, "
            "academic explanation of 2-3 sentences). Ensure the output is strictly a JSON array "
            "without any additional text or markdown fences.\n\n"
            "{topic_lines}"
        )
    )
    st.markdown("---")
    st.caption("Only the top words are sent to the API — no document content leaves your machine.")


# --- Helper: call OpenAI ---------------------------------------------------- #
def label_topics(
    topics: list[dict],
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    frequency_penalty: float,
    presence_penalty: float,
    custom_prompt: str = "") -> tuple[list[dict], str]:
    """
    topics: list of {"topic_id": int/str, "words": [(word, weight), ...]}
    Returns same list enriched with 'label' and 'description' and the prompt used.
    """
    client = openai.OpenAI(api_key=api_key)

    # Build one prompt for all topics to save API calls
    topic_lines = []
    for t in topics:
        words_str = ", ".join(w for w, _ in t["words"])
        topic_lines.append(f'Topic {t["topic_id"]}: {words_str}')

    if custom_prompt.strip():
        prompt = custom_prompt.format(topic_lines="\n".join(topic_lines))
    else:
        prompt = (
            "You are an expert in topic modelling. Below are topics from a Latent Dirichlet "
            "Allocation (LDA) model, each represented by its most probable words.\n\n"
            + "\n".join(topic_lines)
            + "\n\nFor EACH topic return a JSON array (no markdown fences) with objects having "
            'keys "topic_id" (string), "label" (<=5 words), and "description" (1-2 sentences). '
            "Keep labels specific and descriptive. Return only the JSON array."
        )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_completion_tokens=max_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if model added them anyway
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    try:
        results = json.loads(raw)
    except json.JSONDecodeError as e:
        # Re-raise the exception with the raw content for better debugging in the UI
        raise json.JSONDecodeError(f"OpenAI returned non-JSON response: '{raw}'. Original error: {e.msg}", e.doc, e.pos)

    # Merge labels back
    label_map = {str(r["topic_id"]): r for r in results}
    for t in topics:
        info = label_map.get(str(t["topic_id"]), {})
        t["label"] = info.get("label", "—")
        # Try to get 'description', if not found, try 'summary'
        t["description"] = info.get("description", info.get("summary", "—"))
    return topics, prompt


# --- Helper: parse CSV ------------------------------------------------------ #
def parse_csv(df: pd.DataFrame, top_n: int) -> list[dict]:
    """
    Accepts several common LDA CSV layouts:
      A) Long format:  topic | word | weight OR topic_id | word | probability
      B) Wide format:  topic_0_word, topic_0_weight, topic_1_word, ... (gensim style)
      C) Matrix format: rows=topics, columns=words, values=weights
    """
    cols = [c.lower().strip() for c in df.columns]
    df.columns = cols

    topics = []

    # Layout A: Long format check
    topic_col = None
    word_col = None
    weight_col = None

    if {"topic", "word", "weight"}.issubset(set(cols)):
        topic_col = "topic"
        word_col = "word"
        weight_col = "weight"
    elif {"topic", "term", "weight"}.issubset(set(cols)):
        topic_col = "topic"
        word_col = "term"
        weight_col = "weight"
    elif {"topic_id", "word", "probability"}.issubset(set(cols)):
        topic_col = "topic_id"
        word_col = "word"
        weight_col = "probability"

    if topic_col and word_col and weight_col:
        for tid, grp in df.groupby(topic_col):
            grp_sorted = grp.sort_values(weight_col, ascending=False).head(top_n)
            words = list(zip(grp_sorted[word_col], grp_sorted[weight_col]))
            topics.append({"topic_id": tid, "words": words})
        return topics

    # Layout B: gensim wide (topic_0, topic_1 …)
    topic_word_cols = [c for c in cols if re.match(r"topic_\d+$", c)]
    if topic_word_cols:
        for tc in topic_word_cols:
            idx = re.search(r"\d+", tc).group()
            wc = f"topic_{idx}_weight" if f"topic_{idx}_weight" in cols else None
            words_series = df[tc].dropna().head(top_n)
            if wc:
                weights = df[wc].dropna().head(top_n)
                words = list(zip(words_series, weights))
            else:
                words = [(w, None) for w in words_series]
            topics.append({"topic_id": idx, "words": words})
        return topics

    # Fallback to Matrix format (Layout C)
    for idx, row in df.iterrows():
        row_for_words = row.copy()
        # Drop any columns that are clearly identifiers and not words in this context
        if 'topic_id' in row_for_words.index:
            row_for_words = row_for_words.drop('topic_id', errors='ignore')
        if 'topic' in row_for_words.index:
            row_for_words = row_for_words.drop('topic', errors='ignore')

        numeric = row_for_words.apply(lambda x: pd.to_numeric(x, errors="coerce"))
        top_words = numeric.dropna().nlargest(top_n)
        words = [(w, round(float(v), 4)) for w, v in top_words.items()]
        topics.append({"topic_id": idx, "words": words})
    return topics


# --- Helper Function: parse PDF (extract word lists from text) -------------- #
def parse_pdf(file, top_n: int) -> list[dict]:
    if not PDF_SUPPORT:
        st.error("pdfplumber is not installed. Run `pip install pdfplumber`.")
        return []

    topics = []
    with pdfplumber.open(file) as pdf:
        full_text = "\n".join(p.extract_text() or "" for p in pdf.pages)

    # Attempt 1: Try to detect patterns like: "Topic N: word1 word2 ..."
    pattern_a = re.findall(
        r"[Tt]opic\s*(\d+)\s*[:\-]+\s*([^\n]+)", full_text
    )
    if pattern_a:
        for tid, words_str in pattern_a:
            # Handle "0.05*word + 0.03*word2" gensim style
            gensim_matches = re.findall(r"[\d.]+\*\"?(\w+)\"?", words_str)
            if gensim_matches:
                words = [(w, None) for w in gensim_matches[:top_n]]
            else:
                words = [(w.strip().strip('\"\,'), None)
                         for w in re.split(r"[,\s]+", words_str)
                         if w.strip()][:top_n]
            topics.append({"topic_id": tid, "words": words})
        if topics:
            st.success(f"Detected **{len(topics)} topics** using pattern matching.")
            return topics

    # Attempt 2: If pattern matching fails, try to parse the entire text as a CSV
    try:
        # Convert the full_text string into a file-like object
        df_from_pdf_text = pd.read_csv(io.StringIO(full_text), sep=' ')
        topics_from_csv_in_pdf = parse_csv(df_from_pdf_text, top_n)
        if topics_from_csv_in_pdf:
            st.success(f"Detected **{len(topics_from_csv_in_pdf)} topics** by parsing PDF text as CSV.")
            return topics_from_csv_in_pdf
    except Exception as e:
        # Log the error for debugging, but don't show to user unless all else fails
        # st.warning(f"Could not parse PDF content as CSV: {e}") # Optionally uncomment for more verbose debugging
        pass # Continue to final warning if both failed

    st.warning(
        "Could not auto-detect topic structure in the PDF. "
        "Try the manual entry tab or a CSV export."
    )
    return []


# --- Helper Function: display results --------------------------------------- #
def display_results(topics: list[dict], model: str, temperature: float, max_tokens: int, frequency_penalty: float, presence_penalty: float, custom_prompt: str, final_prompt: str):
    st.subheader("📋 Results")

    # Summary table
    rows = []
    for t in topics:
        rows.append({
            "Topic": t["topic_id"],
            "Label": t.get("label", "—"),
            "Description": t.get("description", "—"),
            "Top Words": ", ".join(w for w, _ in t["words"]),
        })
    df_out = pd.DataFrame(rows)
    st.dataframe(df_out, use_container_width=True)

    # Prepare metadata for download
    metadata_string = "\n\n# --- Parameters and Prompt Used ---\n"
    metadata_string += f"# Model: {model}\n"
    metadata_string += f"# Temperature: {temperature}\n"
    metadata_string += f"# Max Tokens: {max_tokens}\n"
    metadata_string += f"# Frequency Penalty: {frequency_penalty}\n"
    metadata_string += f"# Presence Penalty: {presence_penalty}\n"
    metadata_string += f"# Custom Prompt Used: {custom_prompt if custom_prompt else '(Default Prompt)'}\n"
    metadata_string += "# Final Prompt Sent to OpenAI:\n"
    metadata_string += f"# {final_prompt.replace('\n', '\n# ')}\n"

    # Download
    csv_content = df_out.to_csv(index=False).encode() + metadata_string.encode()
    st.download_button(
        "⬇️ Download Results CSV",
        data=csv_content,
        file_name="lda_labels_with_params.csv",
        mime="text/csv",
    )

    # Expandable detail cards
    st.subheader("🔍Topic Details")
    for t in topics:
        with st.expander(f"Topic {t['topic_id']} — {t.get('label', '')}"):
            st.markdown(f"**Description:** {t.get('description', '—')}")
            if t["words"] and t["words"][0][1] is not None:
                word_df = pd.DataFrame(t["words"], columns=["Word", "Weight"])
                st.dataframe(word_df, use_container_width=True, hide_index=True)
            else:
                st.write(", ".join(w for w, _ in t["words"]))

    st.subheader("⚙️ Parameters and Prompt Used")
    st.markdown(f"**Model:** `{model}`")
    st.markdown(f"**Temperature:** `{temperature}`")
    st.markdown(f"**Max Tokens:** `{max_tokens}`")
    st.markdown(f"**Frequency Penalty:** `{frequency_penalty}`")
    st.markdown(f"**Presence Penalty:** `{presence_penalty}`")
    st.markdown("**Custom Prompt Used:**")
    st.code(custom_prompt if custom_prompt else "(Default Prompt)", language="markdown")
    st.markdown("**Final Prompt Sent to OpenAI:**")
    st.code(final_prompt, language="markdown")


# --- Input tabs ------------------------------------------------------------- #
tab_csv, tab_pdf, tab_manual = st.tabs(["📄 CSV Upload", "📑 PDF Upload", "🪶 Manual Entry"])

topics_parsed: list[dict] = []

# --- TAB 1: CSV ------------------------------------------------------------- #
with tab_csv:
    st.markdown(
        "**Accepted formats:**\n"
        "- *Long:* columns `topic`, `word`/`term`, `weight`\n"
        "- *Long:* columns `topic_id`, `word`, `probability`\n"
        "- *Wide (gensim):* columns `topic_0`, `topic_0_weight`, `topic_1`, …\n"
        "- *Matrix:* rows = topics, columns = words, values = weights"
    )
    csv_file = st.file_uploader("Upload CSV", type=["csv"], key="csv")
    if csv_file:
        df_raw = pd.read_csv(csv_file)
        st.write("**Preview:**", df_raw.head())
        topics_parsed = parse_csv(df_raw, top_n)
        if topics_parsed:
            st.success(f"Detected **{len(topics_parsed)} topics**.")

# --- TAB 2: PDF ------------------------------------------------------------- #
with tab_pdf:
    st.markdown(
        "Upload a PDF containing your LDA output. Works best with reports that list "
        "topics line by line, e.g. `Topic 0: word1 word2 …` or gensim-style output." +
        " If direct parsing fails, the app will attempt to interpret the PDF text as a CSV."
    )
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf")
    if pdf_file:
        topics_parsed = parse_pdf(pdf_file, top_n)
        if topics_parsed:
            # Display a preview of the parsed words from the PDF for verification
            st.write("**Topics parsed from PDF:**")
            for t in topics_parsed:
                st.write(f"**Topic {t['topic_id']}:** " + ", ".join(w for w, _ in t["words"]))

# --- TAB 3: Manual ---------------------------------------------------------- #
with tab_manual:
    st.markdown(
        "Enter one topic per line. Separate words with commas or spaces. "
        "Optionally prefix with `Topic N:` (or just paste the word list)."
    )
    manual_text = st.text_area(
        "Topic words",
        placeholder=(                "Topic 0: bank loan interest mortgage credit\n"
            "Topic 1: election vote candidate campaign ballot\n"
            "Topic 2: cancer treatment patient surgery medicine"
        ),
        height=200,
    )
    if manual_text.strip():
        topics_parsed = []
        for i, line in enumerate(manual_text.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            # Strip "Topic N:" prefix
            m = re.match(r"[Tt]opic\s*(\d+)\s*[:\-]?\s*", line)
            if m:
                tid = m.group(1)
                rest = line[m.end():]
            else:
                tid = str(i)
                rest = line
            words = [(w.strip().strip('\"\,'), None)
                     for w in re.split(r"[,\s]+", rest) if w.strip()][:top_n]
            topics_parsed.append({"topic_id": tid, "words": words})
        if topics_parsed:
            st.success(f"Parsed **{len(topics_parsed)} topics**.")

# --- Run labelling ---------------------------------------------------------- #
st.divider()  # Draw a horizontal rule

if topics_parsed:
    if not api_key:
        st.warning("Enter your OpenAI API key in the sidebar to label topics.")
    else:
        if st.button("Label Topics with OpenAI", type="primary"):
            with st.spinner("Calling OpenAI…"):
                try:
                    labeled, final_prompt = label_topics(
                        topics_parsed,
                        api_key,
                        model,
                        temperature,
                        max_tokens,
                        frequency_penalty,
                        presence_penalty,
                        custom_prompt
                        )
                    display_results(labeled, model, temperature, max_tokens, frequency_penalty, presence_penalty, custom_prompt, final_prompt)
                except openai.AuthenticationError:
                    st.error("Invalid API key. Please check your key in the sidebar.")
                except openai.RateLimitError:
                    st.error("Rate limit hit. Wait a moment and try again.")
                except openai.APIError as e:
                    st.error(f"A generic OpenAI API error occurred: {e}")
                except json.JSONDecodeError as e:
                    st.error(f"Could not parse OpenAI response as JSON: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")
else:
    st.info("Upload a file or enter topic words above to get started.")
