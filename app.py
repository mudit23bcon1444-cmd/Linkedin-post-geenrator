import html
import streamlit as st
import config
from retriever import PostRetriever
from generator import generate_post


# Page configuration
st.set_page_config(
    page_title="LinkedIn Post Generator",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for guaranteed high-contrast font visibility across themes
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0A66C2 !important;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
        color: #374151 !important;
        font-weight: 500;
    }
    .post-card {
        background-color: #ffffff !important;
        color: #111827 !important;
        border: 1.5px solid #d1d5db !important;
        border-radius: 12px;
        padding: 1.5rem;
        font-size: 1.08rem;
        line-height: 1.7;
        white-space: pre-wrap;
        word-break: break-word;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.08);
    }
    .post-card * {
        color: #111827 !important;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .sub-header {
            color: #e5e7eb !important;
        }
        .post-card {
            background-color: #1f2937 !important;
            color: #f9fafb !important;
            border-color: #4b5563 !important;
        }
        .post-card * {
            color: #f9fafb !important;
        }
    }
    .meta-tag {
        display: inline-block;
        background-color: rgba(10, 102, 194, 0.15);
        color: #0A66C2 !important;
        border: 1px solid rgba(10, 102, 194, 0.3);
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    div.stButton > button[kind="primary"] {
        background-color: #0A66C2 !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #004182 !important;
        color: #ffffff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_retriever() -> PostRetriever:
    """Load and cache the post retriever instance."""
    return PostRetriever(config.PROCESSED_DATA_PATH)


# Sidebar with instructions and details
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/c/ca/LinkedIn_logo_initials.png", width=48)
    st.title("About Generator")
    
    with st.expander("ℹ️ How it works", expanded=True):
        st.markdown(
            """
            1. **Select Parameters**: Choose your topic, preferred line length, and language (English or Hinglish).
            2. **Smart Retrieval**: The system retrieves up to 2 high-engagement style reference posts using a 4-tier fallback chain.
            3. **Few-Shot LLM Synthesis**: Groq LLM mimics the author's punchy cadence, hook structure, and takeaway without copying text verbatim.
            """
        )

    st.markdown("---")
    st.markdown("💡 **Tip**: Hinglish posts blend Hindi and English in Roman script for high relatability among college peers.")


# Main App Header
st.markdown('<div class="main-header">💼 LinkedIn Post Generator</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Craft authentic, high-impact placement posts with author few-shot style matching.</div>',
    unsafe_allow_html=True,
)

# Verify processed data file exists
if not config.PROCESSED_DATA_PATH.exists():
    st.error(
        f"⚠️ Processed posts dataset not found at `{config.PROCESSED_DATA_PATH}`.\n\n"
        "Please run `python preprocess.py` in your terminal to enrich and prepare the dataset before launching the app."
    )
    st.info("Run command: `python preprocess.py`")
    st.stop()

# Initialize retriever
try:
    retriever = load_retriever()
    available_tags = retriever.get_tags()
except Exception as e:
    st.error(f"Failed to load post retriever: {e}")
    st.stop()

if not available_tags:
    st.warning("No topic tags found in dataset. Please re-run `python preprocess.py`.")
    st.stop()

# Controls row
col1, col2, col3 = st.columns(3)

with col1:
    selected_tag = st.selectbox("📌 Topic", options=available_tags, index=0)

with col2:
    length_options = list(config.LENGTH_RANGES.keys())
    selected_length = st.selectbox(
        "📏 Length",
        options=length_options,
        index=1,
        help="Short (1-5 lines), Medium (6-10 lines), Long (11-15 lines)",
    )

with col3:
    selected_language = st.selectbox(
        "🌐 Language",
        options=config.LANGUAGES,
        index=0,
        help="English or Hinglish (Hindi + English in Roman script)",
    )

generate_clicked = st.button("🚀 Generate Post", type="primary", use_container_width=True)

if generate_clicked:
    with st.spinner("Retrieving style references and generating post with Groq LLM..."):
        try:
            post_text, examples_used, match_level = generate_post(
                length=selected_length,
                language=selected_language,
                tag=selected_tag,
                retriever=retriever,
            )

            st.success("Post generated successfully!")

            # Display generated post
            st.markdown("### 📝 Generated LinkedIn Post")
            st.markdown(f'<div class="post-card">{html.escape(post_text)}</div>', unsafe_allow_html=True)

            # Metadata and match caption
            num_examples = len(examples_used)
            st.caption(
                f"📊 Style examples used: **{num_examples}** (matched on: `{match_level}`)"
            )

            # Copy-friendly block
            st.markdown("#### 📋 Copy-Friendly Text")
            st.code(post_text, language="markdown")

            # Show referenced examples if any
            if examples_used:
                with st.expander(f"🔍 View {num_examples} Reference Style Posts ({match_level})"):
                    for idx, ex in enumerate(examples_used, 1):
                        st.markdown(f"**Reference {idx}** *(Engagement: {ex.get('engagement', 'N/A')}, Length: {ex.get('length', 'N/A')}, Lang: {ex.get('language', 'N/A')})*")
                        st.text(ex.get("text", ""))

        except Exception as err:
            st.error(f"Error generating post: {err}")
