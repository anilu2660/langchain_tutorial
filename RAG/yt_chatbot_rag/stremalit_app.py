import os
import re
import streamlit as st
import requests
from xml.etree import ElementTree as ET

from googleapiclient.discovery import build as yt_build
from googleapiclient.errors import HttpError

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="YouTube Video Chatbot", page_icon="📺")
st.title("📺 Chat with YouTube Video")

with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.info("Get your key from [OpenAI](https://platform.openai.com/)")
    youtube_api_key = st.text_input("YouTube Data API v3 Key", type="password")
    st.info("Get your key from [Google Cloud Console](https://console.cloud.google.com/)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "loaded_video" not in st.session_state:
    st.session_state.loaded_video = ""

def _fetch_caption_text(video_id: str, yt_api_key: str) -> str:
    """Fetch English caption text for a YouTube video using the Data API v3."""
    youtube = yt_build("youtube", "v3", developerKey=yt_api_key)

    # List caption tracks for the video
    captions_response = youtube.captions().list(
        part="snippet",
        videoId=video_id,
    ).execute()

    caption_items = captions_response.get("items", [])
    if not caption_items:
        raise ValueError("No caption tracks found for this video.")

    # Prefer manual English captions, fall back to ASR (auto-generated)
    selected = None
    for item in caption_items:
        lang = item["snippet"].get("language", "")
        kind = item["snippet"].get("trackKind", "")
        if lang == "en" and kind != "asr":
            selected = item
            break
    if selected is None:
        for item in caption_items:
            if item["snippet"].get("language", "") == "en":
                selected = item
                break
    if selected is None:
        raise ValueError("No English caption track found for this video.")

    # Extract track name and language for the timedtext URL
    track_name = selected["snippet"].get("name", "")
    track_lang = selected["snippet"].get("language", "en")

    base_params = f"v={video_id}&lang={track_lang}&name={requests.utils.quote(track_name)}"

    # Try multiple formats in order: srv3 (XML), ttml (XML), plain XML
    resp = None
    for fmt in ["srv3", "ttml", ""]:
        fmt_param = f"&fmt={fmt}" if fmt else ""
        url = f"https://www.youtube.com/api/timedtext?{base_params}{fmt_param}"
        r = requests.get(url, timeout=15)
        if r.status_code == 200 and r.text.strip():
            resp = r
            break

    if resp is None or not resp.text.strip():
        raise ValueError(
            "Could not retrieve caption content. "
            "The video may have captions disabled or they are not publicly accessible."
        )

    # Parse XML and extract plain text
    try:
        root = ET.fromstring(resp.text)
        texts = [elem.text or "" for elem in root.iter("text")]
        transcript = " ".join(t.strip() for t in texts if t.strip())
    except ET.ParseError:
        # Strip XML tags as a fallback
        transcript = re.sub(r"<[^>]+>", " ", resp.text)
        transcript = " ".join(transcript.split())

    if not transcript.strip():
        raise ValueError("Caption content is empty after parsing.")

    return transcript


def build_chain(video_id: str, api_key: str, yt_api_key: str):
    os.environ["OPENAI_API_KEY"] = api_key

    transcript = _fetch_caption_text(video_id, yt_api_key)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    prompt = PromptTemplate(
        template="""You are a helpful assistant.
Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

Context:
{context}

Question: {question}""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough(),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


video_id = st.text_input("Enter YouTube Video ID:", placeholder="e.g. wrcQwMpAirQ")

if st.button("Process Video"):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif not youtube_api_key:
        st.warning("Please enter your YouTube Data API v3 key in the sidebar.")
    elif not video_id.strip():
        st.warning("Please enter a YouTube Video ID.")
    else:
        with st.spinner("Loading transcript and building index..."):
            try:
                chain = build_chain(video_id.strip(), openai_api_key, youtube_api_key)
                st.session_state.qa_chain = chain
                st.session_state.loaded_video = video_id.strip()
                st.session_state.messages = []
                st.success(f"✅ Video `{video_id}` is ready! Ask your questions below.")
            except HttpError as e:
                st.error(f"❌ YouTube API error: {e.reason}")
            except ValueError as e:
                st.error(f"❌ {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

if st.session_state.loaded_video:
    st.caption(f"Active video ID: `{st.session_state.loaded_video}`")

st.divider()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask something about the video..."):
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    elif st.session_state.qa_chain is None:
        st.warning("Please process a YouTube video first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.qa_chain.invoke(prompt)
                except Exception as e:
                    response = f"❌ Error: {e}"
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})