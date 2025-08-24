# YouTube Video Summarizer

A Python tool to download, transcribe, summarize, and query YouTube videos using LLMs and vector search.

---

## Features
- **Download** YouTube videos and extract audio
- **Transcribe** audio using OpenAI Whisper
- **Chunk & Embed** transcript with HuggingFace or Ollama embeddings
- **Summarize** video content using LLMs (Ollama Llama3.2 supported)
- **Conversational Q&A** over the transcript using vector search and LLM

---

## Simple Architecture




## Architecture Diagram

```mermaid
flowchart TD
   A[User] -->|YouTube URL| B[Downloader (yt_dlp)]
   B --> C[Audio File (mp3)]
   C --> D[Transcription (Whisper)]
   D --> E[Transcript Text]
   E --> F[Chunking & Embedding]
   F --> G[ChromaDB (Vector Store)]
   G --> H[Conversational Q&A]
   G --> I[Summarization (LLM)]
   H --> J[User Q&A]
   I --> K[Summary Output]
```

---

## Requirements
- Python 3.8+
- [yt_dlp](https://github.com/yt-dlp/yt-dlp)
- [openai/whisper](https://github.com/openai/whisper)
- [langchain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Ollama](https://ollama.com/) (for local LLM/embeddings)

Install dependencies:
```bash
pip install yt_dlp
pip install -q git+https://github.com/openai/whisper.git
pip install langchain langchain-openai langchain-community chromadb torch
```

---

## Usage

1. **Run the script:**
   ```bash
   python yt_vid_summarizer_copy.py
   ```
2. **Choose LLM and embedding options** (Ollama/Chroma/Nomic)
3. **Enter a YouTube URL**
4. **Get a summary and ask questions** about the video transcript interactively

---

## Example
```
Available LLM Models:
1. Ollama Llama3.2
Choose LLM model (1/2): 1

Available Embeddings:
1. Chroma Default
2. Nomic (via Ollama)
Choose embeddings (1/1): 1

Enter YouTube URL: https://www.youtube.com/watch?v=...
Processing video...

Video Title: ...
Summary:
...

You can now ask questions about the video (type 'quit' to exit)
Your question: What is the main topic?
Answer: ...
```

---

## Notes
- For Ollama/Nomic embeddings, ensure Ollama is running locally.
- Downloads and temporary files are stored in the `downloads/` directory.
- Whisper model defaults to `base` (can be changed in code).

---

## License
MIT License
