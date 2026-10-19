# 🤖 DocMind — AI-Powered Document Assistant

<p align="center">
  <b>📚 Your documents, answered.</b><br>
  Ask questions, get source-backed answers, and test your knowledge with AI-powered quizzes.
</p>

<p align="center">
  <a href="https://docmind-ngseatmduchbpzekzntp2x.streamlit.app/">
    🚀 Live Demo
  </a>
  &nbsp; • &nbsp;
  <a href="https://github.com/shanke3120-code/DocMind">
    💻 GitHub Repository
  </a>
</p>

---

## ✨ Features

- 💬 **AI Chat:** Ask natural-language questions about your documents.
- 📄 **PDF Support:** Upload and interact with PDF documents.
- 🧠 **RAG Pipeline:** Retrieve relevant document chunks to ground AI-generated answers.
- 🔎 **Source References:** View page-level citations and supporting passages.
- 📝 **Interactive Quizzes:** Test your understanding of uploaded content.
- 📊 **Quiz History:** Review your previous attempts during the session.
- 🎨 **Modern UI:** Clean interface with dark and light themes.

## 🛠️ Tech Stack

`Python` · `Streamlit` · `LangChain` · `Groq` · `Hugging Face` · `Sentence Transformers` · `FAISS`

## ⚙️ How It Works

1. 📥 Upload a PDF.
2. ✂️ Extract and split its text into chunks.
3. 🔢 Generate embeddings and index them in FAISS.
4. 🔍 Retrieve relevant passages for your question.
5. 🤖 Generate an answer grounded in the retrieved context.

## 📂 Project Structure

```text
DocMind/
├── app5.py                 # 🚀 Main application
├── 1_chatbot.ipynb         # 🧪 Initial chatbot experimentation
├── 2_AI_assistant.ipynb    # 🧪 AI assistant experimentation
├── quiz_test.py            # 🧪 Quiz feature experiments
├── requirements.txt        # 📦 Dependencies
├── .gitignore              # 🔒 Ignored files and secrets
└── README.md               # 📖 Project documentation
```

> 🧪 **Development Note:** The chatbot notebook, AI assistant notebook, and `quiz_test.py` contain my earlier experiments and practice implementations. I refined these ideas and integrated the resulting features into `app5.py`, the main DocMind application.

## 🚀 Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/shanke3120-code/DocMind.git
cd DocMind
```

**2. Create and activate a virtual environment**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your API key**

Create a `.env` file and add the API key required by your application:

```env
GROQ_API_KEY=your_groq_api_key
```

Use the environment variable name expected by your code. 🔐 Never upload API keys or your `.env` file to GitHub.

**5. Launch DocMind**

```bash
streamlit run app5.py
```

## ☁️ Deployment

🌐 Deployed using **Streamlit Community Cloud**, connected to this GitHub repository.

## 🔮 Future Improvements

- 📚 Support for more document formats
- 🎯 Improved retrieval and answer evaluation
- 💾 Persistent chat and quiz history
- ⚡ Further performance and usability improvements

## 👨‍💻 Author

**Shashank Kumar Jha**

🎓 B.Tech — Computer Science and Engineering (Data Science)

💡 Interested in Data Science, Machine Learning, and Generative AI.

---

⭐ If you find DocMind interesting, consider starring the repository!
