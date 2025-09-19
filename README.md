# RAG-Powered Chatbot with Firestore Memory and Document Indexing

This project implements a Retrieval-Augmented Generation (RAG) powered chatbot that uses the OpenAI API and Firestore to provide context-aware conversations. The chatbot allows users to interact with data from documents (e.g., PDFs) or URLs and stores chat history in Firestore for a persistent, multi-session experience.

---

## Features

* **RAG (Retrieval-Augmented Generation)**: Combines retrieval of relevant information from indexed documents with generation of context-aware responses using OpenAI's GPT models.
* **Persistent Chat Memory**: Stores chat history using Firestore for each session, making it easier to track and reference past interactions.
* **Multi-session Support**: Allows users to start new chats, load previous chats, and interact with different sets of documents or URLs across sessions.
* **Document and URL Indexing**: Users can upload documents (PDFs, etc.) or provide URLs, which are indexed and used as context during conversations.
* **Customizable Chunking**: Provides options to configure chunk size and overlap when indexing documents for optimized retrieval.

---

## Requirements

* **Python 3.7+**
* **Libraries**:

  
  * LangChain
  * OpenAI (via OpenAI API)
  * HuggingFace Transformers
  * Google Cloud Firestore
  * FireCrawl (for URL scraping)
  * dotenv (for environment variables)
  * Gradio

Install the required libraries with the following command:

```bash
pip install -r requirements.txt
```

Make sure to set up environment variables (as detailed below) before running the app.

---

## Setup

### 1. Environment Variables

You’ll need the following environment variables set up for the application to work:

* **OPENAI\_API\_KEY**: Your OpenAI API key (for using GPT models).
* **FIREBASE\_PROJECT\_ID**: Your Google Cloud project ID.
* **FIRECRAWL\_API\_KEY**: Your FireCrawl API key for URL scraping.

You can create a `.env` file in the project root and add these variables:

```env
OPENAI_API_KEY=your-openai-api-key
GOOGLE_CLOUD_PROJECT=your-google-cloud-project-id
FIRECRAWL_API_KEY=your-firecrawl-api-key
```

### 2. Firestore Setup

Ensure you have a Google Cloud Firestore database set up and that your Firestore credentials are configured in your environment. The Firestore database will be used for storing chat history for each session.

---

## How to Run

1. **Run the App**:
   To start the chatbot, run the following command:

   ```bash
   python app2.py
   ```

2. **Access the Chat Interface**:
   Once the app starts, it will be available in your browser at `http://127.0.0.1:7860`. You can then interact with the chatbot, build indexes, upload documents, or provide URLs.

---

## How it Works

### 1. **Session Management**:

* You can create new chat sessions by providing a session ID.
* The chat history for each session is stored in Firestore and can be accessed later.
* The chatbot can load previous sessions and continue conversations from where they left off.

### 2. **Data Ingestion**:

* You can choose between three data ingestion modes:

  * **No Source**: Chatbot works without external data, relying solely on the LLM (Language Learning Model) for responses.
  * **URL**: Allows users to input a URL. The content from the URL will be scraped and indexed for future conversations.
  * **Files**: Users can upload documents (PDFs, etc.), which will be processed and indexed for context-aware responses.

### 3. **Context-Aware Responses**:

* When a user asks a question, the chatbot searches through the indexed documents and provides a response based on the relevant context, using the OpenAI API.
* The chatbot’s responses are powered by GPT, which generates answers based on the retrieved context.

### 4. **Chunking and Overlap**:

* For document indexing, the content is split into smaller chunks (configured via chunk size and overlap sliders), ensuring that the chatbot can retrieve relevant pieces of information efficiently.

---

## Example Usage

1. **Start a New Chat**:

   * Enter a new session ID or choose an existing one.
   * Optionally, build an index by uploading files or providing a URL for the chatbot to reference.

2. **Chat**:

   * Enter a message in the chatbox, and the chatbot will respond using the indexed documents/URLs and chat history.
   * Responses are context-aware and based on the previous messages in the session.

3. **Build/Reset Index**:

   * You can rebuild or reset the index for each session using the "Build/Reset Index" button. This will update the indexed data used for context retrieval.

---

## Example Flow

1. **Create a New Session**:

   * Enter a new session ID and click "Start New Chat."
2. **Index Documents or URLs**:

   * Select a data ingestion mode (URL or Files) and upload the relevant documents or provide a URL for scraping.
   * The chatbot will process and index the data.
3. **Ask Questions**:

   * Start asking questions based on the indexed data, and the chatbot will return contextually relevant responses.

---

## Contributing

Feel free to fork the repository and submit pull requests with improvements or new features!

---

## License

This project is open-source and available under the [MIT License](LICENSE).

---

This README provides clear instructions on setup, usage, and functionality, making it easy for users to understand how to run and interact with your chatbot!
