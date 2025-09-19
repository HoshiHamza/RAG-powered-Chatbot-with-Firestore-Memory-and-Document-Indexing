import os

from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI 
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import FireCrawlLoader


load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_firecrawl")

def create_vector_store():

    api_key=os.getenv("FIRECRAWL_API_KEY")
    if not api_key:
        raise ValueError("FIRECRAWL_API_KEY not found in environment variables. Please set it in your .env file.")
    
    print("Begin Crawling the Website")
    print(api_key)
    loader = FireCrawlLoader(
        api_key=api_key, url="https://apple.com", mode="scrape")
    print('hello')
    docs=loader.load()
    print("Finished scraping the website")

    for doc in docs:
        for key,value in doc.metadata.items():
            if isinstance(value,list):
                doc.metadata[key]=", ".join(value)

    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    split_docs=text_splitter.split_documents(docs)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(split_docs)}")

    embeddings=HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )  # Update to a valid embedding model if needed

    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db=Chroma.from_documents(split_docs, embeddings,persist_directory=persistent_directory)
    print("\n--- Finished creating embeddings ---")
    

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    create_vector_store()
else:
    print(
        f"Vector store {persistent_directory} already exists. No need to initialize.")

embeddings= HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)  # Update to a valid embedding model if needed

db=Chroma(persist_directory=persistent_directory,
          embedding_function=embeddings)

# query = "How do I learn langchain?"

retriever= db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)

llm=ChatOpenAI(model="gpt-4o-mini")

# Contextualize question prompt
# This system prompt helps the AI understand that it should reformulate the question
# based on the chat history to make it a standalone question
contexualize_q_system_prompt=(
    "Given a chat history and the latest user question "
    "which might reference context in the chat history "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contexualize_q_prompt=ChatPromptTemplate.from_messages(
    [
        ("system", contexualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

# Create a history-aware retriever
# This uses the LLM to help reformulate the question based on chat history
history_aware_retreiver=create_history_aware_retriever(
    llm,retriever,contexualize_q_prompt
)


# Answer question prompt
# This system prompt helps the AI understand that it should provide concise answers
# based on the retrieved context and indicates what to do if the answer is unknown
qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt= ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
# `create_stuff_documents_chain` feeds all retrieved context into the LLM
question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain=create_retrieval_chain(history_aware_retreiver,question_answer_chain)

def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []
    while True:
        query= input("You: ")
        if query.lower() == "exit":
            print("Ending the conversation. Goodbye!")
            break
        result=rag_chain.invoke({"input":query,"chat_history":chat_history})
        print("AI:",result['answer'])
        chat_history.append(HumanMessage(content=query))
        chat_history.append(SystemMessage(content=result['answer']))

if __name__=="__main__":
    continual_chat()
