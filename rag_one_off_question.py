import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI 

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)  # Update to a valid embedding model if needed

db = Chroma(persist_directory=persistent_directory,
            embedding_function=embeddings)

query = "How do I learn langchain?"

retriever= db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)
relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

combined_input= (
    "Here are some documents that might help answer the question: "
    + query
    +"\n\n Relevant documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\n Please provide a concise answer based on the above documents. If the answer is found, respond with I am not sure"
)

model=ChatOpenAI(model="gpt-4o-mini")

messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]


# Invoke the model with the combined input
result = model.invoke(messages)

# Display the full result and content only
print("\n--- Generated Response ---")
# print("Full result:")
# print(result)
print("Content only:")
print(result.content)
