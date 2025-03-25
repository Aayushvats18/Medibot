import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Path to the FAISS database storing vectorized documents
DB_FAISS_PATH = "vectorstore/db_faiss"

# 🟢 Function to load the FAISS Vector Store
@st.cache_resource
def load_vectorstore():
    """Loads the FAISS vector database using Hugging Face embeddings."""
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    try:
        vectorstore = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        
        # ✅ Print debugging
        print("Vectorstore loaded successfully:", type(vectorstore))

        return vectorstore
    except Exception as e:
        print("❌ Error loading FAISS vectorstore:", e)
        st.error(f"❌ Failed to load FAISS vector store: {str(e)}")
        return None

# 🟢 Function to define a custom prompt format
def create_custom_prompt():
    """Creates a structured prompt template to ensure the chatbot explains answers clearly."""
    return PromptTemplate(
        template="""
        Use the provided context to answer the user's question **clearly**.  

        - If you don't know the answer, say: **"I'm not sure, but I can provide related information."**  
        - Avoid making up information beyond the provided context.  
        - **Explain the answer in a structured and easy-to-read manner.**  
        - If applicable, provide **numbered points, examples, or bullet points** for clarity.  

        **Context:** {context}  
        **Question:** {question}  

        👉 **Give a structured, well-formatted answer.**
        """,
        input_variables=["context", "question"]
    )

# 🟢 Function to initialize the Hugging Face LLM
def load_llm():
    """Loads the language model from Hugging Face for generating responses."""
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.5,
        model_kwargs={"token": HF_TOKEN, "max_length": 512}
    )

# 🟢 Function to format text with proper line breaks
def format_answer(answer_text):
    """Formats the answer to improve readability with numbered points or bullet points."""
    formatted_text = ""
    points = answer_text.split(". ")  # Splitting sentences into points based on ". "
    
    for idx, point in enumerate(points, start=1):
        formatted_text += f"**{idx}.** {point.strip()}\n\n"  # Adding numbers and new lines

    return formatted_text

# 🟢 Main Streamlit Application
def main():
    st.title("🤖 **Ask Medibot!** – Your AI Medical Assistant")

    # Initialize message history in session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display previous chat messages properly
    for msg in st.session_state.messages:
        role = msg['role']
        content = msg['content']
        st.chat_message(role).markdown(f"🗨️ **{role.capitalize()}:**\n\n{content}\n\n---")

    # Get user input
    user_question = st.chat_input("🔎 Ask me anything about medical topics...")

    if user_question:
        st.chat_message('user').markdown(f"**You asked:** {user_question}")
        st.session_state.messages.append({'role': 'user', 'content': user_question})

        try:
            # Load the vectorstore (retrieves relevant medical knowledge)
            vectorstore = load_vectorstore()
            if vectorstore is None:
                st.error("❌ Failed to load the knowledge database.")
                return

            # Initialize the QA Chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': create_custom_prompt()}
            )

            # Get response from the model (correct key: 'question', NOT 'query')
            response = qa_chain.invoke({'query': user_question})

            # ✅ Print response for debugging
            print("QA Response:", response)

            # Extract the generated answer and source documents
            raw_answer_text = response.get("result", "I couldn't find an answer.")
            source_docs = response.get("source_documents", [])

            # 🔹 Apply the formatting function to break answers into points
            formatted_answer = format_answer(raw_answer_text)

            # 🔹 Improved Structured Answer Formatting
            if source_docs:
                sources_text = "\n".join(
                    f"- **{doc.metadata.get('source', 'Unknown')}**, Page {doc.metadata.get('page', 'N/A')}"
                    for doc in source_docs if isinstance(doc.metadata, dict)
                )
            else:
                sources_text = "No specific sources found."

            structured_answer = f"""
            ### ✅ **Here's what I found:**  

            {formatted_answer}  

            ---  

            ### 📚 **References & Sources:**  

            {sources_text}
            """

            # Display answer in chat format
            st.chat_message('assistant').markdown(structured_answer)
            st.session_state.messages.append({'role': 'assistant', 'content': structured_answer})

        except Exception as e:
            print("🚨 Debugging Error:", e)
            st.error(f"🚨 **Error:** {str(e)}")

# Run the app    
if __name__ == "__main__":
    main()
