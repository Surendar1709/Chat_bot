import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings

#step 1: Load environment variables
load_dotenv()

os.environ["Google_API_KEY"]="AIzaSyCnOloqjSfg3sZV5hJ5Yc0iwAjM9w462lc"

os.environ["Google_API_KEY"] = os.getenv("Google_API_KEY")

# step2 :Load dataset
datapath = r"/home/Surendar.S/Tasks/Chatbot/RAG_MINI_bot/FIR_DATASET.csv"
loader = CSVLoader(datapath, encoding="utf-8")
docs = loader.load()

# step3: Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=20, length_function=len, is_separator_regex=False
)
splits = text_splitter.split_documents(docs)

# step4 : build and   load vector into the vector database 
PERSIST_DIR = "./chroma_store"
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.environ["Google_API_KEY"]
)

if os.path.exists(PERSIST_DIR):
    vector_store = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name="fir_collection"
    )
else:
    vector_store = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
        collection_name="fir_collection"
    )


# step5:intialize the  retriever
retriever = vector_store.as_retriever()


# step 6  Initialize LLM
llm = GoogleGenerativeAI(
    google_api_key=os.environ["Google_API_KEY"],
    model="gemini-1.5-flash",
    temperature=0.1
)

# step 7 creating the  prompt template 
template = """
You are a professional and courteous assistant and **Indian Law Expert** with deep knowledge of the Indian Penal Code (IPC).
When the user sends a greeting message (e.g., "Hi", "Hello", "Good morning" or similar), respond  with  small greeting 

If the user sends a message such as "Thanks", "Thank you", "Welcome" "ok", or similar:
– Respond with a short and professional thank-you reply.
– Reinforce your availability to assist further.

Examples of appropriate replies:
• "You're very welcome! I'm here anytime you need legal support."
• "Thank you for your kind words. Feel free to reach out with any further queries!"
• "Always happy to help. Let me know if you have any more questions."

every try  to mention the  IPC section of the  crime comes under

If the user provides an **IPC section number**, **description of an offense**, or **complaint narrative**, follow these steps:
1. **maintain  professionalism  every time 
2. **Identify if the description indicates an illegal activity under IPC.**
3. If it does, respond with a structured summary of the corresponding IPC section and details.
4. Recommend appropriate **platforms or authorities where the user can file a complaint**.

begin your legal response with:
Based on your input, give the  information to them  try to analyze  user  situation  and  construct your response 
:”
**Answer Format:**
- **IPC Section:**  Offense Classified Under IPC Section

- **Offense Name & Summary:** (1-2) line explanation  
- **Punishment:** Mention  the Punishment  clearly  to  user  and  penalties 
- **Cognizable / Non-Cognizable:**  
- **Bailable / Non-Bailable:**   
- **Complaint Platform:** e.g.[cybercrime.gov.in](https://www.cybercrime.gov.in)

If the input is unclear, respond with:
“I’m sorry, but I need more information to provide a relevant and accurate legal response.”

User Input:  
{question}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n".join(d.page_content for d in docs)

# step8  Implementing the  sequetial chain 
Rag_chain = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough()
} | prompt | llm | StrOutputParser()

# step 9  creating the userinterface with streamlit 

st.set_page_config(page_title="IPC Law Bot", layout="centered")
st.markdown("""<style>
.main {background: linear-gradient(to right, #1f1c2c, #928DAB); color: white;}
h1,h2 {text-align:center; color:#F5F5F5;}
</style>""", unsafe_allow_html=True)

st.markdown("## ⚖️ IPC Law Chatbot")
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Show past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input with chat UI
if user_input := st.chat_input("🔍 Ask about IPC laws or  give  crime description ..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = Rag_chain.invoke(user_input)
            st.markdown(result)
            st.session_state.messages.append({"role": "Assistant", "content": result})
