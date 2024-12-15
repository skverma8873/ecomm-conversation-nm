import asyncio
import gradio as gr
from langchain.agents import Tool
#from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import json
import openai
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DataFrameLoader
import time
import pandas as pd
import whisper
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

#openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = os.getenv("OPENAI_API_KEY")
model_whisper = whisper.load_model("base")

# Transcription function
def transcribe_audio(audio_file):
    result = model_whisper.transcribe(audio_file)
    return result['text']


def load_json_file(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line.strip()} due to error: {e}")
    return data

# # Load Q&A and review data
qa_data = load_json_file("qa_Appliances.json")
review_data = load_json_file("app1.json")
qa_cust=pd.read_csv('./qaecomm.csv')

# Combine 'question' and 'answers' columns to create 'text' and handle NaN values
qa_cust["text"] = qa_cust["question"].fillna("") + " " + qa_cust["answers"].fillna("")
qa_cust = qa_cust.dropna(subset=["text"])  # Drop rows where 'text' is still NaN

# Load the cleaned data
loader = DataFrameLoader(qa_cust, page_content_column="text")
qacust_documents = loader.load()


# Prepare documents
def prepare_documents_qa(data):
    documents = []
    for item in data:
        content = f"Question: {item['question']} Answer: {item['answer']}"
        metadata = {"asin": item.get("asin"), "questionType": item.get("questionType")}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

def prepare_documents_reviews(data):
    documents = []
    for item in data:
        content = f"Title: {item['title']} Review: {item['text']}"
        metadata = {"asin": item.get("asin"), "rating": item.get("rating")}
        documents.append(Document(page_content=content, metadata=metadata))
    return documents

qa_documents = prepare_documents_qa(qa_data)
review_documents = prepare_documents_reviews(review_data)

# Create Vector Store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qa_vectorstore = FAISS.from_documents(qa_documents, embeddings)
reviews_vectorstore = FAISS.from_documents(review_documents, embeddings)
qacust_vectorstore = FAISS.from_documents(qacust_documents, embeddings)

# Async Retrieval for Efficiency
async def async_retrieve_from_vectorstore(vectorstore, query, k=3):
    """Retrieve documents asynchronously from the vector store."""
    return vectorstore.similarity_search(query, k=k)

async def retrieve_all(query):
    """Retrieve from all vector stores in parallel."""
    qa_task = async_retrieve_from_vectorstore(qa_vectorstore, query)
    reviews_task = async_retrieve_from_vectorstore(reviews_vectorstore, query)
    customers_task = async_retrieve_from_vectorstore(qacust_vectorstore, query)
    qa_docs, review_docs, customer_docs = await asyncio.gather(qa_task, reviews_task, customers_task)
    return qa_docs, review_docs, customer_docs

# Define Retrieval Functions
def preprocess_query(query: str) -> str:
    """Refine user query for better vector store matching."""
    # Optional: Apply transformations like stemming, keyword extraction, etc.
    return query.lower().strip()

def retrieve_from_qa(query: str):
    query = preprocess_query(query)
    docs = qa_vectorstore.similarity_search(query, k=5)  # Experiment with k
    if not docs:
        return "No relevant Q&A information found. Please refine your query."
    return "\n".join([doc.page_content for doc in docs])

def retrieve_from_reviews(query: str):
    query = preprocess_query(query)
    docs = reviews_vectorstore.similarity_search(query, k=5)  # Experiment with k
    if not docs:
        return "No relevant reviews found. Please refine your query."
    return "\n".join([doc.page_content for doc in docs])

def retrieve_answer(query: str):
    query = preprocess_query(query)
    docs = qacust_vectorstore.similarity_search(query, k=5)  # Experiment with k
    if not docs:
        return "No relevant customer support information found. Please refine your query."
    return "\n".join([doc.page_content for doc in docs])

# Define Tools
tools = [
    Tool(
        name="Product Q&A",
        func=retrieve_from_qa,
        description="Fetch information from the Q&A knowledge base for specific product-related questions."
    ),
    Tool(
        name="Product Reviews",
        func=retrieve_from_reviews,
        description="Fetch product reviews and customer feedback."
    ),
    Tool(
        name="Customers Q&A",
        func=retrieve_answer,
        description="Fetch information from the customer support Q&A knowledge base for general customer queries."
    ),
]

# Chat Model and Agent
#from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model="gpt-4", temperature=0.2)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)



prompt_template = """You are a smart and empathetic customer support assistant for an e-commerce platform. Your tasks include answering product-related questions, providing recommendations, addressing customer complaints, and resolving issues. Always maintain a professional and friendly tone.

Instructions:
1. Identify Query Type:
   - Product details, specifications, or warranty.
   - Product reviews or recommendations.
   - Order tracking or issue resolution.
   - Other.

2. Empathy & Clarity:
   - Always show understanding of the customer’s needs or frustration.
   - Provide clear, concise, and actionable responses.

3. Use Tools:
   - For retrieving product details, use the "Product Q&A" tool.
   - For finding relevant products, use embeddings to match with the vector store.
   - For handling general complaints, use the "Customers Q&A" tool.
   - For Product reviews and ratings use the "Product Reviews" tool

4. Avoiding Errors:
   - Stick to factual information retrieved from trusted sources.
   - If unsure or if the query is unrelated to e-commerce, politely acknowledge and redirect the customer.

--- 

Examples:

1. Product Q&A:
   - Input: "What’s the battery life of this phone?"
   - Final Answer: "The battery life of this phone is up to 12 hours of video playback. Would you like help finding accessories for it?"

2. Recommendations:
   - Input: "Which laptop is best for gaming under $1500?"
   - Final Answer: "For gaming, I recommend the XGaming Pro 2024. It has a powerful GPU, 16GB RAM, and excellent reviews for $1499. Let me know if you'd like more options!"

3. Out-of-Scope Query:
   - Input: "What’s the weather in New York?"
   - Final Answer: "I’m here to assist with product-related queries. For weather updates, I recommend checking a weather app."

---

Customer Query: {input}

---

Respond thoughtfully and empathetically to every customer query. Always ensure the customer feels heard and supported.

When answering questions:
- Analyze the query to decide if it’s about product details, reviews, or general customer support.
- Respond using one of the tools: ["Product Q&A", "Product Reviews", "Customers Q&A"].
- If unrelated, reply: "I’m here to assist with product-related queries. For weather updates, I recommend checking a weather app."



Question: {input}
Thought: Determine the type of question (product details, reviews, or general customer support).
Action: Select the appropriate tool from ["Product Q&A", "Product Reviews", "Customers Q&A"].
Action Input: Provide input for the tool.
Observation: Review the retrieved information.
Final Answer: Provide a helpful, personalized, and conversational response.

{tools}
"""

prompt = PromptTemplate(
    input_variables=["input", "tools"],
    template=prompt_template,
)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True,
    memory=memory,
    prompt=prompt
)


        
# Define the chatbot UI and logic
def create_gradio_interface():
    with gr.Blocks(elem_id="main-container") as interface:
        gr.Markdown("""
        <div style="text-align: center;">
            <h1 style="color: #005bb5;">🛍️ Applixa Chatbot</h1>
            <p style="font-size: 18px; color: #555;">Your assistant for Product Details,tracking orders, returns, and more!</p>
        </div>
        """)

        chatbot = gr.Chatbot([], elem_id="chatbot", bubble_full_width=True)

        last_interaction = gr.State(time.time())
        session_active = gr.State(True)
        with gr.Column(scale=6):
          with gr.Row():
            msg = gr.Textbox(show_label=False, placeholder="Type your question...", container=False)
            audio_input = gr.Audio(type="filepath", label="🎤 Ask via audio")
            track_order = gr.Button("🔍 Track Order")
            returns = gr.Button("↩️ Returns")
            gift_ideas = gr.Button("🎁 Gift Ideas")
            faqs = gr.Button("❓ FAQs")

        
        with gr.Row():
            clear = gr.Button("Clear")
            restart_session = gr.Button("Restart Session", visible=False)

        # Define functions
        def process_text_input(user_message, history, last_interaction, session_active):
            current_time = time.time()
            if current_time - last_interaction > 120:
                session_active = False
                return history + [("Session Ended", "Your session has timed out.")], last_interaction, session_active, gr.update(visible=True)

            if user_message.lower() in ["hi", "hello"]:
                response = "🤖 Hi, How may I assist you today?"
            else:
                try:
                    response = f"🤖 {agent.run(input=user_message)}"
                except Exception:
                    response = "🤖 I’m here to assist with product-related queries."

            last_interaction = current_time
            history.append((f"👤 {user_message}", response))
            return history, last_interaction, session_active, gr.update(visible=False)

        def process_audio_input(audio_file, history, last_interaction, session_active):
            user_message = transcribe_audio(audio_file)
            return process_text_input(user_message, history, last_interaction, session_active)

        # User message input function with conversation history
        def user_input(user_message, history, last_interaction, session_active):
            current_time = time.time()

            # Check for inactivity timeout (120 seconds)
            if current_time - last_interaction > 120:
                session_active = False
                return history + [("Session Ended", "Your session has timed out due to inactivity. Please restart a new session.")], last_interaction, session_active, gr.update(visible=True)

            # Check for "HI" or "Hello"
            if user_message.lower() in ["hi", "hello"]:
                response = "🤖 Hi, How may I assist you today?"
            else:
                # Use agent to process the message
                try:
                    response = f"🤖 {agent.run(input=user_message)}"
                except Exception as e:
                    response = "🤖 I’m here to assist with only product-related queries"

            # Update last interaction time and history
            last_interaction = current_time
            history.append((f"👤 {user_message}", response))

            return history, last_interaction, session_active, gr.update(visible=False)

        # Handle quick action buttons (Track Order, Returns, etc.)
        def handle_quick_action(action, history):
            if action == "FAQs":
                # Static FAQ response to prevent agent errors
                response = "🤖 Here are some frequently asked questions:\n" \
                           "1. What is your return policy?\n" \
                           "   - You can return items within 30 days of purchase.\n" \
                           "2. How can I track my order?\n" \
                           "   - Use the 'Track Order' button to check your order status.\n" \
                           "3. What payment methods do you accept?\n" \
                           "   - We accept credit/debit cards, net banking, and UPI.\n" \
                           "4. Can I cancel my order?\n" \
                           "   - Yes, you can cancel your order within 24 hours of placing it."
            else:
                # Use agent for other actions
                response = f"🤖 {agent.run(input=f'!{action}')}"
            
            return history + [(f"Quick Action: {action}", response)]

        # Restart session function
        def restart_session_func():
            session_active = True
            return [], time.time(), session_active, gr.update(visible=False)  # Clear history and reset session

        msg.submit(user_input, [msg, chatbot, last_interaction, session_active], [chatbot, last_interaction, session_active, restart_session], queue=False).then(
            lambda: "", None, [msg]
        )

        restart_session.click(restart_session_func, None, [chatbot, last_interaction, session_active, restart_session])

        clear.click(lambda: [], None, chatbot)

        track_order.click(lambda h: handle_quick_action("Track Order", h), chatbot, chatbot)
        returns.click(lambda h: handle_quick_action("Returns", h), chatbot, chatbot)
        gift_ideas.click(lambda h: handle_quick_action("Gift Ideas", h), chatbot, chatbot)
        faqs.click(lambda h: handle_quick_action("FAQs", h), chatbot, chatbot)


        # Bind functions
        msg.submit(process_text_input, [msg, chatbot, last_interaction, session_active], [chatbot, last_interaction, session_active, restart_session], queue=False).then(
            lambda: "", None, [msg]
        )
        audio_input.change(process_audio_input, [audio_input, chatbot, last_interaction, session_active], [chatbot, last_interaction, session_active, restart_session])
        clear.click(lambda: [], None, chatbot)
        restart_session.click(restart_session_func, None, [chatbot, last_interaction, session_active, restart_session])

    return interface

# Main execution
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(share=True)
