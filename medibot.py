import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "0"

import streamlit as st
import datetime

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_FAISS_PATH = os.path.join(BASE_DIR, "vectorstore", "db_faiss")

HUGGINGFACE_REPO_ID = "meta-llama/Meta-Llama-3-8B-Instruct"


def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-content {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .medical-fact {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B6B;
        margin: 1rem 0;
    }
    .emergency-primary {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .emergency-secondary {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: white;
        margin: 0.3rem 0;
    }
    .helpline-card {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        padding: 0.8rem;
        border-radius: 8px;
        color: white;
        margin: 0.3rem 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def get_vectorstore():
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2'
        )
        db = FAISS.load_local(
            DB_FAISS_PATH,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        return db
    except Exception as e:
        st.error(f"Failed to load vectorstore: {str(e)}")
        return None


def create_medical_stats():
    if not PLOTLY_AVAILABLE:
        return None
    try:
        fig = go.Figure()
        categories = ['Cardiology', 'Neurology', 'Pediatrics', 'Oncology', 'Emergency']
        values = [45, 38, 52, 29, 41]
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            text=values,
            textposition='auto',
        ))
        fig.update_layout(
            title="Medical Queries by Specialty",
            xaxis_title="Medical Specialties",
            yaxis_title="Number of Queries",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#2c3e50'),
            showlegend=False,
            height=400
        )
        return fig
    except Exception as e:
        st.error(f"Chart error: {str(e)}")
        return None


def show_alternative_stats():
    st.markdown("### 📊 Medical Query Statistics")
    specialties_data = {
        "🫀 Cardiology": 45,
        "🧠 Neurology": 38,
        "👶 Pediatrics": 52,
        "🔬 Oncology": 29,
        "🚨 Emergency": 41
    }
    cols = st.columns(3)
    for i, (specialty, count) in enumerate(specialties_data.items()):
        with cols[i % 3]:
            st.metric(specialty, count)
    st.markdown("#### Distribution")
    for specialty, count in specialties_data.items():
        st.write(f"{specialty}: {count}")
        st.progress(count / 60)


def show_medical_fact():
    import random
    facts = [
        "🫀 Your heart beats about 100,000 times per day",
        "🧠 The human brain contains approximately 86 billion neurons",
        "🦴 Bones are stronger than steel, pound for pound",
        "👁️ The human eye can distinguish about 10 million colors",
        "🫁 Adults take about 20,000 breaths per day",
        "💧 The human body is about 60% water",
        "🔬 Your body produces 25 million new cells every second"
    ]
    fact = random.choice(facts)
    st.markdown(f"""
    <div class="medical-fact">
        <h4>💡 Did You Know?</h4>
        <p>{fact}</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="MediBot - AI Medical Assistant",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    load_css()

    st.markdown('<h1 class="main-header">🏥 MediBot - AI Medical Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your intelligent healthcare companion powered by advanced AI</p>', unsafe_allow_html=True)

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'query_count' not in st.session_state:
        st.session_state.query_count = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.datetime.now()

    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        st.error("❌ HF_TOKEN not found. Please check your .env file.")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 💬 Chat with MediBot")

        for message in st.session_state.messages:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        prompt = st.chat_input("🩺 Ask your medical question here...")

        if prompt:
            st.session_state.messages.append({'role': 'user', 'content': prompt})
            with st.chat_message('user'):
                st.markdown(prompt)

            with st.chat_message('assistant'):
                with st.spinner("🔍 Analyzing your question..."):

                    # Load vectorstore
                    vectorstore = get_vectorstore()
                    if vectorstore is None:
                        st.error("Failed to load the vector store")
                        st.stop()

                    # Retrieve relevant documents
                    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
                    source_documents = retriever.invoke(prompt)

                    # Build context from retrieved docs
                    context = "\n\n".join([doc.page_content for doc in source_documents])

                    # Build full prompt
                    full_prompt = f"""Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know; don't make up answers.
Don't provide anything outside the given context.

Context: {context}
Question: {prompt}

Start the answer directly. No small talk please.
"""

                    # Create InferenceClient and call API
                    try:
                        llm_client = InferenceClient(
                            provider="novita",
                            api_key=HF_TOKEN
                        )
                        hf_response = llm_client.chat_completion(
                            model=HUGGINGFACE_REPO_ID,
                            messages=[{"role": "user", "content": full_prompt}],
                            max_tokens=512,
                            temperature=0.5,
                            top_p=0.8
                        )
                        result = hf_response.choices[0].message.content
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")
                        st.stop()

                    st.markdown(result)

                    if source_documents:
                        with st.expander("📚 View Source Documents"):
                            for i, doc in enumerate(source_documents):
                                st.write(f"**📄 Source {i+1}:**")
                                content = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                                st.write(content)
                                st.write("---")

                    st.session_state.messages.append({'role': 'assistant', 'content': result})
                    st.session_state.query_count += 1

    with col2:
        st.markdown("### 📊 Dashboard")

        col2_1, col2_2 = st.columns(2)
        with col2_1:
            st.metric("🔢 Questions Asked", st.session_state.query_count)
        with col2_2:
            session_time = datetime.datetime.now() - st.session_state.start_time
            st.metric("⏱️ Session Time", f"{session_time.seconds//60}m")

        show_medical_fact()

        st.markdown("### ⚡ Quick Actions")

        if st.button("🔄 Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

        if st.button("📋 Example Questions"):
            examples = [
                "What are the symptoms of diabetes?",
                "How to prevent heart disease?",
                "What is hypertension?",
                "Signs of stroke to watch for?"
            ]
            for example in examples:
                st.write(f"• {example}")

        st.markdown("### 🏥 Medical Specialties")
        specialties = {
            "🫀 Cardiology": "Heart and cardiovascular system",
            "🧠 Neurology": "Brain and nervous system",
            "🦴 Orthopedics": "Bones and joints",
            "👶 Pediatrics": "Children's health",
            "🔬 Oncology": "Cancer treatment"
        }
        for specialty, description in specialties.items():
            with st.expander(specialty):
                st.write(description)

    st.markdown("---")

    col3, col4 = st.columns([2, 1])

    with col3:
        if PLOTLY_AVAILABLE:
            st.markdown("### 📈 Medical Query Analytics")
            fig = create_medical_stats()
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                show_alternative_stats()
        else:
            show_alternative_stats()

    with col4:
        st.markdown("### 🎯 Key Features")
        features = [
            "🤖 AI-Powered Responses",
            "📚 Source Attribution",
            "🔍 Smart Document Search",
            "💬 Natural Conversation",
            "⚡ Real-time Analysis"
        ]
        for feature in features:
            st.write(f"✅ {feature}")

    with st.sidebar:
        st.markdown("## 🏥 MediBot Info")

        st.markdown("""
        <div class="sidebar-content">
            <h4>💡 Daily Health Tips</h4>
            <p>• Drink 8-10 glasses of water daily</p>
            <p>• Exercise for 30-45 minutes</p>
            <p>• Get 7-8 hours of quality sleep</p>
            <p>• Eat 5 servings of fruits/vegetables</p>
            <p>• Practice meditation for mental wellness</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 🆘 Emergency Contacts - India")

        st.markdown("""
        <div class="emergency-primary">
            🚨 ALL EMERGENCY SERVICES: 112<br>
            <small>(Police, Fire, Ambulance)</small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="emergency-secondary">
            👮‍♂️ Police: 100 | 🚒 Fire: 101 | 🚑 Ambulance: 102 | 🏥 Medical: 108
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📞 Specialized Helplines")

        st.markdown("""
        <div class="helpline-card">
            ☣️ <strong>Poison Control:</strong> 011-2658-8707<br>
            <small>National Poisons Information Centre (AIIMS)</small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="helpline-card">
            🧠 <strong>Mental Health (KIRAN):</strong> 9152987821<br>
            <small>24x7 helpline by Ministry of Social Justice</small>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="helpline-card">
            📞 <strong>Other Helplines:</strong><br>
            Women: 1091 | Child: 1098 | Senior Citizens: 14567
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### ⚠️ Medical Disclaimer")
        st.warning(
            "This chatbot provides general medical information for educational purposes only. "
            "It is not a substitute for professional medical advice, diagnosis, or treatment. "
            "Always consult qualified healthcare providers for medical decisions. "
            "In emergencies, immediately call 112 or visit the nearest hospital."
        )

        st.markdown("### ℹ️ About")
        st.info(
            "MediBot v2.0 🇮🇳\n\n"
            "Powered by:\n"
            "• Mistral 7B AI Model\n"
            "• FAISS Vector Database\n"
            "• LangChain Framework\n"
            "• Streamlit UI\n\n"
            "Features:\n"
            "• Indian Emergency Contacts\n"
            "• AI-Powered Medical Responses\n"
            "• Source-Based Information\n"
            "• Real-time Analysis"
        )


if __name__ == "__main__":
    main()
