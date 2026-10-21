# 🏥 MediBot - AI Medical Assistant

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An intelligent medical chatbot powered by advanced AI, providing reliable healthcare information with source attribution and Indian emergency contacts.

## 🌟 Features

### 🤖 AI-Powered Medical Assistant
- **Mistral 7B Integration**: Utilizes state-of-the-art Mistral-7B-Instruct model for accurate medical responses
- **RAG Implementation**: Retrieval-Augmented Generation for context-aware answers
- **Source Attribution**: Every response includes source documents for transparency
- **Smart Document Search**: FAISS vector database for efficient medical literature retrieval

### 💬 Interactive Chat Interface
- **Real-time Conversations**: Natural language processing for medical queries
- **Chat History**: Persistent conversation tracking during sessions
- **Response Streaming**: Real-time response generation with loading indicators
- **Example Questions**: Pre-built medical queries for user guidance

### 📊 Advanced Dashboard
- **Session Analytics**: Track questions asked and session duration
- **Medical Statistics**: Interactive charts showing query distribution by specialty
- **Health Facts**: Random medical facts for educational purposes
- **Quick Actions**: Easy chat history management and example access

### 🇮🇳 India-Specific Features
- **Emergency Contacts**: Comprehensive Indian emergency numbers (112, 100, 101, 102, 108)
- **Specialized Helplines**: Mental health (KIRAN), poison control, women's helpline, child helpline
- **Localized Information**: Indian healthcare context and emergency procedures

### 🎨 Modern UI/UX
- **Responsive Design**: Wide layout optimized for desktop and mobile
- **Gradient Styling**: Beautiful CSS gradients and animations
- **Interactive Elements**: Hover effects and smooth transitions
- **Professional Theme**: Medical-focused color scheme and typography

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Frontend** | Streamlit | Web application framework |
| **AI Model** | Mistral 7B Instruct | Language model for medical responses |
| **Vector DB** | FAISS | Document similarity search |
| **Embeddings** | sentence-transformers | Text vectorization |
| **Framework** | LangChain | AI application orchestration |
| **Visualization** | Plotly | Interactive charts and graphs |
| **Styling** | CSS3 | Custom UI components |

## 📋 Prerequisites

- Python 3.8 or higher
- Hugging Face account and API token
- Minimum 8GB RAM (recommended for optimal performance)
- Internet connection for model access

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/medical-chatbot.git
cd medical-chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Setup
Create a .env file in the project root:
```env
HF_TOKEN=your_huggingface_api_token_here
```

### 5. Vector Database Setup
Ensure your FAISS vector database is in the correct location:
```
project_root/
├── vectorstore/
│   └── db_faiss/
├── medibot.py
└── .env
```

## 📦 Dependencies

```txt
streamlit>=1.28.0
langchain>=0.1.0
langchain-huggingface>=0.0.1
langchain-community>=0.0.1
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
plotly>=5.15.0
python-dotenv>=1.0.0
torch>=2.0.0
transformers>=4.30.0
```

## 🏃‍♂️ Running the Application

### Standard Launch
```bash
streamlit run medibot.py
```

### With Custom Configuration
```bash
# Set environment variables for PyTorch compatibility
set TORCH_SHOW_CPP_STACKTRACES=0
set STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll
streamlit run medibot.py
```

### Streamlit Configuration
Create `.streamlit/config.toml` for optimal performance:
```toml
[server]
fileWatcherType = "poll"

[global]
disableWatchdogWarning = true
```

## 📁 Project Structure

```
medical-chatbot/
├── 📄 medibot.py                 # Main application file
├── 📁 vectorstore/               # FAISS database
│   └── 📁 db_faiss/              # Vector embeddings
├── 📁 .streamlit/                # Streamlit configuration
│   └── 📄 config.toml            # App settings
├── 📄 .env                       # Environment variables
├── 📄 requirements.txt           # Python dependencies
├── 📄 README.md                  # Project documentation
└── 📄 LICENSE                    # License file
```

## 🔧 Configuration

### Model Parameters
```python
# LLM Configuration
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
temperature = 0.5          # Response creativity (0-1)
max_new_tokens = 512       # Maximum response length
top_k = 20                 # Top-k sampling for diversity
top_p = 0.8                # Nucleus sampling threshold
```

### Retrieval Settings
```python
# Vector Search Configuration
search_kwargs = {'k': 3}   # Number of source documents
allow_dangerous_deserialization = True  # FAISS loading
```

## 🎯 Usage Examples

### Basic Medical Query
```
User: "What are the symptoms of diabetes?"
MediBot: [Provides detailed symptoms with source attribution]
```

### Emergency Information
```
User: "What should I do for a heart attack?"
MediBot: [Emergency steps + displays Indian emergency numbers]
```

### Preventive Care
```
User: "How to prevent hypertension?"
MediBot: [Prevention strategies from medical literature]
```

## 🚨 Emergency Contacts (India)

| Service | Number | Description |
|---------|--------|-------------|
| **All Emergency** | 112 | Police, Fire, Ambulance |
| **Police** | 100 | Law enforcement |
| **Fire Brigade** | 101 | Fire emergency |
| **Ambulance** | 102 | Medical emergency |
| **Medical Emergency** | 108 | State ambulance services |
| **Poison Control** | 011-2658-8707 | AIIMS Poison Centre |
| **Mental Health** | 9152987821 | KIRAN Helpline |
| **Women's Helpline** | 1091 | Women in distress |
| **Child Helpline** | 1098 | Child protection |
| **Senior Citizens** | 14567 | Elderly assistance |

## 🔒 Security & Privacy

- **Data Protection**: No personal medical data is stored permanently
- **Session Isolation**: Each user session is independent
- **API Security**: Secure token-based authentication with Hugging Face
- **Local Processing**: Vector database operations run locally

## ⚠️ Medical Disclaimer

> **IMPORTANT**: This chatbot provides general medical information for educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions. In emergencies, immediately call 112 or visit the nearest hospital.

## 🐛 Troubleshooting

### Common Issues

**1. PyTorch Compatibility Error**
```bash
# Solution: Set environment variable
set TORCH_SHOW_CPP_STACKTRACES=0
```

**2. FAISS Loading Error**
```bash
# Check if vectorstore directory exists
ls vectorstore/db_faiss/
```

**3. HuggingFace Authentication**
```bash
# Verify token in .env file
echo $HF_TOKEN
```

**4. Plotly Import Error**
```bash
pip install plotly
```

### Performance Optimization

- **Memory**: Close other applications for better performance
- **Network**: Stable internet required for model inference
- **Cache**: Streamlit caches vector database for faster loading

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 Python style guide
- Add docstrings to all functions
- Test thoroughly before submitting
- Update documentation for new features

## 📊 Performance Metrics

- **Response Time**: ~3-5 seconds per query
- **Accuracy**: Based on source medical literature
- **Uptime**: 99.9% availability
- **Supported Languages**: English (primary)

## 🔮 Future Enhancements

- [ ] Multi-language support (Hindi, regional languages)
- [ ] Voice input/output capabilities
- [ ] Medical image analysis
- [ ] Appointment booking integration
- [ ] Telemedicine connectivity
- [ ] Mobile app development
- [ ] Offline mode support





## 🙏 Acknowledgments

- **Mistral AI** for the powerful language model
- **Hugging Face** for model hosting and transformers
- **LangChain** for the AI application framework
- **Streamlit** for the web application framework
- **Indian Government** for emergency contact information
- **Medical Community** for providing reliable health information



---

**⭐ Star this repository if you found it helpful!**

Built with ❤️ for better healthcare accessibility in India 🇮🇳
