# AI-Powered Interactive Resume Chatbot

## 🤖 Overview

A conversational AI assistant that transforms my traditional resume into an interactive experience. Visitors can ask questions about my skills, experiences, projects, and qualifications through natural conversation, receiving personalized responses based on my actual documents and credentials.

What makes this chatbot unique is that it also provides access to my personal notes and cheat sheets from various technical domains, allowing visitors to benefit from my learning journey and knowledge collection.

🔗 [Live Demo](https://shruthin-portfolio.onrender.com/)

![Chatbot Demo](./static/Images/chatbot-demo.gif)

## ✨ Key Features

- **Natural Language Understanding**: Understands complex queries about my professional background
- **Semantic Document Search**: Uses vector embeddings to find the most relevant information
- **Context-Aware Responses**: Maintains conversation context for follow-up questions
- **Mobile-Responsive Design**: Works seamlessly across desktop and mobile devices
- **Quick-Access Buttons**: Common questions are just one click away
- **Knowledge Sharing**: Access to my technical notes and cheat sheets (SQL vs PySpark vs Pandas, Git commands, ML algorithms, etc.)

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript
- **Backend**: Flask (Python)
- **Database**: ChromaDB (Vector Database)
- **AI/ML**:
  - Google Gemini API for both text generation and embeddings
  - NLTK and spaCy for text processing
  - Custom prompt engineering for persona-based responses
- **Deployment**: Docker, Render Cloud

## 🏗️ Architecture

1. **Document Processing Pipeline**:
   - Resume, projects, certifications, and other documents are processed and embedded
   - Technical notes and cheat sheets are categorized and embedded
   - Google Gemini API creates semantic embeddings for all documents
   - Text chunks and embeddings are stored with metadata in ChromaDB

2. **Query Processing Flow**:
   - User messages are processed using NLP techniques
   - Queries are converted to embeddings using the same Gemini API
   - ChromaDB compares query embeddings with document embeddings
   - Most relevant document chunks are retrieved based on similarity
   - Context-aware responses are generated using Gemini API

## 🚀 How It Works

1. **Embeddings Creation**: Documents are split into manageable chunks and transformed into vector embeddings using Gemini API.
2. **User Interaction**: Visitors ask questions through a chat interface.
3. **Semantic Search**: The system finds the most contextually relevant information from my documents using embedding similarity.
4. **Response Generation**: The AI generates personalized, conversational responses using the retrieved information.
5. **Dynamic UI**: The chat interface updates in real-time with typing animations and smooth scrolling.

## 💻 Local Development

### Prerequisites

- Python 3.10+
- Google Gemini API key (used for both text generation and embeddings)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/shruthin4/shruthin-portfolio.git
   cd shruthin-portfolio
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m nltk.downloader punkt wordnet stopwords
   python -m spacy download en_core_web_sm
   ```

4. Set up environment variables:
   ```bash
   export GEMINI_API_KEY=your_api_key  # Windows: set GEMINI_API_KEY=your_api_key
   ```

5. Run the application:
   ```bash
   python app.py
   ```

6. Visit `http://localhost:5000` in your browser

## 🐳 Docker Deployment

```bash
# Build the Docker image
docker build -t resume-chatbot .

# Run the container
docker run -p 10000:10000 -e GEMINI_API_KEY=your_api_key resume-chatbot
```

## 📂 Project Structure

```
├── app.py                    # Main Flask application
├── chroma_db.py              # Vector database operations
├── nlp_utils.py              # NLP processing utilities
├── Dockerfile                # Container configuration
├── requirements.txt          # Python dependencies
├── Documents/                # Source documents for embeddings
│   ├── Certifications/       # Certificates and credentials
│   ├── Projects/             # Project descriptions
│   ├── Resume/               # Resume documents
│   └── Notes/                # Technical cheat sheets and notes
├── static/                   # Static assets
│   ├── Images/               # Website images
│   ├── script.js             # Frontend JavaScript
│   └── styling.css           # CSS styles
└── templates/                # HTML templates
    └── Intro.html            # Main portfolio page
```

## 📚 Technical Implementation Details

### Vector Search
The application uses Google Gemini API to generate embeddings and ChromaDB to store and retrieve them. This enables semantic search based on meaning rather than just keywords.

```python
# Example: Generating embeddings with Gemini API
def gemini_embedding_function(texts):
    embeddings = []
    for text in texts:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(response["embedding"])
    return embeddings
```

```python
# Example: Retrieving relevant documents
docs = query_chromadb(user_message, folder_category=folder_cat)
merged_text = combine_docs_text(docs)
```

### Conversational AI
The Google Gemini API is used with custom prompt engineering to generate natural, contextually relevant responses.

```python
# Response generation with context
final_html = generate_llm_response(user_message, merged_text, conversation_history)
```

### Error Handling
Robust error handling ensures the chatbot degrades gracefully when issues occur:

```javascript
.catch(error => {
    console.error("Error:", error);
    messageContent.textContent = "Sorry, I couldn't process your request. Please try again.";
});
```

## 🔍 Future Enhancements

- Multi-language support for international visitors
- Voice interaction capabilities
- Advanced analytics to track common questions
- Additional visualization options for project demonstrations
- Enhanced mobile experience with PWA capabilities
- Expanded collection of technical notes and cheat sheets

## 📫 Connect With Me

- [LinkedIn](https://www.linkedin.com/in/shruthin-reddy-sainapuram/)
- [GitHub](https://github.com/shruthin4)
- [Email](mailto:shruthinreddysainapuram@gmail.com)

