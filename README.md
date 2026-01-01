# AI Customer Support Assistant 🤖

A powerful multilingual AI-powered customer support system for e-commerce platforms with intelligent product search, order tracking, and natural language conversations.

## 🌟 Features

- **Multilingual Support** 🌐 - English and Bengali language support with automatic detection
- **Intelligent Product Search** 🔍 - Vector similarity search with Pinecone integration
- **Customer Reviews & Ratings** ⭐ - Access to real product reviews and ratings
- **Order Tracking** 📦 - Order status and tracking information
- **AI-Powered Conversations** 💬 - Natural language processing using Groq LLM
- **Real-time Database** 🗄️ - MySQL integration for live data
- **RESTful API** 🔌 - FastAPI backend with comprehensive endpoints
- **Vector Database** 🗂️ - Pinecone for efficient similarity search

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- MySQL Database
- [Pinecone](https://www.pinecone.io/) Account
- [Groq](https://groq.com/) API Account
- [Hugging Face](https://huggingface.co/) Account (optional)

### Installation

1. **Clone the repository**

   ```bash
   git clone <your-repository-url>
   cd ai_assistant
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**

   Create a `.env` file in the root directory with your credentials:

   ```env
   # Database Configuration
   DB_USER=your_db_username
   DB_PASSWORD=your_db_password
   DB_HOST=your_db_host
   DB_PORT=3306
   DB_NAME=your_database_name

   # AI Services
   HF_API_TOKEN=your_huggingface_token
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_ENVIRONMENT=us-east-1
   PINECONE_INDEX_NAME=ecommerce-customer-support
   GROQ_API_KEY=your_groq_api_key
   ```

### Database Setup

Ensure your MySQL database has the required tables:

- `Products` - Product information (uid, name, description, price, brand, category, quantity)
- `ProductReview` - Customer reviews (rating, review_text, product_uid, user_name)
- `orders` - Order information (optional, for order tracking)
- `cart_items` - Shopping cart data (optional)

### Vector Database Initialization

Populate Pinecone with your product data:

```bash
python vectorize_database.py
```

This script will:

- Fetch products and reviews from your MySQL database
- Clean and process text data (removes HTML, normalizes text)
- Generate embeddings using HuggingFace models
- Upload vectors to Pinecone for similarity search

### Running the Application

**Development Mode:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**Production Mode:**

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## 📚 API Documentation

Once running, access interactive API documentation:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## 🔌 API Endpoints

### Chat Endpoints

**POST** `/api/v1/ai/chat`

```json
{
  "message": "What are your best selling electronics?",
  "user_id": 123,
  "conversation_id": "optional_chat_id"
}
```

**Response:**

```json
{
  "response": "Our best selling electronics include...",
  "sources": [...],
  "language": "english",
  "chat_id": "chat_abc123",
  "user_type": "authenticated_user",
  "conversation_history": [...],
  "success": true
}
```

**GET** `/api/v1/ai/conversation/{chat_id}`

- Retrieve complete conversation history

### Product Search

**POST** `/api/v1/ai/product-suggestions`

```json
{
  "search_query": "wireless headphones",
  "limit": 5
}
```

**POST** `/api/v1/ai/search-product`

- Direct product search with keyword matching

### Order Management

**POST** `/api/v1/ai/track-order`

```json
{
  "order_id": "ORD123456",
  "user_id": "123"
}
```

### System Health

**GET** `/health`

- Service status and Pinecone connection verification

## 💬 Usage Examples

### 1. Product Inquiry

```bash
curl -X POST "http://localhost:8000/api/v1/ai/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "Show me gaming laptops under $1500", "user_id": 123}'
```

### 2. Review Analysis

```bash
curl -X POST "http://localhost:8000/api/v1/ai/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "What do customers say about Samsung Galaxy reviews?", "user_id": 123}'
```

### 3. Bengali Language Support

```bash
curl -X POST "http://localhost:8000/api/v1/ai/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "আপনার সেরা মোবাইল ফোনগুলি কি?", "user_id": 123}'
```

### 4. Order Status

```bash
curl -X POST "http://localhost:8000/api/v1/ai/track-order" \
  -H "Content-Type: application/json" \
  -d '{"order_id": "ORD789012", "user_id": "123"}'
```

## 🛠️ Configuration

### AI Model Settings

**Groq LLM Configuration:**

- Model: `llama-3.1-8b-instant`
- Temperature: `0.1` (for consistent responses)
- Max Tokens: Configured for optimal performance

**Embedding Model:**

- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: `384`
- Device: `cpu` (auto-switches to GPU if available)

**Pinecone Settings:**

- Index: `ecommerce-customer-support`
- Region: `us-east-1`
- Metric: `cosine` similarity
- Vector Dimension: `384`

### Database Schema

Required table structure:

**Products Table:**

```sql
CREATE TABLE Products (
    uid BINARY(16) PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    price DECIMAL(10,2),
    brand VARCHAR(100),
    category VARCHAR(100),
    quantity INT
);
```

**ProductReview Table:**

```sql
CREATE TABLE ProductReview (
    rating INT,
    review_text TEXT,
    product_uid BINARY(16),
    user_name VARCHAR(100)
);
```

## 🚀 Deployment

### Render.com Deployment

1. **Connect Repository**

   - Link your GitHub repository to Render
   - Choose "Web Service" deployment

2. **Environment Variables**
   Set all environment variables in Render dashboard:

   - `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_NAME`
   - `PINECONE_API_KEY`, `GROQ_API_KEY`, `HF_API_TOKEN`
   - `PINECONE_INDEX_NAME`, `PINECONE_ENVIRONMENT`

3. **Build Settings**

   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port 10000`

4. **Deploy**
   - Render will automatically deploy on git push
   - Monitor build logs for any issues

### Other Platforms

The application can be deployed on:

- **AWS EC2** or **Lambda**
- **Google Cloud Run**
- **Azure App Service**
- **Heroku**
- **DigitalOcean App Platform**

## 🔧 Customization

### Adding New Languages

1. Update `LanguageDetector` class in `ai.py`:

```python
def detect_language(self, text):
    # Add new language detection logic
    if lang == 'es': return 'spanish'
    # Add more languages as needed
```

2. Extend `MultilingualPromptBuilder`:

```python
SPANISH_SYSTEM_PROMPT = """Eres un asistente de soporte al cliente para un sitio web de comercio electrónico..."""
```

### Custom Embedding Models

Change the embedding model in `ai.py`:

```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",  # Higher quality
    model_kwargs={'device': 'cuda'}  # Use GPU if available
)
```

### Extending Product Search

Modify search logic in `search_products_from_db` method:

```python
def search_products_from_db(self, keyword, filters=None):
    # Add custom filters: price range, category, brand, etc.
```

## 📊 Monitoring & Logging

### Health Checks

```bash
curl http://your-domain.com/health
```

Response includes:

- Service status
- Pinecone connection status
- Vector count statistics

### Application Logs

- Request/response logging
- Error tracking
- Performance metrics
- Language detection logs

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError: langchain_huggingface**

   ```bash
   pip install langchain-huggingface
   # Or use alternative import:
   # from langchain_community.embeddings import HuggingFaceEmbeddings
   ```

2. **Pinecone Connection Issues**

   - Verify API key in environment variables
   - Check index exists in Pinecone console
   - Confirm region matches your index location

3. **Database Connection Errors**

   - Verify MySQL credentials in `.env`
   - Check network connectivity to database host
   - Ensure database tables exist with correct schema

4. **Empty Vector Store**

   - Run vectorization script: `python vectorize_database.py`
   - Check if products exist in MySQL database

5. **Groq API Limits**
   - Monitor API usage in Groq console
   - Implement rate limiting if needed
   - Check for valid API key

### Debug Mode

Enable detailed logging by setting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Documentation**: Check `/docs` endpoint when running
- **Issues**: Create a GitHub issue for bugs or feature requests
- **Questions**: Check existing issues or create a new discussion

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for high-performance LLM inference
- [Pinecone](https://www.pinecone.io/) for vector database services
- [Hugging Face](https://huggingface.co/) for embedding models
- [LangChain](https://www.langchain.com/) for AI framework
- [FastAPI](https://fastapi.tiangolo.com/) for web framework

---

**⭐ Star this repository if you find it helpful!**

**📧 Contact:** For enterprise support or custom implementations, please reach out through GitHub issues.
