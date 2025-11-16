# ai_customer_support.py
import os
from dotenv import load_dotenv
load_dotenv()

import pymysql
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import requests
import langdetect
from langdetect import detect, LangDetectException
import uuid
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

_pinecone_instance = None
_pinecone_lock = None

# Text cleaning functions
def clean_html(text):
    """Remove HTML tags and clean up whitespace from text"""
    if not text:
        return ""
    
    # Remove HTML tags
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    
    # Remove extra whitespace, newlines, and special characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple whitespace with single space
    text = re.sub(r'\r\n', ' ', text)  # Replace Windows newlines
    text = re.sub(r'\n', ' ', text)    # Replace Unix newlines
    text = re.sub(r'\t', ' ', text)    # Replace tabs
    text = re.sub(r'\r', ' ', text)    # Replace carriage returns
    
    # Remove any remaining special characters but keep Bengali and English text
    text = re.sub(r'[^\w\s\u0980-\u09FF\.\,\!\?\-\$]', '', text)
    
    return text.strip()

def extract_clean_text(html_text, max_length=300):
    """Extract clean, readable text from HTML"""
    if not html_text:
        return ""
    
    # Remove HTML tags and clean
    clean_text = clean_html(html_text)
    
    # Extract the most meaningful parts (usually the first few sentences)
    sentences = re.split(r'[.!?]+', clean_text)
    meaningful_text = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 20:  # Only take substantial sentences
            meaningful_text += sentence + ". "
            if len(meaningful_text) > max_length:
                break
    
    return meaningful_text.strip() or clean_text[:max_length]

class GroqChat:
    def __init__(self, api_key, model_name="openai/gpt-oss-20b", temperature=0.1):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
    
    def invoke(self, messages):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"❌ Groq API error: {e}")
            return "I apologize, but I'm having trouble processing your request right now."

class LanguageDetector:
    """Detect user language (Bengali or English)"""
    
    @staticmethod
    def detect_language(text):
        """Detect if text is in Bengali or English"""
        try:
            lang = detect(text)
            if lang == 'bn':
                return 'bengali'
            elif lang == 'en':
                return 'english'
            else:
                return 'english'
        except LangDetectException:
            return 'english'

class MultilingualPromptBuilder:
    """Build prompts in appropriate language"""
    
    ENGLISH_SYSTEM_PROMPT = """You are an AI customer support assistant for an ecommerce website. 
    Use the provided context to answer the user's question accurately and helpfully.
    
    Your responsibilities:
    1. Provide product descriptions and information
    2. Answer questions about product reviews and quality
    3. Help customers create and manage orders
    4. Assist with returns, shipping, and policies
    
    CRITICAL RESPONSE FORMAT RULES:
    - Use ONLY plain text format, no markdown, no tables, no asterisks, no special formatting
    - No **bold**, *italic*, | tables, --- lines, or ### headers
    - Use simple line breaks and natural language only
    - Write in clear, conversational English
    - Do not use any special characters for formatting
    
    CONTENT GUIDELINES:
    - If you find relevant products in the context, ALWAYS mention them and provide details
    - Only say you don't know if there is truly no relevant information in the context
    - When suggesting products, include in natural sentences:
      * Product name and key features
      * Price and any discounts
      * Customer ratings and reviews
      * Why it matches their request
    - Focus on providing directly relevant information to the user's specific query"""

    BENGALI_SYSTEM_PROMPT = """আপনি একটি ই-কমার্স ওয়েবসাইটের জন্য একজন AI গ্রাহক সহায়তা সহায়ক।
    ব্যবহারকারীর প্রশ্নের উত্তর দিতে প্রদত্ত প্রসঙ্গ ব্যবহার করুন এবং সঠিক ও সহায়ক উত্তর দিন।
    
    আপনার দায়িত্ব:
    1. পণ্যের বিবরণ এবং তথ্য প্রদান করা
    2. পণ্যের পর্যালোচনা এবং গুণমান সম্পর্কে প্রশ্নের উত্তর দেওয়া
    3. গ্রাহকদের অর্ডার তৈরি এবং পরিচালনায় সহায়তা করা
    4. রিটার্ন, শিপিং এবং নীতি সম্পর্কে সহায়তা করা
    
    গুরুত্বপূর্ণ প্রতিক্রিয়া ফরম্যাট নিয়ম:
    - শুধুমাত্র সাধারণ টেক্সট ফরম্যাট ব্যবহার করুন, মার্কডাউন নয়, টেবিল নয়, তারকাচিহ্ন নয়, বিশেষ ফরম্যাটিং নয়
    - **বোল্ড**, *ইটালিক*, | টেবিল, --- লাইন, বা ### হেডার ব্যবহার করবেন না
    - সহল লাইন ব্রেক এবং প্রাকৃতিক ভাষা ব্যবহার করুন
    - পরিষ্কার, কথোপকথনমূলক বাংলায় লিখুন
    - ফরম্যাটিংয়ের জন্য কোন বিশেষ অক্ষর ব্যবহার করবেন না
    
    বিষয়বস্তু নির্দেশিকা:
    - যদি প্রসঙ্গে প্রাসঙ্গিক পণ্য পাওয়া যায়, তবে সেগুলো অবশ্যই উল্লেখ করুন এবং বিস্তারিত প্রদান করুন
    - শুধুমাত্র তখনই বলুন আপনি জানেন না যখন প্রসঙ্গে সত্যিই কোন প্রাসঙ্গিক তথ্য নেই
    - পণ্য সুপারিশ করার সময় প্রাকৃতিক বাক্যে অন্তর্ভুক্ত করুন:
      * পণ্যের নাম এবং প্রধান বৈশিষ্ট্য
      * মূল্য এবং কোন ডিসকাউন্ট
      * গ্রাহক রেটিং এবং পর্যালোচনা
      * কেন এটি তাদের অনুরোধের সাথে মেলে
    - ব্যবহারকারীর নির্দিষ্ট প্রশ্নের সাথে সরাসরি প্রাসঙ্গিক তথ্য প্রদান করুন"""
    
    @staticmethod
    def build_prompt(user_query, context, language='english', user_context=None):
        """Build language-specific prompt"""
        
        if language == 'bengali':
            system_prompt = MultilingualPromptBuilder.BENGALI_SYSTEM_PROMPT
            if context and "No specific product information" not in context:
                user_content = f"""প্রাসঙ্গিক পণ্য তথ্য:
{context}

ব্যবহারকারীর প্রশ্ন: {user_query}

উপরের তথ্যের উপর ভিত্তি করে একটি সহায়ক প্রতিক্রিয়া প্রদান করুন। নিম্নলিখিত বিষয়গুলি অন্তর্ভুক্ত করুন:
- সরাসরি ব্যবহারকারীর প্রশ্নের উত্তর দিন
- প্রাসঙ্গিক পণ্যগুলো উল্লেখ করুন এবং বিস্তারিত জানান
- প্রাকৃতিক, কথোপকথনমূলক বাংলায় লিখুন
- কোন টেবিল, বোল্ড টেক্সট, বা বিশেষ ফরম্যাটিং ব্যবহার করবেন না
- গ্রাহক পর্যালোচনা এবং রেটিং সম্পর্কে তথ্য দিন
- শুধুমাত্র সরাসরি প্রাসঙ্গিক পণ্য সম্পর্কে কথা বলুন"""
            else:
                user_content = f"""ব্যবহারকারীর প্রশ্ন: {user_query}

দুঃখিত, আমাদের ডাটাবেসে এই বিশেষ পণ্য সম্পর্কে তথ্য নেই। আপনি অন্য কোন পণ্য সম্পর্কে জানতে চান?"""
        else:
            system_prompt = MultilingualPromptBuilder.ENGLISH_SYSTEM_PROMPT
            if context and "No specific product information" not in context:
                user_content = f"""Relevant Product Information:
{context}

User Question: {user_query}

Please provide a helpful response based on the information above. Include:
- Direct answer to user's specific question
- Mention relevant products with details in natural sentences
- Write in conversational, plain English only
- No tables, bold text, or special formatting
- Include information about customer reviews and ratings
- Focus only on directly relevant products"""
            else:
                user_content = f"""User Question: {user_query}

I'm sorry, we don't have information about this specific product in our database. Is there another product you'd like to know about?"""
        
        return system_prompt, user_content

class EcommerceAICustomerSupport:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EcommerceAICustomerSupport, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if EcommerceAICustomerSupport._initialized:
            return
        
        logger.info("🚀 Initializing EcommerceAICustomerSupport (first and only time)")
        
        # Initialize Groq LLM
        self.llm = GroqChat(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="openai/gpt-oss-20b",
            temperature=0.1
        )
        
        self.language_detector = LanguageDetector()
        self.prompt_builder = MultilingualPromptBuilder()
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
        # Initialize database connection
        self.db_config = {
            'host': os.getenv('DB_HOST'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'database': os.getenv('DB_NAME'),
            'port': int(os.getenv('DB_PORT')),
            'charset': 'utf8mb4'
        }
        
        # Simple conversation memory
        self.conversation_history = []
        
        self.pinecone_available = self.setup_pinecone()
        
        EcommerceAICustomerSupport._initialized = True
        
        if self.pinecone_available:
            logger.info("✅ Pinecone initialized successfully (singleton)")
        else:
            logger.warning("⚠️ Pinecone not available - AI will use database queries only")

    def clean_ai_response(self, response):
        """Clean AI response by removing markdown formatting"""
        if not response:
            return response
            
        # Remove markdown bold
        response = re.sub(r'\*\*(.*?)\*\*', r'\1', response)
        # Remove markdown italic
        response = re.sub(r'\*(.*?)\*', r'\1', response)
        # Remove markdown headers
        response = re.sub(r'#+\s*(.*)', r'\1', response)
        # Remove table formatting
        response = re.sub(r'\|.*?\|\n?', '', response)
        response = re.sub(r'\-+\s*\-+', '', response)
        # Remove HTML line breaks
        response = re.sub(r'<br\s*/?>', '\n', response)
        # Remove other markdown elements
        response = re.sub(r'`{1,3}(.*?)`{1,3}', r'\1', response)
        # Clean up extra whitespace
        response = re.sub(r'\n\s*\n', '\n\n', response)
        response = re.sub(r' +', ' ', response)
        response = response.strip()
        
        return response

    def setup_pinecone(self):
        if hasattr(self, 'index') and self.index is not None:
            logger.info("✅ Pinecone already initialized, skipping setup")
            return True
        
        """Create or connect to Pinecone index"""
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                logger.error("❌ PINECONE_API_KEY not found")
                self.index = None
                return False
            
            self.pc = Pinecone(api_key=pinecone_api_key)
            self.index_name = os.getenv("PINECONE_INDEX_NAME")
            
            # Check if index exists
            existing_indexes = self.pc.list_indexes()
            index_names = [index.name for index in existing_indexes.indexes]
            
            if self.index_name not in index_names:
                logger.warning(f"❌ Pinecone index '{self.index_name}' not found")
                logger.info("💡 Index will be created and populated on startup")
                self.index = None
                return False
            
            # Connect to the index
            self.index = self.pc.Index(self.index_name)
            logger.info(f"✅ Connected to Pinecone index: {self.index_name}")
            
            # Test the connection
            stats = self.index.describe_index_stats()
            logger.info(f"📊 Pinecone has {stats.total_vector_count} vectors")
            
            return True
                
        except Exception as e:
            logger.error(f"❌ Error initializing Pinecone: {e}")
            self.index = None
            return False

    def generate_chat_id(self, user_id=None):
        """Generate unique chat ID for conversation"""
        if user_id:
            chat_id = f"user_{user_id}_{uuid.uuid4().hex[:8]}"
        else:
            chat_id = f"guest_{uuid.uuid4().hex[:12]}"
        
        return chat_id

    def search_similar_documents(self, query, k=5):
        """Search for similar documents in Pinecone"""
        try:
            if not hasattr(self, 'index') or self.index is None:
                return []
            
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            documents = []
            for match in results.matches:
                # Use the _preview field which contains the full product information
                preview_content = match.metadata.get("_preview", "")
                if not preview_content:
                    # Fallback to text field if _preview doesn't exist
                    preview_content = match.metadata.get("text", "")
                
                # Clean the content
                clean_content = clean_html(preview_content)
                
                doc = Document(
                    page_content=clean_content,
                    metadata=match.metadata
                )
                documents.append(doc)
            
            logger.info(f"✅ Found {len(documents)} documents via Pinecone")
            return documents
                
        except Exception as e:
            logger.error(f"❌ Error searching Pinecone: {e}")
            return []

    def get_db_connection(self):
        """Create MySQL database connection"""
        return pymysql.connect(**self.db_config)

    def search_products_from_db(self, keyword):
        """Search products directly from database by name or keyword"""
        connection = self.get_db_connection()
        products = []
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                # Search products by name, brand, or description
                query = """
                    SELECT uid, name, description, price, brand, category, quantity 
                    FROM Products 
                    WHERE name LIKE %s OR brand LIKE %s OR category LIKE %s OR description LIKE %s
                    LIMIT 10
                """
                search_term = f"%{keyword}%"
                cursor.execute(query, (search_term, search_term, search_term, search_term))
                raw_results = cursor.fetchall()
                
                for row in raw_results:
                    # Clean the description
                    clean_description = extract_clean_text(row['description'], 200)
                    
                    products.append({
                        'uid': str(uuid.UUID(bytes=row['uid'])) if isinstance(row['uid'], bytes) else str(row['uid']),
                        'name': row['name'],
                        'description': clean_description,
                        'price': row['price'],
                        'brand': row['brand'],
                        'category': row['category'],
                        'quantity': row['quantity']
                    })
                    
        except Exception as e:
            print(f"❌ Error searching products from database: {e}")
        finally:
            connection.close()
        
        return products

    def get_product_recommendations_with_reviews(self, keyword, limit=5):
        """Search for products and include their reviews for recommendations"""
        connection = self.get_db_connection()
        recommendations = []
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                # Search for products
                query = """
                    SELECT uid, name, description, price, brand, category, quantity,
                           (SELECT AVG(rating) FROM ProductReview WHERE product_uid = Products.uid) as avg_rating,
                           (SELECT COUNT(*) FROM ProductReview WHERE product_uid = Products.uid) as review_count
                    FROM Products 
                    WHERE name LIKE %s OR brand LIKE %s OR category LIKE %s OR description LIKE %s
                    ORDER BY review_count DESC, avg_rating DESC
                    LIMIT %s
                """
                search_term = f"%{keyword}%"
                cursor.execute(query, (search_term, search_term, search_term, search_term, limit))
                raw_results = cursor.fetchall()
                
                for row in raw_results:
                    product_uid_str = str(uuid.UUID(bytes=row['uid'])) if isinstance(row['uid'], bytes) else str(row['uid'])
                    
                    # Get top 3 reviews for this product
                    reviews_query = """
                        SELECT rating, review_text, user_name
                        FROM ProductReview 
                        WHERE product_uid = %s
                        ORDER BY rating DESC, created_at DESC
                        LIMIT 3
                    """
                    cursor.execute(reviews_query, (row['uid'],))
                    reviews = cursor.fetchall()
                    
                    # Clean the description
                    clean_description = extract_clean_text(row['description'], 150)
                    
                    # Clean and format reviews
                    clean_reviews = []
                    for r in reviews:
                        clean_review_text = clean_html(r['review_text'])
                        clean_reviews.append({
                            'rating': r['rating'],
                            'text': clean_review_text[:100],
                            'reviewer': r['user_name']
                        })
                    
                    recommendations.append({
                        'product_id': product_uid_str,
                        'name': row['name'],
                        'description': clean_description,
                        'price': row['price'],
                        'brand': row['brand'],
                        'category': row['category'],
                        'available_quantity': row['quantity'],
                        'average_rating': float(row['avg_rating']) if row['avg_rating'] else 0,
                        'total_reviews': row['review_count'] or 0,
                        'top_reviews': clean_reviews
                    })
                    
        except Exception as e:
            print(f"❌ Error getting product recommendations: {e}")
        finally:
            connection.close()
        
        return recommendations

    def get_product_reviews_from_db(self, product_name):
        """Get all reviews for a specific product by name"""
        connection = self.get_db_connection()
        reviews = []
        
        try:
            with connection.cursor(pymysql.cursors.DictCursor) as cursor:
                # First find the product by name
                product_query = """
                    SELECT uid FROM Products WHERE name LIKE %s LIMIT 1
                """
                cursor.execute(product_query, (f"%{product_name}%",))
                product = cursor.fetchone()
                
                if product:
                    product_uid = product['uid']
                    
                    # Get all reviews for this product
                    reviews_query = """
                        SELECT rating, review_text, user_name, created_at
                        FROM ProductReview 
                        WHERE product_uid = %s
                        ORDER BY rating DESC, created_at DESC
                    """
                    cursor.execute(reviews_query, (product_uid,))
                    raw_reviews = cursor.fetchall()
                    
                    for review in raw_reviews:
                        clean_review = clean_html(review['review_text'])
                        reviews.append({
                            'rating': review['rating'],
                            'text': clean_review,
                            'reviewer': review['user_name'],
                            'date': str(review['created_at'])[:10] if review['created_at'] else 'Unknown date'
                        })
                    
                    print(f"✅ Found {len(reviews)} reviews for product: {product_name}")
                else:
                    print(f"❌ Product not found: {product_name}")
                    
        except Exception as e:
            print(f"❌ Error getting product reviews: {e}")
        finally:
            connection.close()
        
        return reviews

    def format_pinecone_context(self, pinecone_docs):
        """Format Pinecone documents into readable context"""
        if not pinecone_docs:
            return ""
        
        context = "Available Products:\n\n"
        
        for i, doc in enumerate(pinecone_docs):
            metadata = doc.metadata
            context += f"Product {i+1}:\n"
            context += f"Name: {metadata.get('name', 'Unknown')}\n"
            context += f"Category: {metadata.get('category', 'Not specified')}\n"
            context += f"Brand: {metadata.get('brand', 'Not specified')}\n"
            context += f"Price: {metadata.get('price', 0)} BDT\n"
            
            discount = metadata.get('discount', 0)
            if discount and discount > 0:
                context += f"Discount: {discount}%\n"
            
            avg_rating = metadata.get('average_rating', 0)
            review_count = metadata.get('review_count', 0)
            context += f"Rating: {avg_rating}/5 ({review_count} reviews)\n"
            
            # Add tags if available
            tags = metadata.get('tags', [])
            if tags and isinstance(tags, list):
                context += f"Tags: {', '.join(tags[:8])}\n"
            
            # Add key features from the preview
            preview = doc.page_content
            if preview:
                # Extract key lines from preview
                lines = preview.split('\n')
                key_info = [line for line in lines if any(keyword in line.lower() for keyword in 
                         ['summary:', 'description:', 'key features', 'attributes:'])]
                if key_info:
                    context += "Key Information:\n"
                    for info in key_info[:3]:  # Limit to 3 key points
                        context += f"- {info}\n"
            
            context += "\n" + "="*50 + "\n\n"
        
        return context

    def get_customer_response(self, user_query, user_context=None, user_id=None, chat_id=None, conversation_history=None, product_context=None):
        """Get AI response with both Pinecone and database integration, with conversation context"""
        try:
            if not chat_id:
                chat_id = self.generate_chat_id(user_id)
        
            detected_language = self.language_detector.detect_language(user_query)
            logger.info(f"💬 Chat ID: {chat_id} | User: {user_id or 'guest'} | Language: {detected_language}")
            logger.info(f"📝 Query: {user_query}")
        
            context = ""
            pinecone_docs = []
            tracked_products = product_context or {}
        
            # First, try Pinecone vector search - this is the primary source
            if self.pinecone_available:
                logger.info("🔍 Searching Pinecone vector store...")
                pinecone_docs = self.search_similar_documents(user_query, k=5)
                
                if pinecone_docs:
                    logger.info(f"✅ Found {len(pinecone_docs)} relevant products in Pinecone")
                    context = self.format_pinecone_context(pinecone_docs)
                else:
                    logger.info("❌ No products found in Pinecone")
                    context = "No specific product information found in our database."
        
            # If no Pinecone results, fall back to database search
            if not context or "No specific product information" in context:
                logger.info("🔍 Falling back to database search...")
                words = user_query.split()
                search_keywords = [w for w in words if len(w) > 2]
                
                for keyword in search_keywords:
                    recommendations = self.get_product_recommendations_with_reviews(keyword, limit=3)
                    if recommendations:
                        logger.info(f"✅ Found {len(recommendations)} products in database")
                        context = "Available Products:\n\n"
                        for product in recommendations:
                            context += f"Product: {product['name']}\n"
                            context += f"Brand: {product['brand']}\n"
                            context += f"Price: {product['price']} BDT\n"
                            context += f"Category: {product['category']}\n"
                            context += f"Rating: {product['average_rating']}/5 ({product['total_reviews']} reviews)\n"
                            context += f"Description: {product['description']}\n\n"
                        break
        
            system_message, user_content = self.prompt_builder.build_prompt(
                user_query=user_query,
                context=context,
                language=detected_language
            )
        
            logger.info("🤖 Calling Groq API...")
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_content}
            ]
        
            response = self.llm.invoke(messages)
            logger.info("✅ Groq response received")
            
            # Clean the response to remove markdown formatting
            clean_response = self.clean_ai_response(response)
            logger.info("✅ Response cleaned and formatted")
        
            # Store conversation with chat_id
            self.conversation_history.append({
                "chat_id": chat_id,
                "user_id": user_id,
                "user": user_query,
                "assistant": clean_response,
                "language": detected_language,
                "timestamp": str(datetime.now())
            })
        
            # Keep only last 50 messages
            if len(self.conversation_history) > 50:
                self.conversation_history = self.conversation_history[-50:]
        
            return {
                "answer": clean_response,
                "source_documents": pinecone_docs,
                "conversation_history": self.conversation_history,
                "language": detected_language,
                "chat_id": chat_id,
                "product_context": tracked_products
            }
        
        except Exception as e:
            logger.error(f"❌ Error in get_customer_response: {e}")
            import traceback
            traceback.print_exc()
        
            detected_language = self.language_detector.detect_language(user_query) if user_query else 'english'
            default_msg = "আমি ক্ষমাপ্রার্থী, কিন্তু আমি এখনই আপনার অনুরোধ প্রক্রিয়া করতে সমস্যা হচ্ছে। পরে আবার চেষ্টা করুন।" if detected_language == 'bengali' else "I apologize, but I'm having trouble processing your request. Please try again later."
        
            return {
                "answer": default_msg,
                "source_documents": [],
                "conversation_history": self.conversation_history,
                "language": detected_language,
                "chat_id": chat_id or "error",
                "product_context": product_context or {}
            }

    def get_user_context(self, user_id=None):
        """Get user-specific context from database"""
        if not user_id:
            return None
        
        connection = self.get_db_connection()
        try:
            with connection.cursor() as cursor:
                # Check if orders table exists and get its structure
                cursor.execute("SHOW TABLES LIKE 'orders'")
                if not cursor.fetchone():
                    return None
                
                # Get user's recent orders
                cursor.execute("""SELECT id, status, total_amount FROM orders WHERE user_id = %s ORDER BY created_at DESC LIMIT 3""", (user_id,))
                recent_orders = cursor.fetchall()
                
                # Get user's cart items if cart_items table exists
                cursor.execute("SHOW TABLES LIKE 'cart_items'")
                if cursor.fetchone():
                    cursor.execute("""SELECT p.name, ci.quantity FROM cart_items ci JOIN products p ON ci.product_id = p.id WHERE ci.user_id = %s""", (user_id,))
                    cart_items = cursor.fetchall()
                else:
                    cart_items = []
                
        except Exception as e:
            print(f"Error getting user context: {e}")
            return None
        finally:
            connection.close()
        
        context = ""
        if recent_orders:
            context += f"User has {len(recent_orders)} recent orders. "
        if cart_items:
            context += f"User has {len(cart_items)} items in cart."
        
        return context if context else None

    def check_pinecone_status(self):
        """Check Pinecone status"""
        try:
            if not hasattr(self, 'index') or self.index is None:
                return {"status": "disconnected", "message": "Pinecone not initialized"}
            
            stats = self.index.describe_index_stats()
            return {
                "status": "connected", 
                "message": "Pinecone is working properly",
                "stats": {
                    "total_vector_count": stats.total_vector_count,
                    "dimension": stats.dimension
                }
            }
        except Exception as e:
            return {"status": "error", "message": f"Pinecone error: {str(e)}"}

# FastAPI Integration
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(
    title="Ecommerce AI Customer Support",
    description="Multilingual AI Customer Support with Product Search & Order Tracking"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    sources: list
    success: bool = True

# Initialize AI support system
ai_support = EcommerceAICustomerSupport()

@app.get("/")
async def root():
    return {"message": "Ecommerce AI Customer Support API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    try:
        user_context = ai_support.get_user_context(request.user_id)
        
        result = ai_support.get_customer_response(
            user_query=request.message,
            user_context=user_context,
            user_id=request.user_id
        )
        
        # Extract source information
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return ChatResponse(
            response=result["answer"],
            sources=sources,
            success=True
        )
    
    except Exception as e:
        return ChatResponse(
            response="Sorry, I encountered an error. Please try again.",
            sources=[],
            success=False
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    pinecone_status = ai_support.check_pinecone_status()
    return {
        "status": "healthy", 
        "service": "AI Customer Support",
        "pinecone": pinecone_status
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)