├── app.py  
├── .env  
├── requirements.txt
├── src/
│ ├── **init**.py
│ ├── config.py  
| ├── vectorize_database.py  
│ ├── database.py  
| ├── main.py
│ ├── core/
│ │ ├── **init**.py
│ │ ├── embeddings.py # Embedding manager
│ │ ├── pinecone_manager.py
│ │ └── database_manager.py
│ ├── utils/
│ │ ├── **init**.py
│ │ ├── banglish_detector.py
│ │ ├── embedding_manager.py
│ │ └── prompt_templates.py
│ └── ai/
│ ├── **init**.py
│ ├── ai_service.py # Main AI service
│ ├── chat_handler.py
│ ├── llm_manager.py
│ ├── context_builder.py
│ ├── language_detector.py
│ ├── prompt_builder.py
│ └── routes.py
