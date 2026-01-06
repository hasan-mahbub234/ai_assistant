import logging
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ChatHandler:
    """Handle chat conversations and state"""
    
    def __init__(self):
        self.conversations = {}
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get conversation by ID"""
        return self.conversations.get(conversation_id)
    
    def create_conversation(self, conversation_id: str, user_id: Optional[int] = None) -> Dict:
        """Create new conversation"""
        conversation = {
            'id': conversation_id,
            'user_id': user_id,
            'messages': [],
            'created_at': str(time.time()),  # Use time.time() instead of logging.time
            'product_context': []
        }
        
        self.conversations[conversation_id] = conversation
        return conversation
    
    def add_message(self, conversation_id: str, role: str, content: str):
        """Add message to conversation"""
        if conversation_id not in self.conversations:
            self.create_conversation(conversation_id)
        
        message = {
            'role': role,
            'content': content,
            'timestamp': str(time.time())  # Use time.time() here too
        }
        
        self.conversations[conversation_id]['messages'].append(message)
        
        # Keep only last 20 messages per conversation
        if len(self.conversations[conversation_id]['messages']) > 20:
            self.conversations[conversation_id]['messages'] = \
                self.conversations[conversation_id]['messages'][-20:]
    
    def update_product_context(self, conversation_id: str, products: List[Dict]):
        """Update product context for conversation"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]['product_context'] = products
    
    def clear_conversation(self, conversation_id: str):
        """Clear conversation messages"""
        if conversation_id in self.conversations:
            self.conversations[conversation_id]['messages'] = []
            self.conversations[conversation_id]['product_context'] = []
    
    def delete_conversation(self, conversation_id: str):
        """Delete conversation"""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]