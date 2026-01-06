import requests
import logging
import re
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class LLMManager:
    """Manage LLM calls to Groq API"""
    
    # Updated Groq available models (correct names)
    MODELS = {
        'fast': 'mixtral-8x7b-32768',
        'balanced': 'llama-3.3-70b-versatile',  # Updated model
        'quality': 'llama-3.1-70b-versatile',   # Updated model
        'latest': 'llama-3.1-8b-instant'        # Updated model
    }
    
    def __init__(self, api_key: str, model: str = 'balanced'):
        self.api_key = api_key
        self.model = self.MODELS.get(model, model)
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.temperature = 0.3
        self.max_tokens = 512
    
    def generate(self, messages: List[Dict], temperature: Optional[float] = None) -> str:
        """Generate response from LLM"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": self.max_tokens,
                "top_p": 0.9
            }
            
            logger.info(f"Calling LLM model: {self.model}")
            
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            response.raise_for_status()
            result = response.json()
            
            content = result["choices"][0]["message"]["content"]
            logger.info(f"LLM response received ({len(content)} chars)")
            
            return content
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from LLM API: {e}")
            if e.response.status_code == 404:
                # Try with a fallback model
                logger.info("Trying fallback model: mixtral-8x7b-32768")
                payload["model"] = "mixtral-8x7b-32768"
                fallback_response = requests.post(
                    self.base_url,
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                fallback_response.raise_for_status()
                result = fallback_response.json()
                return result["choices"][0]["message"]["content"]
            raise
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "I apologize, but I'm having trouble processing your request right now. Please try again."
    
    def format_response(self, text: str) -> str:
        """Clean and format LLM response"""
        if not text:
            return text
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        text = re.sub(r'_(.*?)_', r'\1', text)
        
        # Remove headers
        text = re.sub(r'^#+\s+(.+)$', r'\1', text, flags=re.MULTILINE)
        
        # Remove code blocks
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Clean extra whitespace
        text = re.sub(r'\n\s*\n+', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Remove any remaining special formatting
        text = re.sub(r'[\-\=]{3,}', '', text)
        
        return text.strip()