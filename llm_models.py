"""
LLM Model Integration - Mistral, Qwen3, Llama
"""
import time
from typing import Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import config


class LLMModel:
    """Base class for LLM models"""
    
    def __init__(self, model_name: str, model_config: Dict):
        self.model_name = model_name
        self.config = model_config
        self.model = None
        self.tokenizer = None
        self.device = config.DEVICE if torch.cuda.is_available() else 'cpu'
        
    def load_model(self):
        """Load model and tokenizer"""
        raise NotImplementedError
    
    def generate(self, prompt: str) -> tuple:
        """Generate response and return (response, latency)"""
        raise NotImplementedError
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class LocalLLMModel(LLMModel):
    """Local HuggingFace model"""
    
    def load_model(self, use_quantization: bool = True):
        """Load model with optional quantization"""
        print(f"Loading {self.model_name}...")
        
        # Configure quantization for lower memory usage
        if use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['name'],
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['name'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['name'],
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Model {self.model_name} loaded successfully on {self.device}")
    
    def generate(self, prompt: str) -> tuple:
        """Generate response and measure latency"""
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        if torch.cuda.is_available():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get('max_tokens', 512),
                temperature=self.config.get('temperature', 0.7),
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove prompt)
        response = response[len(prompt):].strip()
        
        latency = time.time() - start_time
        
        return response, latency


class ModelManager:
    """Manages multiple LLM models"""
    
    def __init__(self):
        self.models = {}
        self.current_model = None
    
    def initialize_model(self, model_key: str) -> LocalLLMModel:
        """Initialize a specific model"""
        if model_key not in config.MODELS:
            raise ValueError(f"Model {model_key} not found in config")
        
        if model_key not in self.models:
            model_config = config.MODELS[model_key]
            self.models[model_key] = LocalLLMModel(model_key, model_config)
            self.models[model_key].load_model()
        
        self.current_model = model_key
        return self.models[model_key]
    
    def get_model(self, model_key: str) -> LocalLLMModel:
        """Get a model (initialize if needed)"""
        if model_key not in self.models:
            return self.initialize_model(model_key)
        return self.models[model_key]
    
    def generate_response(self, model_key: str, prompt: str) -> tuple:
        """Generate response using specified model"""
        model = self.get_model(model_key)
        return model.generate(prompt)
    
    def unload_all(self):
        """Unload all models"""
        for model in self.models.values():
            model.unload_model()
        self.models = {}
        self.current_model = None
