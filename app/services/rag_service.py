"""RAG Service - Handles corpus and chat operations"""
import warnings
import sys

# Suppress pathlib warnings from Python 3.14 compatibility
warnings.filterwarnings('ignore')

# Lazy import due to Python 3.14 pathlib issues
vertexai = None
rag = None
GenerativeModel = None
Tool = None

from app.core.config import (
    PROJECT_ID, 
    LOCATION, 
    RAG_CONFIG, 
    CHAT_MODEL, 
    SYSTEM_INSTRUCTION
)
from app.core.auth import authenticate_vertex_ai


def _lazy_imports():
    """Lazy load Vertex AI imports"""
    global vertexai, rag, GenerativeModel, Tool
    
    if vertexai is None:
        import vertexai as vx
        from vertexai import rag as rag_module
        from vertexai.generative_models import GenerativeModel as GM, Tool as T
        
        vertexai = vx
        rag = rag_module
        GenerativeModel = GM
        Tool = T


class RAGService:
    """Service to manage RAG operations"""
    
    def __init__(self):
        self.corpus = None
        self.chat_model = None
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize RAG with corpus and chat model"""
        try:
            # Lazy load imports
            _lazy_imports()
            
            # Authenticate and init Vertex AI
            authenticate_vertex_ai(PROJECT_ID, LOCATION)
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            
            # Load corpus
            corpus_list = list(rag.list_corpora())
            if not corpus_list:
                print("⚠️  Nenhum corpus encontrado")
                return False
            
            self.corpus = corpus_list[0]
            
            # Configure RAG retrieval
            rag_retrieval_config = rag.RagRetrievalConfig(
                top_k=RAG_CONFIG["top_k"],
                filter=rag.Filter(vector_distance_threshold=RAG_CONFIG["vector_distance_threshold"]),
            )
            
            rag_retrieval_tool = Tool.from_retrieval(
                retrieval=rag.Retrieval(
                    source=rag.VertexRagStore(
                        rag_resources=[rag.RagResource(rag_corpus=self.corpus.name)],
                        rag_retrieval_config=rag_retrieval_config,
                    ),
                )
            )
            
            # Initialize chat model
            self.chat_model = GenerativeModel(
                model_name=CHAT_MODEL,
                tools=[rag_retrieval_tool],
                system_instruction=SYSTEM_INSTRUCTION
            )
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Erro ao inicializar RAG: {e}")
            return False
    
    def chat(self, message: str) -> str:
        """Send message and get response"""
        if not self.initialized or not self.chat_model:
            raise RuntimeError("RAG não está inicializado")
        
        if not message or not message.strip():
            raise ValueError("Mensagem vazia")
        
        try:
            response = self.chat_model.generate_content(message)
            return response.text
        except Exception as e:
            raise RuntimeError(f"Erro ao gerar resposta: {e}")
    
    def get_corpus_info(self) -> dict:
        """Get current corpus information"""
        if not self.corpus:
            return None
        
        return {
            "name": self.corpus.display_name,
            "id": self.corpus.name
        }
    
    def list_all_corpus(self) -> list:
        """List all available corpus"""
        try:
            corpus_list = list(rag.list_corpora())
            return [
                {"name": c.display_name, "id": c.name}
                for c in corpus_list
            ]
        except Exception as e:
            raise RuntimeError(f"Erro ao listar corpus: {e}")


# Global instance
rag_service = None


def get_rag_service() -> RAGService:
    """Get or create RAG service instance"""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service
