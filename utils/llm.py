from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class TextChatMessage:
    """Representation of a single text-based chat message in the chat history."""
    role: str # Either "system", "user", or "assistant"
    content: str # The text content of the message (could also be a string.Template instance)
    source: str
    
@dataclass
class KnowledgeUnitResult:
    """Representation of a single text-based result message"""
    content: list
    source: str

class KnowledgeUnits(BaseModel):
    knowledge_units: list[str]
    
class NamedEntityExtraction(BaseModel):
    named_entities: list[str]
