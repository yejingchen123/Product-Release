from pydantic import BaseModel

class Product(BaseModel):
    name: str
    
class Category(BaseModel):
    category:list[str]