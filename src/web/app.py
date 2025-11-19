from fastapi import FastAPI
from web.schemas import Product,Category
from web.service import Classfication_Service
from runer.predict import Predictor
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from configuration import config
import uvicorn

app=FastAPI()

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer=AutoTokenizer.from_pretrained(str(config.PRETRAIN_MODEL_PATH))
model=AutoModelForSequenceClassification.from_pretrained(str(config.CHECKPOINT_BEST_DIR))
predictor=Predictor(model=model,tokenizer=tokenizer,device=device)
service=Classfication_Service(predictor=predictor)

@app.post('/predict')
def predict(product:Product)->Category:
    labels=service.predict(product.name)
    return Category(category=labels)


def web_run():
    uvicorn.run(app, host="0.0.0.0", port=8000)