
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from configuration import config

class Predictor:
    def __init__(self,model,tokenizer,device):
        self.model=model.to(device)
        self.tokenizer=tokenizer
        self.device=device
        
    
    def predict(self,texts:str|list):
        if isinstance(texts,str):
            texts=[texts]#单条预测转为批量预测
        inputs=self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        inputs={k:v.to(self.device) for k,v in inputs.items()}
        self.model.eval()
        with torch.no_grad():
            outputs=self.model(**inputs)
        predictions=torch.argmax(outputs.logits,dim=-1).cpu().numpy().tolist()
        results=[self.model.config.id2label[index] for index in predictions]
        return results
            
        



def predict():
    
    #设备
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #分词器
    tokenizer=AutoTokenizer.from_pretrained(str(config.PRETRAIN_MODEL_PATH))
    #模型
    model=AutoModelForSequenceClassification.from_pretrained(str(config.CHECKPOINT_BEST_DIR))
    #预测器
    predictor=Predictor(model=model,tokenizer=tokenizer,device=device)
    #预测
    ##母婴 办公 粮油速食
    texts=['800G合生元派星较大婴儿配方奶粉','小吉鸭9501滑滑沙','白象方便面大骨面原汁猪骨味75g*24袋整箱装']
    results=predictor.predict(texts)
    print(results)

if __name__=='__main__':
    predict()
    