from runer.train import Trainer,TrainConfig
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score,f1_score
from configuration import config

def evaluate():
    
    model=AutoModelForSequenceClassification.from_pretrained(str(config.CHECKPOINT_BEST_DIR))
    
    #评价函数
    def compute_metrics(y_preds:list,y_labels:list)->dict:
        acc=accuracy_score(y_labels,y_preds)
        f1=f1_score(y_labels,y_preds,average='weighted')
        return {'acc':acc,'f1':f1}
        
    
    trainconfig=TrainConfig()
    
    trainer=Trainer(model,trainconfig,compute_metrics)
    metrics=trainer.evaluate(trainconfig.test_type)
    print(f'测试集指标：{metrics}')

if __name__=='__main__':
    evaluate()