from dataclasses import dataclass
from configuration import config
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from processed.dataset import get_dataloader
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import autocast
from torch.amp import GradScaler
from sklearn.metrics import accuracy_score,f1_score
from pathlib import Path
import time
from tqdm import tqdm

@dataclass
class TrainConfig:
    #设备
    device:str='cuda' if torch.cuda.is_available() else 'cpu'
    #batch_size
    batch_size:int=config.BATCH_SIZE
    #learning_rate
    learning_rate:float=config.LEARNING_RATE
    #epoch
    epoch:int=config.EPOCH
    #保存模型步数
    save_steps:int=config.SAVE_STEPS
    #模型保存路径
    checkpoint_best_dir:str=str(config.CHECKPOINT_BEST_DIR)
    checkpoint_last_dir:str=str(config.CHECKPOINT_LAST_DIR)
    #日志路径
    logs_dir:str=str(config.LOGS_DIR)
    #数据集类型
    train_type:str='train'
    validation_type:str='validation'
    test_type:str='test'
    #预训练模型路径
    pretrain_model:str=str(config.PRETRAIN_MODEL_PATH)
    early_stop_type:str=config.EARLY_STOP_TYPE
    early_stop_patience:int=config.EARLY_STOP_PATIENCE
    #自动混合精度
    use_amp:bool=config.USE_AMP
    
    


class Trainer():
    def __init__(self,model,config,compute_metrics):
        #训练配置
        self.config=config
        #设备
        self.device=torch.device(self.config.device)
        #模型
        self.model=model.to(self.device)
        #数据加载
        self.tokenizer=AutoTokenizer.from_pretrained(self.config.pretrain_model)
        self.data_type=self.config.train_type        
        #优化器
        self.optimizer=torch.optim.Adam(self.model.parameters(),lr=self.config.learning_rate)
        #评价函数
        self.compute_metrics=compute_metrics
        #早停
        self.early_stop_type=self.config.early_stop_type
        self.early_stop_patience=self.config.early_stop_patience
        self.early_stop_counter=0
        self.best_metric=None
        #自动混合精度
        self.scaler=GradScaler(device=self.config.device,enabled=self.config.use_amp)
        
        #迭代步数
        self.step=1
        #tensorboard日志
        self.writer=SummaryWriter(log_dir=str(Path(self.config.logs_dir)/ time.strftime('%Y-%m-%d-%H-%M-%S')))
        #模型保存路径
        self.check_point_best_path=self.config.checkpoint_best_dir
        self.check_point_last_path=self.config.checkpoint_last_dir
        
    
    def train(self):
        #读档
        self.load_checkpoint()
        data_loader=get_dataloader(self.tokenizer,self.data_type)
        self.model.train()
        for epoch in range(1,self.config.epoch+1):
            print(f'========epoch {epoch}========')
            for batch in tqdm(data_loader,desc=f'epoch={epoch}'):
                
                step_loss=self.train_one_step(batch)
                #如果达到保存步数，就记录损失并判断是否保存模型
                if self.step % self.config.save_steps==0:
                    tqdm.write(f'step:{self.step}, loss:{step_loss:.4f}')
                    #记录损失到tensorboard
                    self.writer.add_scalar('loss',step_loss,self.step)
                    
                    metrics=self.evaluate(self.config.validation_type)
                    tqdm.write(f'验证集指标：{metrics}')
                    
                    if self.early_stop_check(metrics,self.early_stop_type):
                        tqdm.write('早停，训练结束！')
                        return
                    
                    #存档
                    self.save_checkpoint()
                    
                self.step+=1
                    
                
    
    def train_one_step(self,batch):
        #将数据移动到设备上
        batch={k:v.to(self.device) for k,v in batch.items()}
        #前向传播
        with autocast(device_type=self.config.device,enabled=self.config.use_amp):#自动混合精度(dvicetype:str)
            output=self.model(**batch)
            l=output.loss
        #反向传播
        self.scaler.scale(l).backward()
        #优化器更新参数
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        
        return l.item()
    
    #验证函数,返回不同指标的字典{accuracy,f1,avg_loss,......}
    def evaluate(self,data_type):
        self.model.eval()
        total_loss=0.0
        preds=[]
        targets=[]
        data_loader=get_dataloader(self.tokenizer,data_type)
        with torch.no_grad():
            for batch in tqdm(data_loader,desc='Evaluating'):
                #将数据移动到设备上
                batch={k:v.to(self.device) for k,v in batch.items()}
                #前向传播
                output=self.model(**batch)
                #损失
                l=output.loss
                total_loss+=l.item()
                #预测结果
                logits=output.logits
                pred=torch.argmax(logits,dim=-1)
                #收集预测结果和真实标签
                preds.extend(pred.cpu().numpy().tolist())
                targets.extend(batch['labels'].cpu().numpy().tolist())
            #计算平均损失
        self.model.train()
        avg_loss=total_loss/len(data_loader)
        metrics=self.compute_metrics(preds,targets)#计算其他指标,返回字典
        metrics['loss']=avg_loss
        return metrics
    
    def early_stop_check(self,metrics,metric_type):
        assert metric_type in ['loss','f1','acc'],f'Unknown early stop metric type {metric_type}'
        metric=metrics[metric_type]
        if metric_type=='loss':
            if self.best_metric is None or metric < self.best_metric:
                self.best_metric=metric
                self.early_stop_counter=0
                #保存模型
                self.model.save_pretrained(self.check_point_best_path)
                tqdm.write(f'模型保存')
            else:
                self.early_stop_counter+=1
        else:
            if self.best_metric is None or metric > self.best_metric:
                self.best_metric=metric
                self.early_stop_counter=0
                #保存模型
                self.model.save_pretrained(self.check_point_best_path)
                tqdm.write(f'模型保存')
            else:
                self.early_stop_counter+=1
        if self.early_stop_counter >= self.early_stop_patience:
            return True
        return False
    
    #存档
    def save_checkpoint(self):
        checkpoint={
            'model_state_dict':self.model.state_dict(),
            'optimizer_state_dict':self.optimizer.state_dict(),
            'scaler_state_dict':self.scaler.state_dict(),
            'step':self.step,
            'best_metric':self.best_metric,
            'early_stop_counter':self.early_stop_counter
        }
        torch.save(checkpoint,str(Path(self.check_point_last_path )/ 'last_checkpoint.pth'))
        tqdm.write("last checkpoint saved.")

    #读档
    def load_checkpoint(self):
        checkpoint_path=Path(self.check_point_last_path) / 'last_checkpoint.pth'
        if checkpoint_path.exists():
            checkpoint=torch.load(str(checkpoint_path),map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            self.step=checkpoint['step']
            self.best_metric=checkpoint['best_metric']
            self.early_stop_counter=checkpoint['early_stop_counter']
            tqdm.write(f'加载断点，继续训练，当前步数：{self.step}')
        else:
            tqdm.write('未找到断点，从头开始训练')
            
def train():
    label_list=[]
    with open(config.PROCESSED_DATA / 'label.txt','r') as f:
        for line in f:
            label=line.strip().split('\t')[1]
            label_list.append(label)
    num_labels=len(label_list)
    #print(f'标签数量：{num_labels}')
    id2label={i:label for i,label in enumerate(label_list)}
    label2id={label:i for i,label in enumerate(label_list)}
    
    model=AutoModelForSequenceClassification.from_pretrained(
        str(config.PRETRAIN_MODEL_PATH),
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    # print(model.config.id2label)
    
    #评价函数
    def compute_metrics(y_preds:list,y_labels:list)->dict:
        acc=accuracy_score(y_labels,y_preds)
        f1=f1_score(y_labels,y_preds,average='weighted')
        return {'acc':acc,'f1':f1}
        
    
    trainconfig=TrainConfig()
    
    trainer=Trainer(model,trainconfig,compute_metrics)
    
    trainer.train() 
    
 

if __name__=='__main__':
    
    train()