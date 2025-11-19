from datasets import load_from_disk
from configuration import config
from transformers import AutoTokenizer,DataCollatorWithPadding
from torch.utils.data import DataLoader


def get_dataloader(tokenizer,datatype='train'):
    
    assert datatype in ['train','test','validation'],f'Unknown datatype {datatype}'
    
    dataset=load_from_disk(str(config.PROCESSED_DATA/ datatype))
    
    #设置数据格式为PyTorch张量
    dataset.set_format(type='torch')
    
    collate_fn=DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True
    )
    
    return DataLoader(dataset,batch_size=config.BATCH_SIZE,shuffle=True,collate_fn=collate_fn)

if __name__=='__main__':
    tokenizer=AutoTokenizer.from_pretrained(str(config.PRETRAIN_MODEL_PATH))
    dataloader=get_dataloader(tokenizer,'train')
    for batch in dataloader:
        for key,value in batch.items():
            print(f'{key}: {value.shape}')
        break
    
    