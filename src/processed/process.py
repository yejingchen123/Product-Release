from configuration import config
from datasets import load_dataset,ClassLabel
from transformers import AutoTokenizer


def process():
    
    #加载数据集
    dataset_dic=load_dataset(
        'csv',
        data_files={
            'train':str(config.RAW_DATA /'train.txt'),
            'test':str(config.RAW_DATA /'test.txt'),
            'validation':str(config.RAW_DATA /'valid.txt'),
        },
        delimiter='\t'
    )
    #print(dataset_dic)
    
    #将label映射为数字
    label_list=list(set(dataset_dic['train']['label']))
    id_to_label={i:label for i,label in enumerate(label_list)}
    
    #将id_to_label保存为label.txt
    with open(config.PROCESSED_DATA / 'label.txt', 'w') as f:
        for i, label in id_to_label.items():
            f.write(f"{i}\t{label}\n")
    
    #print(f'标签列表：{label_list}')
    dataset_dic=dataset_dic.cast_column('label',ClassLabel(names=label_list))

    #加载分词器
    tokenizer=AutoTokenizer.from_pretrained(str(config.PRETRAIN_MODEL_PATH))
    
    #定义分词函数
    def tokenize_function(example):
        intput=tokenizer(
            example['text_a'],
            truncation=True,
        )
        intput['labels']=example['label']
        return intput
    
    #对数据集进行分词
    dataset_dic=dataset_dic.map(tokenize_function,batched=True,remove_columns=['text_a','label'])
    
    #print(dataset_dic['train'][:2])
    
    #保存处理后的数据集
    dataset_dic.save_to_disk(str(config.PROCESSED_DATA))    







    
if __name__=='__main__':
    process()