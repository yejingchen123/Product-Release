from argparse import ArgumentParser


if __name__=='__main__':
    #定义命令行参数解析器
    parser = ArgumentParser(usage="python main.py --mode [train/evaluate/predict/web]")
    parser.add_argument('--mode', type=str, required=True, help='Mode: train, evaluate, predict, web')
    args = parser.parse_args()

    if args.mode == 'train':
        from runer.train import train
        train()
    elif args.mode == 'evaluate':
        from runer.evaluate import evaluate
        evaluate()
    elif args.mode == 'predict':
        from runer.predict import predict
        
        predict()
    elif args.mode == 'web':
        from web.app import web_run
        web_run()
    else:
        print("Invalid mode. Please choose from: train, evaluate, predict, web.")