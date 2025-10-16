import argparse
import subprocess

def main():
    parser = argparse.ArgumentParser(description="LeNet Project Main Script")
    parser.add_argument('--train', action='store_true', help='Run train.py to train the model')
    parser.add_argument('--test', action='store_true', help='Run test.py to test the model')
    parser.add_argument('--fgsmatk', action='store_true', help='Run fgsm_atk.py to perform FGSM attack')
    parser.add_argument('--fgsmtrain', action='store_true', help='Run fgsm_train.py to train with FGSM adversarial examples')
    parser.add_argument('--fgsmatkaftertrain', action='store_true', help='Run fgsm_atk_after_train.py to perform FGSM attack on FGSM-trained model')
    # 可继续添加更多参数

    args = parser.parse_args()

    if args.train:
        print("Running train.py ...")
        subprocess.run(['python', 'train.py'])
    elif args.test:
        print("Running test.py ...")
        subprocess.run(['python', 'test.py'])
    elif args.fgsmatk:
        print("Running fgsm_atk.py ...")
        subprocess.run(['python', 'fgsm_atk.py'])
    elif args.fgsmtrain:
        print("Running fgsm_train.py ...")
        subprocess.run(['python', 'fgsm_train.py'])
    elif args.fgsmatkaftertrain:
        print("Running fgsm_atk_after_train.py ...")
        subprocess.run(['python', 'fgsm_atk_after_train.py'])
    else:
        parser.print_help()

if __name__ == '__main__':
    main()