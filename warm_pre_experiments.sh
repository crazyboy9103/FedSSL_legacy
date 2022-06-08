for exp in simclr simsiam FL
do 
    i=1
    for pretrained in False True
    do 
        for sup_warmup in False True 
        do
            python main.py --model resnet18 --pretrained $pretrained --exp $exp --out_dim 1024 --sup_warmup $sup_warmup --log_path ./logs/$exp/exp$i 2>&1 | tee "$exp"_exp"$i".txt 
            i=$((i+1))
        done    
    done 
done 
## Simclr
#python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup False --log_path ./logs/simclr/exp1 2>&1 | tee simclr_exp1.txt &&
#python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup True --log_path ./logs/simclr/exp2 2>&1 | tee simclr_exp2.txt && 
#python main.py --model resnet18 --pretrained True --exp simclr --out_dim 1024 --sup_warmup False --log_path ./logs/simclr/exp3 2>&1 | tee simclr_exp3.txt &&
#python main.py --model resnet18 --pretrained True --exp simclr --out_dim 1024 --sup_warmup True --log_path ./logs/simclr/exp4 2>&1 | tee simclr_exp4.txt &&
## Simsiam
#python main.py --model resnet18 --pretrained False --exp simsiam --out_dim 1024 --sup_warmup False --log_path ./logs/simsiam/exp1 2>&1 | tee simsiam_exp1.txt &&
#python main.py --model resnet18 --pretrained False --exp simsiam --out_dim 1024 --sup_warmup True --log_path ./logs/simsiam/exp2 2>&1 | tee simsiam_exp2.txt && 
#python main.py --model resnet18 --pretrained True --exp simsiam --out_dim 1024 --sup_warmup False --log_path ./logs/simsiam/exp3 2>&1 | tee simsiam_exp3.txt &&
#python main.py --model resnet18 --pretrained True --exp simsiam --out_dim 1024 --sup_warmup True --log_path ./logs/simsiam/exp4 2>&1 | tee simsiam_exp4.txt &&
## FL
#python main.py --model resnet18 --pretrained False --exp FL --out_dim 1024 --sup_warmup False --log_path ./logs/FL/exp1 2>&1 | tee fl_exp1.txt &&
#python main.py --model resnet18 --pretrained False --exp FL --out_dim 1024 --sup_warmup True --log_path ./logs/FL/exp2 2>&1 | tee fl_exp2.txt && 
#python main.py --model resnet18 --pretrained True --exp FL --out_dim 1024 --sup_warmup False --log_path ./logs/FL/exp3 2>&1 | tee fl_exp3.txt &&
#python main.py --model resnet18 --pretrained True --exp FL --out_dim 1024 --sup_warmup True --log_path ./logs/FL/exp4 2>&1 | tee fl_exp4.txt