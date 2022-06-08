for exp in simclr simsiam FL
do 
    i=5
    for sup_warmup in False True
    do 
        for num_items in 100 300 500
        do
            python main.py --model resnet18 --pretrained False --exp $exp --out_dim 1024 --sup_warmup $sup_warmup --log_path ./logs/$exp/exp$i --epochs 50 --num_class_per_client 2 --num_items $num_items 2>&1 | tee "$exp"_exp"$i".txt 
            i=$((i+1))
        done    
    done 
done 

# # Simclr
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup False --num_items 100 --num_class_per_client 2 --log_path ./logs/simclr/exp5 --epochs 50 2>&1 | tee simclr_exp5.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup False --num_items 300 --num_class_per_client 2 --log_path ./logs/simclr/exp6 --epochs 50 2>&1 | tee simclr_exp6.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup False --num_items 500 --num_class_per_client 2 --log_path ./logs/simclr/exp7 --epochs 50 2>&1 | tee simclr_exp7.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup True --num_items 100 --num_class_per_client 2 --log_path ./logs/simclr/exp8 --epochs 50 2>&1 | tee simclr_exp8.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup True --num_items 300 --num_class_per_client 2 --log_path ./logs/simclr/exp9 --epochs 50 2>&1 | tee simclr_exp9.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup True --num_items 500 --num_class_per_client 2 --log_path ./logs/simclr/exp10 --epochs 50 2>&1 | tee simclr_exp10.txt &&
# # Simsiam
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup False --num_items 100 --num_class_per_client 2 --log_path ./logs/simclr/exp5 --epochs 50 2>&1 | tee simclr_exp5.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup False --num_items 300 --num_class_per_client 2 --log_path ./logs/simclr/exp6 --epochs 50 2>&1 | tee simclr_exp6.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup False --num_items 500 --num_class_per_client 2 --log_path ./logs/simclr/exp7 --epochs 50 2>&1 | tee simclr_exp7.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup True --num_items 100 --num_class_per_client 2 --log_path ./logs/simclr/exp8 --epochs 50 2>&1 | tee simclr_exp8.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup True --num_items 300 --num_class_per_client 2 --log_path ./logs/simclr/exp9 --epochs 50 2>&1 | tee simclr_exp9.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup True --num_items 500 --num_class_per_client 2 --log_path ./logs/simclr/exp10 --epochs 50 2>&1 | tee simclr_exp10.txt &&
# # FL
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup False --num_items 100 --num_class_per_client 2 --log_path ./logs/simclr/exp5 --epochs 50 2>&1 | tee simclr_exp5.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup False --num_items 300 --num_class_per_client 2 --log_path ./logs/simclr/exp6 --epochs 50 2>&1 | tee simclr_exp6.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup False --num_items 500 --num_class_per_client 2 --log_path ./logs/simclr/exp7 --epochs 50 2>&1 | tee simclr_exp7.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup True --num_items 100 --num_class_per_client 2 --log_path ./logs/simclr/exp8 --epochs 50 2>&1 | tee simclr_exp8.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup True --num_items 300 --num_class_per_client 2 --log_path ./logs/simclr/exp9 --epochs 50 2>&1 | tee simclr_exp9.txt &&
# python main.py --model resnet18 --pretrained False --exp simclr --out_dim 1024 --sup_warmup True --num_items 500 --num_class_per_client 2 --log_path ./logs/simclr/exp10 --epochs 50 2>&1 | tee simclr_exp10.txt &&