exp="simclr"

python main.py --exp $exp --freeze False --frac 0.02 --model resnet18 --server_data_frac 0.01 --epochs 100 --log_path ./logs/test/simsiam --lr 0.001 | tee simsiam/FL.txt
