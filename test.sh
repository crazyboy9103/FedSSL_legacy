exp="FL"

python main.py --exp $exp --freeze False --alpha 50000 --frac 0.05 --model resnet18 --server_data_frac 0.05 --epochs 100 --log_path ./logs/test/simclr --lr 0.001 | tee simclr/freeze.txt
