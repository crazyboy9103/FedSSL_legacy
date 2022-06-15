exp="FL"
for frac in "0.1" "0.5" "0.9"
do
    echo Lower bound $exp Server frac $frac
    python main_lower_bound.py --server_data_frac $frac --exp $exp --ckpt_path checkpoint_lower_"$frac".pth.tar --model resnet18 --pretrained False --warmup False --log_path ./logs/$exp/lower/$frac --epochs 100 --num_items 500 2>&1 | tee lower_bound_"$frac".txt 
    echo Upper bound $exp Server frac $frac
    python main.py --server_data_frac $frac --exp $exp --ckpt_path checkpoint_upper_"$frac".pth.tar --model resnet18 --pretrained False --warmup False --log_path ./logs/$exp/upper/$frac --epochs 100 --num_items 500 2>&1 | tee upper_bound_"$frac".txt 
done    

