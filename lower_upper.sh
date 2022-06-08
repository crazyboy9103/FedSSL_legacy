exp="FL"
echo Lower bound $exp
python main_lower_bound.py --exp $exp --ckpt_path checkpoint_lower.pth.tar --model resnet18 --pretrained False --warmup False --log_path ./logs/$exp/lower --epochs 100 --num_class_per_client 5 --num_items 500 2>&1 | tee lower_bound.txt 

for alpha in "0.1" "0.5" "0.9"
do
    echo Upper bound $exp $alpha
    python main.py --exp $exp --ckpt_path checkpoint_upper_"$alpha".pth.tar --model resnet18 --alpha $alpha --pretrained False --warmup False --log_path ./logs/$exp/upper/$alpha --epochs 100 --num_class_per_client 5 --num_items 500 2>&1 | tee upper_bound_"$alpha".txt 
done    

