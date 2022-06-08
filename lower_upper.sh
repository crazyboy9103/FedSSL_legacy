exp="FL"
for alpha in "0.1" "0.5" "0.9"
do
    python main_lower_bound.py --ckpt_path checkpoint_lower_"$alpha".pth.tar --model resnet18 --alpha $alpha --pretrained False --warmup False --log_path ./logs/$exp/lower/$alpha --epochs 100 --num_class_per_client 5 --num_items 500 2>&1 | tee lower_bound_"$alpha".txt 
    python main.py --ckpt_path checkpoint_upper_"$alpha".pth.tar --model resnet18 --alpha $alpha --pretrained False --warmup False --log_path ./logs/$exp/upper/$alpha --epochs 100 --num_class_per_client 5 --num_items 500 2>&1 | tee upper_bound_"$alpha".txt 
done    

