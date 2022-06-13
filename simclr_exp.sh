exp="simclr"
for frac in "0.1" "0.5" "0.9"
do
    for bs in 32 128 512
    do
        python main.py --server_data_frac $frac --exp $exp --model resnet18 --pretrained False --warmup False --log_path ./logs/$exp/$frac/$bs --local_bs $bs --epochs 200 --num_class_per_client 5 --num_items 500 2>&1 | tee "$exp"/"$exp"_"$frac"_"$bs".txt 
    done
done