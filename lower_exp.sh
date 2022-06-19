# Lower = centralized SL training with server data only
epochs="100"
freeze="False"
finetune_epoch="10"
num_items="300"

for frac in "0.01" "0.5" "0.9"
do 
    for model in "resnet18" "resnet50"
    do
        for pretrained in "False" "True"
        do
            for server_data_iid in "False" "True"
            do
                python main_lower_bound.py \
                    --server_data_frac "$frac" \
                    --exp FL \
                    --model "$model" \
                    --pretrained "$pretrained" \
                    --log_path ./logs/FL/lower/"$frac"_"$model"_"$pretrained"_"$server_data_iid" \
                    --epochs "$epochs" \
                    --num_items "$num_items" \
                    --finetune True \
                    --finetune_epoch "$finetune_epoch" \
                    --freeze "$freeze"\
                    --server_data_iid "$server_data_iid"
            done    
        done
    done
done



