# Lower = centralized SL training with server data only
epochs="50"
freeze="False"
finetune_epoch="10"
num_items="300"

for frac in "0.01" "0.5" "0.9"
do 
    for model in "resnet18"
    do
        for pretrained in "False" "True"
        do
            for server_data_iid in "False"
            do
                python main_lower_bound.py \
                    --server_data_frac "$frac" \
                    --exp FL \
                    --model "$model" \
                    --pretrained "$pretrained" \
                    --log_path ./logs/FL/lower/fr_"$frac"_"$model"_pre_"$pretrained"_serveriid_"$server_data_iid" \
                    --epochs "$epochs" \
                    --num_items "$num_items" \
                    --finetune True \
                    --finetune_epoch "$finetune_epoch" \
                    --freeze "$freeze"\
                    --server_data_iid "$server_data_iid"\
                    --num_users 0
            done    
        done
    done
done



