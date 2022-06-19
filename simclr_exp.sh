# SimCLR = UL FL and finetune at the server
epochs="50"

niid_alpha="0.1"
iid_alpha="100000"

num_items="300"
finetune="True"

strength="0.5"
out_dim="512"
temperature="0.5"

for finetune_before_agg in "True" "False"
do
    for finetune_epoch in "1" "5"
    do
        for freeze in "False" "True"
        do 
            for frac in "1"
            do 
                for model in "resnet18" "resnet50"
                do
                    for pretrained in "False" "True"
                    do
                        python main.py \
                            --server_data_frac "$frac" \
                            --exp simclr \
                            --model "$model" \
                            --pretrained "$pretrained" \
                            --epochs "$epochs" \
                            --num_items "$num_items" \
                            --freeze "$freeze" \
                            --finetune "$finetune" \
                            --alpha "$niid_alpha" \
                            --strength "$strength" \
                            --temperature "$temperature" \
                            --out_dim "$out_dim" \
                            --log_path ./logs/simclr/"$finetune_before_agg"_"$finetune_epoch"_"$freeze"_"$frac"_"$model"_"$pretrained"_"$niid_alpha"

                        python main.py \
                            --server_data_frac "$frac" \
                            --exp simclr \
                            --model "$model" \
                            --pretrained "$pretrained" \
                            --epochs "$epochs" \
                            --num_items "$num_items" \
                            --freeze "$freeze" \
                            --finetune "$finetune" \
                            --alpha "$iid_alpha" \
                            --strength "$strength" \
                            --temperature "$temperature" \
                            --out_dim "$out_dim" \
                            --log_path ./logs/simclr/"$finetune_before_agg"_"$finetune_epoch"_"$freeze"_"$frac"_"$model"_"$pretrained"_"$iid_alpha"
                    done
                done
            done
        done
    done
done


