# LOWER BOUND = FINETUNE ONLY
# UPPER BOUND = SL FL

IID_alpha="50000"
nIID_alpha="0.1"
frac="0.01"
num_items="300"
model="resnet18"
pretrained="False"
warmup="False"
epochs="100"

for strength in "0.1" "0.5" "0.9"
do 
    for finetune_epoch in "1" "5" "10"
    do
        for freeze in True False
        do 
                    # LOWER BOUND
            echo Lower bound
            python main_lower_bound.py --freeze "$freeze" --server_data_frac $frac --exp FL --model "$model" --pretrained "$pretrained" --strength "$strength" --finetune_epoch "$finetune_epoch" --warmup "$warmup" --log_path ./logs/FL/hyper/lower/"$frac"_"$strength"_"$finetune_epoch"_"$freeze" --epochs "$epochs" --num_items "$num_items" 2>&1 | tee lower/hyper_lower_bound_"$frac"_"$strength"_"$finetune_epoch"_"$freeze".txt 

            # IID UPPER BOUND 
            echo Upper bound IID
            python main.py --freeze "$freeze" --server_data_frac $frac --exp FL --model "$model" --pretrained "$pretrained" --strength "$strength" --finetune_epoch "$finetune_epoch" --warmup "$warmup" --alpha "$IID_alpha" --log_path ./logs/FL/hyper/upper/"$frac"_iid_"$strength"_"$finetune_epoch"_"$freeze" --epochs "$epochs" --num_items "$num_items" 2>&1 | tee upper/hyper_upper_bound_"$frac"_iid_"$strength"_"$finetune_epoch"_"$freeze".txt 

            # non IID UPPER BOUND
            echo Upper bound non IID
            python main.py --freeze "$freeze" --server_data_frac $frac --exp FL --model "$model" --pretrained "$pretrained" --strength "$strength" --finetune_epoch "$finetune_epoch" --warmup "$warmup" --alpha "$nIID_alpha" --log_path ./logs/FL/hyper/upper/"$frac"_noniid_"$strength"_"$finetune_epoch"_"$freeze" --epochs "$epochs" --num_items "$num_items" 2>&1 | tee upper/hyper_upper_bound_"$frac"_noniid_"$strength"_"$finetune_epoch"_"$freeze".txt 

            # IID simclr 
            echo SimCLR IID
            python main.py --freeze "$freeze" --server_data_frac $frac --exp simclr --model "$model" --pretrained "$pretrained" --strength "$strength" --finetune_epoch "$finetune_epoch" --warmup "$warmup" --alpha "$IID_alpha" --log_path ./logs/simclr/hyper/"$frac"_iid_"$strength"_"$finetune_epoch"_"$freeze" --epochs "$epochs" --num_items "$num_items" 2>&1 | tee simclr/hyper_simclr_"$frac"_iid_"$strength"_"$finetune_epoch"_"$freeze".txt 
            # non IID simclr
            echo SimCLR non IID
            python main.py --freeze "$freeze" --server_data_frac $frac --exp simclr --model "$model" --pretrained "$pretrained" --strength "$strength" --finetune_epoch "$finetune_epoch" --warmup "$warmup" --alpha "$nIID_alpha" --log_path ./logs/simclr/hyper/"$frac"_noniid_"$strength"_"$finetune_epoch"_"$freeze" --epochs "$epochs" --num_items "$num_items" 2>&1 | tee simclr/hyper_simclr_"$frac"_noniid_"$strength"_"$finetune_epoch"_"$freeze".txt 

            # IID simsiam
            echo Simsiam IID
            python main.py --freeze "$freeze" --server_data_frac $frac --exp simsiam --model "$model" --pretrained "$pretrained" --strength "$strength" --finetune_epoch "$finetune_epoch" --warmup "$warmup" --alpha "$IID_alpha" --log_path ./logs/simsiam/hyper/"$frac"_iid_"$strength"_"$finetune_epoch"_"$freeze" --epochs "$epochs" --num_items "$num_items" 2>&1 | tee simsiam/hyper_simsiam_"$frac"_iid_"$strength"_"$finetune_epoch"_"$freeze".txt 

            # non IID simsiam 
            echo Simsiam non IID
            python main.py --freeze "$freeze" --server_data_frac $frac --exp simsiam --model "$model" --pretrained "$pretrained" --strength "$strength" --finetune_epoch "$finetune_epoch" --warmup "$warmup" --alpha "$nIID_alpha" --log_path ./logs/simsiam/hyper/"$frac"_noniid_"$strength"_"$finetune_epoch"_"$freeze" --epochs "$epochs" --num_items "$num_items" 2>&1 | tee simsiam/hyper_simsiam_"$frac"_noniid_"$strength"_"$finetune_epoch"_"$freeze".txt 
# done
        done 
    done
    
done

# for frac in "0.01" "0.3" "0.9"
# do
