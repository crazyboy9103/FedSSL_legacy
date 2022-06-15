# LOWER BOUND = FINETUNE ONLY
# UPPER BOUND = SL FL

IID_alpha="50000"
nIID_alpha="0.1"
frac="0.01"
num_items="400"
model="resnet18"
pretrained="False"
warmup="False"
epochs="100"
# for frac in "0.01" "0.3" "0.9"
# do
# IID LOWER BOUND
echo Lower bound IID 
python main_lower_bound.py --server_data_frac $frac --exp FL --model "$model" --pretrained "$pretrained" --warmup "$warmup" --alpha "$IID_alpha" --log_path ./logs/FL/lower/"$frac"_iid --epochs "$epochs" --num_items "$num_items" 2>&1 | tee lower/lower_bound_"$frac"_iid.txt 

# non IID LOWER BOUND
echo Lower bound non IID 
python main_lower_bound.py --server_data_frac $frac --exp FL --model "$model" --pretrained "$pretrained" --warmup "$warmup" --alpha "$nIID_alpha" --log_path ./logs/FL/lower/"$frac"_noniid --epochs "$epochs" --num_items "$num_items" 2>&1 | tee lower/lower_bound_"$frac"_noniid.txt 

# IID UPPER BOUND 
echo Upper bound IID
python main.py --server_data_frac $frac --exp FL --model "$model" --pretrained "$pretrained" --warmup "$warmup" --alpha "$IID_alpha" --log_path ./logs/FL/upper/"$frac"_iid --epochs "$epochs" --num_items "$num_items" 2>&1 | tee upper/upper_bound_"$frac"_iid.txt 

# non IID UPPER BOUND
echo Upper bound non IID
python main.py --server_data_frac $frac --exp FL --model "$model" --pretrained "$pretrained" --warmup "$warmup" --alpha "$nIID_alpha" --log_path ./logs/FL/upper/"$frac"_noniid --epochs "$epochs" --num_items "$num_items" 2>&1 | tee upper/upper_bound_"$frac"_noniid.txt 

# IID simclr 
echo SimCLR IID
python main.py --server_data_frac $frac --exp simclr --model "$model" --pretrained "$pretrained" --warmup "$warmup" --alpha "$IID_alpha" --log_path ./logs/simclr/"$frac"_iid --epochs "$epochs" --num_items "$num_items" 2>&1 | tee simclr/simclr_"$frac"_iid.txt 
# non IID simclr
echo SimCLR non IID
python main.py --server_data_frac $frac --exp simclr --model "$model" --pretrained "$pretrained" --warmup "$warmup" --alpha "$nIID_alpha" --log_path ./logs/simclr/"$frac"_noniid --epochs "$epochs" --num_items "$num_items" 2>&1 | tee simclr/simclr_"$frac"_noniid.txt 

# IID simsiam
echo Simsiam IID
python main.py --server_data_frac $frac --exp simsiam --model "$model" --pretrained "$pretrained" --warmup "$warmup" --alpha "$IID_alpha" --log_path ./logs/simsiam/"$frac"_iid --epochs "$epochs" --num_items "$num_items" 2>&1 | tee simsiam/simsiam_"$frac"_iid.txt 

# non IID simsiam 
echo Simsiam non IID
python main.py --server_data_frac $frac --exp simsiam --model "$model" --pretrained "$pretrained" --warmup "$warmup" --alpha "$nIID_alpha" --log_path ./logs/simsiam/"$frac"_noniid --epochs "$epochs" --num_items "$num_items" 2>&1 | tee simsiam/simsiam_"$frac"_noniid.txt 
# done