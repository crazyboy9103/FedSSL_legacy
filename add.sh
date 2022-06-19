# LOWER BOUND = FINETUNE ONLY
# UPPER BOUND = SL FL

IID_alpha="50000"
nIID_alpha="0.1"
frac="0.9"
num_items="300"
model="resnet18"
pretrained="False"
warmup="False"
epochs="100"

# IID simsiam
echo Simsiam IID
python main.py --server_data_frac $frac --exp simsiam --model "$model" --pretrained "$pretrained" --warmup "$warmup" --alpha "$IID_alpha" --log_path ./logs/simsiam/"$frac"_iid --epochs "$epochs" --num_items "$num_items" 2>&1 | tee simsiam/simsiam_"$frac"_iid.txt 

# non IID simsiam 
echo Simsiam non IID
python main.py --server_data_frac "0.9" --exp simsiam --model "$model" --pretrained "$pretrained" --warmup "$warmup" --alpha "$nIID_alpha" --log_path ./logs/simsiam/0.9_noniid --epochs "$epochs" --num_items "$num_items" 2>&1 | tee simsiam/simsiam_0.9_noniid.txt 

echo Simsiam non IID
python main.py --server_data_frac "0.3" --exp simsiam --model "$model" --pretrained "$pretrained" --warmup "$warmup" --alpha "$nIID_alpha" --log_path ./logs/simsiam/0.3_noniid --epochs "$epochs" --num_items "$num_items" 2>&1 | tee simsiam/simsiam_0.3_noniid.txt 