# Upper = SL FL with no finetuning
epochs="50"
freeze="False"

niid_alpha="0.1"
iid_alpha="100000"

num_items="300"
finetune="False"
freeze="False"

frac="0.9"
model="resnet18"

for pretrained in "False" "True"
do
    python main.py \
        --exp FL \
        --server_data_frac "$frac" \
        --model "$model" \
        --pretrained "$pretrained" \
        --epochs "$epochs" \
        --num_items "$num_items" \
        --freeze "$freeze" \
        --finetune "$finetune" \
        --alpha "$niid_alpha" \
        --log_path ./logs/FL/upper/"$frac"_"$model"_"$pretrained"_"$niid_alpha"

    python main.py \
        --exp FL \
        --server_data_frac "$frac" \
        --model "$model" \
        --pretrained "$pretrained" \
        --epochs "$epochs" \
        --num_items "$num_items" \
        --freeze "$freeze" \
        --finetune "$finetune" \
        --alpha "$iid_alpha" \
        --log_path ./logs/FL/upper/"$frac"_"$model"_"$pretrained"_"$iid_alpha"
done



