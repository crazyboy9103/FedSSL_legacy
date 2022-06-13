exp="simclr"
frac="0.5"

python main.py --exp $exp --freeze True --server_data_frac 0.5 --epochs 100 --log_path ./logs/test/freeze | tee "test"/"freeze"/"$exp"_"$frac".txt