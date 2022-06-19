Exp: FL, simclr, simsiam
Alpha: 0.01, 1.0, 100000
out_dim: 128 512 2048
Freeze: True False  (Linear evaluation protocol by default is True)
server_data_frac: 0.01 0.5 0.9
Temperature: 0.1 0.3 0.5
num_items: 100, 200, 300
Epochs: 100 200?
local_ep: 10 50 100


log_path
finetune_epoch: 1, 5, 10
Finetune: True for simclr, simsiam
		 False for FL (by default)
 
finetune_before_agg: True False