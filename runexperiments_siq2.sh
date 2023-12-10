#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ml/ft_siq2_100 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=ftsiq2/best_model.pth --blackout --blackout_percent=100 >ml/ft_siq2_100/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ml/ft_siq2_90 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=ftsiq2/best_model.pth --blackout --blackout_percent=90 >ml/ft_siq2_90/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ml/ft_siq2_80 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=ftsiq2/best_model.pth --blackout --blackout_percent=80 >ml/ft_siq2_80/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ml/ft_siq2_70 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=ftsiq2/best_model.pth --blackout --blackout_percent=70 >ml/ft_siq2_70/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ml/ft_siq2_60 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=ftsiq2/best_model.pth --blackout --blackout_percent=60 >ml/ft_siq2_60/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ml/ft_siq2_50 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=ftsiq2/best_model.pth --blackout --blackout_percent=50 >ml/ft_siq2_50/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ml/ft_siq2_40 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=ftsiq2/best_model.pth --blackout --blackout_percent=40 >ml/ft_siq2_40/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ml/ft_siq2_30 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=ftsiq2/best_model.pth --blackout --blackout_percent=30 >ml/ft_siq2_30/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ml/ft_siq2_20 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=ftsiq2/best_model.pth --blackout --blackout_percent=20 >ml/ft_siq2_20/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ml/ft_siq2_10 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=ftsiq2/best_model.pth --blackout --blackout_percent=10 >ml/ft_siq2_10/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc_siq2.py --eval --combine_datasets siq2 --combine_datasets_val siq2 --save_dir=ml/ft_siq2_0 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=ftsiq2/best_model.pth --blackout --blackout_percent=0 >ml/ft_siq2_0/output.log 2>&1
