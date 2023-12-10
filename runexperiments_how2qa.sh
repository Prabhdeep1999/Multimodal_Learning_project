#!/bin/bash
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval --combine_datasets how2qa --combine_datasets_val how2qa --save_dir=ml/ft_how2qa_100 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth --blackout --blackout_percent=100 >ml/ft_how2qa_100/output.log 2>&1
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval --combine_datasets how2qa --combine_datasets_val how2qa --save_dir=ml/ft_how2qa_90 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth --blackout --blackout_percent=90 >ml/ft_how2qa_90/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval --combine_datasets how2qa --combine_datasets_val how2qa --save_dir=ml/ft_how2qa_80 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth --blackout --blackout_percent=80 >ml/ft_how2qa_80/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval --combine_datasets how2qa --combine_datasets_val how2qa --save_dir=ml/ft_how2qa_70 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth --blackout --blackout_percent=70 >ml/ft_how2qa_70/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval --combine_datasets how2qa --combine_datasets_val how2qa --save_dir=ml/ft_how2qa_60 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth --blackout --blackout_percent=60 >ml/ft_how2qa_60/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval --combine_datasets how2qa --combine_datasets_val how2qa --save_dir=ml/ft_how2qa_50 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth --blackout --blackout_percent=50 >ml/ft_how2qa_50/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval --combine_datasets how2qa --combine_datasets_val how2qa --save_dir=ml/ft_how2qa_40 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth --blackout --blackout_percent=40 >ml/ft_how2qa_40/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval --combine_datasets how2qa --combine_datasets_val how2qa --save_dir=ml/ft_how2qa_30 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth --blackout --blackout_percent=30 >ml/ft_how2qa_30/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval --combine_datasets how2qa --combine_datasets_val how2qa --save_dir=ml/ft_how2qa_20 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth --blackout --blackout_percent=20 >ml/ft_how2qa_20/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval --combine_datasets how2qa --combine_datasets_val how2qa --save_dir=ml/ft_how2qa_10 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth --blackout --blackout_percent=10 >ml/ft_how2qa_10/output.log 2>&1 &&
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval --combine_datasets how2qa --combine_datasets_val how2qa --save_dir=ml/ft_how2qa_0 --ds_factor_ff=8 --ds_factor_attn=8 --suffix="." --batch_size_val=32 --max_tokens=512 --load=models/frozenbilm_how2qa.pth --blackout --blackout_percent=0 >ml/ft_how2qa_0/output.log 2>&1