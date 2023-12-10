## Inference
# Cache
Define TRANSFORMERS_CACHE:
```
export TRANSFORMERS_CACHE=/home/admin-guest/Documents/multimodal-ml/FrozenBiLM/tranformers_cache
```
# ZS
From zero-shot, using frozenbilm.pth:
```
python -m torch.distributed.launch --nproc_per_node 2 --use_env mc.py --eval \ 
--combine_datasets siq2 --combine_datasets_val siq2 --save_dir=zssiq2 \ 
--ds_factor_ff=8 --ds_factor_attn=8 --suffix="." \
--batch_size_val=32 --max_tokens=512 --load=models/frozenbilm.pth
```
# FT
From Fine-tuned:
```
python main_siq2_roberta.py \
--num_thread_reader=0 \
--checkpoint_dir=ckpt/ckpt_ft2_how2qa.pth \
--dataset=siq2 \
--skip_transcript_prob=0 \
--pretrain_path=pretrained_models/roberta/best_model.pth \
--zeroshot_eval=1 \
--siq2_questions_features_path=datasets/Social-IQ-2.0/siq2/questions_features.pth \
--siq2_attended_questions_features_path=datasets/Social-IQ-2.0/siq2/attended_questions_features.pth \
--siq2_answers_features_path=datasets/Social-IQ-2.0/siq2/answers_features.pth \
--siq2_transcripts_features_path=datasets/Social-IQ-2.0/siq2/transcript_sentences_features_roberta.pth
```

