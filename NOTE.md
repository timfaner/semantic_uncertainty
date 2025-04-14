

### 常用命令


单独做语义熵计算

python3 semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=v2jgns96 --assign_new_wandb_id --entailment_model=gpt-4o-mini --restore_entity_eval=""  --no-compute_p_ik  --no-compute_p_ik_answerable


### debug 常用配置

- 简单sample


### 对比LLama 和 gpt4o 在  bioasq上的 auroc
python3 semantic_uncertainty/generate_answers.py --model_name=Llama-2-7b-chat --dataset=squad --no-compute_p_ik  --no-compute_p_ik_answerable  --no-compute_p_true


python3 semantic_uncertainty/generate_answers.py --model_name=gpt-4o-mini --dataset=squad --no-compute_p_ik  --no-compute_p_ik_answerable  --no-compute_p_true


// 分别对应
python3 semantic_uncertainty/compute_uncertainty_measures.py --eval_wandb_runid=替换成上两步的id --assign_new_wandb_id --entailment_model=gpt-4o-mini --restore_entity_eval=""  --no-compute_p_ik  --no-compute_p_ik_answerable



