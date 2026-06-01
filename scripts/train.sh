out_dir=cub_visible_all_vit16/lambda_2b/
epoch=1
bsz=32
model=vit16



dataset=pet_visible
aug_data_path=/home/user/Develop/datasets/oxford_pet/oxford-iiit-pet/train_aug_llmdet
lvlm_output=vis_labels/pets_visible/pets_auroc_03_prob_qwen_qwen3-vl-4b-instruct.json
lvlm_output_aug=vis_labels/pets_visible/pets_auroc_aug-llmdet01_prob_qwen_qwen3-vl-4b-instruct.json


lambda_inv=6.0
echo "=============================="
echo " Run ${run} ctr(visible/non-visible)"
echo "=============================="
python src/train_all.py \
--dataset ${dataset} \
--save_path checkpoints_acc4/${out_dir}/ours_lambda${lambda_inv}_ep${epoch}/run_${run} \
--lvlm_output ${lvlm_output} \
--aug_data_path ${aug_data_path} \
--lvlm_output_aug ${lvlm_output_aug} \
--use_concept \
--use_vis_label \
--rich_concept \
--batch_size ${bsz} \
--epochs ${epoch} \
--data_aug \
--warm_up \
--main_lr 2e-6 \
--proj_lr 5e-6 \
--tau 0.1 \
--class_ctr_lr 0 \
--ctr_neg_weight ${lambda_inv} \
--model_type ${model}
