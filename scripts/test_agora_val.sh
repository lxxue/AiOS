CHECKPOINT=$1
OUTPUT_DIR=$2
python -m torch.distributed.launch \
    --nproc_per_node 4 \
    main.py \
    --output_dir "/mnt/AFS_sunqingping/log/aios_log/inference" \
    -c "config/aios_smplx_agora_val.py" \
    --options batch_size=4 epochs=100 lr_drop=55 num_body_points=17 backbone="resnet50" \
    --resume ${CHECKPOINT} \
    --eval \
    --inference \
    --inference_input data/datasets/agora/3840x2160/validation \
    --output_dir test_result/${OUTPUT_DIR}
