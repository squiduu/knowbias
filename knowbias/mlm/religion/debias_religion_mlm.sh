for NUM_EPOCHS in 1; do
    for DEBIAS_RATIO in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99; do
        for NUM_WIKI_WORDS in 10 100 500 1000; do
            TRANSFORMERS_OFFLINE=1 \
            CUDA_VISIBLE_DEVICES="0,1" \
            python religion_trainer_mlm.py \
                --num_wiki_words $NUM_WIKI_WORDS \
                --bias_type religion \
                --model_id bert-base-uncased \
                --student_encoder_id "../../encoder/religion/selected_out/religion_dr0.5_nww1000_nsw40" \
                --output_dir ./out/ \
                --num_gpus 2 \
                --batch_size 1024 \
                --project knowbias-religion \
                --run_name dr$DEBIAS_RATIO-nww$NUM_WIKI_WORDS-unfreezed \
                --seed 42 \
                --lr 2e-5 \
                --weight_decay 0.01 \
                --num_epochs 1 \
                --num_workers 8 \
                --warmup_proportion 0.2 \
                --precision 16 \
                --debias_ratio $DEBIAS_RATIO \
                --freeze_student_encoder False
        done
    done
done