for NUM_EPOCHS in 1; do
    for SEED in 0; do
        for RUN_NAME in div; do
            for NUM_WIKI_WORDS in 100; do
                for BATCH_SIZE in 768; do
                    TRANSFORMERS_OFFLINE=1 \
                    python race_trainer_mlm.py \
                        --num_wiki_words ${NUM_WIKI_WORDS} \
                        --bias_type c \
                        --model_id bert-base-uncased \
                        --student_encoder_id ../../debias_encoder/race/out/race_enc00_0.5_100_20 \
                        --output_dir ./out/ \
                        --num_gpus 2 \
                        --batch_size ${BATCH_SIZE} \
                        --project guidebias-c \
                        --run_name ${RUN_NAME} \
                        --seed ${SEED} \
                        --lr 2e-5 \
                        --weight_decay 0.01 \
                        --num_epochs ${NUM_EPOCHS} \
                        --num_workers 8 \
                        --warmup_proportion 0.2 \
                        --precision 16
                done
            done
        done
    done
done