for NUM_EPOCHS in 1; do
    for DEBIAS_RATIO in 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.99; do
        for RUN_NAME in enc00; do
            for NUM_WIKI_WORDS in 100; do
                for NUM_STER_WORDS in 10 20 30 40 50 60 70 80 90 100 120 150; do
                    TRANSFORMERS_OFFLINE=1 \
                    python race_trainer_encoder.py \
                        --num_wiki_words ${NUM_WIKI_WORDS} \
                        --num_stereo_words ${NUM_STER_WORDS} \
                        --bias_type race \
                        --model_id bert-base-uncased \
                        --output_dir ./out/ \
                        --num_gpus 2 \
                        --batch_size 768 \
                        --proj guidebias-race \
                        --run_name ${RUN_NAME}_${DEBIAS_RATIO}_${NUM_WIKI_WORDS}_${NUM_STER_WORDS} \
                        --seed 0 \
                        --lr 2e-5 \
                        --num_epochs ${NUM_EPOCHS} \
                        --num_workers 8 \
                        --warmup_proportion 0.2 \
                        --debias_ratio ${DEBIAS_RATIO}
                done
            done
        done
    done
done