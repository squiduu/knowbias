for NUM_EPOCHS in 1 10; do
    for SEED in 0; do
        for RUN_NAME in div; do
            for NUM_WIKI_WORDS in 100 5000; do
                for NUM_TARGET_WORDS in 60 318; do
                    for BATCH_SIZE in 32 1024; do
                        for LR in 1e-5 5e-5; do
                            python gender_trainer_mlm_div.py \
                                --num_target_words ${NUM_TARGET_WORDS} \
                                --num_wiki_words ${NUM_WIKI_WORDS} \
                                --bias_type gd \
                                --model_id bert-base-uncased \
                                --student_encoder_id squiduu/guidebias-bert-base-uncased \
                                --model_name bert \
                                --output_dir ./out/ \
                                --num_gpus 2 \
                                --batch_size ${BATCH_SIZE} \
                                --project guidebias \
                                --run_name ${RUN_NAME} \
                                --seed ${SEED} \
                                --lr ${LR} \
                                --weight_decay 0.01 \
                                --num_epochs ${NUM_EPOCHS} \
                                --num_workers 8 \
                                --warmup_proportion 0.2 \
                                --precision 32
                        done
                    done
                done
            done
        done
    done
done