for NUM_EPOCHS in 2; do
    for DEBIAS_RATIO in 0.5; do
        for RUN_NAME in run04; do
            for NUM_WIKI_WORDS in 100; do
                for NUM_STER_WORDS in 10; do
                    python gender_trainer.py \
                        --num_targ_words 60 \
                        --num_wiki_words ${NUM_WIKI_WORDS} \
                        --num_ster_words ${NUM_STER_WORDS} \
                        --bias_type gender \
                        --model_id bert-base-uncased \
                        --model_name bert \
                        --out_dir ./out/ \
                        --num_gpus 2 \
                        --batch_size 1024 \
                        --proj guidebias \
                        --run_name ${RUN_NAME} \
                        --seed 0 \
                        --lr 2e-5 \
                        --num_epochs ${NUM_EPOCHS} \
                        --num_workers 8 \
                        --grad_accum_steps 1 \
                        --warmup_proportion 0.2 \
                        --debias_ratio ${DEBIAS_RATIO}
                done
            done
        done
    done
done