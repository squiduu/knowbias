for NUM_EPOCHS in 1; do
    for DEBIAS_RATIO in 0.0; do
        for NUM_WIKI_WORDS in 10 100 500 1000; do
            for NUM_STER_WORDS in 10 20 30 40 50 60 70 80 90 100 120 150 200; do
                TRANSFORMERS_OFFLINE=1 \
                python religion_trainer_encoder.py \
                    --num_wiki_words $NUM_WIKI_WORDS \
                    --num_stereo_words $NUM_STER_WORDS \
                    --bias_type religion \
                    --model_id bert-base-uncased \
                    --output_dir ./out/ \
                    --num_gpus 2 \
                    --batch_size 1024 \
                    --proj guidebias-religion \
                    --run_name ep$NUM_EPOCHS-dr$DEBIAS_RATIO-nww$NUM_WIKI_WORDS-nsw$NUM_STER_WORDS-ul \
                    --seed 0 \
                    --lr 2e-5 \
                    --weight_decay 0.01 \
                    --num_epochs $NUM_EPOCHS \
                    --num_workers 8 \
                    --warmup_proportion 0.2 \
                    --debias_ratio $DEBIAS_RATIO
            done
        done
    done
done