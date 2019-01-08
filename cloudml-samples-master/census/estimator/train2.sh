gcloud ml-engine local train \
    --module-name trainer.task \
    --package-path trainer/ \
    --job-dir outputdir2 \
    --distributed \
    -- \
    --train-files $TRAIN_DATA \
    --eval-files $EVAL_DATA \
    --train-steps 1000 \
    --eval-steps 100
