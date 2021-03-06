
set -ex
export OMP_NUM_THREADS=1
LR=2e-5
EPOCH=1000000

NEG_WORD=" irrelevant"
POS_WORD=" relevant"
Q=$1
NEG=1
if [ $Q == 'full' ];then
    MAX_STEPS=50000
    devq=500
    LOG_STEP=500
    EVAL_EVERY=5000
    BATCH_SIZE=8
    metric='mrr_cut_10'
fi
if [ $Q == 1000 ];then
    MAX_STEPS=3000
    devq=500
    LOG_STEP=100
    EVAL_EVERY=300
    BATCH_SIZE=8
    metric='mrr_cut_10'
fi 
if [ $Q == 50 ];then
    MAX_STEPS=3000
    devq=50
    LOG_STEP=100
    EVAL_EVERY=300
    BATCH_SIZE=2
    metric='mrr_cut_100'
fi
if [ $Q == 5 ];then
    MAX_STEPS=3000
    devq=5
    LOG_STEP=100
    EVAL_EVERY=300
    BATCH_SIZE=2
    metric='mrr_cut_100'
fi
model="roberta-base-ft"
ckpt="/data/home/scv0540/run/pretrained_models/roberta-base"
dir_path="/data/home/scv0540/run/promptir"

python -m torch.distributed.launch \
--nproc_per_node=4 \
--master_port=3119 \
train.py \
-task classification \
-model roberta \
-qrels $dir_path/collections/msmarco-passage/qrels.train.tsv     \
-train $dir_path/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
-dev $dir_path/dataset/msmarco/dev/$devq-q.jsonl  \
-test $dir_path/dataset/msmarco/test/all-q.jsonl  \
-max_input 80000000 \
-vocab $ckpt  \
-pretrain $ckpt  \
-metric $metric  \
-max_query_len 50  \
-max_doc_len 400 \
-epoch $EPOCH  \
-batch_size $BATCH_SIZE  \
-lr $LR  \
-eval_every $EVAL_EVERY  \
-optimizer adamw   \
-dev_eval_batch_size 200  \
-n_warmup_steps 0  \
-logging_step $LOG_STEP  \
-save $dir_path/checkpoints/$model/q$Q-n-$NEG/  \
-res $dir_path/results/$model/q$Q-n-$NEG.trec  \
-test_res $dir_path/results/$model/test_q$Q-n-$NEG.trec  \
--log_dir=$dir_path/logs/$model/q$Q-n-$NEG/  \
--max_steps=$MAX_STEPS  \
--pos_word=" relevant"  \
--neg_word=" irrelevant"  \
--template='<q> <mask> <d>'  \
-gradient_accumulation_steps 1  \
--prefix='[133, 986, 3645, 16, 10, 25860, 4, 48360, 16, 1437]'   \
--infix='[11167,10]'    \
--suffix='[560, 22053, 257, 991, 3999, 4, 20, 220, 3645, 16, 10, 3780, 4]'   \
--soft_sentence=""  \
