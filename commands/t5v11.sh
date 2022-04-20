
set -ex
export OMP_NUM_THREADS=1
LR=2e-5
EPOCH=300000
Q=$1
if [ $Q == 'full' ];then
    MAX_STEPS=50000
    devq=500
    LOG_STEP=500
    EVAL_EVERY=5000
    BATCH_SIZE=8
    metric='mrr_cut_100'
fi
if [ $Q == 1000 ];then
    MAX_STEPS=3000
    devq=500
    LOG_STEP=100
    EVAL_EVERY=300
    BATCH_SIZE=8
    metric='mrr_cut_100'
fi 
if [ $Q == 50 ];then
    MAX_STEPS=1
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
NEG=1
ckpt="/data/home/scv0540/run/pretrained_models/t5-v11-base"
seed=13
dir_prefix="/data/home/scv0540/run/promptir"
model="t5-v11-base-manual-prompt"
python -m torch.distributed.launch \
         --nproc_per_node=4 \
         --master_port=2517  \
        train.py \
        -task classification  \
        -model t5  \
        -seed $seed    \
        -qrels $dir_prefix/collections/msmarco-passage/qrels.train.tsv     \
        -train $dir_prefix/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
        -dev $dir_prefix/dataset/msmarco/dev/$devq-q.jsonl  \
        -test $dir_prefix/dataset/msmarco/test/all-q.jsonl  \
        -max_input 80000000  \
        -vocab $ckpt   \
        -pretrain $ckpt  \
        -save $dir_prefix/checkpoints/$model/q$Q-n-$NEG/  \
        -res $dir_prefix/results/$model/q$Q-n-$NEG.trec  \
        -test_res $dir_prefix/results/$model/test_q$Q-n-$NEG.trec  \
        --log_dir=$dir_prefix/logs/$model/q$Q-n-$NEG/  \
        -metric $metric  \
        -max_query_len 76  \
        -max_doc_len 290  \
        -epoch $EPOCH  \
        -batch_size $BATCH_SIZE  \
        -lr $LR  \
        -eval_every $EVAL_EVERY  \
        -optimizer adamw  \
        -dev_eval_batch_size  128   \
        -n_warmup_steps 0  \
        -logging_step $LOG_STEP  \
        --max_steps=$MAX_STEPS \
        -gradient_accumulation_steps 1\
        --soft_sentence=""  \
        --template=" Query: <q> Document: <d> Relevant: "   \
        --prefix='[16107, 10, 2588, 8, 20208, 344, 3, 27569, 11, 11167, 5,3,27569,10]'   \
        --infix='[11167,10]'    \
        --suffix='[31484,17,10,1]'   \
        #--soft_prompt   \
        #--template="Task: Find the relevance between Query and Document. Query: <q> Document: <d> Relevant: "
        #--template=" Query: <q> Document: <d> Relevant: "   \





