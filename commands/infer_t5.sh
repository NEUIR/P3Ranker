set -ex
export CUDA_VISIBLE_DEVICES=3,4,5
export OMP_NUM_THREADS=1

LR=1

MAX_STEPS=1
EPOCH=300000

Q='full'
LOG_STEP=500
EVAL_EVERY=2500
 
BATCH_SIZE=2
NEG=1
#TYPE="mono"
TYPE="soft"
#model="t5-lm-adapt-large-$TYPE-prompt"
model="t5-v11-large-$TYPE-prompt"
#ckpt="/data/private/huxiaomeng/pretrained_models/t5-lm-adapt-large"
ckpt="/data/private/huxiaomeng/pretrained_models/t5-v11-large"
#ckpt="t5-large"
dir_prefix="/data/private/huxiaomeng/promptir"
python -m torch.distributed.launch \
         --nproc_per_node=3 \
         --master_port=1617  \
        train.py \
        -task classification  \
        -model t5  \
        -qrels $dir_prefix/collections/msmarco-passage/qrels.train.tsv     \
        -train $dir_prefix/dataset/msmarco/train/$Q-q-$NEG-n.jsonl \
        -dev $dir_prefix/dataset/msmarco/dev/5-q.jsonl  \
        -test $dir_prefix/dataset/msmarco/test/all-q.jsonl  \
        -max_input 80000000  \
        -vocab $ckpt          \
        -pretrain $ckpt  \
        -save $dir_prefix/checkpoints/$model/q$Q-n-$NEG/  \
        -res $dir_prefix/results/$model/q$Q-n-$NEG.trec  \
        -test_res $dir_prefix/results/$model/test_q$Q-n-$NEG.trec  \
        --log_dir=$dir_prefix/logs/$model/q$Q-n-$NEG/  \
        -metric mrr_cut_10  \
        -max_query_len 76  \
        -max_doc_len 290  \
        -epoch $EPOCH  \
        -batch_size $BATCH_SIZE  \
        -lr $LR  \
        -eval_every $EVAL_EVERY  \
        -optimizer adamw  \
        -dev_eval_batch_size  350   \
        -n_warmup_steps 0  \
        -logging_step $LOG_STEP  \
        --max_steps=$MAX_STEPS \
        -gradient_accumulation_steps 1 \
        --soft_sentence=""  \
        --template=" Query: <q> Document: <d> Relevant: "   \
        --prefix='[16107, 10, 2588, 8, 20208, 344, 3, 27569, 11, 11167, 5,3,27569,10]'   \
        --infix='[11167,10]'    \
        --suffix='[31484,17,10,1]'   \
        --soft_prompt   \
        #--template="Task: Find the relevance between Query and Document. Query: <q> Document: <d> Relevant: "
                                                                                                                                                                                                          

