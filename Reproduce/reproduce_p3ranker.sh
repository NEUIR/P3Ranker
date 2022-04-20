cd ..
for i in 5
do
	srun --job-name="reproduce_p3ranker-$i" --nodes=1 --gpus=8 --mem=200G bash commands_bx/p3ranker.sh $i mnli_step 9000
done
