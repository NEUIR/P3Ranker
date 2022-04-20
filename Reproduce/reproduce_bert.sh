cd ..
for i in 5
do
	srun --job-name="reproduce_bert-$i" --nodes=1 --gpus=4 --mem=200G bash commands_bx/bert.sh $i
done
