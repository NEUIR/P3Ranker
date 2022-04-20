cd ..
for i in  5 "full"
do
	srun --job-name="reproduce_roberta-$i" --nodes=1 --gpus=4 --mem=200G bash commands_bx/roberta.sh $i
done
