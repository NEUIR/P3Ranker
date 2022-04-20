for i in 256
do
    bash mnli_ft.sh $i
    bash mnli_prompt.sh $i
done
