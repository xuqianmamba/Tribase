
datasets=("deep1M" "word2vec" "msong" "gist" "tiny5m" "glove2.2m")


for dataset in "${datasets[@]}"
do
    echo "Running query for dataset: $dataset"
    sudo ./release/bin/query --dataset "$dataset" --high_precision_subNN_index --cache
done