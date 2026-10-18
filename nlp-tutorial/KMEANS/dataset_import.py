import kagglehub

path = kagglehub.dataset_download(
    "saquib7hussain/k-mean-cluster-dataset",
    output_dir="./data"
)

print("Dataset downloaded to:", path)