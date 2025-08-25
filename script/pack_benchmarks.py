import os

for dataset in ["fasion_mnist_784", "nuswide", "msong", "sift1m", "glove25", "HandOutlines", "StarLightCurves"]:
    os.system(f"mkdir -p benchmarks-pack/{dataset}")
    os.system(f"cp -r benchmarks/{dataset}/origin benchmarks-pack/{dataset}/")

os.system("zip -r benchmarks-pack.zip benchmarks-pack/")