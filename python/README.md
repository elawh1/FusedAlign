#  Edited parrot code from PlanetAlign 
https://github.com/yq-leo/PlanetAlign/blob/main/PlanetAlign/algorithms/parrot.py

Edits: 
- Exporting datasets by calling "save_for_cpp" function. 
- parameter "outputs" added to train method to specify the path to save exported files


#  Replace this with parrot.py inside the library in order to export the datasets.

- Example of dataset loader
dataset = PlanetAlign.datasets.ACM_DBLP(
    root="data/",    # where to cache
    download=True,   # auto-download if missing
    train_ratio=0.2, # supervision ratio
    seed=42
)

- Example of how to run training: 
model.train(
    dataset=dataset,
    gid1=0, gid2=1,
    outputs="ACM_DBLP",        # Add this to define path for exported datasets
    use_attr=True,           
    inner_iters=5, outer_iters=10,  
    save_log=True, verbose=True
)