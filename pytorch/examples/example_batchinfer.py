
import fingernet as fnet
fnet.run_lightning_inference(
    "/storage/jcontreras/data/datasets/FVC2002Db2A/orig/",
    "/storage/jcontreras/data/datasets/FVC2002Db2A",
    batch_size = 12,
    recursive = True,
    num_cores = 4,
    devices = [0,1,2,3]
)