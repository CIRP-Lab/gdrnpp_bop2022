#Editing Data
run Kais json_gen.py script from inside ipd_bop_data/ to generate models_info.json and run kais other script /ipd_bop_data/train_pbr/000000/combine_cam_data.py to combine the tricamera data into single cam data ( I think this is what that is doing)
#To get my image
Ensure you have Docker and NVCC installed (check by running nvcc --version) if the gpu driver is above a certain threshold, then it will be ok to use with any CUDA
run the following command to pull the docker 
```docker pull cirp/bop_gdrnpp2022```

once it is on the machine, run
```sudo docker run -it --shm-size=16G --gpus all -v /mnt/sda:/gdrnpp_bop2022/datasets/BOP_DATASETS gdrnpp_cirp:working3```

this will place you in an interactive terminal, with GPU's activated, with 16GB of shared memory (you should probably increase that!) with the datasets mounted inside the working repository. Change /mnt/sda if your dataset is at a different spot.

#Running the Model
Once inside the container, you can run the following command to begin training
```./core/gdrn_modeling/train_gdrn.sh configs/gdrn/icbin_pbr/convnext_a6_AugCosyAAEGray_BG05_mlL1_DMask_amodalClipBox_classAware_icbin.py 0```
if you get error like "unknown CUDA error" try resetting both the local machine and docker env (do docker commit then run that image again)

the below command will start training using the ipd dataset
```./core/gdrn_modeling/train_gdrn.sh configs/gdrn/ipd/ipd_config.py 0```

#Adding a custom dataset
Currently not working, below are some things I found that may help to solve the issue, you may also review the repository here to see exactly what I have changed thus far
There are some steps detailed below at the bottom of this page. These were the changes I made, alongside copying a config file (for me I chose icbin ), which had bad results, as well as ycbv.
https://github.com/shanice-l/gdrnpp_bop2022/issues/49#issuecomment-1832903008


#Error
Below is my analysis of the error that I am getting TLDR: I am misnaming something somewhere, but I have not found it yet.
AttributeError: 'ConfigDict' object has no attribute 'DATA_CFG'
From my reading, this is occuring because this if statement is not executing the first half of it from /core/gdrn_modeling/datasets/dataset_factory.py, line 92

```       for name in cfg.DATASETS.get(split, []):
            if name in DatasetCatalog.list():
                continue
            registered = False
            # try to find in pre-defined datasets
            # NOTE: it is better to let all datasets pre-refined
            for _mod_name in _DSET_MOD_NAMES:
                if name in get_available_datasets(_mod_name):
                    register_dataset(_mod_name, name, data_cfg=None)
                    registered = True
                    break
            # not in pre-defined; not recommend
            if not registered:
                # try to get mod_name and data_cfg from cfg
                """load data_cfg and mod_name from file
                cfg.DATA_CFG[name] = 'path_to_cfg'
                """
                breakpoint()
               # assert "DATA_CFG" in cfg and name in cfg.DATA_CFG, "no cfg.DATA_CFG.{}".format(name)
               # assert osp.exists(cfg.DATA_CFG[name])
                data_cfg = mmcv.load(cfg.DATA_CFG[name])
                mod_name = data_cfg.pop("mod_name", None)
               # assert mod_name in _DSET_MOD_NAMES, mod_name
                register_dataset(mod_name, name, data_cfg)

```
you can see the breakpoint is being activated, which means the break statement is not being run to kick us out of the loop. I have to think that the correct name is not being added to DatasetCatalog or _DSET_MOD_NAMES lists. I will go look at these functions more closely to hopefully determine where the error is.
