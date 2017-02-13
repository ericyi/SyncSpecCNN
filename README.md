# sync-spec-cnn
Synchronized Spectral CNN for 3D Shape Segmentation.

### Introduction
This work is based on our [arXiv tech report](https://arxiv.org/abs/1612.00606). In this repository, we release code, data for training Synchronized Spectral CNN for 3D Shape Segmentation. The data we use is from [A Scalable Active Framework for Region Annotation in 3D Shape Collections](http://web.stanford.edu/~ericyi/project_page/part_annotation/index.html), with a slight re-formatting for our training/test purpose. And the training/test split of the data comes from [ShapeNet](https://shapenet.org/).

### Citation
If you find our work useful in your research, please consider citing:

    @article{yi2016syncspeccnn,
      title={SyncSpecCNN: Synchronized Spectral CNN for 3D Shape Segmentation},
      author={Yi, Li and Su, Hao and Guo, Xingwen and Guibas, Leonidas},
      journal={arXiv preprint arXiv:1612.00606},
      year={2016}
    }

If you use the data provided, please also considering citing:

    @article{yi2016scalable,
      title={A scalable active framework for region annotation in 3d shape collections},
      author={Yi, Li and Kim, Vladimir G and Ceylan, Duygu and Shen, I and Yan, Mengyan and Su, Hao and Lu, ARCewu and Huang, Qixing and Sheffer, Alla and Guibas, Leonidas and others},
      journal={ACM Transactions on Graphics (TOG)},
      volume={35},
      number={6},
      pages={210},
      year={2016},
      publisher={ACM}
    }
    @article{chang2015shapenet,
      title={Shapenet: An information-rich 3d model repository},
      author={Chang, Angel X and Funkhouser, Thomas and Guibas, Leonidas and Hanrahan, Pat and Huang, Qixing and Li, Zimo and Savarese, Silvio and Savva, Manolis and Song, Shuran and Su, Hao and others},
      journal={arXiv preprint arXiv:1512.03012},
      year={2015}
    }
    
### Installation

Install <a href="http://torch.ch/docs/getting-started.html" target="_blank">Torch7</a>.

Note that cuDNN and GPU are highly suggested for speed reason. 
You also need to install a few torch packages (if you haven't done so) including `cunn`, `torchx`, `optim`, `matio`.


### Usage
1. Fetch data including point cloud sampled from ShapeNet shapes, point features and segmentation labels:
  
        bash getdata.sh
  
  These data has been split into different categories and is also split into training/test/validation sets for each category. The data file size is 5GB in total.

2. Compute Laplacian basis for individual shapes and compute joint Laplacian basis for each shape category:

        Matlab/data_preprocessing.m
   
   You will need matlab to preprocess the data. There is one sample category having been pre-processed already called `Sample`, which could be directly used for training.
   
3. Train SyncSpecCNN for each category. To see HELP for training script:
        
        cd Lua
        th main.lua -h

   An example training command is as below:
   
        cd Lua
        th main.lua -s Sample -i 33 -o 4 -ntr 3 -nte 1 -nval 1 -e_b1 20 -e 20 -g 0
   
   The segmentation score will be printed as training goes.

### Results 

Please refer to Table 2 in our [arXiv tech report](https://arxiv.org/abs/1612.00606) for segmentation IoUs.

### License
Our code and data are released under MIT License (see LICENSE file for details).

### TODO
Example code for point cloud part label inference.
