# AYANet

<div align="center">
<img src="./images/AYANet_double.png" width=40%>
</div>

<br>
<!-- The official implementation of **AYANet: A Gabor Wavelet-based and CNN-based Double Encoder for Building Change Detection in Remote Sensing** (ICPR 2024) -->
The official implementation of "AYANet: A Gabor Wavelet-based and CNN-based Double Encoder for Building Change Detection in Remote Sensing"
<br>

## Environmental settings

We run the code in an Anaconda virtual environment on Ubuntu 22.04.3 LTS (GNU/Linux 5.15.153.1-microsoft-standard-WSL2 x86_64).

### Requirements
All requirements are listed in `requirements.txt`.
Create a virtual ``conda`` environment named ``ayanet`` with the following command:

```
conda create -n ayanet --file requirements.txt
conda activate ayanet
```
### Clone the Repository

```shell
git clone https://github.com/Ayana-Inria/AYANet.git
cd AYANet
```

### Dataset Preparation
Dataset needs to be structured as follows.

```
"""
data structure
-dataroot
    ├─A
        ├─img1.png
        ...
    ├─B
        ├─img1.png
        ...
    ├─label
        ├─img1.png
        ...
    └─list
        ├─val.txt
        ├─test.txt
        └─train.txt

# In list/ folder, prepare text files of the splits and list down all filenames of each split
   # for example:
       list/train.txt
           img1.png
           img32.png
           ...
       list/test.txt
           img2.png
           img15.png
           ...
       list/val.txt
           img54.png
           img100.png
           ...
"""
```

`A`: pre-change images;

`B`: post-change images;

`label`: binary labels;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.

### Training
1. Set the parameters and hyperparameters for the model training in `run_CD.sh`

```bash
gpu_ids=0
dataset_type=bcd
dataset_root=/home/Dataset/S2Looking/All # The path to the dataset folder
checkpoint_dir=./checkpoints # Name of the folder where you store the weights of the model during the training
tb_dir=./tb_vis
vis_dir=./vis
split='train' # The split you want to train on (adjust according to the file name you put in the 'list' folder of the dataset)
val_split='val'

epoch=300
batch_size=8
optimizer=adamw
encoder_arc=double # 'double': double encoders, 'gaborencoder': Gabor encoder only, '': EfficientNet only
decoder_arc=ayanet
lr=0.0001

project_name=AYANet_S2Looking_efficientnetonly_mtf2iadesv2_${encoder_arc}_${decoder_arc}_${split}_${val_split}_${optimizer}_e${epoch}_b${batch_size}_lr${lr}_newlrlambda

```

2. Run the script for training 
```
sh run_CD.sh
```

### Evaluation
1. The evaluation settings can be modified in `eval.sh`

```bash
gpu_ids=0
dataset_type=bcd
dataset_root=/home/Dataset/S2Looking/All # The path to the dataset folder
test_split='test' # The split you want to test on (adjust according to the file name you put in the 'list' folder of the dataset)
checkpoint_dir=./checkpoints # Name of the folder where you store the weights of the model during the training
vis_dir=./vis # Visualization folder
resultdir=./results
checkpoint='All' # Option 'All' will run the evaluation on all weights stored in checkpoint_dir, if you want to evaluate using only 1 checkpoint, specify the name e.g., 'best_ckpt.pt'
project_name='AYANet_S2Looking_gaborencoderonly_mtf2iadesv2_gaborencoderv2_drtanet_train_val_adamw_e300_b8_lr0.0001_newlrlambda' # Specify which model/experiment you want to do the evaluation with

batch_size=8
encoder_arc=double # 'double': double encoders, 'gaborencoder': Gabor encoder only, '': EfficientNet only
decoder_arc=ayanet

```

2. Run the script for evaluation
```
sh eval.sh
```

## Misc
Our network was tested on three datasets for remote sensing building change detection. 

1. LEVIR-CD
    * Paper: [A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection ](https://www.mdpi.com/2072-4292/12/10/1662)
    * Download: [Link](https://justchenhao.github.io/LEVIR/)

2. WHU-CD
    * Paper: [Fully convolutional networks for multisource building extraction from an open aerial and satellite imagery data set](https://ieeexplore.ieee.org/document/8444434)
    * Download: [Link](https://study.rsgis.whu.edu.cn/pages/download/)
    
3. S2Looking
    * Paper: [S2Looking: A Satellite Side-Looking Dataset for Building Change Detection](https://www.mdpi.com/2072-4292/13/24/5094)
    * Download: [Link](https://github.com/S2Looking/Dataset)

### Reproduce the training data
We also provide the code to crop each dataset to the size we used for training i.e., 256 x 256. 
1. The code can be found in `misc/dataset_tool.py`. We divide the cropping method for each dataset into 3 different functions:  `crop_levir()`, `crop_s2looking()`, `crop_whu()` because each of them has different folder structure originally. Make sure you have the same original folder structure as indicated in the comment of each function

2. Change the path to folder containing original dataset and to the folder you want your cropped images to be stored at. Using an absolute path is recommended
```
    # Path to the root folder of original resolution images
    ori_folder = r"/mnt/c/Dataset/S2Looking"
    # Path to the root folder of cropped images
    cropped_folder = r"/mnt/c/Dataset/S2Looking-Cropped"
```

3. There is no default split for the WHU-CD dataset. You can find the split we used for training in `misc/WHU_split/` or you can randomly split them by ...

## :trollface: License
The code is released under the GPL-3.0-only license. See `LICENSE` file for more details.

## :full_moon_with_face: Citation

If you use this code for your research, please cite our paper (to be updated):

```
@inproceedings{AYANet,
      title={AYANet: A Gabor Wavelet-based and CNN-based Double Encoder for Building Change Detection in Remote Sensing}, 
      author={Priscilla Indira Osa and Josiane Zerubia and Zoltan Kato},
      year={2024},
      
}
```

## Acknowledgement
* https://github.com/Herrccc/DR-TANet/tree/main
* https://github.com/justchenhao/BIT_CD
* https://github.com/jxgu1016/Gabor_CNN_PyTorch  
