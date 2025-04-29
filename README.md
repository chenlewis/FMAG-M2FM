# Moire Spectral Augmentation and Masked Frequency Modeling for Document Presentation Attack Detection
Our work has been accepted for publication in IEEE Transactions on Dependable and Secure Computing.

![Method Overview](figure/image1.png)

## Requirments

For environment setup, please follow the [MFM installation](https://github.com/Jiahao000/MFM/blob/master/docs/INSTALL.md).

## Dataset

### Training Data Preparation

We use  **DM** for training:
- 48 genuine document images
- 386 screen-recaptured images

You can download the dataset [here](https://pan.baidu.com/s/1cSHTpfrWxP8nUyHTRZOZ_g).  🔑Extraction code: `rtw1`

**Preprocessing Steps**:
1. Crop images into 224×224 patches.
2. Apply data-level oversampling:
   - Duplicate genuine samples
   - Maintain class balance
3. Please organise the training set as follows (0-legal; 1-recaptured):
```plaintext
DM/
├── images/
│   ├── 0/         
│   │   ├── HUAWEIP9_0009_1.tif
│   │   └── ...
│   └── 1/   
│       ├── IMG_20230521_231421_0_0.tif
│       └── ...
```

### Testing Data Preparation

We use **SRDID162** for testing. You can download the dataset [here](https://pan.baidu.com/s/1M2GYhMPQHe6af_gvGT1Z1w).   🔑Extraction code: `89kx`

## Frequency-domain Moire AuGmentation (FMAG)

## DPAD Network Trained by FMAG

### Training

1. Modify the data path, backbone type, and training parameters in `config/config1.py`.
2. run `python main_backbone.py train`.

### Testing 
run `python main_backbone.py test`.
