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

You can download the dataset [here]()

**Preprocessing Steps**:
1. Crop images into 224×224 patches.
2. Apply data-level oversampling:
   - Duplicate genuine samples
   - Maintain class balance

**Directory Structure**:
Please organise the training set as follows (0-legal; 1-recaptured):

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

We collect a large-scale and comprehensive document image dataset, named **SRDID162**, which includes 5346 samples of both genuine and recaptured images.

You can download the dataset [here]()

## DPAD Network Trained by FMAG

