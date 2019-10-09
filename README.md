# MaskTextSpotter
This is the code of "Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes" (TPAMI version).
It is an extension of the ECCV version while sharing the same title. For more details, please refer to our [TPAMI paper](https://ieeexplore.ieee.org/document/8812908). 

This repo is inherited from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and follows the same license.

## ToDo List

- [x] Release code
- [x] Document for Installation
- [x] Trained models
- [ ] Document for testing
- [ ] Document for training
- [ ] Demo
- [ ] Evaluation code
- [ ] Release the standalone recognition model

## Installation

### Requirements:
- PyTorch >= 1.0 (1.2 is suggested)
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9 (important)
- OpenCV
- CUDA >= 9.0 (10.0 is suggested)


```bash
  # first, make sure that your conda is setup properly with the right environment
  # for that, check that `which conda`, `which pip` and `which python` points to the
  # right path. From a clean conda env, this is what you need to do

  conda create --name masktextspotter -y
  conda activate masktextspotter

  # this installs the right pip and dependencies for the fresh python
  conda install ipython pip

  # python dependencies
  pip install ninja yacs cython matplotlib tqdm opencv-python shapely scipy tensorboardX

  # install PyTorch
  conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

  export INSTALL_DIR=$PWD

  # install pycocotools
  cd $INSTALL_DIR
  git clone https://github.com/cocodataset/cocoapi.git
  cd cocoapi/PythonAPI
  python setup.py build_ext install

  # install apex (optional)
  cd $INSTALL_DIR
  git clone https://github.com/NVIDIA/apex.git
  cd apex
  python setup.py install --cuda_ext --cpp_ext

  # clone repo
  cd $INSTALL_DIR
  git clone https://github.com/MhLiao/MaskTextSpotter.git
  cd MaskTextSpotter

  # build
  python setup.py build develop


  unset INSTALL_DIR
```

## Models
Download Trained [model](https://drive.google.com/open?id=1pPRS7qS_K1keXjSye0kksqhvoyD0SARz)

## Testing


## Training

## Evaluation

## Citing the related works

Please cite the related works in your publications if it helps your research:

    @article{liao2019mask,
      author={M. {Liao} and P. {Lyu} and M. {He} and C. {Yao} and W. {Wu} and X. {Bai}},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      title={Mask TextSpotter: An End-to-End Trainable Neural Network for Spotting Text with Arbitrary Shapes},
      year={2019},
      volume={},
      number={},
      pages={1-1},
      doi={10.1109/TPAMI.2019.2937086},
      ISSN={},
      month={},
    }
    
    @inproceedings{lyu2018mask,
      title={Mask textspotter: An end-to-end trainable neural network for spotting text with arbitrary shapes},
      author={Lyu, Pengyuan and Liao, Minghui and Yao, Cong and Wu, Wenhao and Bai, Xiang},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
      pages={67--83},
      year={2018}
    }
    

