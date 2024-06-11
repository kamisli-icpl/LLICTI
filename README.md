# Learned Lossless Image Compression Through Interpolation (LLICTI) With Low Complexity 

## Abstract
With the increasing popularity of deep learning in image processing, many learned lossless image compression methods have been proposed recently. One group of algorithms that have shown good performance are based on learned pixel-based auto-regressive models, however, their sequential nature prevents easily parallelized computations and leads to long decoding times. Another popular group of algorithms are based on scale-based auto-regressive models and can provide competitive compression performance while also enabling simple parallelization and much shorter decoding times. However, their major drawback are the used large neural networks and high computational complexity. This paper presents an interpolation based learned lossless image compression method which falls in the scale-based auto-regressive models group. The method achieves better than or on par compression performance with the recent scale-based auto-regressive models, yet requires more than 10x less neural network parameters and encoding/decoding computation complexity. These achievements are due to the contributions/findings in the overall system and neural network architecture design, such as sharing interpolator neural networks across different scales, using separate neural networks for different parameters of the probability distribution model and performing the processing in the YCoCg-R color space instead of the RGB color space.

## Requirements
    pytorch==1.8.2+cu111
    torch-vision==0.9.2+cu111
    PIL==7.0.0
    numpy==1.20.2
    compressai==1.1.8
    torchac==0.9.3
    typing-extensions==3.7.4.3
(Note these are the module versions we used, but the code may/may-not work with other versions)

## How to run and verify (compress/decompress with) the model in the paper
1) Edit the path to test dataset directory with parameter "test_data" in configs/llicti_A.json file
2) Ensure that other path parameters also point to valid directories in the file system (they will not be used during compression/decompression)
3) Run the code with configuration in llicti_A.json and model under experiments folder:
    python main.py configs/llicti_A.json

## Training
To train a (new) model:
1) Edit your json file
2) In the json file, set "mode" as train
3) Set "train_data_1/2/3/4" for the directory(ies) of training sets
4) Set validation/test set directories via "test/valid_data" parameters
5) Edit other parameters if desired (note that all parameter configurations may not be supported)
6) Run the code with configuration in your json file :
    python main.py configs/your.json

## Compression/Decompression
To compress/decompress with trained (or pretrained) model:
1) Use json file that was used to train the model
2) In the json file, set "mode" as eval_model
3) Set "test_data" as the directory of test images.
    
## Citation
@article{kamisli2023learned,
  title={Learned Lossless Image Compression Through Interpolation With Low Complexity},
  author={Kamisli, Fatih},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
