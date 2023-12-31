# DANet

## Introduction

The official implementation of "DANET: DIFFERENCE-AWARE ATTENTION NETWORK FOR SINGING MELODY EXTRACTION FROM POLYPHONIC MUSIC". We propose a difference-aware attention network (DANet) for melody extraction, which can effectively characterize the fundamental frequency based on perceiving and emphasizing harmonic contour. Experimental result demonstrates the effectiveness of the proposed network.

## Important update
### 2023. 09. 04

(i) Update the critical code, and the rest of the code will be released soon.

Uploaded: 

* [model](model/danet.py)
* [add noise](add_noise.py)
* [CFP representation generation](feature_extraction.py)
* [pre-train model](pre-train%20model)

Subsequent update: 

* control group model 

* data generation code 
                   
* complete code with training and testing

### 2023. 09. 10

To esteemed readers and reviewers：

* In the manuscript, we regret to acknowledge that due to spatial limitations, some details in Figures 1 and 2 may appear smaller than desired and require enlargement for clear visibility, which could affect the visual experience. We deeply apologize for any inconvenience this may cause.

### 2023. 09. 21

We also experimented element-wise subtraction between the outputs of convolutions of different sizes on the speech denoising and dereverberation task, which seems to make the spectral boundaries of the target speaker more obvious. On the whole, this simple operation can make the feature boundaries more significant and seems that there can be more possibilities based on this in the future.

## Getting Started

### Download Datasets

* [MIR-1k](https://sites.google.com/site/unvoicedsoundseparation/mir-1k)
* [ADC 2004 & MIREX05](https://labrosa.ee.columbia.edu/projects/melody/)
* [MedleyDB](https://medleydb.weebly.com/)

After downloading the data, use the txt files in the data folder, and process the CFP feature by [feature_extraction.py](feature_extraction.py).

Note that the label data corresponding to the frame shift should be available before generation.

## Model implementation

Refer to the file: [danet.py](model/danet.py)

## Result

### Prediction result

The visualization illustrates that our proposed DANet can reduce the octave errors and the melody detection errors.

<p align="center">
<img src="fig/summary.png" align="center" alt="estimation" width="50%"/>
</p>

* We adopt a visualization approach to explore what types of errors are solved by our model as shown in the above. We choose MSNet to compare due to its structural similarity and popularity. Specifically, we plot the predictive frequencies over the time and ground truths by the DANet and MSNet on two opera songs: “opera male3.wav” and “opera male5.wav” from the ADC2004. We can observe that there are fewer octave errors (i.e., vertical jumps in the contours inside the red circle) in (a) / (c) than (b) / (d). Furthermore, there are fewer melody detection errors around 250ms and 750ms (i.e., predicting a melody frame as a non-melody one) in (c) than (d).

### Supplement of visualization result

<p align="center">
  <img src="fig/visualization1.png" width="100" alt="(a)" />
  <img src="fig/visualization0.png" width="100" alt="(b)" />
  <img src="fig/visualization2.png" width="100" alt="(c)" />
  <img src="fig/visualization3.png" width="100" alt="(d)" />
</p>

* The first and second picture show the output of the time-frequency attention module.
* The third and fourth picture show the output of the calibration fusion module.
* We can find that the features emphasize harmonic and F0 components of the dominant melody in the first and third picture, while the features emphasize accompaniment and noise components in the second and fourth picture (the alternative view is that the features reversely emphasize harmonic and F0 components of the dominant melody).

### Comprehensive result

The scores here are either taken from their respective papers or from the result implemented by us. Experimental results show that our proposed DANet achieves promising performance compared with existing state-of-the-art methods.

<p align="center">
<img src="fig/comprehensive result1.png" align="center" alt="comprehensive result" width="45%"/>
</p>

The models in the above table correspond to paper and codes:

| model | published | paper | code | model | published | paper | code |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
MCDNN|ISMIR2016|[paper](https://www.researchgate.net/profile/Juhan-Nam/publication/305771827_Melody_Extraction_On_Vocal_Segments_Using_Multi-Column_Deep_Neural_Networks/links/57a0a08508ae5f8b25891892/Melody-Extraction-On-Vocal-Segments-Using-Multi-Column-Deep-Neural-Networks.pdf)|[code](https://github.com/LqNoob/MelodyExtraction-MCDNN/blob/master/MelodyExtraction_SCDNN.py)|MSNet|ICASSP2019|[paper](https://ieeexplore.ieee.org/abstract/document/8682389)|[code](https://github.com/bill317996/Melody-extraction-with-melodic-segnet/blob/master/MSnet/model.py)|
FTANet|ICASSP2021|[paper](https://ieeexplore.ieee.org/abstract/document/9413444)|[code](https://github.com/yushuai/FTANet-melodic/tree/main/network)|TONet|ICASSP2022|[paper](https://ieeexplore.ieee.org/abstract/document/9747304)|[code](https://github.com/RetroCirce/TONet/blob/main/model/tonet.py)|
HGNet|ICASSP2022|[paper](https://ieeexplore.ieee.org/abstract/document/9747629)|-|

### Ablation study result

We conducted seven ablations to verify the effectiveness of each design in the proposed network. Due to the page limit, we selected the ADC2004 dataset for ablation study in the paper. More detailed results are presented here.

<p align="center">
<img src="fig/ADC2004_ablation.png" align="center" alt="ablation_ADC2004" width="50%"/>
</p>

<p align="center">
<img src="fig/MIREX05_ablation.png" align="center" alt="ablation_MIREX 05" width="50%"/>
</p>

<p align="center">
<img src="fig/MEDLEY DB_ablation.png" align="center" alt="ablation_MEDLEY DB" width="50%"/>
</p>

### About the application of noisy data

Source of noise database: [Microsoft Scalable Noisy Speech Dataset (MS-SNSD)](https://github.com/Annmixiu/MS-SNSD)

* noisy data for the testing → evaluate the noise immunity and generalization of different models.

<p align="center">
<img src="fig/noise_ADC2004.png" align="center" alt="ablation_ADC2004" width="65%"/>
</p>

<p align="center">
<img src="fig/noise_MIREX05.png" align="center" alt="ablation_MIREX 05" width="65%"/>
</p>

<p align="center">
<img src="fig/noise_MEDLEY DB.png" align="center" alt="ablation_MEDLEY DB" width="65%"/>
</p>

The results show that the DSM model and our model are robust to noise.

* noisy data for the training → typical data augmentation
* noisy data for the training and testing → evaluate the resistance of our model to noise effects after training in a noisy environment

<p align="center">
<img src="fig/noise_all.png" align="center" alt="ablation_MEDLEY DB" width="50%"/>
</p>

DANet with noise for train: Only the training set was randomly added with 0-20db of various types of noise.

DANet with noise for train and test: training and testing sets are randomly added with 0-20db of various types of noise.

## Download the pre-trained model

Refer to the contents of the folder: [pre-train model](pre-train%20model).

## Special thanks

* [Knut(Ke) Chen](https://github.com/RetroCirce)

* [Shuai Yu](https://github.com/yushuai)
