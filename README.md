# DANet

## Introduction

The official implementation of "DANET: DIFFERENCE-VALUE ATTENTION NETWORK FOR SINGING MELODY EXTRACTION FROM POLYPHONIC MUSIC". We propose a difference-value attention network (DANet) for melody extraction, which can effectively characterize the fundamental frequency based on emphasizing harmonic contour. Experimental result demonstrates the effectiveness of the proposed network.

## Important updata
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
<img src="fig/estimation0.png" align="center" alt="estimation0" width="50%"/>
</p>
<p align="center">
<img src="fig/estimation1.png" align="center" alt="estimation1" width="50%"/>
</p>

<p align="center">
<img src="fig/estimation2.png" align="center" alt="estimation2" width="50%"/>
</p>
<p align="center">
<img src="fig/estimation3.png" align="center" alt="estimation3" width="50%"/>
</p>

### Visualization result

<p align="center">
  <img src="fig/visualization0.png" width="100" alt="(a)" />
  <img src="fig/visualization1.png" width="100" alt="(b)" />
  <img src="fig/visualization2.png" width="100" alt="(c)" />
  <img src="fig/visualization3.png" width="100" alt="(d)" />
</p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(a)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(b)&nbsp;(c)&nbsp;(d)
### Comprehensive result

The scores here are either taken from their respective papers or from the result implemented by us. Experimental results show that our proposed DANet achieves promising performance compared with existing state-of-the-art methods.

<p align="center">
<img src="fig/comprehensive result.png" align="center" alt="comprehensive result" width="45%"/>
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

### about the application of noisy data

* noisy data for the testing → evaluate the noise immunity and generalization of different models.

<p align="center">
<img src="fig/noise_ADC2004.png" align="center" alt="ablation_ADC2004" width="55%"/>
</p>

<p align="center">
<img src="fig/noise_MIREX05.png" align="center" alt="ablation_MIREX 05" width="55%"/>
</p>

<p align="center">
<img src="fig/noise_MEDLEY DB.png" align="center" alt="ablation_MEDLEY DB" width="55%"/>
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
