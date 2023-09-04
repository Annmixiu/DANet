# DANet

## Introduction

The official implementation of "DANET: DIFFERENCE-VALUE ATTENTION NETWORK FOR SINGING MELODY EXTRACTION FROM POLYPHONIC MUSIC". We propose a difference-value attention network (DANet) for melody extraction, which can effectively characterize the fundamental frequency based on emphasizing harmonic contour. Experimental result demonstrates the effectiveness of the proposed network.

## Important updata
### 2023. 09. 4

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

The visualization illustrates that our proposed MTANet can reduce the octave errors and the melody detection errors.

<p align="center">
<img src="fig/estimation1.png" align="center" alt="estimation1" width="50%"/>
</p>
<p align="center">
<img src="fig/estimation.png" align="center" alt="estimation" width="50%"/>
</p>

### Comprehensive result

The scores here are either taken from their respective papers or from the result implemented by us. Experimental results show that our proposed DANet achieves promising performance compared with existing state-of-the-art methods.
|ADC2004|VR|VFA|RPA|RCA|ROA|OA|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
|MCDNN|7.12|8.45|7.85|9.43|8.21|
|MSNet [[paper](https://www.researchgate.net/profile/Juhan-Nam/publication/305771827_Melody_Extraction_On_Vocal_Segments_Using_Multi-Column_Deep_Neural_Networks/links/57a0a08508ae5f8b25891892/Melody-Extraction-On-Vocal-Segments-Using-Multi-Column-Deep-Neural-Networks.pdf)]|3.21|4.22|2.25|3.25|3.23|
|FTANet [[paper](https://hal.inria.fr/hal-02293689/document)] [[code](https://github.com/sigsep/open-unmix-pytorch)]|5.23|5.73|4.02|6.32|5.33|
|TONet [[paper](https://arxiv.org/pdf/2002.07016.pdf)] [[code](https://github.com/pfnet-research/meta-tasnet)]|5.58|5.91|4.19|6.40|5.52|
|HGNet [[paper](https://arxiv.org/pdf/1805.02410.pdf)]|5.16|6.41|4.15|6.60|5.58|
|HANME [[paper](https://arxiv.org/pdf/1909.05746.pdf)]|5.25|6.63|4.09|6.61|5.65|
|DSM [[paper](https://arxiv.org/pdf/2010.04228.pdf)] [[code](https://github.com/sony/ai-research-code/tree/master/x-umx)]|5.43|6.47|4.64|6.61|5.79|

|ours|7.92|7.33|4.92|7.37|6.89|

### Ablation study result

We conducted seven ablations to verify the effectiveness of each design in the proposed network. Due to the page limit, we selected the ADC2004 dataset for ablation study in the paper. More detailed results are presented here.

<p align="center">
<img src="fig/ablution_ADC2004.png" align="center" alt="ablution_ADC2004" width="50%"/>
</p>

<p align="center">
<img src="fig/ablution_MIREX 05.png" align="center" alt="ablution_MIREX 05" width="50%"/>
</p>

<p align="center">
<img src="fig/ablution_MEDLEY DB.png" align="center" alt="ablution_MEDLEY DB" width="50%"/>
</p>

## Download the pre-trained model

Refer to the contents of the folder: [pre-train model](pre-train%20model).

## Special thanks

* [Knut(Ke) Chen](https://github.com/RetroCirce)

* [Shuai Yu](https://github.com/yushuai)
