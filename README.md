# <span style="font-variant:small-caps;">ICON</span>: Improving Inter-Report Consistency of Radiology Report Generation via Lesion-aware Mix-up Augmentation

This repository is the implementation of [ICON: Improving Inter-Report Consistency in Radiology Report Generation via Lesion-aware Mixup Augmentation](https://arxiv.org/abs/2402.12844). Before running the code, please install the prerequisite libraries, and follow our instructions to replicate the experiments. Codes and Model Checkpoints are coming soon.

## Overview

Previous research on radiology report generation has made significant progress in terms of increasing the clinical accuracy of generated reports. In this paper, we emphasize another crucial quality that it should possess, i.e., inter-report consistency, which refers to the capability of generating consistent reports for semantically equivalent radiographs. This quality is even of greater significance than the overall report accuracy in terms of ensuring the system's credibility, as a system prone to providing conflicting results would severely erode users' trust. Regrettably, existing approaches struggle to maintain inter-report consistency, exhibiting biases towards common patterns and susceptibility to lesion variants. To address this issue, we propose ICON, which improves the inter-report consistency of radiology report generation. Aiming to enhance the system's ability to capture similarities in semantically equivalent lesions, our approach first involves extracting lesions from input images and examining their characteristics. Then, we introduce a lesion-aware mixup technique to ensure that the representations of the semantically equivalent lesions align with the same attributes, achieved through a linear combination during the training phase. Extensive experiments on three publicly available chest X-ray datasets verify the effectiveness of our approach, both in terms of improving the consistency and accuracy of the generated reports.
![Alt text](figure/overview.png?raw=true "Title")

## Requirements

- `python>=3.9.0`
- `torch==2.1.0`
- `transformers==4.36.2`

Please install dependencies by using the following command:

```
conda env create -f environment.yml # Untested
conda activate icon
```

## Data Preparation and Preprocessing

Please download the three datasets: [IU X-ray](https://openi.nlm.nih.gov/faq), [MIMIC-ABN](https://github.com/zzxslp/WCL/) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/), and put the annotation files into the `data` folder.

- For observation preprocessing, we use [CheXbert](https://arxiv.org/pdf/2004.09167.pdf) to extract relevant observation information. Please follow the [instruction](https://github.com/stanfordmlgroup/CheXbert#prerequisites) to extract the observation tags.
- For CE evaluation, please clone CheXbert into the folder and download the checkpoint [chexbert.pth](https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) into CheXbert:

```
git clone https://github.com/stanfordmlgroup/CheXbert.git
```

## Model Checkpoints

Model checkpoints of two datasets are available at:

- MIMIC-ABN: [Google Drive](https://drive.google.com/drive/folders/1xEvXsXaN_RUIJUCsTZX0iXVkLfJzw9AG?usp=sharing)
- MIMIC-CXR: [Google Drive](https://drive.google.com/drive/folders/1CVDq8qsAy2d1UMji6jZrBSw4-3fNqLqt?usp=sharing)

## Citation

If you use the <span style="font-variant:small-caps;">ICon</span>, please cite our paper:

```bibtex
@inproceedings{hou-etal-2024-icon,
    title = "{ICON}: Improving Inter-Report Consistency in Radiology Report Generation via Lesion-aware Mixup Augmentation",
    author = "Hou, Wenjun and Cheng, Yi and Xu, Kaishuai and Hu, Yan and Li, Wenjie and Liu, Jiang",
}
```
