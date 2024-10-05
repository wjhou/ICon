# <span style="font-variant:small-caps;">ICON</span>: Improving Inter-Report Consistency in Radiology Report Generation via Lesion-aware Mixup Augmentation

This repository is the implementation of [ICON: Improving Inter-Report Consistency in Radiology Report Generation via Lesion-aware Mixup Augmentation](https://arxiv.org/abs/2402.12844). Before running the code, please install the prerequisite libraries, and follow our instructions to replicate the experiments.

## Overview

Previous research on radiology report generation has made significant progress in terms of increasing the clinical accuracy of generated reports. In this paper, we emphasize another crucial quality that it should possess, i.e., inter-report consistency, which refers to the capability of generating consistent reports for semantically equivalent radiographs. This quality is even of greater significance than the overall report accuracy in terms of ensuring the system's credibility, as a system prone to providing conflicting results would severely erode users' trust. Regrettably, existing approaches struggle to maintain inter-report consistency, exhibiting biases towards common patterns and susceptibility to lesion variants. To address this issue, we propose ICON, which improves the inter-report consistency of radiology report generation. Aiming to enhance the system's ability to capture similarities in semantically equivalent lesions, our approach first involves extracting lesions from input images and examining their characteristics. Then, we introduce a lesion-aware mixup technique to ensure that the representations of the semantically equivalent lesions align with the same attributes, achieved through a linear combination during the training phase. Extensive experiments on three publicly available chest X-ray datasets verify the effectiveness of our approach, both in terms of improving the consistency and accuracy of the generated reports.
![Alt text](figure/overview.png?raw=true "Title")

## Requirements

### Basic Requirements
- `python>=3.9.0`
- `torch==2.1.0`
- `transformers==4.36.2`

### Other Requirements
Please install dependencies by using the following command:

```
conda env create -f environment.yml # Untested
conda activate icon
```

## Data Preparation and Preprocessing

### Observation Annotation
Please download the three datasets: [IU X-ray](https://openi.nlm.nih.gov/faq), [MIMIC-ABN](https://github.com/zzxslp/WCL/) and [MIMIC-CXR](https://physionet.org/content/mimic-cxr-jpg/2.0.0/), and put the annotation files into the `data` folder.

- For observation preprocessing, we use [CheXbert](https://arxiv.org/pdf/2004.09167.pdf) to extract relevant observation information. Please follow the [instruction](https://github.com/stanfordmlgroup/CheXbert#prerequisites) to extract the observation tags. _Note that both report-level and sentence-level annotations are required._
- For CE evaluation, please clone CheXbert into the folder and download the checkpoint [chexbert.pth](https://stanfordmedicine.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9) into CheXbert:

```
git clone https://github.com/stanfordmlgroup/CheXbert.git
```

### Attribute Annotation
Attribute annotation is built upon [RadGraph](https://physionet.org/content/radgraph/1.0.0/). We adopt the same attributes released by [Recap](https://github.com/wjhou/Recap/tree/main/data/20240101).

### Semantic Equivalence Retrieval
Semantic equivalances are built based on report similarity. Run the following code to retrieve similar reports:
```
./script_retrieval/run_mimic_cxr.sh
```

## Stage 1: Lesion Extraction
### Stage 1.1 Training Zoomer
Two parameters are required to run the code of the Zoomer:
- debug: whether debugging the code (0 for debugging and 1 for running)
- date: date of running the code (checkpoint identifier)
```
./script_stage1/run_mimic_cxr.sh debug date
```

Checkpoints are saved into `./tmp_stage1/`

Example: `./script_stage1/run_mimic_cxr.sh 1 20240101`

### Stage 1.2 Extracting Lesions
Specify the checkpoint position of Zoomer in the script, and run:
```
./script_xai/run_mimic_cxr.sh
```

## Stage 2: Report Generation
Two parameters are required to run the code of report generation:
- debug: whether debugging the code (0 for debugging and 1 for running)
- date: date of running the code (checkpoint identifier)
```
./script_stage2/run_mimic_cxr.sh debug date
```

Checkpoints are saved into `./tmp_stage2/`

Example: `./script_stage2/run_mimic_cxr.sh 1 20240101`

## Consistency Evaluation

Observation and attribute annotation are required for consistency evaluation. Specify the positions of the outputs a report generation system, and run the following code:
- output_file: the output file of the system 
```
python eval_consistency.py output_file
```

Example: `python eval_consistency.py ./tmp_stage2/mimic_cxr/eval_results.json`

## Model Checkpoints

Model checkpoints of two datasets are available at:

| Dataset |Stage 1|Stage 2|
|---------|-------|-------|
|MIMIC-ABN|[Google Drive](https://drive.google.com/file/d/1-CnFhtdzb-wGN31pUvFcV269JE_KrIyH/view?usp=drive_link)|[Google Drive](https://drive.google.com/file/d/1ICapdG35Qe9VfA9vPE7EktOUU3Gk9emK/view?usp=drive_link)|
|MIMIC-CXR|[Google Drive](https://drive.google.com/file/d/1zd1LXjqBQ_na7LFZ5Rq6segiRCxWRZZF/view?usp=drive_link)|Coming soon|

## Citation

If you use the <span style="font-variant:small-caps;">ICon</span>, please cite our paper:

```bibtex
@inproceedings{hou-etal-2024-icon,
    title = "{ICON}: Improving Inter-Report Consistency in Radiology Report Generation via Lesion-aware Mixup Augmentation",
    author = "Hou, Wenjun and Cheng, Yi and Xu, Kaishuai and Hu, Yan and Li, Wenjie and Liu, Jiang",
}
```
