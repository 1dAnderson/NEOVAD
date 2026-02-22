# Rethinking Open Vocabulary Video Anomaly Detection - Normality Matters
This repository contains the PyTorch implementation of our paper:  [Rethinking Open Vocabulary Video Anomaly Detection - Normality Matters]

![framework](./pic/framework.pdf)

---
## Setup
### Dependencies
Please set up the environment by following the `requirements.txt` file.

## Reproduce 
To reproduce the inference results:
- Change the test list path in `src/configs_base2novel.py`, to all/base/novel test set. The 'All' option is set by default in configs_base2novel.py.

- [Download](https://drive.google.com/drive/folders/1xWK8V0OW58BtBSNQwUl338OLm6OY47kJ?usp=drive_link) and move `ckpt/` to your own path, set the ckpt path in `src/configs_base2novel.py`.


- **Inference**
     ```
    cd src
    python main.py --mode infer --dataset ucf --test best_ckpt --device cuda:0
    ```

if you want to training in scratch:
- Official Dataset Download
The original datasets for [UCF-Crime](https://www.crcv.ucf.edu/research/real-world-anomaly-detection-in-surveillance-videos/), [ShanghaiTech](https://github.com/StevenLiuWen/sRNN_TSC_Anomaly_Detection), [XD-Violence](https://roc-ng.github.io/XD-Violence/), and [UBnormal](https://github.com/lilygeorgescu/UBnormal?tab=readme-ov-file) can be obtained from their official sources.

- Extract the CLIP feature
    The extracted CLIP features for the UCF-Crime, ShanghaiTech and XD-Violence datasets can be obtained from [CLIP-TSA](https://github.com/joos2010kj/CLIP-TSA).


    You can also use the CLIP model to extract features by referring to the scripts under `./scripts/feature_extract`.

The following files need to be modified in order to run the code on your own machine:

- Change the file paths to the CLIP features of the datasets above in `src/list/`, and feel free to change the hyperparameters in `configs_base2novel.py`


- run training command:

```
cd src
python main.py --mode infer --dataset ucf  --test best_ckpt --device cuda:0
```

The `--dataset` option can be `ucf`, `sh`, `xd`, or `ub`, referring to UCF-Crime, ShanghaiTech, XD-Violence, or UBnormal.
`--test` option create new folder for training.
`--device` option asign the GPU 
You could add more options like `--seed` and `--lamda2` to change the training options. Default parameter could be found in main.py.

## Acknowledgement

Our code references:
- [AA-CLIP](t https://github.com/Mwxinnn/AA-CLIP)
- [PLOVAD](https://github.com/ctX-u/PLOVAD)



