# MESED

Pytorch Implementation for AAAI2024 full paper "MESED: A Multi-modal Entity Set Expansion Dataset with Fine-grained Semantic
Classes and Hard Negative Entities".

You can refer to [the arXiv version with Appendix](https://arxiv.org/abs/2307.14878).



## Prerequisites
python == 3.9

pytorch == 2.0.1



## Code

The code files are put under "MESED/MultiExpan/src"



## Data

Run

```
wget -O dataset0 https://cloud.tsinghua.edu.cn/f/5f1c716620bf4c4ca7cb/?dl=1
wget -O dataset1 https://cloud.tsinghua.edu.cn/f/035ee15f2dea4e4285ed/?dl=1
cat dataset* > dataset.tar.gz
```

to get datasets used in our experiments. After downloading the dataset, put them under the folder "MESED/dataset/".



Run

```
wget -O data.tar.gz https://cloud.tsinghua.edu.cn/f/4538ef423b0d43df8b88/?dl=1
```

and put them under the folder "MESED/MultiExpan/data".



## Data Preprocessing

Run
```
python make_entity2sents.py
```
to get the folder "MESED/MultiExpan/data/entity2sents" that contains the preprocessed data.



## Training and Evaluating

To train a model with Masked Entity Prediction task, run
```
python main.py -mode 0 -save_path <model_path>
```



After pretraining,  run

```
python main.py -mode 0 -pretrained_model <model_path> -output <output_path> -result <result_file>
```

to get expansion results with the single MEP loss.



Then run

```
python make_cls2eids.py -path_expand_result <best_epoch>
```

to get hard negative entities, which saved in "MESED/MultiExpan/data/cls2eids.pkl".



Finally, training full model with four loss and getting final expansion results, please run

```
python main.py -mode 4 -save_path <model_path>
python main.py -mode 4 -pretrained_model <model_path> -output <output_path> -result <result_file>
```



## Citation

```
@article{li2023mesed,
  title={MESED: A Multi-modal Entity Set Expansion Dataset with Fine-grained Semantic Classes and Hard Negative Entities},
  author={Li, Yangning and Lu, Tingwei and Li, Yinghui and Yu, Tianyu and Huang, Shulin and Zheng, Hai-Tao and Zhang, Rui and Yuan, Jun},
  journal={arXiv preprint arXiv:2307.14878},
  year={2023}
}
```

