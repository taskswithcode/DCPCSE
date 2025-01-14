# DCPCSE: Deep Continuous Prompt for Contrastive Learning of Sentence Embeddings

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-continuous-prompt-for-contrastive-1/semantic-textual-similarity-on-sick)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sick?p=deep-continuous-prompt-for-contrastive-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-continuous-prompt-for-contrastive-1/semantic-textual-similarity-on-sts12)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts12?p=deep-continuous-prompt-for-contrastive-1)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-continuous-prompt-for-contrastive-1/semantic-textual-similarity-on-sts13)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts13?p=deep-continuous-prompt-for-contrastive-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deep-continuous-prompt-for-contrastive-1/semantic-textual-similarity-on-sts14)](https://paperswithcode.com/sota/semantic-textual-similarity-on-sts14?p=deep-continuous-prompt-for-contrastive-1)

This repository contains the code for our paper [https://arxiv.org/abs/2203.06875](https://arxiv.org/abs/2203.06875). Our code is modified based on [SimCSE](https://github.com/princeton-nlp/SimCSE) and [P-tuning v2](https://github.com/THUDM/P-tuning-v2/). Here we would like to sincerely thank them for their excellent works.

We release our best model checkpoint which acquires **Top 1** results on four STS tasks:

<!-- <img src="https://github.com/YJiangcm/DCPCSE/blob/master/figure/leaderboard.png" width="700" height="380"> -->

|          Model          | STS12 | STS13 | STS14 | STS15 | STS16 | STS-B | SICK-R | Avg. |
|:-----------------------:|:-----:|:----------:|:---------:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  sup-DCPCSE-RoBERTa-large [download](https://drive.google.com/drive/folders/1OWqNgsPtzvsxmDDnOf0WaVuNkELEcatG?usp=sharing)  |  79.14 |88.64| 83.73| 87.33 |84.57| 87.84| 82.07| 84.76|
|  unsup-DCPCSE-BERT-base [download](https://drive.google.com/drive/folders/1OcgJ-7gU_N7J7x5ezrigFLlTU8h7Uvjx?usp=sharing)  |  73.03 |85.18| 76.70| 84.19 |79.69| 80.62| 70.00| 78.49|

If you have any questions, feel free to raise an issue.

## Architecture
<img src="https://github.com/YJiangcm/DCPCSE/blob/master/figure/model%20architecture.png" width="600" height="300">

We add multi-layer trainable dense vectors as continuous prompts to the input sequence, which means the input embeddings as well as each layer's hidden embeddings of prompts are optimized (the orange blocks). Note that all parameters of the pre-trained model are frozen (the blue blocks), thus reducing the number of tunable parameters to around **0.1\%**. The [CLS] token embedding of the last layer is selected as the sentence representation. The contrastive framework is the same as SimCSE.


## Setups
[![Pytorch](https://img.shields.io/badge/pytorch-1.7.1-red?logo=pytorch)](https://pytorch.org/get-started/previous-versions/)

Run the following script to install the remaining dependencies,

```bash
pip install -r requirements.txt
```

## Training

**Data**

Following SimCSE, we use the same datasets to train our unsupervised models and supervised models. You can run `data/download_wiki.sh` and `data/download_nli.sh` to download the two datasets.

**Training scripts**  
(The same as `run_unsup_example.sh`)
```bash
python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir result/my-unsup-dcpcse-bert-base-uncased \
    --num_train_epochs 1 \
    --per_device_train_batch_size 256 \
    --learning_rate 3e-2 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --pre_seq_len 16 \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --fp16
 ```

We provide example training scripts for both unsupervised and supervised DCPCSE. In `run_unsup_example.sh`, we provide a single-GPU (or CPU) example for the unsupervised version, and in `run_sup_example.sh` we give a **multiple-GPU** example for the supervised version. Both scripts call `train.py` for training. We explain the arguments in following:
* `--train_file`: Training file path. We support "txt" files (one line for one sentence) and "csv" files (2-column: pair data with no hard negative; 3-column: pair data with one corresponding hard negative instance). You can use our provided Wikipedia or NLI data, or you can use your own data with the same format.
* `--model_name_or_path`: Pre-trained checkpoints to start with. For now we support BERT-based models (`bert-base-uncased`, `bert-large-uncased`, etc.) and RoBERTa-based models (`RoBERTa-base`, `RoBERTa-large`, etc.).
* `--temp`: Temperature for the contrastive loss.
* `--pooler_type`: Pooling method. It's the same as the `--pooler_type` in the [evaluation part](#evaluation).
* `--mlp_only_train`: We have found that for unsupervised DCPCSE, it works better to train the model with MLP layer but test the model without it. You should use this argument when training unsupervised DCPCSE models.
* `--hard_negative_weight`: If using hard negatives (i.e., there are 3 columns in the training file), this is the logarithm of the weight. For example, if the weight is 1, then this argument should be set as 0 (default value).
* `--do_mlm`: Whether to use the MLM auxiliary objective. If True:
  * `--mlm_weight`: Weight for the MLM objective.
  * `--mlm_probability`: Masking rate for the MLM objective.
* `--pre_seq_len`: The length of deep continuous prompt.
* `--prefix_projection`: Whether apply a two-layer MLP head over the prompt embeddings.
* `--prefix_hidden_size`: The hidden size of the MLP projection head if prefix_projection is used.

All the other arguments are standard Huggingface's `transformers` training arguments. Some of the often-used arguments are: `--output_dir`, `--learning_rate`, `--per_device_train_batch_size`. In our example scripts, we also set to evaluate the model on the STS-B development set (need to download the dataset following the [evaluation](#evaluation) section) and save the best checkpoint.

All our experiments are conducted on two Nvidia 3090 GPUs.

**Hyperparameters**

| **Unsupervised** | BERT-base | BERT-large | RoBERTa-base  | RoBERTa-large |
|:--------------|:-----------:|:--------------:|:---------:|:---------:|
| Batch size    | 256          | 256            | 64       | 64
| Learning rate  | 3e-2 | 3e-2 | 3e-2 | 1e-2 |
| Prompt length | 16 | 10 | 14 | 10 |
| do_mlm | False | False | True | True |
| Epoch |1|1|1|1|
| Valid steps | 125 | 125 | 125 | 125 |

    
| **Supervised** | BERT-base | BERT-large | RoBERTa-base  | RoBERTa-large |
|:--------------|:-----------:|:--------------:|:---------:|:---------:|
| Batch size    | 256          | 256            | 256       | 256
| Learning rate  | 5e-3 | 5e-3 | 1e-2 | 5e-3 |
| Prompt length | 12 | 12 | 10 | 10 |
| do_mlm | False | False | False | False |
| Epoch |10|10|10|10|
| Valid steps | 125 | 125 | 125 | 125 |


## Evaluation
Our evaluation code for sentence embeddings is based on a modified version of [SentEval](https://github.com/facebookresearch/SentEval). It evaluates sentence embeddings on semantic textual similarity (STS) tasks and downstream transfer tasks. For STS tasks, our evaluation takes the "all" setting, and report Spearman's correlation.

Before evaluation, please download the evaluation datasets by running
```bash
cd SentEval/data/downstream/
bash download_dataset.sh
```

Then come back to the root directory, you can evaluate the well trained models using our evaluation code. For example,
```bash
python evaluation.py \
    --model_name_or_path result/my-sup-dcpcse-roberta-large \
    --pooler_type cls \
    --task_set sts \
    --mode test \
    --pre_seq_len 10
```
which is expected to output the results in a tabular format:
```
------ test ------
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness |  Avg. |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
| 79.14 | 88.64 | 83.73 | 87.33 | 84.57 |    87.84     |      82.07      | 84.76 |
+-------+-------+-------+-------+-------+--------------+-----------------+-------+
```

Arguments for the evaluation script are as follows,

* `--model_name_or_path`: The name or path of a `transformers`-based pre-trained checkpoint. 
* `--pooler_type`: Pooling method. Now we support
    * `cls` (default): Use the representation of `[CLS]` token. A linear+activation layer is applied after the representation (it's in the standard BERT implementation). If you use **supervised DCPCSE**, you should use this option.
    * `cls_before_pooler`: Use the representation of `[CLS]` token without the extra linear+activation. If you use **unsupervised DCPCSE**, you should take this option.
    * `avg`: Average embeddings of the last layer. If you use checkpoints of SBERT/SRoBERTa ([paper](https://arxiv.org/abs/1908.10084)), you should use this option.
    * `avg_top2`: Average embeddings of the last two layers.
    * `avg_first_last`: Average embeddings of the first and last layers. If you use vanilla BERT or RoBERTa, this works the best.
* `--mode`: Evaluation mode
    * `test` (default): The default test mode. To faithfully reproduce our results, you should use this option.
    * `dev`: Report the development set results. Note that in STS tasks, only `STS-B` and `SICK-R` have development sets, so we only report their numbers. It also takes a fast mode for transfer tasks, so the running time is much shorter than the `test` mode (though numbers are slightly lower).
    * `fasttest`: It is the same as `test`, but with a fast mode so the running time is much shorter, but the reported numbers may be lower (only for transfer tasks).
* `--task_set`: What set of tasks to evaluate on (if set, it will override `--tasks`)
    * `sts` (default): Evaluate on STS tasks, including `STS 12~16`, `STS-B` and `SICK-R`. This is the most commonly-used set of tasks to evaluate the quality of sentence embeddings.
    * `transfer`: Evaluate on transfer tasks.
    * `full`: Evaluate on both STS and transfer tasks.
    * `na`: Manually set tasks by `--tasks`.
* `--tasks`: Specify which dataset(s) to evaluate on. Will be overridden if `--task_set` is not `na`. See the code for a full list of tasks.
* `--pre_seq_len`: The length of deep continuous prompt.

## Citation

Please cite our paper by:

```bibtex
@misc{jiang2022dcpcse,
      title={Deep Continuous Prompt for Contrastive Learning of Sentence Embeddings}, 
      author={Yuxin Jiang and Wei Wang},
      year={2022},
      eprint={2203.06875},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
