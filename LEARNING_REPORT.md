# Learning Report: Fine-Tuning a Support Ticket Classifier

**Mastering Agentic AI — August 2026 Cohort, Week 5 project**

Evidence: [executed notebook](Finetune_Support_Ticket_Classifier_Qwen3.ipynb), its saved evaluation outputs, and the training logs from this run. See the [README](README.md) for reproduction instructions and the chart correction.

## Objective and approach

I fine-tuned **Qwen/Qwen3-1.7B-Base** to assign an IT support ticket to one of seven categories: Support general, Fileservice, O365, EOL, Software, Active Directory, and Computer-Services. The intended use is to help route tickets to the appropriate queue. This assignment produced a trained classifier and an evaluation; it did not deploy a helpdesk integration or webhook.

I used Google Colab with a T4 GPU and the LLaMA-Factory interface. Instead of updating every model parameter, I trained a LoRA adapter. LoRA adds small trainable matrices while keeping the original model weights frozen. This let me adapt a roughly 1.7-billion-parameter language model by updating **8,716,288 of 1,729,291,264 parameters, approximately 0.504%** of the total reported by the training run.

## Dataset and training configuration

The dataset contained **585 labeled tickets**. The notebook created a stratified split using seed 42: **468 training examples** and **117 held-out validation examples**. Stratification preserved each category's representation across the two partitions. Training examples were converted into ShareGPT conversations containing a system instruction, the ticket text, and the correct category as the assistant response.

I used the `qwen3_nothink` template because the required output was a category label. Matching the training and inference prompt format matters: extra reasoning text or a different response format can interfere with the notebook's label matching.

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-1.7B-Base` |
| Fine-tuning method | LoRA |
| LoRA rank / alpha / dropout | 8 / 16 / 0 |
| LoRA target setting | `all` |
| Learning rate | `5e-5` |
| Epochs | 3 |
| Per-device batch size | 1 |
| Gradient accumulation | 16 |
| Cutoff length | 1,024 tokens |
| Scheduler / warmup steps | Cosine / 0 |
| Precision / seed | FP16 / 42 |

With one GPU, the batch and accumulation settings give a nominal effective batch size of 16. The run completed **90 optimizer steps**. The training logs span 20:37:04 to 20:46:16, approximately **9 minutes 12 seconds**, excluding environment setup, downloads, and evaluation.

The first logged training loss was **3.8701**, and the final logged loss was **2.0716**. These are individual logged values, not average losses for the entire run. Their reduction shows that the training objective improved, but training loss alone does not establish generalization or prove that additional epochs would help.

## Evaluation results

The fine-tuned model classified **84 of 117 validation tickets correctly**, giving **71.8% accuracy**. Its macro F1 was approximately **0.684**, and its weighted F1 was approximately **0.715**. The corrected per-category results were:

| Category | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Support general | 0.533 | 0.800 | 0.640 | 30 |
| Fileservice | 0.955 | 0.750 | 0.840 | 28 |
| O365 | 0.778 | 0.778 | 0.778 | 18 |
| EOL | 1.000 | 1.000 | 1.000 | 12 |
| Software | 0.750 | 0.500 | 0.600 | 12 |
| Active Directory | 0.500 | 0.222 | 0.308 | 9 |
| Computer-Services | 0.625 | 0.625 | 0.625 | 8 |

The five example smoke tests produced **four correct predictions**; the Active Directory example failed. The validation results reinforced that concern: the classifier correctly identified only **2 of 9 Active Directory tickets**, missing seven. Overall accuracy therefore hides a weakness in account and access routing.

EOL achieved perfect scores on this split, but that result covers only 12 tickets. Several categories have small validation samples, so one additional error can noticeably change their reported performance. I would not treat these measurements as a guarantee for future helpdesk traffic.

## What I learned from checking the results

One important lesson was that a correct-looking report can attach metrics to the wrong category. The original code supplied `target_names=LABEL_TOKENS` in a custom order without supplying the matching `labels` order. For string labels, the metrics were calculated in alphabetical order, then displayed under the custom names. The validation support counts exposed the mismatch.

The correction is to make both orders explicit:

```python
classification_report(
    y_true,
    y_pred,
    labels=LABEL_TOKENS,
    target_names=LABEL_TOKENS,
    digits=3,
    zero_division=0,
)
```

Alternatively, omitting `target_names` lets string labels name their own rows. The standalone fine-tuned report was corrected, but the uploaded notebook's comparison helper still needs this correction. Its per-category bars should not be interpreted until that is fixed. Updating these reports does not require rerunning inference, and overall accuracy remains unchanged.

The baseline correctly classified **37 of 117 tickets (31.6%)**, with macro F1 **0.144** and weighted F1 **0.215**. The fine-tuned configuration achieved **47 more correct predictions**, an accuracy improvement of **40.2 percentage points**. However, the baseline scores A–G answer tokens, while the fine-tuned classifier generates category names. The comparison therefore combines changes in training, prompting, and decoding; it does not isolate LoRA's causal contribution.

I also learned to distinguish a confidence score from a calibrated probability. The notebook normalizes scores over the labels' first tokens; this does not mean a prediction with a displayed score of 90% will be correct 90% of the time. Its fallback to Support general for unmatched generated text also makes output-format failures worth tracking separately.

## Proposed next experiments

My next step would be to inspect the misclassified Active Directory tickets and check label consistency, ambiguous wording, and representation in the training data. I would then compare targeted data improvements and a small number of training configurations, rather than assume that more epochs must improve recall.

For a cleaner baseline comparison, I would evaluate both models with the same prompt and decoding protocol. After using validation results to choose changes, I would assess the selected configuration on a **fresh, untouched test set**.

The notebook saved the adapter and merged checkpoint in Colab. Reusing this exact run requires retaining one of those artifacts separately; the GitHub repository currently contains the notebook and dataset, without model weights. Production readiness would require separate validation of routing behavior, artifact persistence, latency, and operating cost; this assignment did not establish those benchmarks.

This project follows the [original course assignment](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router).
