# Fine-Tuning a Support Ticket Classifier with Qwen3

**Mastering Agentic AI — August 2026 Cohort, Week 5 project**

This project fine-tunes `Qwen/Qwen3-1.7B-Base` to assign IT support tickets to seven categories. It uses LLaMA-Factory, LoRA adapters, and a Google Colab Tesla T4 GPU. The notebook covers data preparation, training, adapter merging, inference, and evaluation.

On 117 held-out validation tickets, accuracy increased from **31.6% (37 correct)** for the baseline to **71.8% (84 correct)** for the fine-tuned classifier: **40.2 percentage points** and **47 additional correct predictions**. Active Directory recall remained low at **22.2%**, making this a learning experiment with clear areas for improvement.

[Open the notebook in Colab](https://colab.research.google.com/github/Mathicode-now/Finetuning/blob/main/Finetune_Support_Ticket_Classifier_Qwen3.ipynb) · [Read the learning report](LEARNING_REPORT.md)

## Task and dataset

The classifier returns one category for each ticket. Connecting those predictions to helpdesk queues or webhooks is future work.

The supplied [support_tickets.csv](support_tickets.csv) contains the columns `text` and `category_truth`. The notebook creates a stratified 80/20 split with seed `42`, preserving approximately the same category proportions in both subsets.

| Category | Typical ticket | Training | Validation |
|---|---|---:|---:|
| Active Directory | Account creation and access | 39 | 9 |
| Computer-Services | Printers, scanners, and device support | 33 | 8 |
| EOL | Server retirement and decommissioning | 46 | 12 |
| Fileservice | Shared folders and network-drive permissions | 110 | 28 |
| O365 | Outlook, Teams, OneDrive, and mailbox issues | 74 | 18 |
| Software | Application installation and access | 46 | 12 |
| Support general | General support and manual triage | 120 | 30 |
| **Total** | | **468** | **117** |

Training examples are converted to ShareGPT-style JSON messages containing a system instruction, a ticket, and the correct category. The dataset is registered in LLaMA-Factory as `support_tickets`.

## Training configuration

| Setting | Value used in this run |
|---|---|
| Base model | `Qwen/Qwen3-1.7B-Base` |
| Hardware | Google Colab Tesla T4 |
| Training stage | Supervised fine-tuning |
| Fine-tuning method | LoRA; quantization disabled |
| Chat template | `qwen3_nothink` |
| Compute type | `fp16` |
| Learning rate | `5e-5` |
| Epochs | `3` |
| Per-device batch size | `1` |
| Gradient accumulation | `16` |
| Effective batch size | `16` on one GPU |
| Cutoff length | `1024` tokens |
| LoRA rank / alpha / dropout | `8` / `16` / `0` |
| LoRA targets | `all` supported linear layers |
| Scheduler / warmup steps | `cosine` / `0` |
| Seed | `42` |
| LLaMA Board validation size | `0`; validation was reserved separately |
| Optimizer updates | `90` |

The run trained **8,716,288 parameters**, approximately **0.504%** of the model plus adapter parameters. The first logged training loss was **3.8701** and the final logged loss was **2.0716**. These are logged training values, not validation loss or classification accuracy.

## Results

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Correct predictions | 37 / 117 | 84 / 117 |
| Accuracy | 31.6% | 71.8% |
| Macro F1 | 0.144 | 0.684 |
| Weighted F1 | 0.215 | 0.715 |

The baseline uses an A–G multiple-choice prompt and scores the corresponding first-token logits. The fine-tuned model generates category text. Both use the same validation tickets, but the different prompts and decoding methods mean this comparison measures the complete configurations; it does not isolate the effect of weight updates alone.

The corrected fine-tuned classification report is:

| Category | Precision | Recall | F1 | Validation tickets |
|---|---:|---:|---:|---:|
| Active Directory | 0.500 | 0.222 | 0.308 | 9 |
| Computer-Services | 0.625 | 0.625 | 0.625 | 8 |
| EOL | 1.000 | 1.000 | 1.000 | 12 |
| Fileservice | 0.955 | 0.750 | 0.840 | 28 |
| O365 | 0.778 | 0.778 | 0.778 | 18 |
| Software | 0.750 | 0.500 | 0.600 | 12 |
| Support general | 0.533 | 0.800 | 0.640 | 30 |

The five-ticket smoke test produced four correct predictions. Its account-creation example was routed to `Support general`. The full validation report confirmed a broader weakness: only **2 of 9 Active Directory tickets** were correctly identified. EOL's perfect score covers just 12 examples and should be interpreted with that sample size in mind.

## Run the assignment

1. Open the notebook using the Colab link above. Choose **Runtime → Change runtime type → T4 GPU**, subject to Colab availability.
2. Upload `support_tickets.csv` to `/content`. Run **Install Dependencies**, followed by the GPU check. The separate identity-dataset example is optional for this task.
3. Run **Prepare Support Ticket Dataset**. Confirm `468` training rows, `117` validation rows, and registration of `support_tickets`.
4. Launch the LLaMA Board cell, open its public URL, and enter the configuration above. Select **Preview dataset** and **Preview command** to check the setup, then start training.
5. Wait for training to finish and the adapter to be saved. Stop the web UI cell so the remaining notebook cells can run, keeping the Colab runtime connected.
6. Set `ADAPTER_DIR` to your actual output folder, then run the loss-curve and baseline/merge cells. The merge cell saves a standalone model in `/content/qwen3_merged`.
7. Run the `classify()` smoke test, validation inference, corrected classification report, confusion matrix, and corrected comparison chart.
8. Save the notebook with outputs and back up the adapter or merged model separately from Colab session storage.

Run cells in order: later cells depend on variables and model objects created earlier. If optional dependencies are missing, install the named packages in a Colab code cell; this notebook uses `scikit-learn` and `seaborn` for evaluation. `bitsandbytes` is needed if switching to its quantization workflow.

## Evaluation correction

String labels are sorted by scikit-learn when no explicit label order is supplied. Supplying a differently ordered `target_names` list can attach the wrong category names to the metrics while leaving overall accuracy unchanged.

For the standalone report, use the category strings directly:

```python
print(classification_report(y_true, y_pred, digits=3, zero_division=0))
```

**The uploaded comparison helper still needs the following correction before regenerating its per-category chart.** The standalone classification report is corrected; the overall accuracy bars are unaffected by this labeling issue.

```python
def per_class_f1(y_t, y_p, labels):
    report = classification_report(
        y_t, y_p,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    return {label: report[label]["f1-score"] for label in labels}
```

## Classify a new ticket

After running the merge cell and the cell defining `classify()`, use:

```python
label, confidence = classify("Outlook keeps asking me to sign in.")
print("Predicted category:", label)
print("Relative score:", confidence)
```

The confidence value is a relative softmax score over the labels' first tokens, not a calibrated probability of correctness. Unrecognized generated text falls back to `Support general`, so that category can include output-format failures as well as actual general-support predictions.

## Files and saved artifacts

| File or location | Purpose |
|---|---|
| [Finetune_Support_Ticket_Classifier_Qwen3.ipynb](Finetune_Support_Ticket_Classifier_Qwen3.ipynb) | Training workflow and saved outputs |
| [support_tickets.csv](support_tickets.csv) | Source tickets and labels |
| [LEARNING_REPORT.md](LEARNING_REPORT.md) | Findings, troubleshooting, limitations, and next experiments |
| `/content/val_split.csv` | Validation split generated in Colab |
| `/content/LLaMA-Factory/saves/Qwen3-1.7B-Base/lora/<run>` | Adapter weights, tokenizer, and training logs |
| `/content/qwen3_merged` | Merged model generated in Colab |
| `/content/training_curve.png`, `/content/confusion_matrix.png`, `/content/baseline_vs_finetuned.png` | Generated plots; regenerate the comparison plot after applying the correction |

Model weights are not included in the inspected GitHub repository. To reuse the trained model, retain either the merged checkpoint or the adapter and its matching base model. Colab runtime files require a separate backup from the notebook.

## Learning scope and next experiments

This run demonstrates the complete fine-tuning and evaluation workflow. It has not established production routing reliability, calibrated confidence thresholds, or CPU latency and hosting-cost claims. The small, imbalanced validation set is especially limiting for Active Directory and Computer-Services.

Next experiments should review missed tickets, improve representation and label consistency for weak categories, and compare controlled training settings. An untouched test set should be reserved before using the current validation results to guide further tuning. A prompt-and-decoding-matched baseline would strengthen the comparison.

## Credits and references

- [Original course assignment and dataset](https://github.com/The-Gen-Academy/5A-Fine-Tune-a-Support-Ticket-Router)
- [Qwen3-1.7B-Base model card](https://huggingface.co/Qwen/Qwen3-1.7B-Base)
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [scikit-learn classification report documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
