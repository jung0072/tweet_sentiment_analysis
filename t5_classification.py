
# pip install runpod requests datasets transformers numpy evaluate wandb accelerate scikit-learn

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
import numpy as np
import evaluate
import wandb
import os

os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_LOG_MODEL"] = "true"
# os.environ["WANDB_WATCH"] = "false"

# MODEL_ID = "google/flan-t5-base"
MODEL_ID = "google/flan-t5-large"
# MODEL_ID = "google-bert/bert-base-uncased"
# MODEL_ID = "google-bert/bert-base-cased"
# MODEL_ID = "google-bert/bert-large-uncased"
# MODEL_ID = "google-bert/bert-large-cased"

EPOCH = 10
BATCH_SIZE = 64

def compute_metrics(eval_pred):
    if "t5" in MODEL_ID:
        logits = eval_pred.predictions[0]
        labels = eval_pred.label_ids
    else:
        logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)
    metric = evaluate.load("accuracy")
    return metric.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=3,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

dataset = load_dataset("zeroshot/twitter-financial-news-sentiment")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", max_length=56, truncation=True,)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]

run_name = f"{MODEL_ID}_EPOCH{EPOCH}"
run_name = run_name.replace("/", "").replace("\\", "").replace("#", "").replace("?", "").replace("%", "").replace(":", "")

wandb.init(
    project="t5_classification", 
    entity="minki-jung",
    name=run_name,
    config={
        "model_id": MODEL_ID,
        "epochs": EPOCH,
        "batch_size": BATCH_SIZE,
    },
)


training_args = TrainingArguments(
    output_dir=f"./train_result/{MODEL_ID.split('/')[-1]}_{run_name}",
    evaluation_strategy="steps",
    eval_steps=50,
    report_to="wandb",
    num_train_epochs=EPOCH,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_strategy="epoch",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

print("Training model...")
trainer.train()


