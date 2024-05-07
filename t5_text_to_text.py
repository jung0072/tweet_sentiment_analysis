from datasets import load_from_disk
from transformers import (
    T5ForConditionalGeneration,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import numpy as np
import evaluate
import wandb
import os
import re
import nltk
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

nltk.download('punkt')

os.environ["WANDB_API_KEY"] = ""
os.environ["WANDB_LOG_MODEL"] = "true"
os.environ["WANDB_WATCH"] = "false"

MODEL_ID = "google/flan-t5-large"
EPOCH = 20
BATCH_SIZE = 128
LR = 4e-4
WD = 0.01

project_name = MODEL_ID.split("/")[-1]
run_name = f"E{EPOCH}_B{BATCH_SIZE}_LR{LR}_WD{WD}".replace("/", "_")

metric = evaluate.load("rouge", trust_remote_code=True)

def compute_accuracy(pred, labels):
    exact_match_result = {
        "0_correct": 0,
        "0_wrong": 0,
        "1_correct": 0,
        "1_wrong": 0,
        "2_correct": 0,
        "2_wrong": 0,
    }
    for pred, label in zip(pred, labels):
        if pred == label:
            key = str(label) + "_correct"
            exact_match_result[key] = exact_match_result[key] + 1
        else:
            key = str(label) + "_wrong"
            exact_match_result[key] = exact_match_result[key] + 1
            
    print("Accuracy")
    print("Bearish: ", exact_match_result["0_correct"]/(exact_match_result["0_correct"] + exact_match_result["0_wrong"]))
    print("Bullish: ", exact_match_result["1_correct"]/(exact_match_result["1_correct"] + exact_match_result["1_wrong"]))
    print("Neutral: ", exact_match_result["2_correct"]/(exact_match_result["2_correct"] + exact_match_result["2_wrong"]))

    accurate_result_sum =  sum([v for k, v in exact_match_result.items() if "correct" in k])
    accuracy = accurate_result_sum / len(labels)
    print("Total: ", accuracy)

    return accuracy

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    accuracy = compute_accuracy(decoded_preds, decoded_labels)

    # Rouge expects a newline after each sentence
    decoded_preds = [
        "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
    ]

    # Compute ROUGE scores
    result = metric.compute(
        predictions=decoded_preds, references=decoded_labels
    )

    # Extract ROUGE f1 scores
    result = {key: value * 100 for key, value in result.items()}

    # Add mean generated length to metrics
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
    ]
    result["gen_len"] = np.mean(prediction_lens)
    result["accuracy"] = accuracy

    print(f"==>> result: {result}")
    return {k: round(v, 4) for k, v in result.items()}


model = T5ForConditionalGeneration.from_pretrained(MODEL_ID)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

dataset = load_from_disk("./tokenized_datasets")

train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

wandb.init(
    project=project_name,
    name=run_name,
    entity="minki-jung",
    config={
        "model_id": MODEL_ID,
        "epochs": EPOCH,
        "batch_size": BATCH_SIZE,
    },
)

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./train_result/{project_name}/{run_name}",
    evaluation_strategy="steps",
    eval_steps=25,
    report_to="wandb",
    num_train_epochs=EPOCH,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_strategy="steps",
    logging_steps=100,
    learning_rate=LR,
    save_strategy="epoch",
    save_total_limit=1,
    weight_decay=WD,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer),
    compute_metrics=compute_metrics,
)

print("Training model...")
trainer.train()

# save model
print("Saving model...")
model.save_pretrained(run_name)

eval_result = trainer.evaluate()
print(f"Eval result: {eval_result}")


# HuggingFace Trainer already has evaluation result, however
# if you want to evaluate the model again Use the following code

# model = T5ForConditionalGeneration.from_pretrained("/root/google flan-t5-large_E1_B128")
# model.to(device)  

# correct = 0
# total = len(eval_dataset)

# for element in eval_dataset:
#     input_ids = torch.tensor([element["input_ids"]]).to(device) 
#     output = model.generate(input_ids)  

#     labels = torch.tensor([element["labels"]]).to(device)

#     if torch.equal(output, labels):
#         correct += 1

# accuracy = correct / total
# print(f"Accuracy: {accuracy}")
