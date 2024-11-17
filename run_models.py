from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, pipeline
import random
from tqdm import tqdm
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import time
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MS MARCO dataset
msmarco_version = "v2.1"  # Change to "v1.1" if needed
dataset = load_dataset("ms_marco", msmarco_version)

# Define the models to compare
model_names = [
    "bert-base-uncased",
    "bert-large-uncased",
    "distilbert-base-uncased",
    "roberta-base",
    "roberta-large",
    "albert-base-v2",
    "facebook/bart-base",  # BART also works for MLM
]

# Function to measure speed and resource usage
def measure_speed(func, *args, **kwargs):
    start_time = time.time()

    result = func(*args, **kwargs)

    elapsed_time = time.time() - start_time

    return result, elapsed_time

# Subset the dataset
def subset_msmarco_for_mlm(dataset, subset_size=5, min_passage_length=50):
    # Filter passages with minimum length
    filtered_data = [
        example for example in dataset["validation"] 
        if len(example["passages"]["passage_text"][0]) >= min_passage_length
    ]
    # Randomly sample a subset
    return random.sample(filtered_data, subset_size)

# Get a subset of MS MARCO
subset_size = 100
min_passage_length = 50
subset = subset_msmarco_for_mlm(dataset, subset_size=subset_size, min_passage_length=min_passage_length)

# Extract passages
passages = [example["passages"]["passage_text"][0] for example in subset]
hf_dataset = Dataset.from_dict({"passages": passages})

# Function to mask tokens in a text
def mask_text(text, tokenizer, mask_prob=0.15):
    tokens = tokenizer.tokenize(text)
    num_to_mask = max(1, int(len(tokens) * mask_prob))
    mask_indices = random.sample(range(len(tokens)), num_to_mask)
    masked_tokens = tokens[:]
    for idx in mask_indices:
        masked_tokens[idx] = tokenizer.mask_token
    return tokenizer.convert_tokens_to_string(masked_tokens), mask_indices

# Function to evaluate MLM model
def evaluate_mlm_model(model_name, hf_dataset, top_k, mask_prob=0.15):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name)
    except:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    nlp_pipeline = pipeline("fill-mask", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    # Mask the text for all passages in the dataset
    def mask_passage(example):
        masked_text, mask_indices = mask_text(example["passages"], tokenizer, mask_prob)
        return {"masked_text": masked_text, "mask_indices": mask_indices}

    hf_dataset = hf_dataset.map(mask_passage)

    # Use the pipeline to process all masked texts
    masked_texts = hf_dataset["masked_text"]
    predictions, elapsed_time = measure_speed(nlp_pipeline, masked_texts, batch_size=16)  # Batch size can be adjusted

    total_correct = 0
    total_predictions = 0
    all_true_labels = []
    all_predicted_labels = []
    top_k_correct = [0 for i in range (len(top_k))]

    # Evaluate predictions
    for idx, passage in tqdm(enumerate(hf_dataset)):
        original_tokens = tokenizer.tokenize(passage["passages"])
        mask_indices = passage["mask_indices"]
        predicted_tokens = [pred["token_str"] for preds in predictions[idx] for pred in preds]

        for mask_idx in mask_indices:
            true_token = original_tokens[mask_idx]  # Get the true token

            all_true_labels.append(true_token)
            all_predicted_labels.append(predicted_tokens[0])  # The first prediction for Top-1 Accuracy

            # Top-K Accuracy: Check if the true token is in the top-K predictions
            for i in range(len(top_k)):
                if true_token in predicted_tokens[:top_k[i]]:
                    top_k_correct[i] += 1

            if true_token == predicted_tokens[0]:
                total_correct += 1
            total_predictions += 1


    accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    precision = precision_score(all_true_labels, all_predicted_labels, average='micro', zero_division=0)
    recall = recall_score(all_true_labels, all_predicted_labels, average='micro', zero_division=0)
    f1 = f1_score(all_true_labels, all_predicted_labels, average='micro', zero_division=0)

    top_k_accuracy = [top_k_cor / total_predictions if total_predictions > 0 else 0 for top_k_cor in top_k_correct]
    
    print(f"Accuracy: {accuracy:.4f}")
    for i in range(len(top_k)):
        print(f"Top-{top_k[i]} Accuracy: {top_k_accuracy[i]:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Execution Time: {elapsed_time:.2f} seconds")

    return accuracy, top_k_accuracy, precision, recall, f1, elapsed_time

# Compare models
results = {}
for model_name in model_names:
    print(f"Evaluating {model_name}...")
    top_k = [5,10,20]
    accuracy, top_k_accuracy, precision, recall, f1, elapsed_time = evaluate_mlm_model(model_name, hf_dataset, top_k)
    # Store results
    results[model_name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "elapsed_time": elapsed_time,
    }
    for i in range(len(top_k)):
        results[model_name][f"top_{top_k[i]}_accuracy"] = top_k_accuracy[i]

# Display results
print("\nModel Comparison Results:")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

df = pd.DataFrame.from_dict(results, orient='index')

# Display the resulting DataFrame
print(df)
df.to_csv('results.csv', mode='a')