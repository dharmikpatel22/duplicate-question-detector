from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, DataCollatorWithPadding


def value_to_classlabel(element):
    return {"is_duplicate": class_labels.str2int(element["is_duplicate"])}


def tokenize_function(element):
    return tokenizer(element["sentences1"], element["sentences2"], truncation=True)


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_dataset = load_dataset("quora", trust_remote_code=True)


# Casting the features so that the dataset can later be split in a stratified way.
class_labels = ClassLabel(num_classes=2, names=["Not Duplicate", "Duplicate"])

features_copy = raw_dataset["train"].features.copy()
features_copy["is_duplicate"] = class_labels

raw_dataset["train"] = raw_dataset["train"].cast(features_copy)

# # We are basically performing the following operation using the map function since assignment isn't supported:
# # Equivalent to the following:
# raw_dataset["train"]["is_duplicate"] = class_labels.int2str(raw_dataset["train"]["is_duplicate"])

raw_dataset["train"] = raw_dataset["train"].map(value_to_classlabel, batched=True)


# Separating out the question pairs into different lists (to tokenize later)

sentences1 = []
sentences2 = []

for question_pair in raw_dataset["train"]["questions"]:
    sentences1.append(question_pair["text"][0])
    sentences2.append(question_pair["text"][1])

# Without this, tokenization results in the following error:
# ArrowInvalid: Column 2 named input_ids expected length 1000 but got length 283003

raw_dataset["train"] = raw_dataset["train"].add_column("sentences1", sentences1)
raw_dataset["train"] = raw_dataset["train"].add_column("sentences2", sentences2)


raw_dataset = raw_dataset["train"].train_test_split(
    test_size=0.30, shuffle=True, stratify_by_column="is_duplicate", seed=42
)

validation_test_split = raw_dataset["test"].train_test_split(
    test_size=0.50, shuffle=True, stratify_by_column="is_duplicate", seed=42
)

raw_dataset["validation"] = validation_test_split["train"]
raw_dataset["test"] = validation_test_split["test"]


tokenized_dataset = raw_dataset.map(
    tokenize_function,
    batched=True,
    batch_size=0,  # batch_size=0 corresponds to passing the whole dataset as a batch
    # num_proc=4,
)


# # The following to_tf_dataset methods were just for some tests. They will be used later in the model building script.
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

# tf_train_dataset = tokenized_dataset["train"].to_tf_dataset(
#     columns=["input_ids", "token_type_ids", "attention_mask"],
#     label_cols=["is_duplicate"],
#     shuffle=True,
#     collate_fn=data_collator,
#     batch_size=8,
# )

# tf_validation_dataset = tokenized_dataset["validation"].to_tf_dataset(
#     columns=["input_ids", "token_type_ids", "attention_mask"],
#     label_cols=["is_duplicate"],
#     shuffle=False,
#     collate_fn=data_collator,
#     batch_size=8,
# )

# print(f"type(tokenized_dataset): {type(tokenized_dataset)}")
# print(tokenized_dataset)

tokenized_dataset.save_to_disk("tokenized_quora_duplicates_dataset")
