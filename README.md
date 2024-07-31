# duplicate-question-detector

## Description
The aim of this project is to train an AI model that can reliably detect if any two given questions are duplicates, i.e., paraphrases of each other.

## Dataset
The 'quora' dataset was used for training (See https://huggingface.co/datasets/quora-competitions/quora). 

Following are some highlights about the dataset:
- Total records: 404290
- Features:
    - `questions`:
        - `id`: An integer representing the 'id' of a question.
        - `text`: A pair of questions, possibly duplicates
    - `is_duplicate`: A boolean indicating whether the corresponding pair of questions are duplicates of one another.
- No data cleaning or preprocessing was necessary at all. The data was already very clean and could be used directly.

## Model
A pre-trained transformer model was fine-tuned on the dataset. The exact checkpoint used was `bert-base-uncased`.

Reason for using a transformer model:
    - Although not an impossibility, training any classical model would have been a challenge since the question pairs were quite nuanced. 
    - Even with a successful training, it would be hard for a classical NLP model to beat a transformer model's performance.
