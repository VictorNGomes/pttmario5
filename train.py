import argparse
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
nltk.download("punkt")
from calculate_metric import calculate_metric_on_test_ds
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
import mlflow
import mlflow.pytorch
import pandas as pd
from huggingface_hub import notebook_login
from transformers import DataCollatorForSeq2Seq
import mlflow
import os
from dotenv import load_dotenv

load_dotenv()

# Acesse as variáveis de ambiente
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
HUB_MODEL_TOKEN_READ = os.getenv("HUB_MODEL_TOKEN_READ")
HUB_MODEL_TOKEN_WRITE = os.getenv("HUB_MODEL_TOKEN_WRITE")

device = "cuda" if torch.cuda.is_available() else "cpu"
# Set model e tokenzizer


class Model():
    def __init__(self, model_checkpoint="unicamp-dl/ptt5-base-portuguese-vocab",t5_tokenizer="unicamp-dl/ptt5-base-portuguese-vocab"):
        self.t5_model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer)

# load dataset,
    def load_dataset_hugging_face(self, dataset="VictorNGomes/CorpusTeMario", token=None, text_column='texto', summary_column='sumario'):
        self.text = text_column
        self.summary = summary_column

        print("Loading model from huggingface")
        dataset_samsum = load_dataset(dataset, token)
        print("Dataset loaded")
        split_lengths = [len(dataset_samsum[split]) for split in dataset_samsum]

        print("dataset loaded")
        print(f"Split lengths: {split_lengths}")
        print(f"Features: {dataset_samsum['train'].column_names}")
        print("\nText:")
        print(dataset_samsum["test"][0][self.text])
        print("\nSummary:")
        print(dataset_samsum["test"][0][self.summary])

        return dataset_samsum

    def histogram_tokens(self, dataset_samsum, t5_tokenizer):
        dialogue_token_len = []
        summary_token_len = []

        for i in dataset_samsum['train'][self.text]:
            dialogue_token_len.append(len(self.t5_tokenizer.encode(i)))

        for i in dataset_samsum['train'][self.text]:
            summary_token_len.append(len(self.t5_tokenizer.encode(i)))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(dialogue_token_len, bins=20)
        axes[0].set_title("Text Token Length")
        axes[0].set_xlabel("Length")
        axes[0].set_ylabel("Count")

        axes[1].hist(summary_token_len, bins=20)
        axes[1].set_title("Summary Token Length")
        axes[1].set_xlabel("Length")
        plt.tight_layout()
        plt.show()

    def convert_examples_to_features(self, example_batch):
        prefix = "summarize: "

        inputs = [prefix + doc for doc in example_batch[self.text]]
        input_encodings = self.t5_tokenizer(inputs, max_length=1024, truncation=True)

        with self.t5_tokenizer.as_target_tokenizer():
            target_encodings = self.t5_tokenizer(example_batch['sumario'], max_length=128, truncation=True)

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Script para treinar um modelo e realizar experimento com MLflow.")

    parser.add_argument("--model_checkpoint", required=True, help="Caminho ou nome do modelo checkpoint.")
    parser.add_argument("--tokenizer", required=True, help="Caminho ou nome do tokenizer.")
    parser.add_argument("--dataset", required=True, help="Caminho do conjunto de dados.")
    parser.add_argument("--text_column", required=False, help="Nome da coluna de texto no conjunto de dados.")
    parser.add_argument("--summary_column", required=False, help="Nome da coluna de sumário no conjunto de dados.")
    parser.add_argument("--experiment_name", required=True, help="Nome do experimento no MLflow.")
    parser.add_argument("--push_to_huggingface", action="store_true", help="Flag para decidir se faz push para Hugging Face.")

    return parser.parse_args()


def main():
    args = parse_args()

    model = Model(model_checkpoint=args.model_checkpoint, t5_tokenizer=args.tokenizer)
    dataset = model.load_dataset_hugging_face(args.dataset,args.text_column,args.summary_column)

    dataset_samsum_pt = dataset.map(model.convert_examples_to_features, batched=True)
    seq2seq_data_collator = DataCollatorForSeq2Seq(model.t5_tokenizer, model=model.t5_model)

    rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    rouge_metric = load_metric('rouge')

    # Define your MLflow experiment
    mlflow.set_experiment(args.experiment_name)

    # Start an MLflow run
    with mlflow.start_run():
        # Your machine learning code here
        trainer_args = TrainingArguments(
            output_dir='ptt_temario',
            num_train_epochs=50,
            warmup_steps=500,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            weight_decay=0.01,
            logging_steps=10,
            evaluation_strategy='steps',
            eval_steps=500,
            save_steps=1e6,
            gradient_accumulation_steps=16,
            push_to_hub=args.push_to_huggingface,
            fp16=True,
            hub_model_id='VictorNGomes/pttmario5',
            hub_token=HUB_MODEL_TOKEN_WRITE
        )

        trainer = Trainer(model=model.t5_model, args=trainer_args,
                          tokenizer=model.t5_tokenizer, data_collator=seq2seq_data_collator,
                          train_dataset=dataset_samsum_pt["train"],
                          eval_dataset=dataset_samsum_pt["validation"])

        trainer.train()

        # Log parameters and metrics to MLflow
        mlflow.log_params(trainer_args.to_dict())

        # Log custom metrics (e.g., Rouge scores)
        score = calculate_metric_on_test_ds(
            dataset['test'], rouge_metric, trainer.model, model.t5_tokenizer, batch_size=1, column_text=args.text_column,
            column_summary=args.summary_column
        )

        rouge_dict = {rn: score[rn].mid.fmeasure for rn in rouge_names}
        df_rouge = pd.DataFrame(rouge_dict, index=['ptt5-temario'])
        print(df_rouge)

        # Log specific ROUGE scores using MLflow
        for rouge_type in ["rouge1", "rouge2", "rougeL", "rougeLsum"]:
            rouge_score = rouge_dict.get(rouge_type, 0.0)  # Default to 0.0 if the metric is not present
            mlflow.log_metric(f'{rouge_type}_fmeasure', rouge_score)

        if args.push_to_huggingface:
            trainer.push_to_hub()

        pipe = pipeline("summarization", model=model.model_checkpoint)

        with mlflow.start_run():
            mlflow.transformers.log_model(
                transformers_model=pipe,
                artifact_path="my_pipeline",
            )


if __name__ == '__main__':
    main()
