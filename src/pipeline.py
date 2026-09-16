from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# classifier = pipeline("sentiment-analysis")
# print(classifier("I've been waiting for a HuggingFace course my whole life."))


def run():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    sequences = [
        "I've been waiting for a HuggingFace course my whole life.",
        "So have I!",
    ]

    tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
    output = model(**tokens)
    print(output.logits)


if __name__ == "__main__":
    run()
