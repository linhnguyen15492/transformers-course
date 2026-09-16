import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from bertviz import head_view


def scaled_dot_product_attention(Q, K, V):
    d_k = Q.size(-1)  # the dimension of the key vectors
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights


model_name = "bert-base-uncased"  # We will be using the Bert model
model = BertModel.from_pretrained(model_name, output_attentions=True)
tokenizer = BertTokenizer.from_pretrained(model_name)
model.eval()  # As we are not training, we have set the model to evaluation mode so no updates are made

# This is our input sentence
sentence = "The quick brown fox jumps over the lazy dog."

# Using the tokenizer to convert the sentence into tokens
inputs = tokenizer(sentence, return_tensors="pt")
input_ids = inputs["input_ids"]

# Forward pass with attentions
with torch.no_grad():
    outputs = model(**inputs)
    attentions = (
        outputs.attentions
    )  # Tuple of (num_layers, batch, num_heads, seq_len, seq_len)

# Here we decode the tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
head_view(attentions, tokens)  # Visualization
