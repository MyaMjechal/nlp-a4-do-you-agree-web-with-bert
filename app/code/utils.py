import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bert_args = pickle.load(open("models/bert_args.pkl", "rb"))
vocab_size = bert_args['vocab_size']
word2id = bert_args['word2id']
batch_size = bert_args['batch_size']
max_mask = bert_args['max_mask']
max_len = bert_args['max_len']
n_layers = bert_args['n_layers']
n_heads = bert_args['n_heads']
d_model = bert_args['d_model']
d_ff = bert_args['d_ff']
d_k = bert_args['d_k']
d_v = bert_args['d_v']
n_segments = bert_args['n_segments']

sbert_args = pickle.load(open("models/sbert_args.pkl", "rb"))
segment_ids = sbert_args['segment_ids']
masked_pos = sbert_args['masked_pos']


# embedding
class Embedding(nn.Module):
    def __init__(self, vocab_size, max_len, n_segments, d_model, device):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos_embed = nn.Embedding(max_len, d_model)      # position embedding
        self.seg_embed = nn.Embedding(n_segments, d_model)  # segment(token type) embedding
        self.norm = nn.LayerNorm(d_model)
        self.device = device

    def forward(self, x, seg):
        #x, seg: (bs, len)
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long).to(self.device)
        pos = pos.unsqueeze(0).expand_as(x)  # (len,) -> (bs, len)
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


# scaled dot attention
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, device):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([d_k])).to(device)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


# multihead attention
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, device):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_k
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, self.d_v * n_heads)
        self.device = device
    def forward(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1,2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1,2)  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = ScaledDotProductAttention(self.d_k, self.device)(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v) # context: [batch_size x len_q x n_heads * d_v]
        output = nn.Linear(self.n_heads * self.d_v, self.d_model, device=self.device)(context)
        return nn.LayerNorm(self.d_model, device=self.device)(output + residual), attn # output: [batch_size x len_q x d_model]


# poswisefeedforwardnet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, len_seq, d_model) -> (batch_size, len_seq, d_ff) -> (batch_size, len_seq, d_model)
        return self.fc2(F.gelu(self.fc1(x)))


# encoder
class EncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, d_k, device):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(n_heads, d_model, d_k, device)
        self.pos_ffn       = PoswiseFeedForwardNet(d_model, d_ff)

    def forward(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs, attn


# attention mask
def get_attn_pad_mask(seq_q, seq_k, device):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1).to(device)  # batch_size x 1 x len_k(=len_q), one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


class BERT(nn.Module):
    def __init__(self, n_layers, n_heads, d_model, d_ff, d_k, n_segments, vocab_size, max_len, device):
        super(BERT, self).__init__()
        self.params = {'n_layers': n_layers, 'n_heads': n_heads, 'd_model': d_model,
                       'd_ff': d_ff, 'd_k': d_k, 'n_segments': n_segments,
                       'vocab_size': vocab_size, 'max_len': max_len}
        self.embedding = Embedding(vocab_size, max_len, n_segments, d_model, device)
        self.layers = nn.ModuleList([EncoderLayer(n_heads, d_model, d_ff, d_k, device) for _ in range(n_layers)])
        self.fc = nn.Linear(d_model, d_model)
        self.activ = nn.Tanh()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, 2)
        # decoder is shared with embedding layer
        embed_weight = self.embedding.tok_embed.weight
        n_vocab, n_dim = embed_weight.size()
        self.decoder = nn.Linear(n_dim, n_vocab, bias=False)
        self.decoder.weight = embed_weight
        self.decoder_bias = nn.Parameter(torch.zeros(n_vocab))
        self.device = device

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)
        # output : [batch_size, len, d_model], attn : [batch_size, n_heads, d_mode, d_model]

        # 1. predict next sentence
        # it will be decided by first token(CLS)
        h_pooled   = self.activ(self.fc(output[:, 0])) # [batch_size, d_model]
        logits_nsp = self.classifier(h_pooled) # [batch_size, 2]

        # 2. predict the masked token
        masked_pos = masked_pos[:, :, None].expand(-1, -1, output.size(-1)) # [batch_size, max_pred, d_model]
        h_masked = torch.gather(output, 1, masked_pos) # masking position [batch_size, max_pred, d_model]
        h_masked  = self.norm(F.gelu(self.linear(h_masked)))
        logits_lm = self.decoder(h_masked) + self.decoder_bias # [batch_size, max_pred, n_vocab]

        return logits_lm, logits_nsp, output

    def get_last_hidden_state(self, input_ids, segment_ids):
        output = self.embedding(input_ids, segment_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids, self.device)
        for layer in self.layers:
            output, enc_self_attn = layer(output, enc_self_attn_mask)

        return output


class CustomBERTTokenizer:
  def __init__(self, word2id):
      self.word2id = word2id
      self.id2word = {v: k for k, v in self.word2id.items()}
      self.vocab_size = len(self.word2id)
      self.max_len = max_len

  def encode(self, sentences):
      output = {}
      output['input_ids'] = []
      output['attention_mask'] = []
      for sentence in sentences:
        # TODO don't forget to change UNK back
          input_ids = [self.word2id.get(word, self.word2id['[UNK]']) for word in sentence.split()]
          n_pad = self.max_len - len(input_ids)
          input_ids.extend([0] * n_pad)
          att_mask = [1 if idx != 0 else 0 for idx in input_ids]  # Create attention mask
          output['input_ids'].append(torch.tensor(input_ids))  # Convert to tensor
          output['attention_mask'].append(torch.tensor(att_mask))  # Convert to tensor
      return output

  def decode(self, ids):
      return ' '.join([self.id2word.get(idx.item(), '[UNK]') for idx in ids])


tokenizer = CustomBERTTokenizer(word2id)

# Define Labels
nli_labels = ["Entailment", "Neutral", "Contradiction"]

classifier_head = torch.nn.Linear(768*3, 3).to(device)

# define mean pooling function
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool


def calculate_similarity(model, tokenizer, sentence_a, sentence_b, device):
    # Tokenize and convert sentences to input IDs and attention masks
    inputs_a = tokenizer.encode([sentence_a])
    inputs_b = tokenizer.encode([sentence_b])

    # Move input IDs and attention masks to the active device
    inputs_ids_a = inputs_a['input_ids'][0].unsqueeze(0).to(device)
    attention_a = inputs_a['attention_mask'][0].unsqueeze(0).to(device)
    inputs_ids_b = inputs_b['input_ids'][0].unsqueeze(0).to(device)
    attention_b = inputs_b['attention_mask'][0].unsqueeze(0).to(device)

    # Define `segment_ids` and `masked_pos` to prevent errors in the embedding layer
    segment_ids = torch.zeros_like(inputs_ids_a, dtype=torch.long).to(device)
    masked_pos = torch.zeros_like(inputs_ids_a, dtype=torch.long).to(device)

    # Extract sentence embeddings
    with torch.no_grad():
        _, _, u = model(inputs_ids_a, segment_ids, masked_pos)
        _, _, v = model(inputs_ids_b, segment_ids, masked_pos)

    # Get the mean-pooled vectors
    u = mean_pool(u, attention_a).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim
    v = mean_pool(v, attention_b).detach().cpu().numpy().reshape(-1)  # batch_size, hidden_dim

    # Calculate cosine similarity
    similarity_score = cosine_similarity(u.reshape(1, -1), v.reshape(1, -1))[0, 0]

    return similarity_score


def predict_nli(model, premise, hypothesis):
    """Predicts the NLI label using a custom-trained Sentence Transformer model."""
    model.eval()
    
    # Tokenize and encode the premise and hypothesis using `.encode()`
    inputs_a = tokenizer.encode([premise])
    inputs_b = tokenizer.encode([hypothesis])

    # Move input IDs and attention masks to the active device
    inputs_ids_a = inputs_a["input_ids"][0].unsqueeze(0).to(device)
    attention_a = inputs_a["attention_mask"][0].unsqueeze(0).to(device)
    inputs_ids_b = inputs_b["input_ids"][0].unsqueeze(0).to(device)
    attention_b = inputs_b["attention_mask"][0].unsqueeze(0).to(device)

    # Define `segment_ids` and `masked_pos` to prevent errors in the embedding layer
    segment_ids = torch.zeros_like(inputs_ids_a, dtype=torch.long).to(device)
    masked_pos = torch.zeros_like(inputs_ids_a, dtype=torch.long).to(device)

    # Extract sentence embeddings
    with torch.no_grad():
        _, _, u = model(inputs_ids_a, segment_ids, masked_pos)
        _, _, v = model(inputs_ids_b, segment_ids, masked_pos)

    # Mean pooling
    u_mean_pool = mean_pool(u, attention_a)
    v_mean_pool = mean_pool(v, attention_b)

    # Concatenate u, v, and |u - v|
    x = torch.cat([u_mean_pool, v_mean_pool, torch.abs(u_mean_pool - v_mean_pool)], dim=-1)

    # Pass through classifier head
    logits = classifier_head(x)

    # Get predicted label
    predicted_label = torch.argmax(logits, dim=1).item()

    return nli_labels[predicted_label]
