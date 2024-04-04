import torch
import torch.nn as nn
from transformers import BartTokenizer, BartForConditionalGeneration
import math
import copy
from tqdm import tqdm

bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
device = torch.device("cpu")

class BART(nn.Module):
  def __init__(self, tokenizer, pretrained_bart):
    """
    Initializer. Creates network modules and loss function.
    Arguments:
        tokenizer: BART tokenizer
        pretrained_bart: pretrained BART
    """
    super(BART, self).__init__()

    self.tokenizer = tokenizer

    self.V_tgt = len(bart_tokenizer)

    # Get special word ids
    self.padding_id_tgt = tokenizer.pad_token_id

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.bart = pretrained_bart.to(self.device)

    self.loss_function = nn.CrossEntropyLoss(reduction="sum",
                                             ignore_index=self.padding_id_tgt)

  def forward(self, src, src_lengths, tgt_in):
    """
    Performs forward computation, returns logits.
    Arguments:
        src: src batch of size (batch_size, max_src_len)
        src_lengths: src lengths of size (batch_size)
        tgt_in:  a tensor of size (batch_size, tgt_len)
    """
    # BART assumes inputs to be batch-first
    # This single function is forwarding both encoder and decoder (w/ cross attn),
    # using `input_ids` as encoder inputs, and `decoder_input_ids` as decoder inputs.
    logits = self.bart(input_ids=src,
                       decoder_input_ids=tgt_in,
                       use_cache=False
                      ).logits
    return logits

  def evaluate_ppl(self, iterator):
    """Returns the model's perplexity on a given dataset `iterator`."""
    self.eval()
    total_loss = 0
    total_words = 0
    for batch in iterator:
      # Input and target
      src = batch['src_ids']              # bsz, max_src_len
      src_lengths = batch['src_lengths']  # bsz
      tgt_in = batch['tgt_ids'][:, :-1]   # Remove <eos> for decode input
      tgt_out = batch['tgt_ids'][:, 1:]   # Remove <bos> as target        
      # Forward to get logits
      logits = self.forward(src, src_lengths, tgt_in) # bsz, tgt_len, V_tgt
      # Compute cross entropy loss
      loss = self.loss_function(logits.reshape(-1, self.V_tgt), tgt_out.reshape(-1))
      total_loss += loss.item()
      total_words += tgt_out.ne(self.padding_id_tgt).float().sum().item()
    return math.exp(total_loss/total_words)

  def train_all(self, train_iter, val_iter, epochs=10, first_n_batches=None, learning_rate=0.001):
    """Train the model."""
    # Switch the module to training mode
    self.train()
    # Use Adam to optimize the parameters
    optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
    best_validation_ppl = float('inf')
    best_model = None
    # Run the optimization for multiple epochs
    for epoch in range(epochs):
      total_words = 0
      total_loss = 0.0
      i = 0
      # Iterate over each batch in train data iterator
      for batch in tqdm(train_iter):
        # If maximum number of batches has been set and
        # that number of batches has been processed,
        # stop processing further batches
        if first_n_batches and i >= first_n_batches:
          break
        # Zero the parameter gradients
        self.zero_grad()
        # Input and target
        tgt = batch['tgt_ids']              
        src = batch['src_ids']              # bsz, max_src_len
        src_lengths = batch['src_lengths']  # bsz
        tgt_in = tgt[:, :-1].contiguous()   # Remove <eos> for decode input
        tgt_out = tgt[:, 1:].contiguous()   # Remove <bos> as target       
        bsz = tgt.size(0)
        # Run forward pass and compute loss
        logits = self.forward(src, src_lengths, tgt_in)
        loss = self.loss_function(logits.view(-1, self.V_tgt), tgt_out.view(-1))
        # Training stats
        num_tgt_words = tgt_out.ne(self.padding_id_tgt).float().sum().item()
        total_words += num_tgt_words
        total_loss += loss.item()
        # Perform backpropagation
        loss.div(bsz).backward()
        optim.step()
        i += 1

      # Evaluate and track improvements on the validation dataset
      validation_ppl = self.evaluate_ppl(val_iter)
      self.train()
      if validation_ppl < best_validation_ppl:
        best_validation_ppl = validation_ppl
        self.best_model = copy.deepcopy(self.state_dict())
      epoch_loss = total_loss / total_words
      print (f'Epoch: {epoch} Training Perplexity: {math.exp(epoch_loss):.4f} '
             f'Validation Perplexity: {validation_ppl:.4f}')

  def predict(self, tokens, K=1, max_T=400):
    """
    Generates the target sequence given the source sequence using beam search decoding.
    Note that for simplicity, we only use batch size 1.
    Arguments:
        tokens: the source sentence.
        max_T: at most proceed this many steps of decoding
    Returns:
        a string of the generated target sentence.
    """
    # Tokenize and map to a list of word ids
    inputs = torch.LongTensor(self.tokenizer([tokens])['input_ids'][:1024]).to(device)
    # The `transformers` package provides built-in beam search support
    prediction = self.bart.generate(inputs,
                                    num_beams=K,
                                    max_length=max_T,
                                    early_stopping=True,
                                    no_repeat_ngram_size=0,
                                    decoder_start_token_id=49179,
                                    use_cache=True)[0]
    return self.tokenizer.decode(prediction, skip_special_tokens=True)