Character-level language model trained on 2 material science books(1695508 characters).

This model was trained for about 10 hours on a one a100 GPU.


To use this model simply install the requirements file and run the Demo.ipynb notebook
```bash
pip install -r requirements.txt
```

# Hyperparameters:
- emb_dim = 384 
- context_length = 256 
- n_heads = 6
- n_layers = 8
- batch_size = 512

# Architectrue:
Nothing too fancy 
```Transformer(
  (token_embedding): Embedding(104, 384)
  (positional_encoding): PositionalEncoding()
  (layers): ModuleList(
    (0-7): 8 x TransformerBlock(
      (multihead): CausalSelfAttention(
        (c_attn): Linear(in_features=384, out_features=1152, bias=True)
        (c_proj): Linear(in_features=384, out_features=384, bias=True)
        (attn_dropout): Dropout(p=0.2, inplace=False)
        (resid_dropout): Dropout(p=0.2, inplace=False)
      )
      (layer_norm1): RMSNorm()
      (layer_norm2): RMSNorm()
      (ffd): Sequential(
        (0): Linear(in_features=384, out_features=1536, bias=True)
        (1): GELU(approximate='none')
        (2): Linear(in_features=1536, out_features=384, bias=True)
        (3): Dropout(p=0.2, inplace=False)
      )
    )
  )
  (ffd): Linear(in_features=384, out_features=104, bias=True)
```

# Further Improvements:
- Right now the data quality is very bad. It could be improved a lot. 
- The model is trained on a very small dataset. It could be trained on a much larger dataset of more books/exams.
- The architecture could be improved by adding rotary embeddings for example. 
- and a lot more...