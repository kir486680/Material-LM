Character-level language model trained on 2 material science books(1695508 characters)


To use this model simple install the requirements file and run the Demo.ipynb notebook.
```bash
pip install -r requirements.txt
```

Hyperparameters:
emb_dim = 384 
context_length = 256 
n_heads = 6
n_layers = 8
batch_size = 512



Further Improvements:
- Right now the data quality is very bad. It could be improved a lot. 
- The model is trained on a very small dataset. It could be trained on a much larger dataset of more books/exams.
- The architecture could be improved by adding rotary embeddings for example. 