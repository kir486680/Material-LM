from tokenizers import Tokenizer, models, pre_tokenizers, decoders
import json 
class CustomTokenizer:
    def __init__(self, chars=None, json_file=None):
        if json_file:
            with open(json_file, 'r') as f:
                mapping = json.load(f)
                self.stoi = mapping['stoi']
                self.itos = mapping['itos']
        else:
            self.stoi = {ch: i for i, ch in enumerate(chars)}
            self.itos = {i: ch for i, ch in enumerate(chars)}

        self._tokenizer = Tokenizer(models.Unigram())
        self._tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        self._tokenizer.decoder = decoders.WordPiece()

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, tokens):
    
        return ''.join([self.itos[str(i)] for i in tokens])

    def save(self, path, name):
        self._tokenizer.save(f"{path}/{name}")
        mapping = {'stoi': self.stoi, 'itos': self.itos}
        with open(f"{path}/{name}_mapping.json", 'w') as f:
            json.dump(mapping, f)

    def load(self, path):
        self._tokenizer = Tokenizer.from_file(path)
