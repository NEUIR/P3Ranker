from typing import List

from nltk import word_tokenize

from src.data.tokenizers import Tokenizer

class WordTokenizer(Tokenizer):
    def tokenize(self, text: str) -> List[str]:
        tokens = word_tokenize(text)
        return tokens
