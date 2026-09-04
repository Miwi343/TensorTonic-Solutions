class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        self.word_to_id[self.pad_token] = 0
        self.id_to_word[0] = self.pad_token
        self.word_to_id[self.unk_token] = 1
        self.id_to_word[1] = self.unk_token
        self.word_to_id[self.bos_token] = 2
        self.id_to_word[2] = self.bos_token
        self.word_to_id[self.eos_token] = 3
        self.id_to_word[3] = self.eos_token 

        vocab_words = set()
        for sentence in texts:
            vocab_words.update(sentence.lower().split())
        
        sorted_vocab = sorted(list(vocab_words))

        i = 3
        for word in sorted_vocab:
            if word in self.word_to_id:
                continue
            i+=1
            self.word_to_id[word] = i
            self.id_to_word[i] = word

        self.vocab_size = i + 1
            

    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """        
        words = text.lower().split()
        token_ids = []
        for word in words:
            id = self.word_to_id.get(word, 1)
            token_ids.append(id)

        return token_ids

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        output_list = []

        for id in ids:
            token = self.id_to_word.get(id, self.unk_token)
            output_list.append(token)

        return " ".join(output_list)
            