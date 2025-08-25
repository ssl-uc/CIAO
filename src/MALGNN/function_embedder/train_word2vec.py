import os
import random
from gensim.models import Word2Vec



class ChunkedESILInstructions:
    def __init__(self, chunk_dir, chunk_ratio=1.0, shuffle=True, seed=42):
        self.chunk_dir = chunk_dir
        self.chunk_files = sorted(f for f in os.listdir(chunk_dir) if f.endswith(".pkl"))
        self.seed = seed

        if shuffle:
            random.seed(seed)
            random.shuffle(self.chunk_files)

        if chunk_ratio < 1.0:
            keep_n = max(1, int(len(self.chunk_files) * chunk_ratio))
            self.chunk_files = self.chunk_files[:keep_n]

    def __iter__(self):
        for fname in self.chunk_files:
            path = os.path.join(self.chunk_dir, fname)
            try:
                with open(path, "rb") as f:
                    chunk_data = pickle.load(f)
                    for sentence in chunk_data["data"]:
                        if isinstance(sentence, list) and any(t.strip() for t in sentence):
                            yield sentence
            except Exception as e:
                print(f"Failed to load {fname}: {e}")
                
       
def train_word2vec_on_chunks(
    chunk_dir,
    vector_size=100,
    window=4,
    min_count=1,
    epochs=10,
    chunk_ratio=1,
    seed=42,
    save_path="model_cbow_custom/esil_w2v.model"
):
    print("Initializing iterator over instruction chunks...")
    sentences = lambda: ChunkedESILInstructions(chunk_dir, chunk_ratio=chunk_ratio, seed=seed)

    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0, 
        workers=4  
    )

    print("Building vocabulary...")
    model.build_vocab(sentences())

    print(f"Training for {epochs} epochs...")
    model.train(sentences(), total_examples=model.corpus_count, epochs=epochs)

    model.save(save_path)
    print(f"Model saved to: {save_path}")
    
def train(chunk_dir, vector_size, windows, min_count, epochs, save_path):
    train_word2vec_on_chunks(
    chunk_dir=chunk_dir,
    vector_size=vector_size,
    window=window,
    min_count=min_count,
    epochs=epochs,
    chunk_ratio=1,
    save_path=save_path)
