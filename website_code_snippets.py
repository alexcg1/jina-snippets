# Documents

from jina import Document, DocumentArray

doc1 = Document(text="hello world")
doc2 = Document(text="hi there planet")

docs = DocumentArray([doc1, doc2])


# Executors

from jina import Executor, requests, DocumentArray
import numpy as np

class CharEmbed(Executor):  # a simple character embedding with mean-pooling
    offset = 32  # letter `a`
    dim = 127 - offset + 1  # last pos reserved for `UNK`
    char_embd = np.eye(dim) * 1  # one-hot embedding for all chars

    @requests
    def encode(self, docs: DocumentArray, **kwargs):
        for doc in docs:
            r_emb = [ord(c) - self.offset if self.offset <= ord(c) <= 127 else (self.dim - 1) for c in doc.text]
            doc.embedding = self.char_embd[r_emb, :].mean(axis=0)  # average pooling

# Flows

from jina import Flow

flow = Flow().add(uses=CharEmbed).add(uses="jinahub+docker://SimpleIndexer")

with flow:
    flow.index()
    flow.search(Document(text="howdy earth"))
