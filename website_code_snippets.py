# Documents

from jina import Document, DocumentArray

doc1 = Document(text="hello world")
doc2 = Document(text="hi there planet")

docs = DocumentArray([doc1, doc2])


# Executors

from jina import Executor, requests, DocumentArray
import numpy as np

class NumpyEmbed(Executor):
    @requests
    def encode(self, docs, **kwargs):
        for d in docs:
            d.embedding = np.random.random(10)


# Flows

from jina import Flow

flow = Flow().add(uses=NumpyEmbed).add(uses="jinahub+docker://SimpleIndexer")

with flow:
    flow.index(docs)
    flow.search(Document(text="howdy earth"))
