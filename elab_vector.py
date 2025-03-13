import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from elab_api import ElabFTWAPI
import json
import ollama
from pydantic import BaseModel
from typing import List

class WorkingMemory(BaseModel):
    role: str
    content: str

class MemoryFoam(BaseModel):
    messages: List[WorkingMemory] = []

class ElabVector:
    elab_client = None
    raw_data = None
    documents = None
    experiments_mapping = None
    embedding_model = None
    embeddings = None
    index = None
    text_data = None
    titles = []
    individual_summaries = []

    def __init__(self):
        self.elab_client = ElabFTWAPI(
            "https://prp-electronic-lab.areasciencepark.it/",
            "36-4a5923032a7feb980689671e57a5e1fb061c29b8bd2694d852f8b89c9d0fe083556194f593bdf2d4fa2a36"
        )
        # Retrieve and format experiments data
        self.raw_data = self.elab_client.get_experiments()

        # Print the list of experiments returned
        print(self.raw_data)

        self.titles = []
        for experiment in self.raw_data:
            self.titles.append(experiment['title'])

        self.documents = [
            f"ID: {entry['id']}, Title: {entry['title']}, Date: {entry['date']}, Body: {entry['body']}"
            for entry in self.raw_data
        ]

        # Create a mapping to store full experiment data by index
        self.experiments_mapping = {i: entry for i, entry in enumerate(self.raw_data)}

        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Convert text to embeddings
        self.embeddings = np.array([
            self.embedding_model.encode(doc)
            for doc in self.documents
        ], dtype=np.float32)

        # Store embeddings in FAISS for fast retrieval
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

        # Store original text for lookup
        self.text_data = {i: self.documents[i] for i in range(len(self.documents))}

    def retrieve_experiments_with_mention(self, mention: str, k: int = 10):
        """
        Retrieve experiments that are semantically similar to the given mention,
        then filter the results to include only experiments whose title or body
        actually contains the mention (if needed).
        """
        # Compute the embedding for the mention
        mention_embedding = self.embedding_model.encode(mention).reshape(1, -1)
        # Retrieve the top k experiments from the FAISS index
        _, indices = self.index.search(mention_embedding, k)

        filtered_experiments = []
        for idx in indices[0]:
            experiment_data = self.experiments_mapping[idx]
            # Optional: only append if mention actually appears
            if mention.lower() in experiment_data['title'].lower() or mention.lower() in experiment_data['body'].lower():
                filtered_experiments.append(experiment_data)

        return filtered_experiments

    def make_a_map(self, data, query):
        """
        Implementation of a 'Map-Reduce'-like approach:
        Summarize each experiment individually, then combine results.
        """
        client = ollama.Client(host='http://10.128.2.165:11434')

        self.individual_summaries = []
        for exp in data:
            if query:
                prompt = f"""
                Summarize the following experiment answering this query "{query}". The answer must not exceed 2 rows.
                ID: {exp['id']}
                Title: {exp['title']}
                Author: {exp['fullname']}
                Body: {exp['body']}
                """
            else:
                prompt = f"""
                Summarize the following experiment:
                ID: {exp['id']}
                Title: {exp['title']}
                Body: {exp['body']}
                """

            response = client.generate(
                model='llama3.3:latest',
                prompt=prompt,
                stream=False
            )

            summary_text = response['response'].strip()

            # Store each summary for later
            self.individual_summaries.append({
                'id': exp['id'],
                'summary': summary_text
            })

        # REDUCE STEP
        summaries_text = "\n".join([
            f"Experiment ID: {summ['id']}\nSummary: {summ['summary']}\n"
            for summ in self.individual_summaries
        ])

        summaries_json = json.dumps(self.individual_summaries, indent=2)

        # Write the JSON to an external file
        filename = 'query_summary.json' if query else 'summaries.json'
        with open(filename, "w") as json_file:
            json_file.write(summaries_json)

        return summaries_text

