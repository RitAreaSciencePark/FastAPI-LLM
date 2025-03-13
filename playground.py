import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from elab_api import ElabFTWAPI
import json

class ElabVector:
    elab_client = None
    raw_data = None
    documents = None
    experiment_mapping = None
    embedding_model = None
    embeddings = None
    index = None
    text_data = None

    def __init__(self):
        self.elab_client = ElabFTWAPI("https://prp-electronic-lab.areasciencepark.it/", "36-4a5923032a7feb980689671e57a5e1fb061c29b8bd2694d852f8b89c9d0fe083556194f593bdf2d4fa2a36")

        # Retrieve and format experiments data
        self.raw_data = self.elab_client.get_experiments()
        self.documents = [
            f"ID: {entry['id']}, Title: {entry['title']}, Date: {entry['date']}, Body: {entry['body']}"
            for entry in self.raw_data
        ]
    """
        # Create a mapping to store full experiment data by index
        self.experiments_mapping = {i: entry for i, entry in enumerate(self.raw_data)}

        # Initialize embedding model
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        # Convert text to embeddings
        self.embeddings = np.array([self.embedding_model.encode(doc) for doc in self.documents], dtype=np.float32)

        # Store embeddings in FAISS for fast retrieval
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

        # Store original text for lookup
        self.text_data = {i: self.documents[i] for i in range(len(self.documents))}
"""

def try_me_cosine(data):
    # Implementation of Strategy 3: Embeddings + Similarity Search with Ollama

    import ollama
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    # Initialize Ollama client
    client = ollama.Client(host='http://10.128.2.165:11434')

    # List of experiment data
    #data = [
    #    {'id': 154, 'title': 'ORID0036_Neuromed_Takara_scell', 'body': 'Details of experiment 154...'},
    #    {'id': 155, 'title': 'Experiment XYZ', 'body': 'Details of experiment 155...'},
        # Add more experiments
    #]

    # Generate embeddings for each experiment
    experiment_texts = [exp['body'] for exp in data]
    embeddings = [client.embeddings(model='llama3.3:latest', prompt=text)['embedding'] for text in experiment_texts]

    # Query from user
    query = "Summarize the experiments with Nanopore."

    # Generate embedding for the query
    query_embedding = client.embeddings(model='llama3.3:latest', prompt=query)['embedding']

    # Calculate similarities
    similarities = cosine_similarity([query_embedding], embeddings)[0]

    # Find top N most similar experiments
    N = 10  # adjust based on your preference
    similar_indices = similarities.argsort()[-N:][::-1]

    # Build a prompt with most relevant experiments
    relevant_experiments = "".join(
        [f"Experiment ID: {data[idx]['id']}\nDetails: {data[idx]['body']}\n\n" for idx in similar_indices])

    final_prompt = f"""
    Given the following experiments:
    {relevant_experiments}
    Answer the query: {query}
    """

    # Invoke the LLM
    response = client.generate(
        model='llama3.3:latest',
        prompt=final_prompt,
        stream=False
    )

    # Print the response
    print(response['response'])

def make_a_map(data, query):
    # Implementation of the "Mapping" approach (sometimes called Map-Reduce)
    # Using Ollama to summarize each experiment individually, then combining results.

    import ollama

    # Initialize Ollama client
    client = ollama.Client(host='http://10.128.2.165:11434')

    # List of experiment data
    """
    data = [
        {
            'id': 154,
            'title': 'ORID0036_Neuromed_Takara_scell',
            'body': 'Details of experiment 154...'
        },
        {
            'id': 155,
            'title': 'Experiment XYZ',
            'body': 'Details of experiment 155...'
        },
        # Add more experiments as needed
    ]
    """

    ############################
    # MAP STEP
    # Summarize each experiment individually
    ############################

    individual_summaries = []

    for exp in data:
        # Build a prompt for each experiment
        if query is not None and query != "":
            prompt = f"""
            Summarize the following experiment answering this query "{query}", if you do not find valuable information, return the title and information not preset:
            ID: {exp['id']}
            Title: {exp['title']}
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
            model='llama3.3:latest',  # or your preferred model
            prompt=prompt,
            stream=False
        )

        summary_text = response['response'].strip()
        print(summary_text)

        # Store the summary for later use
        individual_summaries.append({
            'id': exp['id'],
            'summary': summary_text
        })

    ############################
    # REDUCE STEP
    # Combine individual summaries into a single, aggregated analysis
    ############################

    # We'll build a prompt that includes all individual summaries,
    # then ask for a combined analysis or final summary.

    summaries_text = "\n".join([
        f"Experiment ID: {summ['id']}\nSummary: {summ['summary']}\n" for summ in individual_summaries
    ])

    summaries_json = json.dumps(individual_summaries, indent=2)

    # Write the JSON to an external file

    if query is not None and query != "":
        filename = 'query_summary.json'
    else:
        filename = 'summaries.json'

    with open(filename, "w") as json_file:
        json_file.write(summaries_json)


def query_the_summary(query, filepath):

    with open(filepath) as fp:
        summaries_text = json.load(fp)
    import ollama

    # Initialize Ollama client
    client = ollama.Client(host='http://10.128.2.165:11434')
    reduce_prompt = f"""
    Below are individual experiment summaries:

    {summaries_text}
    
     with this knowledge answer this user query: {query}.
    """

    final_response = client.generate(
        model='llama3.3:latest',
        prompt=reduce_prompt,
        stream=False
    )

    print("\n===== AGGREGATED SUMMARY =====")
    print(final_response['response'])


if __name__ == "__main__":
    import ollama
    client = ollama.Client(host='http://10.128.2.165:11434')

    initial_query = "I need to make a data management plan using data from the experiments"
    llm_input = f"the user searching for something in this query: {initial_query}, extract the filter/condition, write only the content of filter/condition"
    query = client.generate(
        model='llama3.3:latest',
        prompt=llm_input,
        stream=False
    )
    print(query['response'])
    elab_vector = ElabVector()
    print(elab_vector.raw_data)
    make_a_map(elab_vector.raw_data, query['response'])
    query_the_summary(initial_query, 'query_summary.json')
    # try_me_maps(elab_vector.raw_data)
#    mention = "manina"
#    vector = ElabVector()
#    experiments = vector.retrieve_experiments_with_mention(mention, k=10)
#    for experiment in experiments:
#        print(experiment)
