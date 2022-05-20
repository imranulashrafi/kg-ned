'''
This module implements the KGvec2go REST API.

KGvec2go is a semantic resource consisting of RDF2Vec knowledge graph 
embeddings trained currently on 4 different knowledge graphs.

For each concept/entity/word KGvec2go provides a word embedding which
can be used for different downstream tasks.

Reference: http://kgvec2go.org/
'''

import requests


def get_concept_embedding(concept):
    '''
    Gets the concept embedding from the kgvec2go graph embedding api.

    Parameters
    ----------

    concept: str
        The word for which the graph embedding will be returned.

    Returns
    -------

    vector: array_like
        An 200 dimension float array representing the embedding of
        the concept word.
    '''

    if type(concept) != str:
        return "Concept word should be of String type"

    url = f"http://kgvec2go.org/rest/get-vector/alod/{concept}"
    resposne = requests.get(url).json()

    if len(resposne) == 0:
        return None

    return resposne['vector']
