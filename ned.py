'''
This module implements the Named Entity Disambiguation model.

Given a keyword and a list of candidate entities, the NED model first
fetches the graph embeddings for all the candidates as well as the 
keyword and calculates the similarity between them. From the similarities,
the top N matches are then returned as the target entities.
'''

import torch
import torch.nn as nn
import numpy as np
from kgvec2go import get_concept_embedding


class NED:
    def calculate_similarity(self, emb1, emb2):
        '''
        Calculates the similarity score between two embeddings.

        Parameters
        ----------

        emb1: array_like
            One dimensional word embedding.

        emb3: array_like
            One dimensional word embedding.

        Returns
        -------

        score: float
            The similarity score between two embedings.
        '''

        cos = nn.CosineSimilarity(dim=1)
        return round(cos(emb1, emb2).item(), 2)

    def get_word_embedding(self, word):
        '''
        Preprocess and split words and return mean embeddings.

        Parameters
        ----------

        word: str
            The word for which the graph embedding will be returned.

        Returns
        -------

        vector: array_like
            An 200 dimension float array representing the mean embedding
            of the word.
        '''

        if len(word.split(' ')) > 1:
            splitted_word = word.split(' ')
            splitted_word_embeddings = []

            for word in splitted_word:
                embedding = get_concept_embedding(word)
                if embedding is not None:
                    splitted_word_embeddings.append(embedding)

            if len(splitted_word_embeddings) > 0:
                return np.mean(splitted_word_embeddings)
            else:
                return None
        else:
            return get_concept_embedding(word)

    def predict_entities(self, concept_entities, keyword):
        '''
        Returns top entities scores in sorted order given a keyword by 
        extracting the graph embeddings for each candidate and keyword and
        calculating the similarity between them.

        Parameters
        ----------

        concept_entities: array_like
            List of string words which are candidate entities for keyword.

        keyword: str


        Returns
        -------

        vector: array_like
            An 200 dimension float array representing the mean embedding
            of the word.
        '''

        candidate_entities = concept_entities[concept_entities['entity'] == keyword]
        candidate_entities = list(candidate_entities['concept'])

        keyword_embedding = self.get_word_embedding(keyword)

        if keyword_embedding is None:
            return "Keyword embedding was not found"
        keyword_embedding = torch.tensor(keyword_embedding).unsqueeze(0)

        entity_scores = {}

        for candidate in candidate_entities:
            candidate_embedding = self.get_word_embedding(candidate)

            if candidate_embedding is not None:
                candidate_embedding = torch.tensor(
                    candidate_embedding).unsqueeze(0)
                entity_scores[candidate] = self.calculate_similarity(
                    candidate_embedding, keyword_embedding)

        if len(entity_scores) == 0:
            return "No concept entity embedding was found"

        entity_scores = {k: v for k, v in sorted(
            entity_scores.items(), key=lambda item: item[1], reverse=True)}

        return entity_scores
