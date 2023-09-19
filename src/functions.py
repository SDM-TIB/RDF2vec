import requests
import json
import pandas as pd
import csv
import time
import re

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.embedders import Word2Vec
from pyrdf2vec.graphs import KG
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.walkers import WLWalker
from pyrdf2vec.walkers import HALKWalker



from components.Logger import Logger

logger = Logger('extract - function')

endpoints = {
    'tib-wikidata': 'http://node3.research.tib.eu:4010/sparql',
    'tib-wikidata-2': 'http://node3.research.tib.eu:4012/sparql',
    'wikidata': 'https://query.wikidata.org/sparql',
    'dbpedia': 'https://dbpedia.org/sparql',
    'tib-dbpedia': 'http://node3.research.tib.eu:4002/sparql',
    'tib-lucat': 'https://labs.tib.eu/sdm/p4-lucat-v2/sparql',
    'tib-node2': 'http://node2.research.tib.eu:41111/sparql',
    'clarify': 'https://labs.tib.eu/sdm/clarify_kg/sparql',
    'clarify1': 'https://labs.tib.eu/sdm/sub_clarify_v1.0/sparql'
}


def extract(source, max_depth, max_walks, walker, size, epochs, entities_path='results/entities_from_query.csv'):
    df = pd.read_csv(entities_path,
                     sep=',',
                     error_bad_lines=False, encoding='cp1252')
    
    entities = df['Unique_Entities'].tolist()
    entities = entities[:]
    print(entities)
    
    

    # %%
    # print(entities)

    # %%
    walkersM=[RandomWalker(max_depth, max_walks, with_reverse=False, n_jobs=2)]
    if(walker == 1):
        walkersM=[RandomWalker(max_depth, max_walks, with_reverse=False, n_jobs=2)]
    elif(walker==2):
        walkersM=[HALKWalker(max_depth, max_walks, with_reverse=False, n_jobs=2)]
    elif(walker==3):
        walkersM=[WLWalker(4, 10, with_reverse=False, n_jobs=2)]

    transformer = RDF2VecTransformer(
        Word2Vec(epochs=epochs, vector_size=size),
        walkers=walkersM,
        verbose=1
    )
    
    logger.log('-------- Extracting Embeddings...')
    embeddings, literals = transformer.fit_transform(
        KG(
            endpoints[source], is_remote=True,
            literals=[
            ],
            skip_predicates={
            }
        ),
        entities
    )
    logger.log('--------- Done')

    # %%

    # print(embeddings)

    # parse to list (json-serializable) before return
    embeddings = [embedding.tolist() if not isinstance(embedding, list) else embedding for embedding in embeddings]
    return entities, literals, embeddings
