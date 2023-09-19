import re
from flask import abort
import json
import requests
from components.Logger import Logger


logger = Logger('QueryService')


class QueryService:
    # To add support to more endpoints, add here:
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

    @classmethod
    def get_class_entities_query(cls, source, class_, limit=None, query_path='data/query.txt'):
        limit_str = 'LIMIT %s' % limit if limit else ''
        if 'dbpedia' or 'tib-lucat' or 'tib-node2' or 'wikidata' in source:
            query = ''
            with open(query_path, 'rt') as f:
                query = f.read()
            return query
        elif 'TIB' in source:
            return '''
            select distinct ?s where {?s ?p <http://p4-lucat.eu/vocab/Disorder>}
            '''

    @classmethod
    def get_class_entities(cls, source, class_, limit, query_path='data/query.txt'):

        endpoint = cls.endpoints[source]
        query = cls.get_class_entities_query(source, class_, limit, query_path)

        s = requests.Session()

        headers = {
            'Accept': 'application/json'
        }
        data = {'query': query}
        s.headers.update(headers)

        response = s.post(endpoint, data=data, headers=headers)

        if response.status_code != 200:
            logger.log('Query to endpoint %s, returned code %s' % (endpoint, response.status_code))
            logger.log(response.text)
            abort(400, response.text)

        content = json.loads(response.text)

        if 'results' not in content or 'bindings' not in content['results']:
            logger.log('Query for class %s returned no entities' % class_)
            abort(400)

        results = content['results']['bindings']

        if len(results[0].keys()) == 1:
            entities_uris = [result['entity']['value'] for result in results]
        else:
            entities_uris = []
            for result in results:
                row = []
                for key in result.keys():
                    row.append(result[key]['value'])
                entities_uris.append(row)

        return entities_uris