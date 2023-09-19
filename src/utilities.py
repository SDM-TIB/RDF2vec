'''
All necessary utility functions are put in this module. 
'''

from QueryService import QueryService
from functions import extract

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import pandas as pd
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import requests
import json
import networkx as nx

def savefig_embeddings(embeddings):
    embeddings = np.asarray(embeddings)
    embeddings = embeddings.T
    viridis = cm.get_cmap('Greens', 512)
    cmp = ListedColormap(viridis(np.linspace(0.15, 1.0, 256)))
    am = plt.matshow(embeddings, cmap=cmp)
    plt.colorbar(am)
    plt.title('Embeddings')
    plt.savefig('results/embeddings.png')

def savefig_sim_mat(sim_mat):
    viridis = cm.get_cmap('plasma', 512)
    cmp = ListedColormap(viridis(np.linspace(0.15, 1.0, 256)))
    am = plt.matshow(sim_mat, cmap=cmp)
    plt.colorbar(am)
    plt.title('Similarity')
    plt.savefig('results/sim_mat.png')

def similarity_calculation(vectors):
    vec = np.asarray(vectors)
    vec_normalized = np.zeros(vec.shape)
    for i in range(vec.shape[0]):
        vec_normalized[i] = vec[i] / np.sqrt(vec[i] @ vec[i].T)
    sim_mat = np.abs(vec_normalized @ vec_normalized.T)

    return sim_mat.tolist()

# for all entities of sim_mat, entities are list of strings
def find_most_similar(sim_mat, entities, threshold=0.7):
    sim_mat = np.array(sim_mat)
    most_similar_dict = dict((entity, []) for entity in entities)
    row, col = np.where((threshold < sim_mat) & (sim_mat < 1 - 1e-4))
    for r, c in zip(row, col):
        most_similar_dict[entities[r]].append(entities[c])

    return most_similar_dict

# for one specific entity, entity without brackets
def most_similar_entities(sim_mat, entity, threshold=0.7):
    sim_mat = np.array(sim_mat)
    sim_table = pd.read_csv('results/similarity.csv')
    row_num = sim_table[sim_table['0'] == entity].index.values[0]  # get the row number of entity in sim_mat
    row = sim_mat[row_num, :]
    col = np.where((threshold < row) & (row < 1 - 1e-4))

    entities = []
    for c in col[0]:
        entities.append(sim_table['0'][c])
    
    return entities

# choose num entities with highest similarity, entity without angle brackets
def highest_similar_entities(sim_mat, entity, num=4):
    sim_mat = np.array(sim_mat)
    sim_table = pd.read_csv('results/similarity.csv')
    row_num = sim_table[sim_table['0'] == entity].index.values[0]  # get the row number of entity in sim_mat
    row = sim_mat[row_num, :]
    row_dict = {}
    for e, v in zip(sim_table['0'].values, row):
        row_dict[e] = v
    entities = list({k: v for k, v in sorted(row_dict.items(), key=lambda item: item[1], reverse=True)[1:num+1]}.keys())

    return entities

# entities are list of strings
def sim_mat_to_csv(sim_mat, entities):
    sim_mat = np.asarray(sim_mat)
    tabular = np.empty((sim_mat.shape[0], sim_mat.shape[1] + 1), dtype=object)
    for i in range(sim_mat.shape[0]):
        tabular[i][0] = entities[i]
        for j in range(1, sim_mat.shape[1] + 1):
            tabular[i][j] = entities[j - 1] + '(' + str(sim_mat[i][j - 1]) + ')'
    tabular = pd.DataFrame(tabular)

    return tabular.to_csv('results/similarity.csv')

# concatenate new queries to original query, entity_to_replace without angle brackets
def query_expansion(query, entity_to_replace, entities_to_add=[]):
    if type(query) is not str:
        raise TypeError('query must be string!')
    # add angle brackets to the endpoints of entities
    if entity_to_replace[0] != '<':
        entity_to_replace = '<' + entity_to_replace + '>'
    for i in range(len(entities_to_add)):
        entities_to_add[i] = '<' + entities_to_add[i] + '>'
    new_query = query
    part_within_braces = re.findall('\{.*\}', query, flags=re.DOTALL)[0]
    for entity in entities_to_add:
        new_query = new_query + '\nunion\n' + part_within_braces.replace(entity_to_replace, entity)
    slice_point = [match for match in re.finditer('\{', new_query)][0].start()-1
    new_query = new_query[:slice_point] + '{' + new_query[slice_point:] + '}'
    
    return new_query

# returned entity without angle brackets
def extract_last_entity_from_query(query):
    return re.findall('<.*>', query)[-1][1:-1]

# function to generate new_query.txt from query.txt
def generate_new_query(sim_mat, entities, num=4, original_query='data/query1.txt'):
    sim_mat_to_csv(sim_mat, entities)
    with open(original_query, 'rt') as f:
        query = f.read()
    add_entities = highest_similar_entities(sim_mat, extract_last_entity_from_query(query), num=num)
    a = query_expansion(query, extract_last_entity_from_query(query), add_entities)
    with open('results/new_query.txt', 'wt') as f:
        f.write(a)

# get entities of certain number from .csv file
def get_entities_from_csv(path, start=None, end=None):
    df = pd.read_csv(path, sep=',', error_bad_lines=False, encoding='cp1252')
    entities = df['Unique_Entities'].tolist()
    if start is not None and end is not None:
        entities = entities[start:end]

    return entities

# save sim_mat as sim_mat.npy file in data directory
def save_sim_mat(sim_mat):
    np.save('data/sim_mat.npy', sim_mat)

# read sim_mat from sim_mat.npy file in data directory
def read_sim_mat():
    return np.load('data/sim_mat.npy')

# write entities queried from query.txt or new_query.txt to a csv file
def write_entities_to_csv(entities, path_to_write='results/entities_from_query.csv', write_mode='wt'):
    with open(path_to_write, write_mode) as f:
        if isinstance(entities[0], str):
            if write_mode == 'wt':
                f.write('Unique_Entities\n')
            for entity in entities:
                f.write(entity + '\n')
        else:
            title = ''
            for i in range(len(entities[0])):
                title += 'Unique_Entities,'
                f.write(title[:-1])
            f.write('\n')
            for row in entities:
                str_to_write = ''
                for entity in row:
                    str_to_write += entity + ','
                f.write(str_to_write[:-1] + '\n')

def query_and_write_entities_to_csv(source, query_path='data/query.txt', path_to_write='results/entities_from_query.csv', write_mode='wt'):
    entities = QueryService.get_class_entities(source, None, None, query_path=query_path)
    write_entities_to_csv(entities, path_to_write=path_to_write, write_mode=write_mode)

def query_expansion_pipeline(source, max_depth, max_walks, walker, size, epochs, query_path='data/query.txt', query1_path='data/query1.txt', path_to_write='results/entities_from_query.csv'):
    # retrieve entities by quering query.txt
    # query.txt only to get all properties, which are used for query expansion
    # the query need to be expanded is in query1.txt
    query_and_write_entities_to_csv(source, query_path, path_to_write)
    query_and_write_entities_to_csv(source, query1_path, 'results/entities_from_query1.csv')
    entities, embeddings = extract(source,
                                max_depth,
                                max_walks,
                                walker,
                                size,
                                epochs,
                                path_to_write)
    sim_mat = similarity_calculation(embeddings)
    savefig_sim_mat(sim_mat)
    save_sim_mat(sim_mat)
    # sim_mat = read_sim_mat()
    generate_new_query(sim_mat, entities, num=4, original_query=query1_path)
    # retrieve entities by quering new_query.txt
    query_and_write_entities_to_csv(source, query_path='results/new_query.txt', path_to_write='results/entities_from_new_query.csv')
    # entities, embeddings = extract(source,
    #                             max_depth,
    #                             max_walks,
    #                             walker,
    #                             size,
    #                             epochs,
    #                             'results/entities_from_new_query.csv')

# write all vectorial data into csv file
def write_vector_data_to_csv(data, path='results/vector_data.csv'):
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)

# read all vectorial data from csv file and convert to ndarray
def read_vector_data_from_csv(path='results/vector_data.csv'):
    df = pd.read_csv(path)
    data = df.to_numpy()
    return data

# PCA analysis
def PCA_analysis(data, component_num=2, visualize=False):
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)
    standardized_data = StandardScaler().fit_transform(data)
    pca = PCA(n_components=component_num)
    reduced_dim_data = pca.fit_transform(standardized_data)

    if visualize:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(reduced_dim_data[:, 0], reduced_dim_data[:, 1])
        fig.savefig('results/PCA_result.png')

    return reduced_dim_data

# generate all graph files that are mandatory for semEP
def generate_bigraph(vertices1: list, vertices2: list, edges: list, outdir='/mnt/c/Users/SongZ/Downloads/repositories/semep/test/p4lucat/', vec1=None, vec2=None):
    """
    Generate all graph files that are mandatory for semEP
    :param vertices1: a list contains elements of type MyOWLLogicalEntity
    :param vertices2: a list contains elements of type MyOWLLogicalEntity
    :param edges: a list conatins tuples, which describe the edge between vertices
                    for example: [(vertex in vertices1, vertex in vertices2, weight of edge)]
    :param outdir: the directory of output files
    
    :return: None
    """

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # path of all necessary files
    vertices1_file = outdir + 'vertices1.txt'
    vertices1_simmat_file = outdir + 'vertices1_simmat.txt'
    vertices2_file = outdir + 'vertices2.txt'
    vertices2_simmat_file = outdir + 'vertices2_simmat.txt'
    bipartite_graph_file = outdir + 'bigraph.txt'

    with open(vertices1_file, 'wt') as v1f:
        v1f.write('{:d}\n'.format(len(vertices1)))
        for v1 in vertices1[:-1]:
            v1f.write('{}\n'.format(str(v1)))
        v1f.write('{}'.format(str(vertices1[-1])))
    
    with open(vertices1_simmat_file, 'wt') as v1sf:
        v1sf.write('{:d}\n'.format(len(vertices1)))
        if vec1 is None:
            for v11 in vertices1:
                for v12 in vertices1:
                    sim = 1.0
                    v1sf.write('{:.6f} '.format(sim))
                    # print(str(v11), str(v12), sim)
                v1sf.write('\n')
        else:
            for v11 in vec1:
                for v12 in vec1:
                    v1, v2 = np.array(v11), np.array(v12)
                    v1_n, v2_n = v1 / np.sqrt(v1 @ v1.T), v2 / np.sqrt(v2 @ v2.T)
                    sim = np.abs(v1_n @ v2_n.T)
                    v1sf.write('{:.6f} '.format(sim))
                    # print(str(v11), str(v12), sim)
                v1sf.write('\n')
    
    with open(vertices2_file, 'wt') as v2f:
        v2f.write('{:d}\n'.format(len(vertices2)))
        for v2 in vertices2[:-1]:
            v2f.write('{}\n'.format(str(v2)))
        v2f.write('{}'.format(str(vertices2[-1])))
    
    with open(vertices2_simmat_file, 'wt') as v2sf:
        v2sf.write('{:d}\n'.format(len(vertices2)))
        if vec2 is None:
            for v21 in vertices2:
                for v22 in vertices2:
                    sim = 1.0
                    v2sf.write('{:.6f} '.format(sim))
                    # print(str(v11), str(v12), sim)
                v2sf.write('\n')
        else:
            for v21 in vec2:
                for v22 in vec2:
                    v1, v2 = np.array(v21), np.array(v22)
                    v1_n, v2_n = v1 / np.sqrt(v1 @ v1.T), v2 / np.sqrt(v2 @ v2.T)
                    sim = np.abs(v1_n @ v2_n.T)
                    v2sf.write('{:.6f} '.format(sim))
                    # print(str(v11), str(v12), sim)
                v2sf.write('\n')
    
    with open(bipartite_graph_file, 'wt') as bgf:
        bgf.write('{:d}\n'.format(len(edges)))
        for v1, v2, w in edges:
            if w >= 0.0:
                bgf.write('{}\t{}\t{:.8f}\n'.format(str(v1), str(v2), w))

# plot communities on 2D coordinate system
def plot_communities(entity, embed, community_dir):
    """
    params:
        entity: entity e.g. 'http://...'
        embed: embedding e.g. [0.2, 0.3, ...]
        community_dir: directory of files that represent communities
    return:
        None
    """
    embeddings = np.array(embed)
    ### dimension reduction ###
    standardized_data = StandardScaler().fit_transform(embeddings)
    pca = PCA(n_components=2)
    reduced_dim_data = pca.fit_transform(standardized_data)
    entity_2d = {}  # dict stores entity embedding pairs
    for i in range(len(entity)):
        entity_2d[entity[i]] = reduced_dim_data[i]
    
    ### get communities from files ###
    com_files = os.listdir(community_dir)
    com_num = len(com_files)  # number of communities
    types = []
    colormap = []
    for i, cf in enumerate(com_files):
        entity_set = set()
        with open(community_dir + '/' + cf) as cfile:
            lines = cfile.readlines()
            for line in lines:
                entity_set.update(line.split('\t')[:2])
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.set_title('Community ' + str(i+1))
        points = []
        categories = []
        for e in entity_set:
            # query over endpoint
            q = '''
                    PREFIX clarify: <http://research.tib.eu/clarify2020/vocab/>
                    PREFIX clarifyE: <http://research.tib.eu/clarify2020/entity/>
                    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                    select distinct ?entity
                    where {
                        $e$ a clarify:LCPatient.
                        $e$ clarify:hasFamilyHistory ?family.
                        ?family clarify:familyType ?entity.
                    }
                '''.replace('$e$', '<' + e + '>')
            s = requests.Session()
            headers = {
                'Accept': 'application/json'
            }
            data = {'query': q}
            s.headers.update(headers)
            endpoints = {
                'clarify': 'https://labs.tib.eu/sdm/clarify_kg/sparql',
                'clarify1': 'https://labs.tib.eu/sdm/sub_clarify_v1.0/sparql'
            }
            endpoint = endpoints['clarify']
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
            
            # add category to point
            if len(entities_uris) == 1:
                uri = entities_uris[0]
                if uri in types:
                    points.append(entity_2d[e])
                    idx = types.index(uri)
                    categories.append(idx)
                else:
                    points.append(entity_2d[e])
                    types.append(uri)
                    categories.append(len(types) - 1)
                    colormap.append(np.random.random(3))
            else:
                for uri in entities_uris:
                    if uri in types:
                        points.append(entity_2d[e] + np.random.random(2) * 8e-2)
                        idx = types.index(uri)
                        categories.append(idx)
                    else:
                        points.append(entity_2d[e])
                        types.append(uri)
                        categories.append(len(types) - 1)
                        colormap.append(np.random.random(3))
        points = np.array(points)
        m = ax.scatter(points[:, 0], points[:, 1], s=30, c=np.array(colormap)[np.array(categories)])

        # plot colormap
        ax = fig.add_subplot(2, 1, 2)
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())  # get bbox of ax
        ax_width, ax_height = bbox.width, bbox.height  # width and height of ax
        ax.set_axis_off()
        n_row, n_col = 3, 3
        r_width, r_height = 1.0 / (2 * n_col + 1), 1.0 / (2 * n_row + 1)
        for i, cm in enumerate(colormap):
            r, c = np.floor(i / n_col), i % n_col
            ax.add_patch(
                Rectangle(xy=((2*c+1)*r_width, (2*r+1)*r_height), width=r_width, height=r_height, facecolor=cm, edgecolor='0.7')
            )
            t = types[i]
            cname = t[(t.rindex('/')+1):]
            ax.text((2*c+1.5)*r_width, (2*r+0.5)*r_height, cname, fontsize=8, horizontalalignment='center', verticalalignment='center')

        fig_path = community_dir + '/' + cf[:-4] + '.png'
        fig.savefig(fig_path)

# print info of communities
def print_communities(entity, community_dir, tofile=None):
    if tofile:
        f = open(tofile, 'wt')

    ### get communities from files ###
    com_files = os.listdir(community_dir)
    com_num = len(com_files)  # number of communities
    types = []
    for i, cf in enumerate(com_files):
        entity_set = set()
        with open(community_dir + '/' + cf) as cfile:
            lines = cfile.readlines()
            for line in lines:
                entity_set.update(line.split('\t')[:2])
        if tofile:
            print(f'Community {i}', file=f)
            print('---------------------------------------------', file=f)
        print(f'Community {i}')
        print('---------------------------------------------')
        
        points = []
        categories = []
        for e in entity_set:
            # query over endpoint
            q = '''
                    PREFIX clarify: <http://research.tib.eu/clarify2020/vocab/>
                    PREFIX clarifyE: <http://research.tib.eu/clarify2020/entity/>
                    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                    select distinct ?entity, ?smokingHabits, ?Stages, ?bio, ?gender, ?relapseprogression
                    where {
                        $e$ a clarify:LCPatient.
                        $e$ clarify:hasFamilyHistory ?family.
                        ?family clarify:hasFamilyCancerType ?entity.
                        $e$ clarify:hasDiagnosis ?date.
                        ?date clarify:hasDiagnosisStage ?Stages.
                        # $e$  clarify:age ?Age.
                        $e$ clarify:hasSmokingHabit ?smokingHabits.
                        $e$ clarify:hasBio ?bio.
                        $e$ clarify:sex ?gender.
                        $e$ clarify:hasProgressionRelapseAssessment ?rp.
                        ?rp clarify:hasProgressionOrRelapse ?relapseprogression.
                    }
                '''.replace('$e$', '<' + e + '>')
            s = requests.Session()
            headers = {
                'Accept': 'application/json'
            }
            data = {'query': q}
            s.headers.update(headers)
            endpoints = {
                'clarify': 'https://labs.tib.eu/sdm/clarify_kg/sparql',
                'clarify1': 'https://labs.tib.eu/sdm/sub_clarify_v1.0/sparql'
            }
            endpoint = endpoints['clarify']
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
            
            # add category to entity
            if isinstance(entities_uris[0], str):
                for uri in entities_uris:
                    uri = uri[(uri.rindex('/')+1):]
                    cat = []
                    if uri in types:
                        idx = types.index(uri)
                        cat.append(idx)
                    else:
                        types.append(uri)
                        cat.append(len(types) - 1)
                    categories.append(tuple(cat))
            elif isinstance(entities_uris[0], list):
                for uris in entities_uris:
                    uris = [u[(u.rindex('/')+1):] for u in uris]
                    uris2str = ', '.join(uris)
                    cat = []
                    if uris2str in types:
                        idx = types.index(uris2str)
                        cat.append(idx)
                    else:
                        types.append(uris2str)
                        cat.append(len(types) - 1)
                    categories.append(tuple(cat))
            
        # numerate entities wrt. each category
        cat2num = {}
        for cat in categories:
            for c in cat:
                t = types[c]
                if t not in cat2num:
                    cat2num[t] = 1
                else:
                    cat2num[t] += 1
        
        # print cat2num
        if tofile:
            for cat, n in cat2num.items():
                print(f'{cat}: {n}', file=f)
        
            print('---------------------------------------------', file=f)
        for cat, n in cat2num.items():
            print(f'{cat}: {n}')
        
        print('---------------------------------------------')
    
    if tofile:
        f.close()

# plot communities in bar
def bar_communities(entity, community_dir, top_n=2):
    # delete *.png, *.html
    for f in os.listdir(community_dir):
        if re.search('.*.png', f) or re.search('.*.html', f):
            os.remove(community_dir + '/' + f)

    ### get communities from files ###
    com_files = os.listdir(community_dir)
    com_num = len(com_files)  # number of communities
    types = []
    bar_data = []
    for i, cf in enumerate(com_files):
        entity_set = set()
        with open(community_dir + '/' + cf) as cfile:
            lines = cfile.readlines()
            for line in lines:
                entity_set.update(line.split('\t')[:2])
        print(f'Community {i}')
        print('---------------------------------------------')

        categories = []
        for e in entity_set:
            # query over endpoint
            q = '''
                    PREFIX clarify: <http://research.tib.eu/clarify2020/vocab/>
                    PREFIX clarifyE: <http://research.tib.eu/clarify2020/entity/>
                    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                    select distinct ?entity, ?smokingHabits, ?Stages, ?bio, ?gender, ?relapseprogression
                    where {
                        $e$ a clarify:LCPatient.
                        $e$ clarify:hasFamilyHistory ?family.
                        ?family clarify:hasFamilyCancerType ?entity.
                        $e$ clarify:hasDiagnosis ?date.
                        ?date clarify:hasDiagnosisStage ?Stages.
                        # $e$  clarify:age ?Age.
                        $e$ clarify:hasSmokingHabit ?smokingHabits.
                        $e$ clarify:hasBio ?bio.
                        $e$ clarify:sex ?gender.
                        $e$ clarify:hasProgressionRelapseAssessment ?rp.
                        ?rp clarify:hasProgressionOrRelapse ?relapseprogression.
                    }
                '''.replace('$e$', '<' + e + '>')
            s = requests.Session()
            headers = {
                'Accept': 'application/json'
            }
            data = {'query': q}
            s.headers.update(headers)
            endpoints = {
                'clarify': 'https://labs.tib.eu/sdm/clarify_kg/sparql',
                'clarify1': 'https://labs.tib.eu/sdm/sub_clarify_v1.0/sparql'
            }
            endpoint = endpoints['clarify']
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
            
            # add category to entity
            if isinstance(entities_uris[0], str):
                for uri in entities_uris:
                    uri = uri[(uri.rindex('/')+1):]
                    cat = []
                    if uri in types:
                        idx = types.index(uri)
                        cat.append(idx)
                    else:
                        types.append(uri)
                        cat.append(len(types) - 1)
                    categories.append(tuple(cat))
            elif isinstance(entities_uris[0], list):
                for uris in entities_uris:
                    uris = [u[(u.rindex('/')+1):] for u in uris]
                    uris2str = '-'.join(uris)
                    cat = []
                    if uris2str in types:
                        idx = types.index(uris2str)
                        cat.append(idx)
                    else:
                        types.append(uris2str)
                        cat.append(len(types) - 1)
                    categories.append(tuple(cat))
    
        # numerate entities wrt. each category
        cat2num = {}
        for cat in categories:
            for c in cat:
                t = types[c]
                if t not in cat2num:
                    cat2num[t] = 1
                else:
                    cat2num[t] += 1
        
        cat = cat2num.keys()
        cat = sorted(cat, key=lambda x: cat2num[x], reverse=True)[:top_n]
        num = [cat2num[c] for c in cat]

        fig_path = community_dir + '/' + cf[:-4] + '.png'

        bar_data.append((fig_path, cat, num))
    
    cat_set = set()
    for _, c, _ in bar_data:
        for x in c:
            cat_set.add(x)
    cat2color = {}
    for c in cat_set:
        cat2color[c] = (np.random.random(), np.random.random(), np.random.random(), 1)

    for f, c, n in bar_data:
        # plot bar
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(c)), n, color=tuple(cat2color[x] for x in c))

        fig.savefig(f)
    
    # use html show color scheme
    with open(community_dir + '/color-scheme.html', 'wt') as f:
        html_template = ['''
            <!doctype html>
            <html>
                <head>
                    <title>Colors-Attributes</title>

                    <style>
                        .box {
                            float: left;
                            height: 20px;
                            width: 20px;
                            margin-bottom: 15px;
                            border: 1px solid black;
                            clear: both;
                        }'''
                        ,
                    '''</style>
                </head>
                <body>'''
                ,
                '''</body>
            </html>
        ''']

        def rgb2hex(r, g, b):
            return '#{:02x}{:02x}{:02x}'.format(r, g, b)
        
        style_items = []
        body_items = []
        for ctg, col in cat2color.items():
            s_it = r'.' + ctg + r'{background-color: ' + \
                rgb2hex(int(col[0]*255), int(col[1]*255), int(col[2]*255)) + r';}'
            style_items.append(s_it)
            b_it = r"<div><div class='box " + ctg + r"'></div> = " + ctg + r" </div><br>"
            body_items.append(b_it)
        style_str = ' '.join(style_items)
        body_str = ' '.join(body_items)

        final_html = html_template[0] + style_str + html_template[1] + body_str + html_template[2]

        f.write(final_html)
            
# calculate conductance of communities (inter-cluster conductance)
def comm_conductance(entities, sim_mat, community_dir, threshold=0.0):
    num_nodes = sim_mat.shape[0]

    # create graph from similarity matrix
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sim_mat[i, j] >= threshold:
                G.add_edge(i, j, weight=sim_mat[i, j])
    
    ### get communities from files ###
    communities = []

    com_files = os.listdir(community_dir)
    com_num = len(com_files)  # number of communities
    for i, cf in enumerate(com_files):
        entity_set = set()
        with open(community_dir + '/' + cf) as cfile:
            lines = cfile.readlines()
            for line in lines:
                entity_set.update(line.split('\t')[:2])
        print(f'Community {i}')
        print('---------------------------------------------')
        comm = [entities.index(e) for e in entity_set]
        communities.append(comm)
    
    conductance_clusters = []
    for comm in communities:
        con = nx.algorithms.cuts.conductance(G, comm)
        conductance_clusters.append(con)
    
    conductance = 1 - np.max(conductance_clusters)

    return conductance

# calculate coverage of communities
def comm_coverage(entities, sim_mat, community_dir, threshold=0.0):
    num_nodes = sim_mat.shape[0]

    # create graph from similarity matrix
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sim_mat[i, j] >= threshold:
                G.add_edge(i, j, weight=sim_mat[i, j])
    
    ### get communities from files ###
    communities = []

    com_files = os.listdir(community_dir)
    com_num = len(com_files)  # number of communities
    for i, cf in enumerate(com_files):
        entity_set = set()
        with open(community_dir + '/' + cf) as cfile:
            lines = cfile.readlines()
            for line in lines:
                entity_set.update(line.split('\t')[:2])
        print(f'Community {i}')
        print('---------------------------------------------')
        comm = [entities.index(e) for e in entity_set]
        communities.append(comm)
    
    # coverage
    coverage = nx.algorithms.community.quality.coverage(G, communities)
    print(coverage)

# calculate modularity of communities
def comm_modularity(entities, sim_mat, community_dir, threshold=0.0):
    num_nodes = sim_mat.shape[0]

    # create graph from similarity matrix
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sim_mat[i, j] >= threshold:
                G.add_edge(i, j, weight=sim_mat[i, j])
    
    ### get communities from files ###
    communities = []

    com_files = os.listdir(community_dir)
    com_num = len(com_files)  # number of communities
    for i, cf in enumerate(com_files):
        entity_set = set()
        with open(community_dir + '/' + cf) as cfile:
            lines = cfile.readlines()
            for line in lines:
                entity_set.update(line.split('\t')[:2])
        print(f'Community {i}')
        print('---------------------------------------------')
        comm = [entities.index(e) for e in entity_set]
        communities.append(comm)
    
    # coverage
    modularity = nx.community.modularity(G, communities)
    
    return modularity

# calculate performance of communities
def comm_performance(entities, sim_mat, community_dir, threshold=0.0):
    num_nodes = sim_mat.shape[0]

    # create graph from similarity matrix
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sim_mat[i, j] >= threshold:
                G.add_edge(i, j, weight=sim_mat[i, j])
    
    ### get communities from files ###
    communities = []

    com_files = os.listdir(community_dir)
    com_num = len(com_files)  # number of communities
    for i, cf in enumerate(com_files):
        entity_set = set()
        with open(community_dir + '/' + cf) as cfile:
            lines = cfile.readlines()
            for line in lines:
                entity_set.update(line.split('\t')[:2])
        print(f'Community {i}')
        print('---------------------------------------------')
        comm = [entities.index(e) for e in entity_set]
        communities.append(comm)
    
    # coverage
    performance = nx.algorithms.community.quality.performance(G, communities)
    
    return performance

# calculate normalized total cut of communities
def comm_normalized_total_cut(entities, sim_mat, community_dir, threshold=0.0):
    num_nodes = sim_mat.shape[0]

    # create graph from similarity matrix
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if sim_mat[i, j] >= threshold:
                G.add_edge(i, j, weight=sim_mat[i, j])
    
    ### get communities from files ###
    communities = []

    com_files = os.listdir(community_dir)
    com_num = len(com_files)  # number of communities
    for i, cf in enumerate(com_files):
        entity_set = set()
        with open(community_dir + '/' + cf) as cfile:
            lines = cfile.readlines()
            for line in lines:
                entity_set.update(line.split('\t')[:2])
        print(f'Community {i}')
        print('---------------------------------------------')
        comm = [entities.index(e) for e in entity_set]
        communities.append(comm)
    
    total_cut = 0
    for i in range(len(communities)):
        for j in range(i + 1, len(communities)):
            total_cut += nx.cut_size(G, communities[i], communities[j], weight='weight')
    normalized_total_cut = total_cut / nx.volume(G, G, weight='weight')
    
    return normalized_total_cut

# call it for plotting radar chart
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

