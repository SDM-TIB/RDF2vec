import os
from re import S
import sys

from components.Logger import Logger
from flask import Flask, request

from functions import extract

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from pprint import pprint

from utilities import *
import graph_util as gu
import requests

from pykeen.pipeline import pipeline
from pykeen.models import TransE
from pykeen.triples import TriplesFactory
import torch


app = Flask(__name__)
logger = Logger('app')


@app.route('/api/extract', methods=['GET'])
def extract_entities_embeddings():
    source = request.args.get('source')
    source = 'dbpedia' if source is None else source
    max_depth = request.args.get('max_depth')
    max_depth = 4 if max_depth is None else int(max_depth)
    max_walks = request.args.get('max_walks')
    max_walks = 10 if max_walks is None else int(max_walks)
    walker = request.args.get('walker')
    walker = 1 if walker is None else int(walker)
    size = request.args.get('size')
    size = 100 if size is None else int(size)
    epochs = request.args.get('epochs')
    epochs = 1 if epochs is None else int(epochs)

    # query_and_write_entities_to_csv(source, path_to_write='results/LCPatients.csv', write_mode='wt')
    # entities, literals, embeddings = extract(source,
    #                             max_depth=max_depth,
    #                             max_walks=max_walks,
    #                             walker=walker,
    #                             size=size,
    #                             epochs=epochs, entities_path='results/LCPatients.csv')
    # write_vector_data_to_csv(embeddings, path='results/LCPatients_embeddings.csv')

    # entities = get_entities_from_csv('results/patients.csv')
    # embeddings = read_vector_data_from_csv('results/patients_embeddings.csv')
    # string_to_write = ''
    # for ent, emb in zip(entities, embeddings):
    #     string_to_write += ent + ','
    #     for e in emb:
    #         string_to_write += str(e) + ','
    #     string_to_write = string_to_write[:-1] + '\r\n'
    # with open('results/patients_entities_embeddings.csv', 'wt') as f:
    #     f.write(string_to_write)

    # embeddings = read_vector_data_from_csv('results/male_embeddings.csv')
    # male2d = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv('results/female_embeddings.csv')
    # female2d = PCA_analysis(embeddings)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(male2d[:, 0], male2d[:, 1])
    # ax.scatter(female2d[:, 0], female2d[:, 1])
    # ax.grid()
    # ax.legend(['male', 'female'])
    # fig.savefig('results/PCA_analysis.png')

    # embeddings = read_vector_data_from_csv(path='results/smoker_embeddings.csv')
    # smoker2d = PCA_analysis(embeddings)
    # smokers2d = PCA_analysis(embeddings)
    # smoker2d = smokers2d[:400]
    # embeddings = read_vector_data_from_csv(path='results/pre-smoker_embeddings.csv')
    # pre_smoker2d = PCA_analysis(embeddings)
    # pre_smoker2d = smokers2d[400:800]
    # embeddings = read_vector_data_from_csv(path='results/no-smoker_embeddings.csv')
    # no_smoker2d = PCA_analysis(embeddings)
    # no_smoker2d = smokers2d[800:]
    # embeddings = read_vector_data_from_csv(path='results/male_embeddings.csv')
    # male2d = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv(path='results/female_embeddings.csv')
    # female2d = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv(path='results/older50_embeddings.csv')
    # older502d = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv(path='results/younger50_embeddings.csv')
    # younger502d = PCA_analysis(embeddings)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(smoker2d[:, 0], smoker2d[:, 1])
    # ax.scatter(pre_smoker2d[:, 0], pre_smoker2d[:, 1])
    # ax.scatter(no_smoker2d[:, 0], no_smoker2d[:, 1], color='tab:green')
    # ax.scatter(male2d[:, 0], male2d[:, 1])
    # ax.scatter(female2d[:, 0], female2d[:, 1])
    # ax.scatter(older502d[:, 0], older502d[:, 1])
    # ax.scatter(younger502d[:, 0], younger502d[:, 1])
    # ax.legend(['smokers', 'pre-smokers', 'no-smokers', 'male', 'female', 'older50', 'younger50'])
    # ax.legend(['smokers', 'pre-smokers', 'no-smokers'])
    # ax.grid()
    # fig.savefig('results/PCA_analysis.png')

    # embeddings = read_vector_data_from_csv('results/SmokerMaleOlder50_embeddings.csv')
    # SMO = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv('results/SmokerMaleYounger50_embeddings.csv')
    # SMY = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv('results/SmokerFemaleOlder50_embeddings.csv')
    # SFO = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv('results/Pre-smokerMaleOlder50_embeddings.csv')
    # PMO = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv('results/Pre-smokerMaleYounger50_embeddings.csv')
    # PMY = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv('results/Pre-smokerFemaleOlder50_embeddings.csv')
    # PFO = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv('results/Pre-smokerFemaleYounger50_embeddings.csv')
    # PFY = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv('results/No-smokerMaleOlder50_embeddings.csv')
    # NMO = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv('results/No-smokerFemaleOlder50_embeddings.csv')
    # NFO = PCA_analysis(embeddings)
    # embeddings = read_vector_data_from_csv('results/No-smokerFemaleYounger50_embeddings.csv')
    # NFY = PCA_analysis(embeddings)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(SMO[:, 0], SMO[:, 1])
    # ax.scatter(SMY[:, 0], SMY[:, 1])
    # ax.scatter(SFO[:, 0], SFO[:, 1])
    # ax.scatter(PMO[:, 0], PMO[:, 1])
    # ax.scatter(PMY[:, 0], PMY[:, 1])
    # ax.scatter(PFO[:, 0], PFO[:, 1])
    # ax.scatter(PFY[:, 0], PFY[:, 1])
    # ax.scatter(NMO[:, 0], NMO[:, 1])
    # ax.scatter(NFO[:, 0], NFO[:, 1])
    # ax.scatter(NFY[:, 0], NFY[:, 1])
    # ax.legend([
    #     'SmokerMaleOlder50',
    #     'SmokerMaleYounger50',
    #     'SmokerFemaleOlder50',
    #     'Pre-smokerMaleOlder50',
    #     'Pre-smokerMaleYounger50',
    #     'Pre-smokerFemaleOlder50',
    #     'Pre-smokerFemaleYounger50',
    #     'No-smokerMaleOlder50',
    #     'No-smokerFemaleOlder50',
    #     'No-smokerFemaleYounger50',
    #     ])
    # ax.grid()
    # fig.savefig('results/PCA_analysis.png')

    # entities = get_entities_from_csv('results/LCPatients.csv')
    # embeds = read_vector_data_from_csv('results/LCPatients_embeddings.csv')
    # sim_mat = similarity_calculation(embeds)
    # savefig_sim_mat(sim_mat)
    # pca = PCA_analysis(embeds)
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(pca[:, 0], pca[:, 1])
    # ax.grid()
    # fig.savefig('results/PCA_analysis.png')
    # sim_mat_to_csv(sim_mat, entities)

    ################ pykeen ################
    # # train
    # tf = TriplesFactory.from_path('data/Clarify_Prediction_Biomarker/clarify.tsv')
    # training, testing = tf.split()
    # # pipeline_result = pipeline(
    # #     training=training,
    # #     testing=testing,
    # #     model=TransE
    # # )
    # # pipeline_result.save_to_directory('results/Trans-models')

    # # load model
    # pykeen_model = torch.load('results/Trans-models/trained_model.pkl')

    # # get entities
    # entities = get_entities_from_csv('results/LCPatients.csv')

    # # as prefixes are not consistent
    # # change prefix "http://research.tib.eu/clarify2020/entity/" to "http://clarify2020.eu/entity/"
    # entities = ['<http://clarify2020.eu/entity/' + e[e.rfind(r'/')+1:] + '>' for e in entities]

    # # get id of entities
    # entity_ids = torch.as_tensor(tf.entities_to_ids(entities))
    
    # # get embeddings using id
    # entity_representation_modules: List['pykeen.nn.Representation'] = pykeen_model.entity_representations
    # entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
    # embeddings_tensor: torch.FloatTensor = entity_embeddings(indices=entity_ids)
    # embeddings: np.ndarray = embeddings_tensor.detach().cpu().numpy()
    ################ pykeen ################

    ################ generate files for METIS #################
    # entities = get_entities_from_csv('results/LCPatients.csv')
    # embeddings = read_vector_data_from_csv('results/LCPatients_embeddings.csv')
    # embeddings = embeddings[:]
    # sim_mat = similarity_calculation(embeddings)
    # save_sim_mat(sim_mat)
    # savefig_sim_mat(sim_mat)
    # sim_mat = read_sim_mat()
    # gu.sim_mat_to_graph(sim_mat)
    # gu.visualize_graph('results/sim_mat.gp')
    # gu.group_entities_into_csvfiles('results/sim_mat.gp.part.6', 'results/LCPatients.csv')
    # gu.communities_in_bar('/mnt/c/Users/SongZ/Downloads/repositories/RDF2vec-target-based/src/results/metis-results', top_n=10)
    ################ generate files for METIS #################

    ################# calculate structural indices of communities (METIS) #################
    entities = get_entities_from_csv('results/LCPatients.csv')
    # embeddings = read_vector_data_from_csv('results/LCPatients_embeddings.csv')
    # embeddings = embeddings[:]
    # sim_mat = similarity_calculation(embeddings)
    # save_sim_mat(sim_mat)
    # savefig_sim_mat(sim_mat)
    sim_mat = read_sim_mat()
    community_dir = '/mnt/c/Users/SongZ/Downloads/repositories/RDF2vec-target-based/src/results/metis-results'
    inv_conductance = 1 - gu.conductance_comm(entities, sim_mat, community_dir)
    coverage = gu.coverage_comm(entities, sim_mat, community_dir)
    modularity = (gu.modularity_comm(entities, sim_mat, community_dir) + 0.5) / 1.5
    inv_performance = 1 - gu.performance_comm(entities, sim_mat, community_dir)
    inv_ntc = 1 - gu.normalized_total_cut_comm(entities, sim_mat, community_dir)

    # plot radar chart
    structural_indices = [
        inv_conductance,
        coverage,
        modularity,
        inv_performance,
        inv_ntc
    ]
    indices_name = [
        'Inv. Conductance',
        'Coverage',
        'Norm. Modularity',
        'Inv. Performance',
        'Inv. Norm. Total Cut'
    ]
    theta = gu.radar_factory(len(structural_indices), frame='polygon')
    fig, ax = plt.subplots(subplot_kw=dict(projection='radar'))
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.plot(theta, structural_indices)
    ax.fill(theta, structural_indices)
    ax.set_varlabels(indices_name)
    plt.savefig('results/structural-indices.png')
    ################# calculate structural indices of communities (semEP) #################

    ################ generate files for semEP #################
    # entities = get_entities_from_csv('results/LCPatients.csv')
    # embeds = read_vector_data_from_csv('results/LCPatients_embeddings.csv')
    # entities = entities[:]
    # embeds = embeds[:]
    # vertices1 = entities[:int(len(entities)/2)]
    # embeds1 = embeds[:int(len(entities)/2)]
    # vertices2 = entities[int(len(entities)/2):]
    # embeds2 = embeds[int(len(entities)/2):]
    # edges = []
    # for i, v1 in enumerate(vertices1):
    #     for j, v2 in enumerate(vertices2):
    #         e1, e2 = np.array(embeds1[i]), np.array(embeds2[j])
    #         e1_n, e2_n = e1 / np.sqrt(e1 @ e1.T), e2 / np.sqrt(e2 @ e2.T)
    #         sim = np.abs(e1_n @ e2_n.T)
    #         edges.append((v1, v2, sim))
    #         print(v1, v2, sim)
    # generate_bigraph(vertices1, vertices2, edges, outdir='results', vec1=embeds1, vec2=embeds2)
    ################ generate files for semEP #################

    ################ plot communities generated by semEP #################
    # entities = get_entities_from_csv('results/LCPatients.csv')
    # embeds = read_vector_data_from_csv('results/LCPatients_embeddings.csv')
    # entities = entities[:]
    # embeds = embeds[:]
    # community_dir = '/mnt/c/Users/SongZ/Downloads/repositories/RDF2vec-target-based/src/results/semEP-results/bigraph-0.7500-0.7500-Clusters'
    # # plot_communities(entities, embeds, community_dir)
    # # print_communities(entities, community_dir, tofile='/mnt/c/Users/SongZ/Downloads/repositories/RDF2vec-target-based/src/results/semEP-results/CD.txt')
    # bar_communities(entities, community_dir, top_n=10)
    ################ plot communities generated by semEP #################

    ################# calculate structural indices of communities (semEP) #################
    # entities = get_entities_from_csv('results/LCPatients.csv')
    # # embeddings = read_vector_data_from_csv('results/LCPatients_embeddings.csv')
    # # embeddings = embeddings[:]
    # # sim_mat = similarity_calculation(embeddings)
    # # save_sim_mat(sim_mat)
    # # savefig_sim_mat(sim_mat)
    # sim_mat = read_sim_mat()
    # community_dir = '/mnt/c/Users/SongZ/Downloads/repositories/RDF2vec-target-based/src/results/semEP-results/bigraph-0.7500-0.7500-Clusters'
    # comm_coverage(entities, sim_mat, community_dir)
    ################# calculate structural indices of communities (semEP) #################


    return {
        'entities': entities,
        # 'literals': literals,
        'embeddings': embeddings
    }


@app.route('/api/up', methods=['GET'])
def up():
    return {'msg': 'ok'}


def main(*args):

    if len(args) == 1:
        myhost = args[0]
    else:
        myhost = "0.0.0.0"

    debug = os.environ.get('APP_DEBUG', 'true').lower() == 'true'
    app.run(debug=debug, host=myhost, port=5000)


if __name__ == '__main__':
    main(*sys.argv[1:])