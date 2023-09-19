import utilities
from graph_tool.all import *
import numpy as np
import pandas as pd
import os
import re
import requests
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import networkx as nx

# visualize graph file of METIS's format
def visualize_graph(filename):
    g = Graph(directed=False)

    with open(filename, 'rt') as f:
        line = f.readline()  # the header line
        info = line.split(' ')  # either [n, m], [n, m, fmt], or [n, mn fmt, ncon]
        n = int(info[0])  # the number of vertices
        m = int(info[1])  # the number of edges
        if len(info) >= 3:
            fmt = info[2]  # the information about vertex sizes, vertex weights, edge weights
            if len(info) == 4:
                ncon = int(info[3])  # the number of vertex weights associated with each vertex
        else:
            fmt = '000'
            ncon = '0'
        
        # process all nodes and edges of graph
        vlist = list(g.add_vertex(n))  # references of all vertices
        eprop = g.new_edge_property('string')  # weight property
        i = 0
        while line:
            # print(i)
            line = f.readline()[:-1]
            if i == n:
                break
            line = line.split(' ')
            if line == ['']:
                i += 1
                continue
            if fmt[0] == '1':
                size = line[0]  # vertex size
                line = line[1:]  # consider the rest of line
            if fmt[1] == '1':
                weights = line[:ncon]  # vertex weights
                line = line[ncon:]  # the rest is all about nodes and edges(if weights of edges applied)
            if fmt[2] == '1':
                for j in range(int(len(line)/2)):
                    edges = [(edge.source(), edge.target()) for edge in g.edges()]
                    if (vlist[i], vlist[int(line[2*j])-1]) not in edges and (vlist[int(line[2*j])-1], vlist[i]) not in edges:
                        e = g.add_edge(vlist[i], vlist[int(line[2*j])-1])
                        eprop[e] = str(int(line[2*j+1]) / 1e6)
                g.edge_properties['similarity'] = eprop
            else:
                for v in line:
                    edges = [(edge.source(), edge.target()) for edge in g.edges()]
                    if (vlist[i], vlist[int(v)-1]) not in edges and (vlist[int(v)-1], vlist[i]) not in edges:
                        g.add_edge(vlist[i], vlist[int(v)-1])
            i += 1
    graph_draw(g, vertex_text=g.vertex_index, edge_text=g.edge_properties['similarity'], output='results/graphhh.png')

# visualize partitioned graph in terms of original graph file and partition file
def visualize_partitioned_graph(graph_file, partition_file):
    g = Graph(directed=False)

    with open(partition_file, 'rt') as f:
        line = f.readline()[:-1]
        groups = []  # groups in which each vertex is assigned
        groups.append(int(line))
        while line:
            line = f.readline()[:-1]
            if len(line) == 0:
                break
            groups.append(int(line))
        groups_num = max(groups) + 1  # the number of groups
        color_table = [(np.random.random(), np.random.random(), np.random.random(), 1.0) for _ in range(groups_num)]
        colors = [color_table[group] for group in groups]  # each color corresponds to each group

    with open(graph_file, 'rt') as f:
        line = f.readline()  # the header line
        info = line.split(' ')  # either [n, m], [n, m, fmt], or [n, mn fmt, ncon]
        n = int(info[0])  # the number of vertices
        m = int(info[1])  # the number of edges
        if len(info) >= 3:
            fmt = info[2]  # the information about vertex sizes, vertex weights, edge weights
            if len(info) == 4:
                ncon = int(info[3])  # the number of vertex weights associated with each vertex
        else:
            fmt = '000'
            ncon = '0'
        
        # process all nodes and edges of graph
        vlist = list(g.add_vertex(n))  # references of all vertices
        eprop = g.new_edge_property('string')  # weight property
        vprop = g.new_vertex_property('vector<float>')  # group property
        g.vertex_properties['group'] = vprop
        for color, v in zip(colors, vlist):
            vprop[v] = list(color)
        i = 0
        while line:
            line = f.readline()[:-1]
            if i == n:
                break
            line = line.split(' ')
            if line == ['']:
                i += 1
                continue
            if fmt[0] == '1':
                size = line[0]  # vertex size
                line = line[1:]  # consider the rest of line
            if fmt[1] == '1':
                weights = line[:ncon]  # vertex weights
                line = line[ncon:]  # the rest is all about nodes and edges(if weights of edges applied)
            if fmt[2] == '1':
                for j in range(int(len(line)/2)):
                    edges = [(edge.source(), edge.target()) for edge in g.edges()]
                    if (vlist[i], vlist[int(line[2*j])-1]) not in edges and (vlist[int(line[2*j])-1], vlist[i]) not in edges:
                        e = g.add_edge(vlist[i], vlist[int(line[2*j])-1])
                        eprop[e] = str(int(line[2*j+1]) / 1e6)
                g.edge_properties['similarity'] = eprop
            else:
                for v in line:
                    edges = [(edge.source(), edge.target()) for edge in g.edges()]
                    if (vlist[i], vlist[int(v)-1]) not in edges and (vlist[int(v)-1], vlist[i]) not in edges:
                        g.add_edge(vlist[i], vlist[int(v)-1])
            i += 1
    graph_draw(g, vertex_text=g.vertex_index, vertex_fill_color=g.vertex_properties['group'], edge_text=g.edge_properties['similarity'], output='results/graphhh_3parts.png')

# write similarity matrix to a graph file of METIS's format
def sim_mat_to_graph(sim_mat, output_path='results/sim_mat.gp'):
    sim_mat = np.array(sim_mat)
    n = str(sim_mat.shape[0])  # the number of vertices
    fmt = '001'  # the format of graph
    ncon = '0'  # the number of multi-constraints of vertices

    m = 0  # the number of edges
    graph_structure = ''
    threshold = 0.3  # only consider edges with weights greater than threshold
    for i, row in enumerate(sim_mat):
        v_w_list = []
        for v, w in enumerate(row):
            if w >= threshold and i != v:
                v_w_list.append(str(v+1))
                v_w_list.append(str(int(w*1e6)))  # weights in graph file must be integer, so scale 10 times
                m += 1
        graph_structure += ' '.join(v_w_list) + '\n'
    m = int(m / 2)  # rule out duplicated edges
    string_to_write = n + ' ' + str(m) + ' ' + fmt + ' ' + ncon + '\n' + graph_structure

    with open(output_path, 'wt') as f:
        f.write(string_to_write)

# extract entities from entities csv file into groups according to METIS group file (each group is also a csv file containing entities)
def group_entities_into_csvfiles(partition_file, entities_to_group='results/entities_from_query.csv'):
    df = pd.read_csv(entities_to_group, sep=',', error_bad_lines=False, encoding='cp1252')
    entities = df['Unique_Entities'].to_list()
    groups = {}  # group dictionary, each key has all entities from one group
    with open(partition_file, 'rt') as f:
        lines = f.readlines()
        lines = [line[0] for line in lines]
        for i, line in enumerate(lines):
            if line not in groups.keys():
                groups[line] = []
            groups[line].append(entities[i])
    for k in groups.keys():
        utilities.write_entities_to_csv(groups[k], 'results/metis-results/group' + k + '.csv')

# plot communities in bar
def communities_in_bar(community_dir, top_n=2):
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
        df = pd.read_csv(community_dir + '/' + cf, sep=',', error_bad_lines=False, encoding='cp1252')
        entity_set = set(df['Unique_Entities'].to_list())
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
                abort(400, response.text)
            content = json.loads(response.text)
            if 'results' not in content or 'bindings' not in content['results']:
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
def conductance_comm(entities, sim_mat, community_dir, threshold=0.0):
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
        df = pd.read_csv(community_dir + '/' + cf, sep=',', error_bad_lines=False, encoding='cp1252')
        entity_set = set(df['Unique_Entities'].to_list())
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
def coverage_comm(entities, sim_mat, community_dir, threshold=0.0):
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
        df = pd.read_csv(community_dir + '/' + cf, sep=',', error_bad_lines=False, encoding='cp1252')
        entity_set = set(df['Unique_Entities'].to_list())
        print(f'Community {i}')
        print('---------------------------------------------')
        comm = [entities.index(e) for e in entity_set]
        communities.append(comm)
    
    # coverage
    coverage = nx.algorithms.community.quality.coverage(G, communities)
    
    return coverage

# calculate modularity of communities
def modularity_comm(entities, sim_mat, community_dir, threshold=0.0):
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
        df = pd.read_csv(community_dir + '/' + cf, sep=',', error_bad_lines=False, encoding='cp1252')
        entity_set = set(df['Unique_Entities'].to_list())
        print(f'Community {i}')
        print('---------------------------------------------')
        comm = [entities.index(e) for e in entity_set]
        communities.append(comm)
    
    # coverage
    modularity = nx.community.modularity(G, communities)
    
    return modularity

# calculate performance of communities
def performance_comm(entities, sim_mat, community_dir, threshold=0.0):
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
        df = pd.read_csv(community_dir + '/' + cf, sep=',', error_bad_lines=False, encoding='cp1252')
        entity_set = set(df['Unique_Entities'].to_list())
        print(f'Community {i}')
        print('---------------------------------------------')
        comm = [entities.index(e) for e in entity_set]
        communities.append(comm)
    
    # coverage
    performance = nx.algorithms.community.quality.performance(G, communities)
    
    return performance

# calculate normalized total cut of communities
def normalized_total_cut_comm(entities, sim_mat, community_dir, threshold=0.0):
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
        df = pd.read_csv(community_dir + '/' + cf, sep=',', error_bad_lines=False, encoding='cp1252')
        entity_set = set(df['Unique_Entities'].to_list())
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

if __name__ == '__main__':
    # visualize_graph('../test.gp')
    # visualize_graph('results/sim_mat.gp')
    # visualize_partitioned_graph('results/sim_mat.gp', 'results/sim_mat.gp.part.3')
    group_entities_into_csvfiles('results/sim_mat.gp.part.3')