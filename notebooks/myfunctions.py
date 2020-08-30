from multiprocessing import cpu_count
def extract_freqs(graph, size, random=0, algo='gtrie', threads=cpu_count(), output='../output/'):
    """
    Calls execution of GTscanner algorithm with parameters:
    
    '../gtscanner_modified/gtries/undir4.gt'

    graph - path to the txt file of graph

    size - size of motif to extract 

    random - number of random graph to generate (better 100+)
    """
    assert (size >=3) and (size<=8), 'Size of motifs should be 3 <= size <= 8'
    assert (algo == 'fase') or (algo=='gtrie'), 'Algorithm should be \'fase\' or \'gtrie\''
    
    import os
    from bs4 import BeautifulSoup
    import pickle


    path_to_examples = '../output/temp/'
    graph_data = nx.to_pandas_edgelist(graph)
    unq_elem = np.unique(graph_data)
    graph_data = graph_data.applymap(lambda x: np.where(x == unq_elem)[0][0]+1)
    prep_data = os.path.join(path_to_examples, 'temp.txt')
    graph_data.to_csv(prep_data, sep=' ', header=None, index=False)

    cmd = '../gtscanner_modified/./GTScanner -s ' \
    + str(size) \
    + ' -m ' + algo + ' ../gtscanner_modified/gtries/undir'+str(size)+'.gt' \
    + ' -g ' + os.path.abspath(prep_data) \
    + ' -f simple' \
    + ' -t html' \
    + ' -o ' + os.path.join(output, 'temp.html') \
    + ' -r ' + str(random) \
    + ' -th ' + str(threads) #change to format string
    answer = os.popen(cmd).read()
    #print(answer)
    
    # Parsing output html file
    
    

    adjs = []
    freqs = []
    zs = []

    soup = BeautifulSoup(open(os.path.join(output, 'temp.html')).read())
    content = soup.find_all('tr')[1:]

    for motif in content:
        adjs.append(np.matrix([list(x) for x in motif.find('td', attrs={'class':'pre'}).text.split('\n')], dtype=int))
        stats = motif.find_all('td')[2:4]
        freqs.append(float(stats[0].text))
        zs.append(float(stats[1].text))

    ans = list(zip(adjs, freqs, zs))
    
    pickleFile = open('../gtscanner_modified/dict_4.pkl', 'rb')
    dict_4 = pickle.load(pickleFile)
    pickleFile.close()
    
    dict_4_z = dict(dict_4) 
    dict_4_f = dict(dict_4)


    for elem in ans:
        dict_4_z[str(elem[0])] = 0
        dict_4_f[str(elem[0])] = 0

        if (elem[2] not in [float('inf'), float('-inf')]) and (not np.isnan(elem[2])):
            dict_4_z[str(elem[0])] = elem[2]
        else: 
            dict_4_z[str(elem[0])] = 0

        if (elem[1] not in [float('inf'), float('-inf')]) and (not np.isnan(elem[2])):
            dict_4_f[str(elem[0])] = elem[1]
        else: 
            dict_4_f[str(elem[0])] = 0

    z_scores_4 = list(dict_4_z.values())
    freqs_4 = list(dict_4_f.values())

    sum_of_freqs_4 = np.sum(freqs_4)
    normed_freqs_4 = freqs_4 / sum_of_freqs_4

    normed_z_scores_4 = z_scores_4 / np.sqrt(np.sum([x**2 for x in z_scores_4])) if np.sum(z_scores_4) != 0 else np.zeros(6)
    
    return list(normed_freqs_4)#, list(normed_z_scores_4)

def calc_iter(g, beta=0.2, percentage_infected=0.01, estimations=10):    
    import ndlib.models.epidemics as ep
    import ndlib.models.ModelConfig as mc
    len_nodes = len(g.nodes())
    
    list_of_iter = []
    
    for i in range(estimations):
        model = ep.SIModel(g)
        cfg = mc.Configuration()
        cfg.add_model_parameter('beta', beta)
        cfg.add_model_parameter("percentage_infected", percentage_infected)
        model.set_initial_status(cfg)
    
        iteration = model.iteration() #initialization
    
        while (iteration['node_count'][1]<len_nodes):
            iteration = model.iteration()
        
        list_of_iter.append(iteration['iteration'])
        
    return np.mean(list_of_iter)