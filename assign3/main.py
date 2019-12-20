import numpy as np
import pandas as pd
import util
import vis
from scipy.sparse.csgraph import minimum_spanning_tree
from collections import defaultdict
import os

np.set_printoptions(precision=4)
pd.set_option('precision', 2)

SEP = os.sep

def gen_full_path(dir, file):
    return SEP.join([dir, file])

def main():
    vocabolary_threshold = 400
    # open image dataset
    oid_data = gen_full_path('data', 'annotations-machine.csv')
    classes_fn = gen_full_path('data', 'class-descriptions.csv')
    # images_w_url = 'data\\images.csv'

    # classes_display_name - 
    # Mapping between class id (label) and class name 
    # class_name_to_class_id -
    # Mapping between class name to class id (label)
    classes_display_name, class_name_to_class_id = util.load_display_names(classes_fn)
    # id_url = util.image_to_url(images_w_url)
    annotations = pd.read_csv(oid_data)
    img_to_labels, freq_labels = util.image_to_labels(annotations, vocabolary_threshold)

    # Return a dictionary with mapping between each Node and its childern nodes.
    # Use for each node the class label
    chow_liu_tree = chow_lio_model(img_to_labels, freq_labels)

    for name in ["Face", "Sports", "Vehicle"]:
        root = class_name_to_class_id[name]
        sub_tree = defaultdict(list)
        marked = { root }
        extract_sub_graph(chow_liu_tree, root, sub_tree, marked)
        vis.plot_network(sub_tree, classes_display_name, name)

    vis.plot_network(chow_liu_tree, classes_display_name)

# extract sub graph of a given a root. 'level' defines the traversal depth 
def extract_sub_graph(graph, root, sub_graph, marked, level=2):
    if level == 0:
        return

    children = graph[root]
    for child in children:
        if child in marked:
            continue
        marked.add(child)
        sub_graph[root].append(child)
        extract_sub_graph(graph, child, sub_graph, marked, level - 1)

# Build the chow liu graph
# Return max spanning tree
def chow_lio_model(img_to_labels, freq_labels):
    label_to_ind = dict() # label is the class id

    ind = 0
    ind_to_label = []
    for lbl in freq_labels:
        label_to_ind[lbl] = ind
        ind_to_label.append(lbl)
        ind += 1

    num_images = len(img_to_labels)
    num_labels = len(freq_labels)

    weighted_graph = np.array(np.zeros((num_labels, num_labels), dtype=float))
    
    for _, labels in img_to_labels.items():
        V = []
        for lbl in labels:
            V.append(label_to_ind[lbl])
        for i in V:
            for j in V:
                if i == j or i < j:
                    weighted_graph[i][j] += 1

    weighted_graph /= num_images

    for i in range(num_labels):
        for j in range(i + 1, num_labels):
            # data (empirical) distribution of a single label
            pdi = weighted_graph[i][i] 
            pdj = weighted_graph[j][j]
            # joint data distribution of two labels
            pdij = weighted_graph[i][j]
            weighted_graph[i][j] = 0
            if pdi > 0 and pdj > 0 and pdij > 0:
                weighted_graph[i][j] = pdij * np.log(pdij / (pdi * pdj))

    for i in range(num_labels):
        weighted_graph[i][i] = 0 # set to zero the diagonal 

    # since we want the maximum spanning tree, we multiply the weights by -1
    weighted_graph *= -1
    # Return spanning tree matrix in SCR format
    Tscr = minimum_spanning_tree(weighted_graph)
    Tscr *= -1 # back to origin positive weights
    # SCR to np array
    Tarr = Tscr.toarray()

    # Build a graph in a dictionary structure where each vertex is a key and 
    # the value of the key is the neighbors of that vertex
    Tdict = defaultdict(list)
    for i in range(num_labels):
        for j in range(i + 1, num_labels):
            if Tarr[i][j] > 0:
                lbli = ind_to_label[i] # label_i
                lblj = ind_to_label[j] # label_j
                Tdict[lbli].append(lblj)
                Tdict[lblj].append(lbli)
    return Tdict

if __name__ == '__main__':
    main()
