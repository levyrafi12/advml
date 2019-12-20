import numpy as np
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
from collections import OrderedDict
from collections import Counter
from collections import defaultdict
import numpy as np
import pandas as pd
np.set_printoptions(precision=4)

# Turn the JSON file to a dictionary {lbl, parent}
def walk(node, res={}):
    if 'children' in dict.keys(node):
        kids_list = node['children']
        for curr in kids_list:
            res.update({curr['name']: node['name']})
            walk(curr)
    else:
        return
    return res


# Map a label mid to its display name
def load_display_names(classes_filename):
    classes = pd.read_csv(classes_filename, names=['mid', 'name'])
    display_names = dict(zip(classes.mid, classes.name))
    cls_name_to_cls_id = dict(zip(classes.name, classes.mid))
    return display_names, cls_name_to_cls_id


# Map { image id --> url }
def image_to_url(images_path):
    urls = pd.read_csv(images_path)
    id_url = dict(zip(urls.ImageID, urls.Thumbnail300KURL))
    return id_url


# Parse a DF (data frame) into a dict {image -> associated labels}
def image_to_labels(annotations, vocab_threshold):
    freq_labels = build_freq_labels(annotations, vocab_threshold)
    img_to_labels, col_name = defaultdict(list), 'ImageID'
    images = annotations[col_name].unique().tolist()
    for i in range(len(annotations)):
        label = annotations['LabelName'][i]
        if label in freq_labels:
            img_id = annotations[col_name][i]
            img_to_labels[img_id].append(label)
    return img_to_labels, freq_labels

# Return 'vocan_treshold' most frequent labels 
def build_freq_labels(annotations, vocab_threshold):
    labels_count = defaultdict(int)
    freq_labels = set()
    for i in range(len(annotations)):
        label = annotations['LabelName'][i]
        labels_count[label] += 1
    freq_labels = sorted(labels_count.items(), key=lambda x: x[1], reverse=True)
    freq_labels = freq_labels[:vocab_threshold]
    return [label for label, _ in freq_labels]

# Load train, test, validation image - url files into df.
def load_urls_to_df(path_train, path_val, path_test):
    df_train = pd.read_csv(path_train)
    df_val = pd.read_csv(path_val)
    df_test = pd.read_csv(path_test)
    urls = pd.concat([df_train, df_val, df_test])
    urls.set_index('ImageID', inplace=True)
    return urls


def plot_px_vs_entropy(singles, num_images):
    font_size = 'x-large'
    p, h, xy = OrderedDict(), OrderedDict(), OrderedDict()
    for k, v in singles.items():
        px = float(v)/float(num_images)
        p[k] = px
        h[k] = -px*np.log2(px)-(1-px)*np.log2(1-px)
        xy[k] = (p[k], h[k])
    x_val = [x[0] for x in xy.values()]
    y_val = [x[1] for x in xy.values()]
    plt.scatter(x_val, y_val)
    plt.xlabel('p', fontsize=font_size)
    plt.ylabel('H(p)', fontsize=font_size)
    plt.show()
