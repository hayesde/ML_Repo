import numpy as np
from flask import jsonify, render_template, request, redirect, url_for
from joblib import load
import pydotplus
import sklearn
import pydotplus
from sklearn import tree
from sklearn.tree import export_graphviz
from flask import send_file
import os


with open('fetal_health.pkl','rb') as file:
        my_tree = load(file)
x_cols = ['baseline value','accelerations','fetal_movement','uterine_contractions','light_decelerations','severe_decelerations','prolongued_decelerations','abnormal_short_term_variability','mean_value_of_short_term_variability','percentage_of_time_with_abnormal_long_term_variability','mean_value_of_long_term_variability','histogram_width','histogram_min','histogram_max','histogram_number_of_peaks','histogram_number_of_zeroes','histogram_mode','histogram_mean','histogram_median','histogram_variance','histogram_tendency']
y_col = ['healthy','suspect','pathological']


dot_data = tree.export_graphviz(my_tree, out_file=None, feature_names= x_cols, class_names= y_col, filled=True, rounded=True, special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)

def download(patient_id):
        for node in graph.get_node_list():
            if node.get_attributes().get('label') is None:
                continue
            if 'samples = ' in node.get_attributes()['label']:
                labels = node.get_attributes()['label'].split('<br/>')
                for i, label in enumerate(labels):
                        if label.startswith('samples = '):
                                labels[i] = 'samples = 0'
                        node.set('label', '<br/>'.join(labels))
                        node.set_fillcolor('white')

        samples = np.array(patient_id).reshape(1,-1)
        decision_paths = my_tree.decision_path(samples)

        for decision_path in decision_paths:
                for n, node_value in enumerate(decision_path.toarray()[0]):
                        if node_value == 0:
                                continue
                        node = graph.get_node(str(n))[0]
                        node.set_fillcolor('green')
                        labels = node.get_attributes()['label'].split('<br/>')
                        for i, label in enumerate(labels):
                                if label.startswith('samples = '):
                                        labels[i] = 'samples = {}'.format(int(label.split('=')[1]) + 1)
                        node.set('label', '<br/>'.join(labels))
        
       
        filename = 'tree_path.jpg'
        with open(filename, "wb") as f:
                f.write(graph.create_png())
        
        UPLOAD_FOLDER = '.'
        path = os.path.join(UPLOAD_FOLDER, filename)
        return send_file(path, as_attachment=True)
 
        #return "Upload success"

