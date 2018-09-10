#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 14:27:16 2018

@author: raghuvansh
"""

import numpy as np
from math import log
from data_loader import read_data

class Node:
    
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = []
        self.answer = ''
        
    def __str__(self):
        return self.attribute
    
def subtables(data, col, delete):
    dictionary = dict()
    items = np.unique(data[:, col])
    count = np.zeros((items.shape[0], 1), dtype=np.int64)
    
    for x in range(items.shape[0]):
        for y in range(data.shape[0]):
            if data[y, col] == items[x]:
                count[x] += 1
            
    for x in range(items.shape[0]):
        dictionary[items[x]] = np.empty((count[x][0], data.shape[1]), dtype='S32')
        pos = 0
        for y in range(data.shape[0]):
            if data[y, col] == items[x]:
                dictionary[items[x]][pos] = data[y]
                pos += 1
    
        if delete:
            dictionary[items[x]] = np.delete(dictionary[items[x]], col, 1)
        
    return items, dictionary

def entropy(S):
    items = np.unique(S)
    
    if items.size == 1:
        return 0
    
    counts = np.zeros((items.shape[0], 1))
    sums = 0
    
    for x in range(items.shape[0]):
        counts[x] = sum(S == items[x]) / S.shape[0]
        
    for count in counts:
        sums += -1 * count * log(count, 2)
        
    return sums

def gain_ratio(data, col):
    items, dictionary = subtables(data, col, delete=False)
    
    total_size = data.shape[0]
    entropies = np.zeros((items.shape[0], 1))
    intrinsic = np.zeros((items.shape[0], 1))
    
    for x in range(items.shape[0]):
        ratio = dictionary[items[x]].shape[0] / total_size
        entropies[x] = ratio * entropy(dictionary[items[x]][:, -1])
        intrinsic[x] = ratio * log(ratio, 2)
        
    total_entropy = entropy(data[:, -1])
    iv = -1 * sum(intrinsic)
    
    for x in range(entropies.shape[0]):
        total_entropy -= entropies[x]
        
    return (total_entropy / iv)

def build_tree(data, metadata):
    if np.unique(data[:, -1]).shape[0] == 1:
        node = Node('')
        node.answer = np.unique(data[:, -1])[0]
        return node
    
    gains = np.zeros((data.shape[1] - 1, 1))
    for col in range(data.shape[1] - 1):
        gains[col] = gain_ratio(data, col)
        
    split = np.argmax(gains)
    node = Node(metadata[split])
    metadata = np.delete(metadata, split, 0)
    
    items, dictionary = subtables(data, split, delete= True)
    
    for x in range(items.shape[0]):
        child = build_tree(dictionary[items[x]], metadata)
        node.children.append((items[x], child))
        
    return node

def empty(size):
    s = ""
    for x in range(size):
        s += "   "
    return s

def print_tree(node, level):
    if node.answer != "":
        print(empty(level), node.answer)
        return
        
    print(empty(level), node.attribute)
    
    for value, n in node.children:
        print(empty(level + 1), value)
        print_tree(n, level + 2)
       
filename = input()
metadata, traindata = read_data(filename)
data = np.asarray(traindata)
node = build_tree(data, metadata)
print_tree(node, 0)