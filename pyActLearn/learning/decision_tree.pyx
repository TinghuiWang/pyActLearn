import sys
import time
import logging
cimport cython
import numpy as np
cimport numpy as np
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from libc.stdio cimport stdout
from cpython.string cimport PyString_AsString

cdef extern from "math.h":
    cdef float log2f(float x)

cdef struct DecisionTreeNode:
    # Pointer to Parent Node
    DecisionTreeNode* parent
    # Pointer to Children Node Pointers
    DecisionTreeNode** children
    # Number of children of current node
    int numChildren
    # Attribute Index used to divide the tree
    int attribute
    # Number of instances used to determine the attribute of current node
    unsigned long numInstances
    # array of instance ids used to determine the attribute of current node
    unsigned long *instances
    # Threshold used to classify current node
    float threshold
    # the Majority class of current node (if leaf node - this is the class)
    int classId
    # Number of instances that belong to majority class
    int numRight
    # Best entropy
    float entropy
    # Distribution
    float *adist


cdef struct SplitInfo:
    int valid
    float threshold
    float infoGain
    float gainRatio

cdef class DecisionTree:
    """
    DecisionTree class implements model interface for decision tree
    structure.

    The class is callable in python codes, but cdef parameters are
    not accessible.

    Parameters:
    root: Pointer to DecisionTreeNode Root
    num_classes: Number of Activities (class label from 0 to num_classes - 1)
    feature_num: Number of Features
    """
    cdef DecisionTreeNode* root
    cdef int num_classes
    cdef int feature_num
    cdef float[:,:] x
    cdef long[:] y
    cdef unsigned long *train_array
    cdef unsigned long training_num
    cdef int minimum_object_of_leaf

    cdef unsigned long *class_frequency

    cdef unsigned long *probability_left
    cdef unsigned long *probability_right

    cdef float *possible_threshold
    cdef SplitInfo *split_info_array
    cdef object logger

    def __cinit__(self, feature_num, num_classes, log_level=logging.INFO):
        """
        Intialize Data Structure. Callable in Python.
        :param feature_num: Enabled Total Feature Count
        :param num_classes: Enabled Activity Count
        :return:
        """
        self.root = <DecisionTreeNode *> NULL
        self.num_classes = num_classes
        self.feature_num = feature_num
        self.train_array = <unsigned long *> NULL
        self.training_num = 0
        self.minimum_object_of_leaf = 2
        self.logger = logging.getLogger('DecisionTree')
        self.logger.setLevel(log_level)
        # Some local parameters
        self.class_frequency = <unsigned long *> PyMem_Malloc(self.num_classes * sizeof(unsigned long))
        self.probability_left = <unsigned long *> PyMem_Malloc(self.num_classes*sizeof(unsigned long))
        self.probability_right = <unsigned long *> PyMem_Malloc(self.num_classes*sizeof(unsigned long))
        self.possible_threshold = <float *> NULL
        self.split_info_array = <SplitInfo *> PyMem_Malloc(self.feature_num*sizeof(SplitInfo))
        self.logger.info('Features: %d, Class Number: %d' % (self.feature_num, self.num_classes))

    def build(self, np.ndarray x, np.ndarray y):
        self.logger.info('Start to build decision tree')
        self.x = x
        self.y = y.astype(int)
        self.training_num = self.x.shape[0]
        self.train_array = <unsigned long *> PyMem_Malloc(self.training_num * sizeof(unsigned long))
        self.possible_threshold = <float *> PyMem_Malloc(self.training_num*sizeof(float))
        for i in range(self.training_num):
            self.train_array[i] = i
        self.logger.info('Got %d training instances' % self.training_num)
        self.logger.info('Features: %d, Class Number: %d' % (self.feature_num, self.num_classes))
        self.root = self.build_tree_node(self.train_array, self.training_num, NULL)
        self.logger.info('Finished building Decision Tree')

    cdef DecisionTreeNode * build_tree_node(self, unsigned long *train_array,
                         unsigned long num_train, DecisionTreeNode * parent):
        """Recursive Method for building tree node
        :param train_array: Lists of training array
        :param start_pos: Start position in the training array
        :param num_train: number of training samples for current node
        :param parent: pointer to parent node. NULL if it is the root node
        :return:
        """
        cdef DecisionTreeNode *node
        cdef int major_class
        cdef int need_split
        cdef unsigned long major_class_count
        cdef unsigned long num_left
        # Allocate space for current node
        node = self.allocate_tree_node()
        # self.logger.debug('Node Allocated at %x' % (<unsigned long> node))
        # Set Parents
        node.parent = parent
        # Records the number of training instances within the tree
        node.numInstances = num_train
        if num_train == 0:
            # No more training examples left, make it leaf
            node.classId = node.parent.classId
            # self.logger.debug('No More Training instances. This is a leaf node')
        else:
            # Populate array with a list of training example IDs
            node.instances = train_array
            # self.logger.debug([train_array[i] for i in range(num_train)])
            # Find Majority Class, whether to further split current node and the count of the majority class
            (major_class, need_split, major_class_count) = self.find_major_class(train_array, num_train)
            node.classId = major_class
            node.numRight = major_class_count
            # self.logger.debug('num_train: %d, Major class: %d, Number of instances: %d, split: %d' %
            #                  (<int> num_train, major_class, <int> major_class_count, need_split))
            # If all examples in training array belong to same class or examples too small
            # make current node a leaf node (return here directly)
            # Otherwise, find the attribute and best split and recursively build the sub-tree
            if major_class_count != num_train and need_split != 0:
                # self.logger.debug('Start Splitting')
                if self.select_attributes(train_array, num_train, node):
                    # sort train_array according to the best attribute and threshold
                    self.index_sort(train_array, num_train, node.attribute)
                    # Get the number of instances whose attribute is smaller than threshold
                    num_left = 0
                    while self.x[train_array[num_left]][node.attribute] < node.threshold:
                        num_left += 1
                    node.numChildren = 2
                    node.children = <DecisionTreeNode **> PyMem_Malloc(2 * sizeof(DecisionTreeNode *))
                    # Recursively construct sub-tree
                    self.logger.info('Create Sub Tree: left %d, right %d' % (num_left, num_train - num_left))
                    node.children[0] = self.build_tree_node(node.instances, num_left, node)
                    if node.children[0].numChildren > 0:
                        if node.children[0].children[0] == NULL or node.children[0].children[1] == NULL:
                            self.logger.error("NULL Children Found. node instances:")
                            self.logger.error([node.instances[k] for k in range(node.numInstances)])
                    node.children[1] = self.build_tree_node(&(node.instances[num_left]), num_train - num_left, node)
                    if node.children[1].numChildren > 0:
                        if node.children[1].children[0] == NULL or node.children[1].children[1] == NULL:
                            self.logger.error("NULL Children Found. node instances:")
                            self.logger.error([node.instances[k] for k in range(node.numInstances)])
                    self.logger.info('Sub Tree Finished: left %d, right %d' % (num_left, num_train - num_left))
                else:
                    self.logger.info('Leaf Node: instances %d', node.numInstances)
            else:
                self.logger.info('Leaf Node: instances %d', node.numInstances)
        return node

    cdef select_attributes(self, unsigned long *train_array,
                           unsigned long num_train, DecisionTreeNode* dt_node):
        """
        Perform a greddy search to select the attribute and threshold of the split point for current
        node according to the best information gain ratio.
        :param train_array: array of indexes to data
        :param num_train: number of training examples
        :return:
        """
        cdef int attribute_index = -1
        cdef float avg_info_gain = 0.0
        cdef unsigned long valid_split_count = 0
        cdef int best_attribute_index = -1
        cdef float best_gain_ratio = 0.0
        cdef float best_threshold = 0.0
        # Clear best split array to store the information for each attribute
        self.logger.debug('Clear Split Info Array for each feature')
        for i in range(self.feature_num):
            self.split_info_array[i].valid = 0
            self.split_info_array[i].threshold = 0
            self.split_info_array[i].infoGain = 0
            self.split_info_array[i].gainRatio = 0
        # Initialize default entropy in current decision tree node
        dt_node.entropy = self.default_class_entropy(train_array, num_train)
        self.logger.debug('Default Entropy %10.5f' % dt_node.entropy)
        # Go through every attribute and calculate their information gain with best numeric split
        valid_split_count = 0
        self.logger.debug('Go through attributes to find the best numeric split')
        for attribute_index in range(self.feature_num):
            self.find_best_numeric_split(train_array, num_train, dt_node, attribute_index, self.split_info_array)
            if self.split_info_array[attribute_index].valid != 0:
                avg_info_gain += self.split_info_array[attribute_index].infoGain
                valid_split_count += 1
                self.logger.debug('Avg Information Gain %10.5f, valid split %d' % (avg_info_gain, valid_split_count))
        # Search through all features to find the best feature and split point with the best information gain
        best_gain_ratio = 0.
        best_attribute_index = -1
        if valid_split_count > 0:
            avg_info_gain = avg_info_gain/valid_split_count - 0.00001
            for attribute_index in range(self.feature_num):
                self.logger.debug('attribute %d, valid %d, avg_info_gain %f, infoGain %f, gainRatio %f, diff_gain %f' %
                                  (attribute_index,
                                   self.split_info_array[attribute_index].valid,
                                   avg_info_gain,
                                   self.split_info_array[attribute_index].infoGain,
                                   self.split_info_array[attribute_index].infoGain - avg_info_gain,
                                   self.split_info_array[attribute_index].gainRatio
                                   ))
                if self.split_info_array[attribute_index].valid != 0 and \
                    self.split_info_array[attribute_index].infoGain >= avg_info_gain and \
                    self.split_info_array[attribute_index].gainRatio > best_gain_ratio:
                    best_attribute_index = attribute_index
                    best_threshold = self.split_info_array[attribute_index].threshold
                    best_gain_ratio = self.split_info_array[attribute_index].gainRatio
                    # self.logger.debug('current best: index: %d, threshold %.5f, gain %.5f' %
                    #                   (best_attribute_index, best_threshold, best_gain_ratio))
            # Update Info for current tree node
            if best_gain_ratio > 0.:
                dt_node.attribute = best_attribute_index
                """Need to Update the threshold calculation function C45SplitPoint"""
                dt_node.threshold = best_threshold
                self.logger.debug("select_attributes: best attribute %d, threshold %f" %
                          (best_attribute_index, best_threshold))
            return True
        else:
            # No Valid split count
            return False

    cdef find_best_numeric_split(self, unsigned long * train_array,
                                 unsigned long num_train, DecisionTreeNode* dt_node,
                                 unsigned long attribute_index, SplitInfo* split_info_array):
        """ Find the best numeric split for a specific attribute
        and calculate the corresponding information gain
        :param train_array:
        :param num_train:
        :param dt_node:
        :param attribute_index:
        :param split_info_array:
        :return:
        """
        cdef float last_threshold = 0.
        cdef float current_threshold = 0.
        cdef unsigned long border = 0
        cdef unsigned long left_count = 0
        cdef unsigned long right_count = 0
        cdef float entropy_left = 0.
        cdef float entropy_right = 0.

        cdef float default_entropy = dt_node.entropy
        cdef float best_info_gain = 0.0
        cdef float info_gain = 0.0
        cdef float entropy = 0.0
        cdef float best_threshold = 0.0

        # Sort train_array according to some attribute
        # self.logger.debug('Start Sorting along attribute %d' % attribute_index)
        self.index_sort(train_array, num_train, attribute_index)
        # self.logger.debug('Sorted training array index: ')
        # self.logger.debug([train_array[i] for i in range(num_train)])

        # Count the number of thresholds and add the number to the threshold array
        # The split point is selected according to the following criterias:
        # 1. If number of training instances per class is below 25, then the split
        #    point starts from 2 or 10% of the average training instances per class
        # 2. If the average training instances per class is massive (>25), then
        #    the split point starts at the point where 25 instances are at.
        minimum_split = max(min(0.1 * (num_train / self.num_classes), 25), 2)
        # self.logger.debug('minimum split is %d' % minimum_split)
        threshold_count = 0
        last_threshold = self.x[train_array[0]][attribute_index]
        # Start searching
        self.possible_threshold[0] = last_threshold
        for i in range(num_train):
            current_threshold = self.x[train_array[i]][attribute_index]
            if last_threshold < current_threshold and \
                i >= minimum_split and (num_train - i) >= minimum_split:
                # Count current threshold if it meets the following criterias:
                # 1. current threshold is a step up from the previous recorded threshold
                # 2. i (the number of instances whose attribute is smaller than threshold) is greater than
                #    the minimum split requirement (serve as a start point)
                # 3. (num_train - i) (the number of instances whose attribute is greater than threshold) is
                #    greater than the minimum split requirement (serve as the end point)
                self.possible_threshold[threshold_count] = (current_threshold + last_threshold)/2
                if self.possible_threshold[threshold_count] == last_threshold:
                    self.possible_threshold[threshold_count] = last_threshold
                # self.logger.debug('Threshold found, i = %d, threshold is %10.5f' % (i, self.possible_threshold[threshold_count]))
                threshold_count += 1
            last_threshold = current_threshold
        # Go through each threshold, calculate the information gain and return the best threshold
        border = 0
        left_count = 0
        right_count = num_train
        # self.logger.debug('Start calculating information gain for each threshold. Total %d' % threshold_count)
        # Clean the probability array
        for i in range(self.num_classes):
            self.probability_left[i] = 0
            self.probability_right[i] = 0
        for i in range(num_train):
            self.probability_right[self.y[train_array[i]]] += 1
        # Start for each threshold count and calculate the information gain
        # Record the threshold with the best information gain
        for i in range(threshold_count):
            while border < num_train and \
                self.x[train_array[border]][attribute_index] <= self.possible_threshold[i]:
                # Record (for each class) the instances whose attribute is smaller than threshold to left
                # and the ones whose attribute is greater than threshold to right
                self.probability_left[self.y[train_array[border]]] += 1
                self.probability_right[self.y[train_array[border]]] -= 1
                left_count += 1
                right_count -= 1
                border += 1
            # Calculate the entropy
            entropy_left = self.entropy(self.probability_left, self.num_classes, left_count)
            entropy_right = self.entropy(self.probability_right, self.num_classes, right_count)
            entropy = (left_count / <float> num_train) * entropy_left + \
                      (right_count / <float> num_train) * entropy_right
            info_gain = default_entropy - entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_threshold = self.possible_threshold[i]
                best_split_gain = self.split_entropy(left_count, right_count, num_train)
            # self.logger.debug('threshold %10.5f, border %d, left_count %d, right_count %d, info_gain %10.5f' %
            #                  (self.possible_threshold[i], border, left_count, right_count, info_gain))
            # self.logger.debug('left_entropy %10.5f, right_entropy %10.5f, entropy %10.5f' %
            #                  (entropy_left, entropy_right, entropy))
        # Finished recording information about best gain in best_info_gain, best_threshold and best_split_gain
        # Now, fill in the the Split Info array
        if best_info_gain > 0.0:
            self.split_info_array[attribute_index].valid = 1
            self.split_info_array[attribute_index].threshold = best_threshold
            self.split_info_array[attribute_index].infoGain = best_info_gain
            if best_split_gain == 0.0:
                self.split_info_array[attribute_index].gainRatio = 0.0
            else:
                self.split_info_array[attribute_index].gainRatio = best_info_gain / best_split_gain
            # self.logger.debug('Attribute %d, Valid %d, Threshold %10.5f, Info Gain %10.5f, Gain Ratio %10.5f' %
            #                  (attribute_index,
            #                   self.split_info_array[attribute_index].valid,
            #                   self.split_info_array[attribute_index].threshold,
            #                   self.split_info_array[attribute_index].infoGain,
            #                   self.split_info_array[attribute_index].gainRatio))
        return

    cdef index_sort(self, unsigned long *train_array, unsigned long num_train, unsigned long attribute):
        """
        Merge sort data array (self.x) based on attribute column and put the index in train_array
        without affecting data in self.x
        :param train_array:
        :param num_train:
        :param attribute:
        :return:
        """
        cdef unsigned long* tmp_array = <unsigned long*> PyMem_Malloc(num_train * sizeof(unsigned long))
        self.index_msort(tmp_array, train_array, num_train, attribute)
        PyMem_Free(tmp_array)

    cdef index_msort(self, unsigned long *tmp_array, unsigned long *train_array,
                     unsigned long num_train, unsigned long attribute):
        """
        Merge sort data array (self.x) based on attribute column and put the index in train_array
        without affecting data in self.x
        :param tmp_array:
        :param train_array:
        :param num_train:
        :param attribute:
        :return:
        """
        cdef unsigned long num_left = 0
        cdef unsigned long num_right = 0
        cdef unsigned long index_left = 0
        cdef unsigned long index_right = 0
        cdef unsigned long index = 0
        self.logger.debug([train_array[i] for i in range(num_train)])
        # If there is only one item - no need to sort
        if num_train == 1:
            return
        # Sort two sub-array recursively
        num_left = num_train/2
        num_right = num_train - num_left
        # sort left
        self.index_msort(tmp_array, train_array, num_left, attribute)
        self.index_msort(tmp_array, &train_array[num_left], num_right, attribute)
        # Merge Left and Right
        index = 0
        index_left = 0
        index_right = num_left
        self.logger.debug("merge: num_train %d" % num_train)
        self.logger.debug("before merge: " + str([train_array[i] for i in range(num_train)]))
        for index in range(num_train):
            # Compare Left and right
            if index_left == num_left:
                # Left done
                tmp_array[index] = train_array[index_right]
                index_right += 1
            elif index_right == num_train:
                # Right Done
                tmp_array[index] = train_array[index_left]
                index_left += 1
            elif self.x[train_array[index_left]][attribute] > self.x[train_array[index_right]][attribute]:
                # Right one is smaller, take right one
                tmp_array[index] = train_array[index_right]
                index_right += 1
            else:
                # Left one is smaller, take left one
                tmp_array[index] = train_array[index_left]
                index_left += 1
        for index in range(num_train):
            train_array[index] = tmp_array[index]
        self.logger.debug("after merge: " + str([train_array[i] for i in range(num_train)]))

    cdef entropy(self, unsigned long *list, unsigned long list_len, unsigned long total):
        """
        Calculate entropy given frequency list and total item counted in the list
        :param list:
        :param list_len:
        :param total:
        :return:
        """
        cdef float entropy = 0.
        cdef unsigned long i = 0
        # self.logger.debug('entropy: ' + str([list[i] for i in range(list_len)]))
        for i in range(list_len):
            if list[i] != 0:
                entropy -= (list[i] / <float> total) * log2f(list[i] / <float> total)
        return entropy

    cdef default_class_entropy(self, unsigned long *train_array, unsigned long num_train):
        """
        Calculate default entropy of a node before searching through the attributes
        :param train_array:
        :param num_train:
        :return:
        """
        for i in range(self.num_classes):
            self.class_frequency[i] = 0
        for i in range(num_train):
            self.class_frequency[self.y[train_array[i]]] += 1
        return self.entropy(self.class_frequency, self.num_classes, num_train)

    def split_entropy(self, unsigned long num_left, unsigned long num_right, unsigned long total):
        """
        Get the entropy of the split without each class
        :param num_left:
        :param num_right:
        :param total:
        :return:
        """
        cdef float percent_left = 0.
        cdef float percent_right = 0.
        cdef float entropy_left = 0.
        cdef float entropy_right = 0.
        percent_left = <float> num_left / <float> total
        percent_right = <float> num_right / <float> total
        if percent_left > 0.000001:
            entropy_left = percent_left * log2f(percent_left)
        if percent_right > 0.000001:
            entropy_right = percent_right * log2f(percent_right)
        return - entropy_left - entropy_right

    cdef find_major_class(self, unsigned long *train_array, unsigned long num_train):
        """
        Find the majority class in the training array and return with
        a tuple of class id, whether the node needs further split,
        and the frequency of the majority class
        :param train_array: array of training examples
        :param num_train: number of training samples in the array
        :return: tuple (int classId, bool is_smaller, int frequency)
        """
        cdef unsigned long max_frequency = 0
        cdef int major_class = -1
        cdef int need_split = 0
        # Reinitialize activity count
        for i in range(self.num_classes):
            self.class_frequency[i] = 0
        # Go through all the training examples, log the frequency of each class label
        for i in range(num_train):
            self.class_frequency[self.y[train_array[i]]] += 1
        # Go through the logged class frequency and find the majority
        for i in range(self.num_classes):
            if self.class_frequency[i] > max_frequency:
                major_class = i
                max_frequency = self.class_frequency[i]

        # To determine whether the current node needs further splitting
        # We compare the amount of major class instances to a static number
        # If smaller, it does not need to be split
        if max_frequency > self.minimum_object_of_leaf:
            need_split = 1
        else:
            need_split = 0

        return major_class, need_split, max_frequency

    cdef DecisionTreeNode * allocate_tree_node(self):
        """ Allocate and Initialize a decision tree node
        :return:
        """
        cdef DecisionTreeNode *node
        node = <DecisionTreeNode *> PyMem_Malloc(sizeof(DecisionTreeNode))
        if node == NULL:
            self.logger.error('PyMem_Malloc Failed! Running Out of Memory')
        else:
            node.parent = NULL
            node.children = NULL
            node.numChildren = 0
            node.attribute = -1
            node.numInstances = 0
            node.instances = NULL
            node.threshold = 0
            node.classId = -1
            node.numRight = 0
            node.entropy = 0
            node.adist = NULL
        return node

    def print_tree(self):
        self.print_subtree(NULL, 0)

    cdef print_subtree(self, DecisionTreeNode *node=NULL, unsigned long level=0):
        """
        Print the decision tree
        :param node:
        :param level:
        :return:
        """
        if node == NULL:
            self.print_subtree(self.root)
        else:
            for i in range(level):
                sys.stdout.write('\t')
            sys.stdout.write('%5d: %10.5f\n' % (node.attribute, node.threshold))
            if node.numChildren != 0:
                self.print_subtree(node.children[0], level + 1)
                self.print_subtree(node.children[1], level + 1)

    def export_to_graphviz(self):
        """
        Print the decision tree in graph viz format
        :return: String
        """
        num = 0
        tmp_out = ""
        output = "digraph Tree {\n"
        (tmp_out, num) = self.export_node_to_graphviz(NULL, 0)
        output += tmp_out
        output += "}\n"
        self.logger.info('exported %d nodes' % num)
        return output

    cdef export_node_to_graphviz(self, DecisionTreeNode *node=NULL, unsigned long id=0):
        """
        Export current sub-tree as graphviz tree direct graph content.
        Returns the number of items in the tree as well as the size of the tree
        :param node:
        :param id:
        :return:
        """
        cdef unsigned long num = 1
        cdef unsigned long tmp_num = 0
        output = ""
        tmp_out = ""
        if node == NULL:
            (output, num)  = self.export_node_to_graphviz(self.root, 0)
        else:
            output += "%u [label=\"X[%d]<=%.5f\\ninstances=%u\\ngain=%.5f\\nclass:%d\\n\" shape=\"box\"]\n" % (
                id, node.attribute, node.threshold, node.numInstances, node.entropy, node.classId
            )
            if node.numChildren != 0:
                (tmp_out, tmp_num) = self.export_node_to_graphviz(node.children[0], id+num)
                output += tmp_out
                output += "%d -> %d\n" % (id, id+num)
                num += tmp_num
                (tmp_out, tmp_num) = self.export_node_to_graphviz(node.children[1], id+num)
                output += tmp_out
                output += "%d -> %d\n" % (id, id+num)
                num += tmp_num
        return output, num

    def classify(self, np.ndarray x):
        """
        :type x: np.ndarray
        :param x: test data
        :return:
        """
        cdef DecisionTreeNode *node = self.root
        predicted_y = np.zeros((x.shape[0],))
        for i in range(x.shape[0]):
            node = self.root
            while node.numChildren != 0:
                if node.attribute == -1:
                    self.logger.error('Classify: Split Node, Attribute = -1, instances: %d' % node.numInstances)
                if node.children[0] == NULL or node.children[1] == NULL:
                    self.logger.error('Classify - found NULL Child Node, attribute: %d, threshold: %f, instances: %d' %
                                      (node.attribute, node.threshold, node.numInstances))
                if x[i][node.attribute] < node.threshold:
                    node = node.children[0]
                else:
                    node = node.children[1]
                # self.logger.debug('Reach Node: instances %d, classId %d, attribute %d, threshold %f' %
                #                  (node.numInstances, node.classId, node.attribute, node.threshold))
            predicted_y[i] = node.classId
        return predicted_y

    def export_to_dict(self):
        """
        Save the decision to a dict structure
        The decision tree is stored in pre-order transversal
        :return: dict
        """
        cdef DecisionTreeNode* root
        cdef int num_classes
        cdef int feature_num
        cdef np.ndarray x
        cdef np.ndarray y
        cdef unsigned long *train_array
        cdef unsigned long training_num
        cdef int minimum_object_of_leaf

        cdef unsigned long *class_frequency

        cdef unsigned long *probability_left
        cdef unsigned long *probability_right

        cdef float *possible_threshold
        cdef SplitInfo *split_info_array
        cdef object logger

        data = {
            'type': 'model',
            'name': 'DecisionTree',
            'modified': time.time(),
            'data': {
                'num_classes': self.num_classes,
                'feature_num': self.feature_num,
                'training_num': self.training_num,
                'root': self.node_to_dict(self.root)
            }
        }
        return data

    cdef node_to_dict(self, DecisionTreeNode *node=NULL):
        """
        Export Node into dict structure. Skip instances array (as it is only needed
        during training phase)
        :param node:
        :return:
        """
        if node == NULL:
            return {}
        data = {
            'numChildren': node.numChildren,
            'attribute': node.attribute,
            'numInstances': node.numInstances,
            'threshold': node.threshold,
            'classId': node.classId,
            'numRight': node.numRight,
            'entropy': node.entropy
        }
        if node.numChildren > 0:
            data['children'] = [
                self.node_to_dict(node.children[0]),
                self.node_to_dict(node.children[1])
            ]
        return data

    def load_from_dict(self, data):
        """
        Load Decision Tree from Dictionary Structure
        :type data: dict
        :param data:
        :return:
        """
        # Check Dict Type, Name
        if not {'type', 'name', 'data', 'modified'}.issubset(data.keys()):
            self.logger.error('Dict Keys Asserssion Error: got %s' % str(data.keys()))
            return
        if data['type'] != 'model':
            self.logger.error('Dict Type should be model but got %s instead' % data['type'])
            return
        if data['name'] != 'DecisionTree':
            self.logger.error('Model Name should be DecisionTree but got %s instead' % data['name'])
            return
        # Load Decision Tree
        if self.root == NULL:
            self.num_classes = data['data']['num_classes']
            self.feature_num = data['data']['feature_num']
            self.training_num = data['data']['training_num']
            self.root = <DecisionTreeNode *> self.dict_to_node(data['data']['root'])
        else:
            self.logger.error("Cannot Load Decision Tree Since the root is not NULL")

    cdef DecisionTreeNode * dict_to_node(self, object data):
        """
        Create Node and load from data
        :type data: dict
        :param data:
        :return:
        """
        cdef DecisionTreeNode *node = NULL
        node = self.allocate_tree_node()
        if 'numChildren' not in data.keys():
            self.logger.error('Cannot find num_children in dict:')
            self.logger.error(data)
        node.numChildren = data['numChildren']
        node.attribute = data['attribute']
        node.numInstances = data['numInstances']
        node.instances = NULL
        node.classId = data['classId']
        node.numRight = data['numRight']
        node.entropy = data['entropy']
        node.threshold = data['threshold']
        if node.numChildren > 0:
            node.children = <DecisionTreeNode **> PyMem_Malloc(node.numChildren * sizeof(DecisionTreeNode *))
            node.children[0] = self.dict_to_node(data['children'][0])
            node.children[1] = self.dict_to_node(data['children'][1])
        return node
