import random
import numpy as np
from sklearn.neighbors import kneighbors_graph


class Segmenter:

    def __init__(self, values, t):
        random.seed(123)
        self.values = values
        self.label_count = 1
        self.labels = np.zeros((values.shape[0], 1))  # labels for segments
        self.b_thresh = t  # boundary threshold


    def segment(self, k):
        """ Follows steps in exercise 1 description for segmentation """
        # Calculate the Knn conectivity graph
        A = kneighbors_graph(self.values, k, mode='connectivity', include_self=False).toarray()

        # search
        self.bfs(A)

        # set boundaries to -1 for visualization
        self.labels[self.labels == 0] = -1

        return np.squeeze(self.labels)

    def bfs(self, A):

        # random initial node
        rand_start = random.randint(0, (A.shape[0] - 1))

        visited, queue = set(), [rand_start]

        while queue:

            node_idx = queue.pop(0)

            if node_idx in visited:
                if self.labels[node_idx] != self.label_count:
                    previous_label = self.labels[node_idx]
                    self.labels[self.labels == previous_label] = self.label_count
            else:
                # mark node as visited and set a segment (or label)
                visited.add(node_idx)

                # if the point is a boundary
                if self.values[node_idx, 7] > self.b_thresh:
                    continue

                self.labels[node_idx] = self.label_count
                # Queue connected nodes for search
                queue.extend(np.where(A[node_idx] == 1)[0])

            # if queue empty but still unvisited nodes select next random start
            if not queue and len(visited) < A.shape[0]:
                rand_node = random.randint(0, (A.shape[0] - 1))
                while self.labels[rand_node] != 0:
                    rand_node = random.randint(0, A.shape[0] - 1)
                queue.extend([rand_node])
                self.label_count += 1

        print(len(visited))
        print(A.shape[0])
        print("is the end of the world as we know it")
