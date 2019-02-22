import cv2
import numpy as np
import queue


class Node:
    def __init__(self, deep, index, rotation, q):
        self.rotation = rotation
        self.deep = deep
        self.index = index
        self.leftSon = None
        self.rightSon = None
        self.q = q


class Marker:
    def __init__(self, index, matrix=None):
        self.index = index
        self.matrix = matrix


class DetectedMarker:
    def __init__(self, index, p, h, r):
        self.index = index
        self.points = p
        self.h = h
        self.rotation = r


class MarkerTracker:
    def __init__(self, marker_size, block_count):
        self.__marker_size = marker_size
        self.__block_count = block_count
        self.__h_src = np.float32(
            [[0, 0], [self.__marker_size, 0], [self.__marker_size, self.__marker_size], [0, self.__marker_size]])
        self.__root = Node(0, -1, 0, queue.Queue(self.__block_count * self.__block_count))
        self.__size = 0

    def add_marker(self, marker):
        for r in range(3, -1, -1):
            marker.matrix = np.rot90(marker.matrix)
            node = self.__root
            q = queue.Queue(self.__block_count * self.__block_count)
            for i in range(0, self.__block_count):
                for j in range(0, self.__block_count):
                    q.put(marker.matrix[i][j])
            self.__add_marker_step(q, node, marker.index, 0, r)

    def __add_marker_step(self, q, node, index, deep, r):
        if deep == self.__block_count * self.__block_count:
            return
        if node.index == -1:
            switch = q.get()
            if switch == 0:
                if node.leftSon is None:
                    node.leftSon = Node(deep, index, r, q)
                    self.__size += 1
                else:
                    self.__add_marker_step(q, node.leftSon, index, deep + 1, r)
            else:
                if node.rightSon is None:
                    node.rightSon = Node(deep, index, r, q)
                    self.__size += 1
                else:
                    self.__add_marker_step(q, node.rightSon, index, deep + 1, r)
        else:
            switch = node.q.get()
            if switch == 0:
                node.leftSon = Node(node.deep + 1, node.index, node.rotation, node.q)
            else:
                node.rightSon = Node(node.deep + 1, node.index, node.rotation, node.q)
            node.index = -1
            self.__add_marker_step(q, node, index, deep, r)

    def __find_marker(self, thresh, h):
        block_size = (self.__marker_size + 1) / (self.__block_count + 2)
        start = block_size / 2
        end = start + block_size * (self.__block_count + 1)
        if thresh[self.transform_h([start, start], h, True)] == 0 and \
                thresh[self.transform_h([start, end], h, True)] == 0 and \
                thresh[self.transform_h([end, end], h, True)] == 0 and \
                thresh[self.transform_h([end, start], h, True)] == 0:
            node = self.__root
            pos_y = start
            for i in range(0, self.__block_count):
                pos_y += block_size
                pos_x = start
                for j in range(0, self.__block_count):
                    pos_x += block_size
                    if thresh[self.transform_h([pos_x, pos_y], h, True)] == 255:
                        node = node.leftSon
                    else:
                        node = node.rightSon
                    if node is None:
                        return -1, 0
                    else:
                        if node.index != -1:
                            return node.index, node.rotation
        return -1, 0

    def find_markers(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), cv2.THRESH_BINARY)
        ret, thresh = cv2.threshold(blurred, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detected_marker = []

        for cnt in contours:
            arc_length = cv2.arcLength(cnt, True)
            if arc_length > 500:
                approx = cv2.approxPolyDP(cnt, 0.01 * arc_length, True)
                if len(approx) == 4:
                    points = [approx[0][0], approx[1][0], approx[2][0], approx[3][0]]
                    top_left_index, min_top_left = 0, points[0][0] + points[0][1]
                    for i in range(1, 4):
                        cur_sum = points[i][0] + points[i][1]
                        if cur_sum < min_top_left:
                            min_top_left, top_left_index = cur_sum, i
                        elif cur_sum == min_top_left and points[i][0] < points[top_left_index][0]:
                            top_left_index = i
                    points[top_left_index], points[0] = points[0], points[top_left_index]

                    top_right_index, min_top_right_y = 1, 10000
                    for i in range(1, 4):
                        if min_top_right_y > points[i][1]:
                            min_top_right_y, top_right_index = points[i][1], i
                    points[top_right_index], points[1] = points[1], points[top_right_index]

                    if points[2][0] < points[3][0]:
                        points[2], points[3] = points[3], points[2]
                    h = cv2.findHomography(self.__h_src, np.float32(points))[0]
                    index, rotation = self.__find_marker(thresh, h)
                    if index != -1:
                        detected_marker.append(DetectedMarker(index, points, h, rotation))
        return detected_marker

    @staticmethod
    def transform_h(p, h, inv=False):
        w = (h[2][0] * p[0] + h[2][1] * p[1] + 1)
        x = (h[0][0] * p[0] + h[0][1] * p[1] + h[0][2]) / w
        y = (h[1][0] * p[0] + h[1][1] * p[1] + h[1][2]) / w
        if inv:
            return int(y), int(x)
        return int(x), int(y)


