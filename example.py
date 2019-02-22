import cv2
from MarkerTracking import *

def draw_circle(img, points):
    for i in range(0, 4):
        cv2.circle(img, tuple(points[i]), 3, (0, 255, 0), 3)


if __name__ == '__main__':
    marker1 = Marker(0)
    marker1.matrix = [[1, 0, 0, 0, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0]]

    marker_database = [marker1]
    markerTracker = MarkerTracker(349, 5)
    for m in marker_database:
        markerTracker.add_marker(m)

    img = cv2.imread('fotka6.jpg')
    detected_markers = markerTracker.find_markers(img)

    for marker in detected_markers:
        draw_circle(img, marker.points)
        px0 = py0 = markerTracker.transform_h([175, 175], marker.h)
        px1 = markerTracker.transform_h([125, 175], marker.h)
        py1 = markerTracker.transform_h([175, 125], marker.h)

        print(px0)
        print(px1)

        cv2.line(img, tuple(px0), tuple(px1), (0, 0, 255), 3)
        cv2.line(img, tuple(py0), tuple(py1), (255, 0, 0), 3)

        #moving on x
        x_move_x = px1[0] - px0[0];
        x_move_y = px1[1] - px0[1];

        cv2.circle(img, (px0[0] + x_move_x * 2, px0[1] + x_move_y * 2), 5, (0, 0, 255), 5)

        # moving on y
        y_move_x = px1[0] - py0[0];
        y_move_y = py1[1] - py0[1];


    cv2.imshow('detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
