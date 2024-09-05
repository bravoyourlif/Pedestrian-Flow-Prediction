import numpy as np
import tracker
from detector import Detector
import cv2

if __name__ == '__main__':

    # Create a polygon for collision line detection based on video dimensions
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)

    # Initialize two polygons for collision detection
    list_pts_blue = [[204, 305], [227, 431], [605, 522], [1101, 464], [1900, 601], [1902, 495], [1125, 379], [604, 437],
                     [299, 375], [267, 289]]
    ndarray_pts_blue = np.array(list_pts_blue, np.int32)
    polygon_blue_value_1 = cv2.fillPoly(mask_image_temp, [ndarray_pts_blue], color=1)
    polygon_blue_value_1 = polygon_blue_value_1[:, :, np.newaxis]

    # Fill the second polygon
    mask_image_temp = np.zeros((1080, 1920), dtype=np.uint8)
    list_pts_yellow = [[181, 305], [207, 442], [603, 544], [1107, 485], [1898, 625], [1893, 701], [1101, 568],
                       [594, 637], [118, 483], [109, 303]]
    ndarray_pts_yellow = np.array(list_pts_yellow, np.int32)
    polygon_yellow_value_2 = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=2)
    polygon_yellow_value_2 = polygon_yellow_value_2[:, :, np.newaxis]

    # Mask for collision detection, containing two polygons (values range 0, 1, 2)
    polygon_mask_blue_and_yellow = polygon_blue_value_1 + polygon_yellow_value_2

    # Resize from 1920x1080 to 960x540
    polygon_mask_blue_and_yellow = cv2.resize(polygon_mask_blue_and_yellow, (960, 540))

    # Blue color plate b,g,r
    blue_color_plate = [255, 0, 0]
    # Blue polygon image
    blue_image = np.array(polygon_blue_value_1 * blue_color_plate, np.uint8)

    # Yellow color plate
    yellow_color_plate = [0, 255, 255]
    # Yellow polygon image
    yellow_image = np.array(polygon_yellow_value_2 * yellow_color_plate, np.uint8)

    # Color image (values range 0-255)
    color_polygons_image = blue_image + yellow_image
    # Resize from 1920x1080 to 960x540
    color_polygons_image = cv2.resize(color_polygons_image, (960, 540))

    # List overlapping with blue polygon
    list_overlapping_blue_polygon = []

    # List overlapping with yellow polygon
    list_overlapping_yellow_polygon = []

    # Number of entries
    down_count = 0
    # Number of exits
    up_count = 0

    font_draw_number = cv2.FONT_HERSHEY_SIMPLEX
    draw_text_postion = (int(960 * 0.01), int(540 * 0.05))

    # Initialize yolov5
    detector = Detector()

    #list_bboxs = []

    # Open video
    # capture = cv2.VideoCapture('./video/test.mp4')
    capture = cv2.VideoCapture('./test_clipped.MP4')

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use 'XVID', 'DIVX', 'MJPG', etc.
    out = cv2.VideoWriter('output_demo.mp4', fourcc, 30.0, (960, 540))  # Adjust frame size (960, 540) to your output frame size

    while True:
        # Read each frame
        _, im = capture.read()
        if im is None:
            break

        # Resize from 1920x1080 to 960x540
        im = cv2.resize(im, (960, 540))

        list_bboxs = []
        bboxes = detector.detect(im)

        # Initialize mask2former
        # result = inference_detector(model, im)
            
        # Extract bounding boxes and labels from result
        # bboxes = result.pred_instances.bboxes.cpu().numpy() if hasattr(result, 'pred_instances') else []
        # labels = result.pred_instances.labels.cpu().numpy() if hasattr(result, 'pred_instances') else []
        # confs = result.pred_instances.scores.cpu().numpy() if hasattr(result, 'pred_instances') else []

        if len(bboxes) > 0:
            # for bbox, label, conf in zip(bboxes, labels, confs):
            #    x1, y1, x2, y2 = bbox   
            #    list_bboxs.append([x1, y1, x2, y2, label, conf])  # Track ID can be set to None initially

            # Update tracker with detected bounding boxes
            list_bboxs = tracker.update(bboxes, im)

            # Draw bounding boxes
            output_image_frame = tracker.draw_bboxes(im, list_bboxs, line_thickness=None)
        else:
            output_image_frame = im

        # Combine with polygon mask
        output_image_frame = cv2.add(output_image_frame, color_polygons_image)

        if len(list_bboxs) > 0:
            for item_bbox in list_bboxs:
                x1, y1, x2, y2, label, track_id = item_bbox

                y1_offset = int(y1 + ((y2 - y1) * 0.6))
                y, x = y1_offset, x1

                if polygon_mask_blue_and_yellow[y, x] == 1:
                    if track_id not in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.append(track_id)

                    if track_id in list_overlapping_yellow_polygon:
                        up_count += 1
                        print(f'Category: {label} | id: {track_id} | Exit collision | Total exit collisions: {up_count} | Exit id list: {list_overlapping_yellow_polygon}')
                        list_overlapping_yellow_polygon.remove(track_id)

                elif polygon_mask_blue_and_yellow[y, x] == 2:
                    if track_id not in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.append(track_id)

                    if track_id in list_overlapping_blue_polygon:
                        down_count += 1
                        print(f'Category: {label} | id: {track_id} | Entry collision | Total entry collisions: {down_count} | Entry id list: {list_overlapping_blue_polygon}')
                        list_overlapping_blue_polygon.remove(track_id)

            list_overlapping_all = list_overlapping_yellow_polygon + list_overlapping_blue_polygon
            for id1 in list_overlapping_all:
                is_found = False
                for _, _, _, _, _, bbox_id in list_bboxs:
                    if bbox_id == id1:
                        is_found = True
                        break

                if not is_found:
                    if id1 in list_overlapping_yellow_polygon:
                        list_overlapping_yellow_polygon.remove(id1)
                    if id1 in list_overlapping_blue_polygon:
                        list_overlapping_blue_polygon.remove(id1)
            list_overlapping_all.clear()

            list_bboxs.clear()
        else:
            list_overlapping_blue_polygon.clear()
            list_overlapping_yellow_polygon.clear()

        text_draw = f'DOWN: {down_count} , UP: {up_count}'
        output_image_frame = cv2.putText(img=output_image_frame, text=text_draw,
                                         org=draw_text_postion,
                                         fontFace=font_draw_number,
                                         fontScale=1, color=(255, 255, 255), thickness=2)

        out.write(output_image_frame)

        cv2.imshow('demo', output_image_frame)
        cv2.waitKey(1)
        
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    capture.release()
    out.release()
    cv2.destroyAllWindows()