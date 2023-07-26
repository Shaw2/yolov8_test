# ### 23.07.25~ 테스트

# import cv2
# import argparse

# from ultralytics import YOLO
# import supervision as sv
# import numpy as np

# # 640, 480으로 수정해야할 것 같음
# # ZONE의 크기를 고정하지 않고 입력받은 카메라 resolution에 따라서 자동적으로 변하게 세팅 0.5
# ZONE_POLYGON = np.array([
#     # [0, 0],
#     # [640 // 2, 0],
#     # [640 // 2, 480],
#     # [0, 480]
#     [0, 0],
#     [0.5, 0],
#     [0.5, 1],
#     [0, 1]    
# ])

# def parse_arguments() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="YOLOv8 live")
#     parser.add_argument(
#         "--webcam-resolution", 
#         default=[1280, 720], 
#         nargs=2, 
#         type=int
#     )
#     args = parser.parse_args()
#     return args


# def main():
#     args = parse_arguments()
#     frame_width, frame_height = args.webcam_resolution

#     cap = cv2.VideoCapture(0)
#     # 왜인지는 모르겠지만 .set()이 적용이 안된다. 800, 480을 쳐도 640, 480으로 나옴. 우선 진행
#     # 다른것 때문에 pip uninstall opencv-python pip install opencv-python 했는데 여전히 안됨 (error: (-215:Assertion failed))
#     # 위에꺼 해결할 겸 opencv를 다운그레이드 해보자!
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

#     model = YOLO("yolov8s.pt")

#     box_annotator = sv.BoxAnnotator(
#         thickness=2,
#         text_thickness=2,
#         text_scale=1
#     )

#     # 지금은 webcam resolution을 따로 받고있어서 써주지 않으면 default로 되어있기 때문에
#     # 1280, 720의 절반인 640, 360의 zone이 만들어짐. 이러면 640, 480에서는 전체를 의미하므로
#     # 실행시 python -m main --webcam-resolution 640 480 으로 켜줘서 절반만 인식하도록 만들기 
#     zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
#     zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
#     zone_annotator = sv.PolygonZoneAnnotator(
#         zone=zone,
#         color=sv.Color.red(),
#         thickness=2,
#         text_thickness=4,
#         text_scale=2
#     )

#     while True:
#         ret, frame = cap.read()

#         # 가끔 한 사물에 2개씩 적용되는 것을 하나만 되게 바꾸는 방법 
#         result = model(frame, agnostic_nms=True)[0]
#         detections = sv.Detections.from_yolov8(result)
        
#         ### detection에서 person class를 제거하는 것. 손 제거용
#         # detections = detections[detections.class_id != 0]

#         ### 이걸 detecting할 class로 바꾸면 그것만 보게 할 수 있음
#         detections = detections[detections.class_id == 47]

#         # 여기는 pip install supervision==0.3.0으로 다운그레이드필요
#         labels = [
#             f"{model.model.names[class_id]} {confidence:0.2f}"
#             for _, confidence, class_id, _
#             in detections
#         ]

#         # 여기는 opencv-python 삭제 후 재설치 필요
#         frame = box_annotator.annotate(
#             scene=frame, 
#             detections=detections,
#             labels=labels            
#             )

#         zone.trigger(detections=detections)
#         frame = zone_annotator.annotate(scene=frame)

#         cv2.imshow("yolov8", frame)

#         if(cv2.waitKey(30)==27):
#             break

# if __name__ == "__main__":
#     main()


import cv2
import argparse

from ultralytics import YOLO
import supervision as sv
import numpy as np


ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution



    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8l.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone, 
        color=sv.Color.red(),
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    while True:
        ret, frame = cap.read()

        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _
            in detections
        ]
        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections, 
            labels=labels
        )

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)      
        
        cv2.imshow("yolov8", frame)

        print(frame.shape)
        #break

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()