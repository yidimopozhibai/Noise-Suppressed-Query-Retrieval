import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# rtdetr与yolov8一致均采用yolo头
# 最终论文的参数量和计算量统一以这个脚本运行出来的为准

if __name__ == '__main__':
    model = RTDETR('runs/train/best.pt')
    model.val(data='dataset/data.yaml',
              split='val', # split可以选择train、val、test 根据自己的数据集情况来选择.
              imgsz=640,
              batch=4,
              save_json=False, # if you need to cal coco metrice
              project='runs/val',
              name='exp',
              )
