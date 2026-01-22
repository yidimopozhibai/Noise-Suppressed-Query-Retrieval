import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

# 预测框粗细和颜色修改问题可看<使用说明.md>下方的<YOLOV8源码常见疑问解答小课堂>第六点

if __name__ == '__main__':
    model = RTDETR('runs/train/exp19/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/images/test',
                  conf=0.25,
                  project='runs/detect',
                  name='exp',
                  save=True,
                  # visualize=True # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=True, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                  )