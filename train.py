# import warnings
# warnings.filterwarnings('ignore')
# from ultralytics import YOLO
#
# if __name__ == '__main__':
#     model = YOLO('D:/w/2O/jyz/ultralytics-main+SPDConv+SE+WIOU/ultralytics/cfg/models/11/yolo11_SPDConv+SE.yaml')
#     # model.load('yolov8n.pt') # loading pretrain weights
#     model.train(data='D:/w/2O/jyz/ultralytics-main+SPDConv+SE+WIOU/ultralytics/cfg/datasets/my_detect.yaml',
#                 cache=True,
#                 imgsz=640,
#                 epochs=300,
#                 batch=64,
#                 close_mosaic=10,
#                 workers=16,
#                 device='0',
#                 optimizer='SGD', # using SGD
#                 # resume='runs/GF2_seg/SMLS-YOLO-enhance/weights/last.pt', # last.pt path
#                 # amp=False, # close amp
#                 # fraction=0.2,
#                 conf=0.02,
#                 project='runs/1217',
#                 name='yolo11_SPDConv+SE',
#                 )


import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

# BILIBILI UP 魔傀面具
# 训练参数官方详解链接：https://docs.ultralytics.com/modes/train/#resuming-interrupted-trainings:~:text=a%20training%20run.-,Train%20Settings,-The%20training%20settings

# 指定显卡和多卡训练问题 统一都在<YOLOV11配置文件.md>下方常见错误和解决方案。
# 训练过程中loss出现nan，可以尝试关闭AMP，就是把下方amp=False的注释去掉。

if __name__ == '__main__':
    model = YOLO(r'D:/w/2O/jyz/ultralytics-main+SPDConv+SE+WIOU/ultralytics/cfg/models/11/yolo11_SPDConv+SE.yaml')
    # model.load('yolo11n.pt') # loading pretrain weights
    model.train(data=r'D:/w/2O/jyz/ultralytics-main+SPDConv+SE+WIOU/ultralytics/cfg/datasets/my_detect.yaml',
                cache=False,
                # imgsz=640,
                epochs=300,
                batch=64,
                close_mosaic=0,
                workers=16,
                # device='0',
                optimizer='SGD', # using SGD
                # patience=0, # close earlystop
                # resume=True, # 断点续训,YOLO初始化时选择last.pt
                amp=True, # close amp
                # fraction=0.2,
                lr0=0.01,
                lrf=0.01,
                project='runs/1217.5',
                name='our',
                )