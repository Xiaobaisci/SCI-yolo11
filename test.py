# from ultralytics import YOLO
#
# # Load a pretrained YOLOv10n model
# model = YOLO("E:/200/c2f_pki_ultralytics-main/ultralytics-main/runs/detect/train_v82/weights/best.pt")
#
# # Perform object detection on an image
# # results = model("test1.jpg")
# results = model.predict("D:/Users/wjy70/Desktop/val/crazing_20.jpg")
#
# # Display the results
# results[0].show()
import os
from ultralytics import YOLO

# 加载预训练的YOLO模型
model = YOLO("./best_ys.pt")

# 要检测的图片文件夹路径
input_folder = "D:/w/2O/jyz/val"
output_folder = os.path.join(input_folder, "sci-yolo11")

# 如果结果文件夹不存在，创建它
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历文件夹中的所有图片文件
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 检查图片格式
        img_path = os.path.join(input_folder, filename)

        # 对图片进行目标检测
        results = model.predict(img_path)

        # 保存每张图片的检测结果到新文件夹
        for i, result in enumerate(results):
            result_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_result_{i}.jpg")
            result.save(result_path)

print(f"检测完成，结果保存在 {output_folder} 文件夹中。")

