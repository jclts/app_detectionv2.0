from flask import Flask, request, jsonify
import os
from PIL import Image
import torch
import io

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Tải mô hình YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Hoặc thay 'yolov5s' bằng mô hình khác

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Kiểm tra xem có file trong request không
        if 'image' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Lưu file
        image_path = os.path.join(UPLOAD_FOLDER, 'image.jpg')
        file.save(image_path)

        # Mở và xử lý ảnh
        image = Image.open(image_path)

        # Tiến hành nhận diện đối tượng
        result = detect_objects(image)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

def detect_objects(image):
    # Chuyển ảnh thành định dạng phù hợp cho YOLOv5
    results = model(image)  # YOLOv5 model dự đoán

    # Lấy kết quả dự đoán
    detections = results.pandas().xywh[0].to_dict(orient="records")

    # Chuyển kết quả thành định dạng JSON
    detection_result = {
        'detections': [{'label': detection['name'], 'confidence': detection['confidence']} for detection in detections]
    }

    return detection_result

if __name__ == '__main__':
    app.run(debug=True, port=5000)