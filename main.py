"""
============================================================
라즈베리 파이 Flask YOLO CCTV 시스템 (멀티 카메라 지원)
============================================================
기능:
1. 사용 가능한 카메라 자동 검색 (USB/내장)
2. 웹 페이지에서 카메라 선택 후 [ON]
3. 실시간 YOLO 감지 및 스트리밍
============================================================
"""

from flask import Flask, Response, jsonify, request
import cv2
import json
import threading
from datetime import datetime
import os
import sys
import io

# [필수] 윈도우 터미널 인코딩 오류 방지
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

# ============================================================
# [1단계] 시스템 초기화
# ============================================================

print("=" * 60)
print("🖥️  라즈베리 파이 Flask YOLO CCTV 시스템 (카메라 선택 가능)")
print("=" * 60)

# YOLO 모델 로드
try:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    YOLO_AVAILABLE = True
    print("[✅] YOLO 모델 로드 성공!")
except Exception as e:
    print(f"[❌] YOLO 모델 로드 실패: {e}")
    YOLO_AVAILABLE = False
    model = None

# 사용 가능한 카메라 인덱스 찾기
def get_available_cameras():
    """연결된 카메라 인덱스 리스트 반환 (0~3번 검색)"""
    available_cameras = []
    # 0번부터 3번 포트까지 빠르게 스캔
    for i in range(4):
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        except:
            pass
    return available_cameras

print("=" * 60)

app = Flask(__name__)

# ============================================================
# [2단계] 카메라 매니저 클래스
# ============================================================

class CameraManager:
    def __init__(self):
        self.camera = None
        self.is_running = False
        self.lock = threading.Lock()
        self.detection_count = 0
        self.current_camera_index = 0  # 현재 선택된 카메라 번호
        
    def start(self, camera_index=0):
        """선택한 카메라 인덱스로 시작"""
        with self.lock:
            if self.is_running:
                return {"success": True, "message": "이미 실행 중입니다"}
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 📹 카메라 {camera_index}번 연결 시도...")
            
            # 선택된 카메라 연결
            self.camera = cv2.VideoCapture(camera_index)
            
            if not self.camera.isOpened():
                return {
                    "success": False, 
                    "message": f"❌ {camera_index}번 카메라 연결 실패!"
                }
            
            # 카메라 설정
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.camera.set(cv2.CAP_PROP_FPS, 15)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_running = True
            self.current_camera_index = camera_index
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ 카메라 {camera_index}번 ON")
            
            return {"success": True, "message": f"✅ {camera_index}번 카메라 시작됨"}
    
    def stop(self):
        with self.lock:
            if not self.is_running:
                return {"success": True, "message": "이미 중지 상태입니다"}
            
            self.is_running = False
            if self.camera:
                self.camera.release()
                self.camera = None
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏹️ 카메라 OFF")
            return {"success": True, "message": "⏹️ 카메라 종료됨"}
    
    def get_frame(self):
        if not self.is_running or self.camera is None:
            return None
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                return None
            
            if YOLO_AVAILABLE and model is not None:
                frame = self.detect_objects(frame)
            
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                return jpeg.tobytes()
        except Exception:
            pass
        return None
    
    def detect_objects(self, frame):
        try:
            results = model(frame, conf=0.5, verbose=False)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    class_name = model.names[int(box.cls[0])]
                    
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {confidence:.1%}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    self.log_detection(class_name, confidence)
        except:
            pass
        return frame
    
    def log_detection(self, class_name, confidence):
        # (로그 저장 로직은 동일하게 유지 - 생략 가능하나 전체 코드 완성을 위해 포함)
        log_file = "detection_log.json"
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            new_log = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "class": class_name,
                "confidence": round(confidence, 4),
                "camera_idx": self.current_camera_index
            }
            logs.append(new_log)
            if len(logs) > 1000: logs = logs[-1000:]
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            
            self.detection_count += 1
        except:
            pass

    def get_status(self):
        return {
            "camera_on": self.is_running,
            "current_idx": self.current_camera_index,
            "yolo_available": YOLO_AVAILABLE,
            "detection_count": self.detection_count
        }

camera_manager = CameraManager()

# ============================================================
# [3단계] Flask 라우트
# ============================================================

@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>라즈베리 파이 YOLO CCTV</title>
        <meta charset="utf-8">
        <style>
            body { 
                font-family: 'Segoe UI', sans-serif; 
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: white; text-align: center; padding: 20px;
            }
            .container { max-width: 800px; margin: 0 auto; }
            
            /* 카메라 선택 드롭다운 */
            select {
                padding: 10px 15px;
                font-size: 16px;
                border-radius: 8px;
                border: 2px solid #00d4ff;
                background: #16213e;
                color: white;
                margin-right: 10px;
                cursor: pointer;
            }
            
            button {
                padding: 10px 30px; margin: 10px; font-size: 18px;
                border: none; border-radius: 10px; cursor: pointer;
                font-weight: bold; transition: 0.3s;
            }
            .btn-on { background: #00d4ff; color: #1a1a2e; }
            .btn-on:hover { background: #00ff88; box-shadow: 0 0 15px #00ff88; }
            .btn-off { background: #ff4444; color: white; }
            .btn-off:hover { background: #ff6b6b; box-shadow: 0 0 15px #ff4444; }
            
            img { 
                width: 100%; max-width: 640px; border-radius: 15px;
                border: 3px solid #00d4ff; box-shadow: 0 0 20px rgba(0,212,255,0.3);
            }
            .status-box { 
                background: rgba(255,255,255,0.1); padding: 15px; 
                border-radius: 15px; margin-bottom: 20px; 
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎥 CCTV 모니터링 시스템</h1>
            
            <div class="status-box" id="status-display">
                상태 로딩 중...
            </div>
            
            <div>
                <select id="camera-select">
                    <option value="0">카메라 검색 중...</option>
                </select>
                
                <button class="btn-on" onclick="cameraOn()">▶️ Start</button>
                <button class="btn-off" onclick="cameraOff()">⏹️ Stop</button>
            </div>
            
            <br>
            <img id="video" src="/video_feed" alt="Camera OFF">
        </div>

        <script>
            // 페이지 로드 시 카메라 목록 가져오기
            window.onload = function() {
                loadCameras();
                updateStatus();
                setInterval(updateStatus, 2000);
            };

            function loadCameras() {
                fetch('/cameras')
                    .then(r => r.json())
                    .then(cams => {
                        const select = document.getElementById('camera-select');
                        select.innerHTML = '';
                        if (cams.length === 0) {
                            select.innerHTML = '<option value="-1">❌ 카메라 없음</option>';
                            return;
                        }
                        cams.forEach(camIdx => {
                            let option = document.createElement('option');
                            option.value = camIdx;
                            option.text = `📷 Camera ${camIdx}`;
                            select.appendChild(option);
                        });
                    });
            }

            function cameraOn() {
                const select = document.getElementById('camera-select');
                const camIdx = parseInt(select.value);
                
                if (camIdx < 0) {
                    alert("사용 가능한 카메라가 없습니다.");
                    return;
                }

                fetch('/control', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: 'on', camera_index: camIdx})
                })
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    updateStatus();
                    document.getElementById('video').src = '/video_feed?' + Date.now();
                });
            }

            function cameraOff() {
                fetch('/control', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({action: 'off'})
                })
                .then(r => r.json())
                .then(data => {
                    alert(data.message);
                    document.getElementById('video').src = ""; // 영상 끄기
                    updateStatus();
                });
            }

            function updateStatus() {
                fetch('/status')
                    .then(r => r.json())
                    .then(data => {
                        let statusText = `
                            상태: ${data.camera_on ? '🟢 <b>ON</b> (Cam ' + data.current_idx + ')' : '🔴 <b>OFF</b>'} | 
                            YOLO: ${data.yolo_available ? '✅' : '❌'} | 
                            탐지: ${data.detection_count}회
                        `;
                        document.getElementById('status-display').innerHTML = statusText;
                    });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/cameras')
def list_cameras():
    """사용 가능한 카메라 목록 반환 API"""
    cams = get_available_cameras()
    return jsonify(cams)

@app.route('/control', methods=['POST'])
def control():
    """카메라 제어 API"""
    data = request.get_json()
    action = data.get('action', '').lower()
    
    if action == 'on':
        # 클라이언트가 선택한 카메라 번호를 받음 (기본값 0)
        idx = int(data.get('camera_index', 0))
        result = camera_manager.start(idx)
    elif action == 'off':
        result = camera_manager.stop()
    else:
        result = {"success": False, "message": "잘못된 요청"}
    
    return jsonify(result)

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = camera_manager.get_frame()
            if frame is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                import time
                time.sleep(0.1)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify(camera_manager.get_status())

# ============================================================
# [4단계] 서버 실행
# ============================================================

if __name__ == '__main__':
    print("🚀 서버 시작... (http://localhost:5000)")
    # 초기 카메라 스캔 (정보 표시용)
    cams = get_available_cameras()
    print(f"🔎 감지된 카메라 인덱스: {cams}")
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)