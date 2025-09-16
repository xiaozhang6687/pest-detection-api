#!/usr/bin/env python3
"""
病虫害检测后端API服务 - app.py
集成优化模型 + 智能决策系统
"""

import warnings
import os
import sys
import time
import json
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import base64
import io
from PIL import Image
import torch
from ultralytics import YOLO
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
import traceback

# 忽略警告
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask应用
app = Flask(__name__)
CORS(app)

class PestDetectionAPI:
    """病虫害检测API核心类"""
    
    def __init__(self):
        # 模型配置 - 使用绝对路径
        self.model_path = r"D:\ultralytics-main\optimization_results\models\optimized_best.pt"
        self.backup_model_path = r"D:\ultralytics-main\runs\train\exp2\weights\best.pt"
        
        # 优化后的参数
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.5
        self.input_size = 416
        
        # 加载模型和类别映射
        self.model = None
        self.class_names = {}
        self.load_model()
        self.load_class_names()
        
        # 初始化数据库
        self.init_database()
        
        logger.info("🚀 病虫害检测API初始化完成")
    
    def load_model(self):
        """加载YOLO模型"""
        try:
            # 首先尝试加载优化模型
            if os.path.exists(self.model_path):
                self.model = YOLO(self.model_path)
                logger.info(f"✅ 优化模型加载成功: {self.model_path}")
            elif os.path.exists(self.backup_model_path):
                self.model = YOLO(self.backup_model_path)
                logger.info(f"✅ 备用模型加载成功: {self.backup_model_path}")
                # 如果使用备用模型，调整参数
                self.confidence_threshold = 0.4
                self.iou_threshold = 0.7
                self.input_size = 640
            else:
                logger.error(f"❌ 未找到模型文件")
                logger.error(f"   优化模型路径: {self.model_path}")
                logger.error(f"   备用模型路径: {self.backup_model_path}")
                self.model = None
                return
                
            # 模型预热
            dummy_image = np.random.randint(0, 255, (self.input_size, self.input_size, 3), dtype=np.uint8)
            self.model(dummy_image, conf=self.confidence_threshold, verbose=False)
            logger.info("✅ 模型预热完成")
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            self.model = None
    
    def load_class_names(self):
        """加载类别名称映射"""
        # IP102数据集的类别名称（根据您的数据集调整）
        self.class_names = {
            0: "稻飞虱", 1: "稻叶蝉", 2: "稻螟虫", 3: "稻纵卷叶螟", 4: "稻瘟病",
            5: "玉米螟", 6: "玉米叶斑病", 7: "小麦蚜虫", 8: "小麦条纹花叶病", 9: "棉铃虫",
            10: "棉花枯萎病", 11: "大豆食心虫", 12: "大豆花叶病", 13: "蚜虫", 14: "红蜘蛛",
            15: "白粉虱", 16: "烟粉虱", 17: "蓟马", 18: "叶螨", 19: "介壳虫",
            20: "粉虱", 21: "斜纹夜蛾", 22: "甜菜夜蛾", 23: "小菜蛾", 24: "菜青虫",
            25: "豆荚螟", 26: "豌豆象", 27: "蝼蛄", 28: "地老虎", 29: "金针虫",
            30: "蛴螬", 31: "叶蝉", 32: "粒黑粉病", 33: "水稻条纹叶枯病", 34: "稻瘟病",
            35: "玉米大斑病", 36: "玉米小斑病", 37: "小麦赤霉病", 38: "小麦白粉病", 39: "棉花黄萎病"
            # 更多类别...可以扩展到102个
        }
        
        # 尝试从文件加载完整映射
        class_file = "../static/class_names.json"
        if os.path.exists(class_file):
            try:
                with open(class_file, 'r', encoding='utf-8') as f:
                    file_names = json.load(f)
                    self.class_names.update(file_names)
                logger.info(f"✅ 从文件加载类别映射: {len(self.class_names)} 个类别")
            except Exception as e:
                logger.warning(f"⚠️ 无法从文件加载类别映射: {e}")
    
    def init_database(self):
        """初始化数据库"""
        try:
            os.makedirs('../logs', exist_ok=True)
            conn = sqlite3.connect('../logs/pest_detection.db')
            cursor = conn.cursor()
            
            # 创建检测历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS detection_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_name TEXT,
                    detected_pests TEXT,
                    confidence_scores TEXT,
                    detection_count INTEGER,
                    inference_time REAL,
                    client_ip TEXT
                )
            ''')
            
            # 创建系统统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE DEFAULT CURRENT_DATE,
                    total_detections INTEGER DEFAULT 0,
                    total_requests INTEGER DEFAULT 0,
                    avg_inference_time REAL DEFAULT 0
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("✅ 数据库初始化完成")
            
        except Exception as e:
            logger.error(f"❌ 数据库初始化失败: {e}")
    
    def preprocess_image(self, image_data):
        """图像预处理"""
        try:
            if isinstance(image_data, str):
                # 处理base64编码
                if 'data:image' in image_data:
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            else:
                image = image_data
            
            # 转换格式
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            return cv_image
            
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            return None
    
    def detect_pests(self, image_data, client_ip="unknown"):
        """执行病虫害检测"""
        if self.model is None:
            return {"success": False, "error": "模型未加载，请检查模型文件路径"}
        
        try:
            # 预处理图像
            processed_image = self.preprocess_image(image_data)
            if processed_image is None:
                return {"success": False, "error": "图像处理失败，请检查图像格式"}
            
            # 模型推理
            start_time = time.time()
            results = self.model(
                processed_image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                imgsz=self.input_size,
                verbose=False
            )
            inference_time = time.time() - start_time
            
            # 解析结果
            detections = []
            pest_names = []
            confidences = []
            
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    class_id = int(box.cls)
                    confidence = float(box.conf)
                    bbox = box.xyxy[0].tolist()
                    
                    pest_name = self.class_names.get(class_id, f"未知害虫{class_id}")
                    
                    detection = {
                        "class_id": class_id,
                        "class_name": pest_name,
                        "confidence": round(confidence, 3),
                        "bbox": [int(x) for x in bbox]
                    }
                    detections.append(detection)
                    pest_names.append(pest_name)
                    confidences.append(confidence)
            
            # 生成防治建议
            treatment_advice = self.generate_treatment_advice(detections)
            
            # 绘制检测结果
            result_image = self.draw_detections(processed_image, detections)
            
            # 保存检测历史
            self.save_detection_history(pest_names, confidences, inference_time, client_ip)
            
            return {
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "detection_count": len(detections),
                "detections": detections,
                "treatment_advice": treatment_advice,
                "result_image": result_image,
                "inference_time": round(inference_time, 3),
                "model_params": {
                    "confidence_threshold": self.confidence_threshold,
                    "iou_threshold": self.iou_threshold,
                    "input_size": self.input_size
                }
            }
            
        except Exception as e:
            logger.error(f"检测失败: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": f"检测失败: {str(e)}"}
    
    def generate_treatment_advice(self, detections):
        """生成防治建议"""
        if not detections:
            return """🎉 恭喜！未检测到明显的病虫害，作物状况良好。

🌱 预防性管理建议：
• 定期巡查作物生长情况，观察叶片颜色和形态变化
• 保持田间清洁，及时清除杂草和病残体
• 合理施肥，增强作物抗性，避免偏施氮肥
• 注意田间通风透光，合理密植
• 适时灌溉，避免田间积水
• 做好轮作倒茬，减少病虫害基数

💡 监测要点：
• 重点关注新叶和嫩梢部位
• 注意观察虫卵和幼虫
• 关注天气变化对病虫害发生的影响"""
        
        # 按害虫类型分组
        pest_groups = {}
        for det in detections:
            pest_name = det['class_name']
            if pest_name not in pest_groups:
                pest_groups[pest_name] = []
            pest_groups[pest_name].append(det)
        
        # 生成建议
        advice = "🔍 检测结果分析：\n"
        advice += f"发现 {len(detections)} 处病虫害，涉及 {len(pest_groups)} 种类型\n\n"
        
        advice += "🐛 检测到的病虫害：\n"
        for pest_name, pest_list in pest_groups.items():
            avg_conf = np.mean([p['confidence'] for p in pest_list])
            severity = "严重" if avg_conf > 0.7 else "中等" if avg_conf > 0.4 else "轻微"
            advice += f"• {pest_name}: {len(pest_list)} 处 (平均置信度: {avg_conf:.1%}, 危害程度: {severity})\n"
        
        advice += "\n💡 综合防治建议：\n"
        advice += "🚨 紧急措施：\n"
        advice += "• 立即隔离受害植株，防止病虫害扩散\n"
        advice += "• 清除病残体和周围杂草，减少病虫源\n"
        advice += "• 加强田间巡查，监测扩散情况\n\n"
        
        advice += "🛡️ 防治策略：\n"
        advice += "• 优先采用生物防治：释放天敌昆虫、使用生物农药\n"
        advice += "• 物理防治：黄板诱杀、灯光诱杀、人工捕杀\n"
        advice += "• 化学防治：选择对症药剂，注意轮换用药\n"
        advice += "• 农业防治：调整栽培措施，增强作物抗性\n\n"
        
        advice += "💊 用药指导：\n"
        advice += "• 选择晴朗无风天气，避开高温时段\n"
        advice += "• 严格按照标签用量，不得随意增减\n"
        advice += "• 注意药剂轮换，防止产生抗药性\n"
        advice += "• 遵守安全间隔期，确保农产品安全\n\n"
        
        advice += "📅 后续管理：\n"
        advice += "• 施药后3-5天检查防治效果\n"
        advice += "• 持续监测7-10天，防止复发\n"
        advice += "• 做好防治记录，为下次防治提供参考\n"
        advice += "• 加强田间管理，改善作物生长环境\n\n"
        
        advice += "⚠️ 重要提醒：\n"
        advice += "• 本建议仅供参考，具体用药请咨询当地植保部门\n"
        advice += "• 不同作物、不同地区的防治方法可能不同\n"
        advice += "• 建议采用综合防治策略(IPM)，减少农药使用\n"
        advice += "• 如病虫害严重，请及时联系专业技术人员指导"
        
        return advice
    
    def draw_detections(self, image, detections):
        """绘制检测结果"""
        try:
            result_image = image.copy()
            
            # 定义颜色
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色  
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 紫色
            ]
            
            for i, det in enumerate(detections):
                bbox = det['bbox']
                x1, y1, x2, y2 = bbox
                
                # 选择颜色
                color = colors[i % len(colors)]
                
                # 绘制边界框
                cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 3)
                
                # 绘制标签
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                
                # 标签背景
                cv2.rectangle(result_image, (x1, y1-label_size[1]-15), 
                            (x1+label_size[0]+10, y1), color, -1)
                
                # 标签文字
                cv2.putText(result_image, label, (x1+5, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 添加序号
                cv2.circle(result_image, (x1+10, y1+20), 15, color, -1)
                cv2.putText(result_image, str(i+1), (x1+5, y1+25), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # 转换为base64
            _, buffer = cv2.imencode('.jpg', result_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            return f"data:image/jpeg;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"绘制检测结果失败: {e}")
            return None
    
    def save_detection_history(self, pest_names, confidences, inference_time, client_ip):
        """保存检测历史"""
        try:
            conn = sqlite3.connect('../logs/pest_detection.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO detection_history 
                (detected_pests, confidence_scores, detection_count, inference_time, client_ip)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                json.dumps(pest_names, ensure_ascii=False),
                json.dumps(confidences),
                len(pest_names),
                inference_time,
                client_ip
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"保存检测历史失败: {e}")
    
    def get_system_stats(self):
        """获取系统统计信息"""
        try:
            conn = sqlite3.connect('../logs/pest_detection.db')
            cursor = conn.cursor()
            
            # 获取今日统计
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(detection_count) as total_detections,
                    AVG(inference_time) as avg_inference_time
                FROM detection_history 
                WHERE DATE(timestamp) = DATE('now')
            ''')
            
            today_stats = cursor.fetchone()
            
            # 获取总体统计
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_requests,
                    SUM(detection_count) as total_detections,
                    AVG(inference_time) as avg_inference_time
                FROM detection_history
            ''')
            
            overall_stats = cursor.fetchone()
            
            conn.close()
            
            return {
                "today": {
                    "total_requests": today_stats[0] or 0,
                    "total_detections": today_stats[1] or 0,
                    "avg_inference_time": round(today_stats[2] or 0, 3)
                },
                "overall": {
                    "total_requests": overall_stats[0] or 0,
                    "total_detections": overall_stats[1] or 0,
                    "avg_inference_time": round(overall_stats[2] or 0, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"获取系统统计失败: {e}")
            return {"today": {}, "overall": {}}

# 创建API实例
api = PestDetectionAPI()

# ==================== API路由定义 ====================

@app.route('/')
def index():
    """API首页"""
    model_status = "已加载" if api.model else "未加载"
    return f"""
    <h1>🌾 智能农作物病虫害检测API</h1>
    <h2>服务状态: 正常运行</h2>
    <h2>模型状态: {model_status}</h2>
    <h3>核心功能:</h3>
    <ul>
        <li>✅ YOLOv11模型检测病虫害 (支持{len(api.class_names)}种类别)</li>
        <li>🧠 智能防治建议生成</li>
        <li>📊 检测历史记录</li>
        <li>⚡ 优化推理性能 (conf={api.confidence_threshold}, iou={api.iou_threshold}, size={api.input_size})</li>
    </ul>
    <h3>API接口:</h3>
    <ul>
        <li><b>POST /api/detect</b> - 病虫害检测</li>
        <li><b>GET /api/health</b> - 健康检查</li>
        <li><b>GET /api/stats</b> - 系统统计</li>
        <li><b>GET /api/classes</b> - 支持的类别</li>
        <li><b>GET /api/test</b> - API测试</li>
    </ul>
    <p>当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>模型路径: {api.model_path if api.model else '模型未加载'}</p>
    """

@app.route('/api/detect', methods=['POST'])
def detect_pests():
    """病虫害检测接口"""
    try:
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({"success": False, "error": "缺少图像数据"}), 400
        
        # 获取客户端IP
        client_ip = request.remote_addr
        
        # 执行检测
        result = api.detect_pests(data['image'], client_ip)
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"API错误: {e}")
        return jsonify({
            "success": False, 
            "error": f"服务器错误: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "success": True,
        "status": "healthy",
        "model_loaded": api.model is not None,
        "model_path": api.model_path if api.model else None,
        "confidence_threshold": api.confidence_threshold,
        "iou_threshold": api.iou_threshold,
        "input_size": api.input_size,
        "supported_classes": len(api.class_names),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取系统统计信息"""
    try:
        stats = api.get_system_stats()
        return jsonify({
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """获取支持的病虫害类别"""
    return jsonify({
        "success": True,
        "classes": api.class_names,
        "total_classes": len(api.class_names),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_api():
    """API测试接口"""
    return jsonify({
        "success": True,
        "message": "API测试成功！",
        "model_status": "已加载" if api.model else "未加载",
        "system_info": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "working_directory": os.getcwd()
        },
        "timestamp": datetime.now().isoformat()
    })

# 错误处理
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "接口不存在"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "服务器内部错误"}), 500

if __name__ == '__main__':
    print("🚀 启动病虫害检测API服务...")
    print(f"🧠 模型状态: {'已加载' if api.model else '未加载'}")
    if api.model:
        print(f"📊 优化参数: conf={api.confidence_threshold}, iou={api.iou_threshold}, size={api.input_size}")
        print(f"🏷️ 支持类别: {len(api.class_names)} 种")
    print(f"📡 访问地址: http://localhost:5000")
    print(f"📚 API文档: http://localhost:5000")
    print(f"📁 当前目录: {os.getcwd()}")
    
    # 启动Flask服务
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # 生产环境设为False
        threaded=True
    ) 
