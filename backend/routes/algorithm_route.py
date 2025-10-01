from flask import Blueprint, request, jsonify
from flask_socketio import emit
from backend.services.algorithm_service import handle_algorithm_request
import logging

logger = logging.getLogger(__name__)

# 创建蓝图
algorithm_bp = Blueprint('algorithm', __name__, url_prefix='/api/algorithm')


# HTTP接口（用于同步算法请求）
@algorithm_bp.route('/run', methods=['POST'])
def run_algorithm():
    try:
        request_data = request.get_json()
        logger.info(f"收到HTTP算法请求: {request_data['algorithm']} - {request_data['dataset']}")

        response = handle_algorithm_request(request_data)
        return jsonify(response)

    except Exception as e:
        logger.error(f"HTTP接口错误: {str(e)}")
        return jsonify({"code": 500, "message": f"接口错误：{str(e)}", "data": {}})


# WebSocket事件处理
def register_socket_events(socketio):
    @socketio.on('connect')
    def handle_connect():
        logger.info('客户端已连接')
        emit('connection_response', {'message': '连接成功', 'status': 'connected'})

    @socketio.on('disconnect')
    def handle_disconnect():
        logger.info('客户端已断开连接')

    @socketio.on('run_algorithm')
    def handle_algorithm_socket(data):
        """WebSocket实时处理算法请求"""
        try:
            algorithm = data.get('algorithm')
            dataset = data.get('dataset')
            logger.info(f"收到WebSocket算法请求 - 算法: {algorithm}, 数据集: {dataset}")

            # 发送处理中状态
            emit('algorithm_status', {'status': 'processing', 'message': '算法执行中...'})

            # 调用算法服务
            response = handle_algorithm_request(data)

            if response["code"] == 200:
                emit('algorithm_result', response["data"])
                logger.info(f"算法执行完成: {algorithm}")
            else:
                emit('algorithm_error', {'error': response["message"]})

        except Exception as e:
            error_msg = f"算法执行错误: {str(e)}"
            logger.error(error_msg)
            emit('algorithm_error', {'error': error_msg})

    @socketio.on('ping')
    def handle_ping():
        emit('pong', {'message': 'pong', 'timestamp': __import__('time').time()})