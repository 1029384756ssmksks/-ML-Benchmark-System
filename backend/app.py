from flask import Flask, render_template
from flask_socketio import SocketIO
import logging
import socket

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'ml-benchmark-secret-key'

# 使用 threading 模式替代 eventlet
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    async_mode='threading'
)

# 注册蓝图
from backend.routes.algorithm_route import algorithm_bp

app.register_blueprint(algorithm_bp)

# 注册 SocketIO 事件
from backend.routes.algorithm_route import register_socket_events

register_socket_events(socketio)


# 获取本机IP地址
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    local_ip = get_local_ip()
    logger.info("=" * 50)
    logger.info("机器学习算法基准系统启动成功!")
    logger.info(f"本地访问: http://localhost:5000")
    logger.info(f"网络访问: http://{local_ip}:5000")
    logger.info("=" * 50)

    # 添加 allow_unsafe_werkzeug=True 参数
    socketio.run(
        app,
        debug=True,
        host='0.0.0.0',
        port=5000,
        allow_unsafe_werkzeug=True
    )