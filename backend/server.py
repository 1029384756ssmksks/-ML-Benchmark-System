from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import logging
import time
import json
import socket

# 配置日志
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)


# 获取本机IP地址
def get_local_ip():
    try:
        # 创建一个临时socket连接来获取本机IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


# 模拟的深度学习算法（与之前相同）
class DeepLearningSimulator:
    @staticmethod
    def linear_regression(dataset):
        """模拟线性回归算法"""
        logger.info(f"执行线性回归算法，数据集: {dataset}")
        time.sleep(2)
        return {
            "algorithm": "linear_regression",
            "dataset": dataset,
            "results": {
                "coefficients": [0.5, -1.2, 0.8],
                "intercept": 2.1,
                "r_squared": 0.89,
                "mse": 0.45
            },
            "status": "completed"
        }

    # ... 其他算法方法保持不变，为简洁起见省略 ...
    @staticmethod
    def logistic_regression(dataset):
        logger.info(f"执行逻辑回归算法，数据集: {dataset}")
        time.sleep(2)
        return {
            "algorithm": "logistic_regression",
            "dataset": dataset,
            "results": {"accuracy": 0.85, "precision": 0.82, "recall": 0.87, "f1_score": 0.84},
            "status": "completed"
        }

    @staticmethod
    def decision_tree(dataset):
        logger.info(f"执行决策树算法，数据集: {dataset}")
        time.sleep(3)
        return {
            "algorithm": "decision_tree",
            "dataset": dataset,
            "results": {"accuracy": 0.92, "max_depth": 5, "feature_importance": [0.3, 0.25, 0.2, 0.15, 0.1]},
            "status": "completed"
        }

    @staticmethod
    def random_forest(dataset):
        logger.info(f"执行随机森林算法，数据集: {dataset}")
        time.sleep(4)
        return {
            "algorithm": "random_forest",
            "dataset": dataset,
            "results": {"accuracy": 0.95, "n_estimators": 100, "oob_score": 0.93},
            "status": "completed"
        }

    @staticmethod
    def svm(dataset):
        logger.info(f"执行SVM算法，数据集: {dataset}")
        time.sleep(3)
        return {
            "algorithm": "svm",
            "dataset": dataset,
            "results": {"accuracy": 0.88, "support_vectors": 150, "kernel": "rbf"},
            "status": "completed"
        }

    @staticmethod
    def kmeans(dataset):
        logger.info(f"执行K均值算法，数据集: {dataset}")
        time.sleep(2)
        return {
            "algorithm": "kmeans",
            "dataset": dataset,
            "results": {"n_clusters": 3, "inertia": 45.2, "iterations": 10},
            "status": "completed"
        }

    @staticmethod
    def neural_network(dataset):
        logger.info(f"执行神经网络算法，数据集: {dataset}")
        time.sleep(5)
        return {
            "algorithm": "neural_network",
            "dataset": dataset,
            "results": {"accuracy": 0.96, "loss": 0.15, "epochs": 50, "architecture": "3-layer MLP"},
            "status": "completed"
        }

    @staticmethod
    def pca(dataset):
        logger.info(f"执行PCA算法，数据集: {dataset}")
        time.sleep(2)
        return {
            "algorithm": "pca",
            "dataset": dataset,
            "results": {"explained_variance_ratio": [0.45, 0.25, 0.15], "n_components": 3},
            "status": "completed"
        }

    @staticmethod
    def knn(dataset):
        logger.info(f"执行KNN算法，数据集: {dataset}")
        time.sleep(2)
        return {
            "algorithm": "knn",
            "dataset": dataset,
            "results": {"accuracy": 0.87, "n_neighbors": 5, "distance_metric": "euclidean"},
            "status": "completed"
        }

    @staticmethod
    def gradient_boosting(dataset):
        logger.info(f"执行梯度提升算法，数据集: {dataset}")
        time.sleep(4)
        return {
            "algorithm": "gradient_boosting",
            "dataset": dataset,
            "results": {"accuracy": 0.94, "learning_rate": 0.1, "n_estimators": 100},
            "status": "completed"
        }


# 可用算法映射
ALGORITHMS = {
    "linear_regression": DeepLearningSimulator.linear_regression,
    "logistic_regression": DeepLearningSimulator.logistic_regression,
    "decision_tree": DeepLearningSimulator.decision_tree,
    "random_forest": DeepLearningSimulator.random_forest,
    "svm": DeepLearningSimulator.svm,
    "kmeans": DeepLearningSimulator.kmeans,
    "neural_network": DeepLearningSimulator.neural_network,
    "pca": DeepLearningSimulator.pca,
    "knn": DeepLearningSimulator.knn,
    "gradient_boosting": DeepLearningSimulator.gradient_boosting
}


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    logger.info('客户端已连接')
    emit('connection_response', {'message': '连接成功', 'status': 'connected'})


@socketio.on('disconnect')
def handle_disconnect():
    logger.info('客户端已断开连接')


@socketio.on('run_algorithm')
def handle_run_algorithm(data):
    try:
        algorithm = data.get('algorithm')
        dataset = data.get('dataset')

        logger.info(f"收到算法执行请求 - 算法: {algorithm}, 数据集: {dataset}")

        if not algorithm or not dataset:
            emit('algorithm_error', {'error': '算法和数据集不能为空'})
            return

        if algorithm not in ALGORITHMS:
            emit('algorithm_error', {'error': f'不支持的算法: {algorithm}'})
            return

        emit('algorithm_status', {'status': 'processing', 'message': '算法执行中...'})

        algorithm_func = ALGORITHMS[algorithm]
        result = algorithm_func(dataset)

        logger.info(f"算法执行完成: {algorithm}")
        emit('algorithm_result', result)

    except Exception as e:
        logger.error(f"算法执行错误: {str(e)}")
        emit('algorithm_error', {'error': f'算法执行错误: {str(e)}'})


@socketio.on('ping')
def handle_ping():
    logger.debug('收到ping请求')
    emit('pong', {'message': 'pong', 'timestamp': time.time()})


if __name__ == '__main__':
    local_ip = get_local_ip()
    logger.info("=" * 50)
    logger.info("深度学习算法服务器启动成功!")
    logger.info(f"本地访问: http://localhost:5000")
    logger.info(f"网络访问: http://{local_ip}:5000")
    logger.info("=" * 50)

    socketio.run(app, debug=True, host='0.0.0.0', port=5000)