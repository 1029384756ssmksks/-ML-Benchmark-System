import json
import numpy as np
import logging
from backend.algorithms.ml10 import serve_request
from backend.dataset_loader import load_dataset

logger = logging.getLogger(__name__)


def handle_algorithm_request(request_data: dict) -> dict:
    """
    对接前端请求与ml10算法模块的核心服务函数
    """
    try:
        # 1. 解析前端请求参数
        algo_name = request_data["algorithm"]
        action = request_data["action"]
        dataset_name = request_data["dataset"]
        params = request_data.get("params", {})
        state = request_data.get("state", None)

        logger.info(f"处理算法请求 - 算法: {algo_name}, 动作: {action}, 数据集: {dataset_name}")

        # 2. 加载数据集
        X, y = load_dataset(dataset_name)
        X = X.tolist()  # 转为list，适配ml10.py的输入要求
        y = y.tolist() if y is not None else None

        # 3. 构造ml10.py的serve_request输入
        ml10_payload = {
            "algo": algo_name,
            "action": action,
            "X": X,
            "y": y,
            "params": params,
            "state": state
        }

        # 4. 调用组员A的算法模块
        ml10_result = serve_request(ml10_payload)

        if not ml10_result["ok"]:
            return {"code": 400, "message": f"算法运行失败：{ml10_result['message']}", "data": {}}

        # 5. 整理基础结果
        response_data = {
            "code": 200,
            "message": "success",
            "data": {
                "basic_info": {
                    "algorithm": algo_name,
                    "dataset": dataset_name,
                    "task_type": ml10_result["task_type"],
                    "action": action
                },
                "metrics": ml10_result["metrics"] or {},
                "y_pred": ml10_result["y_pred"] or [],
                "y_proba": ml10_result["y_proba"] or [],
                "state": ml10_result["state"] or None
            }
        }

        # 6. 针对特殊算法补充差异化结果
        if algo_name == "pca" and action == "train":
            # 从ml10的PCA模型state中提取解释方差比
            pca_mean = ml10_result["state"]["mean"]
            pca_var = ml10_result["state"]["var"]
            n_components = params.get("n_components", 2)
            explained_variance_ratio = [var / sum(pca_var) for var in pca_var[:n_components]]
            cumulative_var = np.cumsum(explained_variance_ratio).tolist()

            response_data["data"]["metrics"]["explained_variance_ratio"] = explained_variance_ratio
            response_data["data"]["metrics"]["cumulative_explained_variance"] = cumulative_var

        elif algo_name == "kmeans" and action == "train":
            # 提取KMeans聚类中心
            response_data["data"]["cluster_centers"] = ml10_result["state"]["centers"]

        logger.info(f"算法执行成功: {algo_name}, 指标: {response_data['data']['metrics']}")
        return response_data

    except Exception as e:
        logger.error(f"算法服务错误: {str(e)}")
        return {"code": 500, "message": f"服务端错误：{str(e)}", "data": {}}