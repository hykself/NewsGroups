# 20 Newsgroups 文本分类项目

## 项目结构说明

### 核心代码模块
1. `textPreprocessor.py` - 文本预处理模块  
   - 功能：实现数据加载、分词、停用词处理、词干提取等预处理功能
   - 依赖：nltk, sklearn

2. `knn_train.py` - KNN分类器训练模块  
   - 功能：训练K近邻分类器并保存模型
   - 输出：`knn_classifier.pkl`（分类器）、`knn_vectorizer.pkl`（特征提取器）

3. `dl_train.py` - 深度学习模型训练模块  
   - 功能：构建深度神经网络模型并进行训练
   - 输出：`dnn_model.h5`（Keras模型）、`label_encoder.pkl`（标签编码器）

4. `compare_model.py` - 模型对比分析模块  
   - 功能：对比不同模型的准确率和F1分数
   - 输出：`accuracy_comparison.png`、`f1_comparison.png`

5. `evaluate.py` - 模型评估模块  
   - 功能：生成分类报告、混淆矩阵可视化
   - 依赖：matplotlib, seaborn

### 生成文件
- `model_performance/` - 模型性能可视化目录
- `accuracy_comparison.png` - 准确率对比图
- `f1_comparison.png` - F1分数对比图
- `*.pkl` - 序列化的模型和预处理对象
  - `label_encoder.pkl` - 标签编码器
  - `knn_classifier.pkl` - KNN分类器
  - `knn_vectorizer.pkl` - KNN特征提取器
  - `tfidf_vectorizer.pkl`     - 贝叶斯模型专用TF-IDF特征提取器
  - `nb_classifier.pkl`        - 训练好的朴素贝叶斯分类器模型
  - `knn_vectorizer.pkl`       - KNN模型专用TF-IDF特征提取器 
  - `knn_classifier.pkl`       - 训练好的K近邻分类器模型
  - `label_encoder.pkl`        - 深度学习标签编码器（将文本标签转换为数字）
  - `dnn_model.h5`             - 保存的Keras神经网络模型权重

### 数据文件
- `20_newsgroups/` - 原始数据集目录
- `processed_data.txt` - 预处理后的文本数据
- `train_data.txt`/`test_data.txt` - 训练/测试集划分

### 评估输出
- `accuracy_comparison.png`  - 模型准确率对比图
- `f1_comparison.png`        - F1分数对比图

### 系统依赖
- `requirement.txt`           - Python依赖清单
- `nltk_data`              - NLTK资源缓存（自动生成）

## 使用指南

### 安装依赖
```bash
# 训练所有模型
python bys_train.py
python knn_train.py
python dl_train.py

# 执行模型对比
python compare_model.py