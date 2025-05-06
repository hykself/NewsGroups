# 20 Newsgroups 文本分类项目

## 项目结构说明

### 核心代码模块
````
├── textPreprocessor.py       # 文本预处理模块
│   ├── 数据加载与清洗
│   ├── 停用词过滤/词干提取
│   └── 数据集划分(train/test)
├── knn_train.py              # KNN分类器训练
│   └── 输出knn_classifier.pkl和knn_vectorizer.pkl
├── dl_train.py               # 深度神经网络训练 
│   └── 输出dnn_model.h5和label_encoder.pkl
├── bys_train.py              # 朴素贝叶斯分类器训练
│   └── 输出nb_classifier.pkl和tfidf_vectorizer.pkl 
├── evaluate.py               # 模型评估模块
│   ├── 生成分类报告
│   ├── 混淆矩阵可视化
│   └── 保存评估图表
├── compare_model.py          # 模型对比分析
│   ├── 准确率/F1分数对比
│   └── 生成性能对比图表
````
### 生成文件
````
├── model_performance/         # 模型性能可视化
│   ├── *_confusion_matrix.png # 混淆矩阵
│   ├── accuracy_comparison.png 
│   └── f1_comparison.png      
├── *.pkl                      # 序列化对象
│   ├── nb_classifier.pkl      # 朴素贝叶斯分类器 (bys_train生成)
│   ├── knn_classifier.pkl     # K近邻分类器 (knn_train生成)
│   ├── tfidf_vectorizer.pkl   # TF-IDF特征提取器 (bys_train/knn_train共用)
│   └── label_encoder.pkl       # 标签编码器 (dl_train生成)
└── dnn_model.h5              # 神经网络模型权重 (dl_train生成)
````
### 数据文件
````
├── 20_newsgroups/            # 原始数据集
├── processed_data.txt        # 完整预处理数据
│   └── 格式：每行包含[标签] [预处理文本]，如：
│       comp.graphics 23423 3d render comput graph...
├── train_data.txt            # 训练集(80%)
│   └── 用于模型训练的特征数据
├── test_data.txt             # 测试集(20%) 
│   └── 保留用于最终模型评估
````
### 评估报告
````
├── classification_report.txt  # 详细评估指标
│   └── 包含每个类别的precision/recall/f1-score，如：
│               precision  recall  f1-score  support
│   comp.graphics       0.89      0.87      0.88       319
│        sci.med       0.92      0.85      0.88       298
└── model_performance/
    ├── NaiveBayes_report.txt  # 贝叶斯模型详细报告
    ├── KNN_report.txt         # KNN模型详细报告
    └── DNN_report.txt         # 神经网络详细报告
````
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