# 20 Newsgroups 文本分类项目

## 新增文件说明

### 新增核心代码文件
1. `knn_train.py` - KNN分类器训练模块
   - 功能：使用K近邻算法进行模型训练
   - 输出：`knn_classifier.pkl`、`knn_vectorizer.pkl`

2. `dl_train.py` - 深度神经网络训练模块
   - 功能：构建并训练深度神经网络
   - 输出：`dnn_model.h5`、`label_encoder.pkl`

3. `compare_model.py` - 模型对比分析模块
   - 功能：对比不同模型的准确率和F1分数
   - 输出：`accuracy_comparison.png`、`f1_comparison.png`

### 新增生成文件
1. `dnn_model.h5` - 保存的Keras神经网络模型
2. `label_encoder.pkl` - 类别标签编码器
3. `cv_scores.log` - 交叉验证结果日志
4. `model_performance/` - 模型对比可视化目录
   ├── accuracy_comparison.png
   └── f1_comparison.png

## 更新后的使用流程
```bash
# 训练所有模型
python bys_train.py
python knn_train.py
python dl_train.py

# 执行模型对比
python compare_model.py