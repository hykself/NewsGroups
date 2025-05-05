# 20 Newsgroups 文本分类项目

## 项目结构说明

### 核心代码文件
1. `textPreprocessor.py` - 文本预处理模块
   - 功能：数据清洗、分词、去停用词、词干提取
   - 输入：原始新闻文本数据
   - 输出：预处理后的文本和标签
   - 关键方法：
     - `clean_text()`：去除特殊字符和数字
     - `load_data()`：加载并处理整个数据集

2. `train.py` - 模型训练模块
   - 功能：特征工程(TF-IDF)、模型训练和保存
   - 输入：预处理后的文本数据
   - 输出：训练好的朴素贝叶斯模型(`.pkl`文件)
   - 关键参数：
     - `max_features=5000`：限制TF-IDF特征维度
     - `alpha=0.1`：平滑系数

3. `evaluate.py` - 模型评估模块
   - 功能：模型性能评估和错误分析
   - 输出：
     - 分类报告(`.txt`)
     - 混淆矩阵(`.png`)
     - 错误样本分析(`.txt`)

### 生成文件
1. `tfidf_vectorizer.pkl` - 保存的TF-IDF向量化器
2. `nb_classifier.pkl` - 训练好的朴素贝叶斯分类器
3. `train_data.txt`/`test_data.txt` - 预处理后的数据集划分
4. `processed_data.txt` - 完整预处理数据集备份

## 使用指南

### 安装依赖
```bash
pip install -r requirements.txt
