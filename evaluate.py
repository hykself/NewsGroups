from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow.keras.models import load_model

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 假设 TextPreprocessor 已经定义在 textPreprocessor 模块中
from textPreprocessor import TextPreprocessor


def load_test_data(preprocessor):
    """加载预处理后的原始数据用于测试"""
    return preprocessor.load_data()


def evaluate_models():
    # 定义模型列表，包含模型名称、模型文件路径和向量化器文件路径
    models = [
        ('Naive Bayes', 'nb_classifier.pkl', 'tfidf_vectorizer.pkl'),
        ('KNN', 'knn_classifier.pkl', 'knn_vectorizer.pkl'),
        ('DNN', 'dnn_model.h5', 'tfidf_vectorizer.pkl')
    ]

    # 初始化预处理类（需与训练时相同配置）
    preprocessor = TextPreprocessor(
        dataset_path="F:/study/three/NLP/pythonProject3/20_newsgroups",
        use_stopwords=True,
        use_stemming=True
    )

    # 加载并预处理数据
    texts, labels = load_test_data(preprocessor)

    for model_name, model_file, vectorizer_file in models:
        # 加载向量化器
        vectorizer = joblib.load(vectorizer_file)

        if model_name == 'DNN':
            # 加载深度学习模型
            classifier = load_model(model_file)
            le = joblib.load('label_encoder.pkl')
            X = vectorizer.transform(texts).toarray()  # 转换为密集矩阵以适应DNN模型
            y_pred_prob = classifier.predict(X)
            y_pred = le.inverse_transform(y_pred_prob.argmax(axis=1))
        else:
            # 加载传统机器学习模型
            classifier = joblib.load(model_file)
            X = vectorizer.transform(texts)
            y_pred = classifier.predict(X)

        # 生成分类报告字符串
        report = classification_report(labels, y_pred, digits=4)
        print(f"{model_name} 评估报告：")
        print(report)

        # 保存分类报告到文件
        with open(f'{model_name}_classification_report.txt', 'w', encoding='utf-8') as f:
            f.write(f"{model_name} 分类评估报告\n")
            f.write("=" * 60 + "\n")
            f.write(report)

        # 可视化混淆矩阵
        plt.figure(figsize=(15, 12))
        cm = confusion_matrix(labels, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} 混淆矩阵')
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'final_confusion_matrix_{model_name}.png', dpi=200, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    evaluate_models()