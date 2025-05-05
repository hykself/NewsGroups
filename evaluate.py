from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from textPreprocessor import TextPreprocessor


def load_test_data(preprocessor):
    """加载预处理后的原始数据用于测试"""
    return preprocessor.load_data()

def evaluate_saved_model():
    # 加载已保存的模型
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    classifier = joblib.load('nb_classifier.pkl')
    
    # 初始化预处理类（需与训练时相同配置）
    preprocessor = TextPreprocessor(
        dataset_path="F:/study/three/NLP/pythonProject3/20_newsgroups",
        use_stopwords=True,
        use_stemming=True
    )
    
    # 加载并预处理数据
    texts, labels = preprocessor.load_data()
    
    # 特征工程（使用训练时的向量化器）
    X = vectorizer.transform(texts)
    
    # 预测与评估
    y_pred = classifier.predict(X)
    
    # 生成分类报告
    print("完整数据集评估报告：")
    print(classification_report(labels, y_pred))
    
    # 可视化混淆矩阵
    plt.figure(figsize=(15, 12))
    cm = confusion_matrix(labels, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classifier.classes_,
                yticklabels=classifier.classes_)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig('final_confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    evaluate_saved_model()
