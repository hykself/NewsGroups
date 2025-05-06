import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import load_model
import joblib
from textPreprocessor import TextPreprocessor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

# ============ 添加以下配置，解决中文显示问题 ============
from matplotlib import rcParams

# 设置中文字体和解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体（支持中文）
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号 '-' 显示为方块的问题


# ===================================================

def compare_models():
    # 加载数据
    preprocessor = TextPreprocessor(
        dataset_path="F:/study/three/NLP/pythonProject3/20_newsgroups",
        use_stopwords=True,
        use_stemming=True
    )
    texts, labels = preprocessor.load_data()

    # 划分训练集和测试集
    _, X_test, _, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # 初始化结果存储
    results = []

    # 朴素贝叶斯交叉验证
    nb_vec = TfidfVectorizer(max_features=5000)
    nb_X = nb_vec.fit_transform(texts)
    nb_cv_mean, nb_cv_std = cross_validate(MultinomialNB(alpha=0.1), nb_X, labels)

    # 使用已保存的朴素贝叶斯模型进行测试集预测
    nb_model = joblib.load('nb_classifier.pkl')
    nb_pred = nb_model.predict(nb_vec.transform(X_test))
    results.append(('Naive Bayes', nb_cv_mean, nb_cv_std,
                    accuracy_score(y_test, nb_pred),
                    f1_score(y_test, nb_pred, average='weighted')))

    # KNN交叉验证
    knn_vec = TfidfVectorizer(max_features=5000)
    knn_X = knn_vec.fit_transform(texts)
    knn_cv_mean, knn_cv_std = cross_validate(KNeighborsClassifier(metric='cosine'), knn_X, labels)

    # 使用已保存的KNN模型进行测试集预测
    knn_model = joblib.load('knn_classifier.pkl')
    knn_pred = knn_model.predict(knn_vec.transform(X_test))
    results.append(('KNN', knn_cv_mean, knn_cv_std,
                    accuracy_score(y_test, knn_pred),
                    f1_score(y_test, knn_pred, average='weighted')))

    # 深度学习模型预测
    dnn_model = load_model('dnn_model.h5')
    le = joblib.load('label_encoder.pkl')
    dnn_vec = joblib.load('tfidf_vectorizer.pkl')
    dnn_X_test = dnn_vec.transform(X_test).toarray()  # 转换为密集矩阵
    dnn_pred = le.inverse_transform(dnn_model.predict(dnn_X_test).argmax(axis=1))
    results.append(('DNN',
                    None, None,  # DNN模型未进行交叉验证
                    accuracy_score(y_test, dnn_pred),
                    f1_score(y_test, dnn_pred, average='weighted')))

    # 生成可视化
    df = pd.DataFrame(results, columns=['Model', 'CV Accuracy Mean', 'CV Accuracy Std', 'Test Accuracy', 'F1 Score'])

    plt.figure(figsize=(10, 6))
    ax=sns.barplot(x='Model', y='Test Accuracy', data=df)
    plt.title('模型准确率对比')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.savefig('accuracy_comparison.png')

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Model', y='F1 Score', data=df)
    plt.title('模型F1分数对比')
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')
    plt.savefig('f1_comparison.png')


def cross_validate(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean(), scores.std()


if __name__ == "__main__":
    compare_models()