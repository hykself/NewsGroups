import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyparsing import results
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import load_model
import joblib

from textPreprocessor import TextPreprocessor
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split


def compare_models():
    # 加载数据
    preprocessor = TextPreprocessor(
        dataset_path="F:/study/three/NLP/pythonProject3/20_newsgroups",
        use_stopwords=True,
        use_stemming=True
    )
    texts, labels = preprocessor.load_data()
    
    # 加载测试集
    _, X_test, _, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # 初始化结果存储
    results = []
    
    # 比较朴素贝叶斯
    nb_model = joblib.load('nb_classifier.pkl')
    nb_vec = joblib.load('tfidf_vectorizer.pkl')
    nb_pred = nb_model.predict(nb_vec.transform(X_test))
    results.append(('Naive Bayes', 
                   accuracy_score(y_test, nb_pred),
                   f1_score(y_test, nb_pred, average='weighted')))
    
    # 比较KNN
    knn_model = joblib.load('knn_classifier.pkl') 
    knn_vec = joblib.load('knn_vectorizer.pkl')
    knn_pred = knn_model.predict(knn_vec.transform(X_test))
    results.append(('KNN', 
                   accuracy_score(y_test, knn_pred),
                   f1_score(y_test, knn_pred, average='weighted')))
    
    # 比较深度学习模型
    dnn_model = load_model('dnn_model.h5')
    le = joblib.load('label_encoder.pkl')
    dnn_vec = joblib.load('tfidf_vectorizer.pkl')
    dnn_pred = le.inverse_transform(dnn_model.predict(dnn_vec.transform(X_test)).argmax(axis=1))
    results.append(('DNN',
                   accuracy_score(y_test, dnn_pred),
                   f1_score(y_test, dnn_pred, average='weighted')))
    
    # 生成可视化
    df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1 Score'])
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=df)
    plt.title('模型准确率对比')
    plt.savefig('accuracy_comparison.png')
    
    plt.figure(figsize=(10, 6)) 
    sns.barplot(x='Model', y='F1 Score', data=df)
    plt.title('模型F1分数对比')
    plt.savefig('f1_comparison.png')

if __name__ == "__main__":
    compare_models()


def cross_validate(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    return scores.mean(), scores.std()

# 在 compare_models() 函数中添加以下代码：
# 在加载数据后添加：
from sklearn.feature_extraction.text import TfidfVectorizer

# 交叉验证准备
all_texts, all_labels = preprocessor.load_data()

# 朴素贝叶斯交叉验证
nb_vec = TfidfVectorizer(max_features=5000)
nb_X = nb_vec.fit_transform(all_texts)
nb_cv_mean, nb_cv_std = cross_validate(MultinomialNB(alpha=0.1), nb_X, all_labels)

# KNN交叉验证
knn_vec = TfidfVectorizer(max_features=5000)
knn_X = knn_vec.fit_transform(all_texts)
knn_cv_mean, knn_std = cross_validate(KNeighborsClassifier(metric='cosine'), knn_X, all_labels)

# 更新结果列表
results.append(('Naive Bayes', nb_cv_mean, nb_cv_std,
               accuracy_score(y_test, nb_pred),
               f1_score(y_test, nb_pred, average='weighted')))    compare_models()
