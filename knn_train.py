from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from textPreprocessor import TextPreprocessor
import joblib

def knn_train():
    # 初始化预处理类（与朴素贝叶斯相同配置）
    preprocessor = TextPreprocessor(
        dataset_path="F:/study/three/NLP/pythonProject3/20_newsgroups",
        use_stopwords=True,
        use_stemming=True
    )
    
    # 加载数据
    texts, labels = preprocessor.load_data()
    
    # 特征工程（TF-IDF）
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42
    )
    
    # 训练KNN分类器
    knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
    knn.fit(X_train, y_train)
    
    # 模型评估
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # 保存模型
    joblib.dump(vectorizer, 'knn_vectorizer.pkl')
    joblib.dump(knn, 'knn_classifier.pkl')

if __name__ == "__main__":
    knn_train()
