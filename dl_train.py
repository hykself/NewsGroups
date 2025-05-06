from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from textPreprocessor import TextPreprocessor
import joblib


def dl_train():
    # 初始化预处理类
    preprocessor = TextPreprocessor(
        dataset_path="F:/study/three/NLP/pythonProject3/20_newsgroups",
        use_stopwords=True,
        use_stemming=True
    )

    # 加载数据
    texts, labels = preprocessor.load_data()

    # 特征工程（复用朴素贝叶斯的TF-IDF）
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    X = vectorizer.transform(texts)  # 得到稀疏矩阵

    # ⚠️ 关键修改：转为密集矩阵以避免稀疏索引错误
    X = X.toarray()

    # 标签编码
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # 手动拆分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 构建深度神经网络
    model = Sequential([
        Dense(512, activation='relu', input_shape=(X.shape[1],)),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])

    # 编译模型
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    # 训练模型（使用手动拆分的验证集）
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(X_val, y_val)
    )

    # 保存模型和标签编码器
    model.save('dnn_model.h5')
    joblib.dump(le, 'label_encoder.pkl')


if __name__ == "__main__":
    dl_train()