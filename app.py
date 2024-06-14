import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb
from scipy.stats import zscore

# Tiêu đề của ứng dụng
st.title("LightGBM Classifier Model")

df = pd.read_csv('liver_cirrhosis.csv')

# Lưu trữ các giá trị ban đầu
df_original = df.copy()

df = df.drop_duplicates()

df['Age'] = (df['Age'] / 365.25).astype(int)

# Mã hóa các feature dạng chuỗi
categorical_features = df.select_dtypes(include=['object']).columns
le_dict = {}
for feature in categorical_features:
    le = LabelEncoder()
    df[feature] = le.fit_transform(df[feature])
    le_dict[feature] = le

# Loại bỏ các giá trị ngoại lệ
numerical_columns = ['Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper']
z_scores = df[numerical_columns].apply(zscore)
threshold = 3
df = df[(z_scores.abs() < threshold).all(axis=1)]

# Chia dữ liệu thành X và y
X = df.drop('Stage', axis=1)
y = df['Stage']

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Mã hóa nhãn
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Chia tập dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Huấn luyện mô hình LightGBM với các tham số cố định
lgbm = lgb.LGBMClassifier(random_state=42, n_estimators=100, subsample=0.8)
lgbm.fit(X_train, y_train)

# Dự đoán và đánh giá mô hình
y_pred_lgbm = lgbm.predict(X_test)
accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
report_lgbm = classification_report(y_test, y_pred_lgbm, target_names=[str(cls) for cls in label_encoder.classes_])

# Giao diện thay đổi các feature đầu vào với giá trị ban đầu
st.header("Dự đoán với giá trị đầu vào khác nhau")

input_features = {}
for feature in df_original.columns:
    if feature == 'Stage':
        continue
    if feature == 'Age':
        input_features[feature] = st.slider(f"Input {feature} (years)", min_value=int(df_original[feature].min() / 365.25), max_value=int(df_original[feature].max() / 365.25), value=int(df_original[feature].mean() / 365.25))
    elif df_original[feature].dtype in [int, float]:
        input_features[feature] = st.slider(f"Input {feature}", min_value=float(df_original[feature].min()), max_value=float(df_original[feature].max()), value=float(df_original[feature].mean()))
    else:
        input_features[feature] = st.selectbox(f"Input {feature}", options=list(df_original[feature].unique()), key=f"{feature}_select")

# Tạo DataFrame từ các giá trị đầu vào
input_df = pd.DataFrame([input_features])

# Mã hóa các giá trị đầu vào
for feature in categorical_features:
    if feature in input_df.columns:
        le = le_dict[feature]
        input_df[feature] = le.transform(input_df[feature])

# Chuyển đổi tuổi từ năm sang ngày
input_df['Age'] = input_df['Age'] * 365.25

# Chuẩn hóa các giá trị đầu vào
input_df_scaled = scaler.transform(input_df)

# Dự đoán nhãn với các giá trị đầu vào tùy chỉnh
custom_pred = lgbm.predict(input_df_scaled)
custom_pred_label = label_encoder.inverse_transform(custom_pred)

# Hiển thị kết quả dự đoán
st.write("Giai đoạn:", custom_pred_label[0])
