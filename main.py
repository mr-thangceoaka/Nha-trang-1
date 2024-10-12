import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV với xử lý encoding
file_path = r'C:\Users\MSI-PC\Downloads\Road bus CSV\79B01617_X.csv'

# Thử nhiều encoding khác nhau để xử lý lỗi khi đọc file CSV
try:
    df = pd.read_csv(file_path, encoding='utf-8', sep=',')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', sep=',')
    except Exception as e:
        print(f"Lỗi khi đọc file CSV với nhiều encoding: {e}")
        exit()

# In ra 5 dòng đầu tiên của dữ liệu để kiểm tra
print(df.head())

# Chuyển đổi cột 'Ngày giờ' thành kiểu thời gian để dễ xử lý
df['Ngày giờ'] = pd.to_datetime(df['Ngày gi?'])

# Trích xuất các thông tin từ cột ngày giờ
df['Giờ'] = df['Ngày giờ'].dt.hour  # Trích xuất giờ
df['Ngày trong tuần'] = df['Ngày giờ'].dt.weekday  # Trích xuất ngày trong tuần
# Đánh dấu giờ cao điểm (7-9h sáng, 17-19h chiều)
df['Có phải giờ cao điểm'] = df['Giờ'].apply(lambda x: 1 if (7 <= x <= 9 or 17 <= x <= 19) else 0)

# Tạo cột thời gian chờ giả định (tính từ sự chênh lệch giữa thời gian của các lần đo)
df['Thời gian chờ'] = df['Ngày giờ'].diff().dt.total_seconds() / 60  # Thời gian chờ tính bằng phút

# Loại bỏ các giá trị NaN
df = df.dropna()

# Giả sử bạn có cột 'Thời gian chờ' trong dữ liệu
X = df[['Giờ', 'Ngày trong tuần', 'Có phải giờ cao điểm', 'V?n t?c GPS', 'Km']]  # Km là khoảng cách từ file CSV
y = df['Thời gian chờ']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sử dụng Random Forest để dự đoán
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tính toán sai số trung bình tuyệt đối (MAE)
mae = mean_absolute_error(y_test, y_pred)

# In ra kết quả
print(f'MAE (Mean Absolute Error): {mae}')

# Vẽ đồ thị so sánh giữa giá trị thực tế và giá trị dự đoán (chỉ lấy 5 dòng đầu)
plt.figure(figsize=(10, 6))
plt.plot(y_test.values[:5], label='Thực tế', marker='o')
plt.plot(y_pred[:5], label='Dự đoán', marker='x')
plt.title('So sánh giữa giá trị Thực tế và Dự đoán (Thời gian chờ) - 5 dòng đầu')
plt.xlabel('Chỉ số dòng')
plt.ylabel('Thời gian chờ (phút)')
plt.legend()
plt.grid(True)
plt.show()
