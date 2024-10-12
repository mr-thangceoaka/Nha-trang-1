import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV với xử lý encoding
file_path = r'C:\Users\MSI-PC\Downloads\Road bus CSV\79B01744_X.csv'

# Thử nhiều encoding khác nhau để xử lý lỗi khi đọc file CSV
try:
    df = pd.read_csv(file_path, encoding='utf-8', sep=',')
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='ISO-8859-1', sep=',')
    except Exception as e:
        print(f"Lỗi khi đọc file CSV với nhiều encoding: {e}")
        exit()

# In danh sách các cột để kiểm tra
print(df.columns)

# Chuyển đổi cột 'Ngày giờ' thành kiểu thời gian để dễ xử lý
df['Ngày giờ'] = pd.to_datetime(df['Ngày gi?'])

# Trích xuất các thông tin từ cột ngày giờ
df['Giờ'] = df['Ngày giờ'].dt.hour  # Trích xuất giờ
df['Ngày trong tuần'] = df['Ngày giờ'].dt.weekday  # Trích xuất ngày trong tuần
# Đánh dấu giờ cao điểm (7-9h sáng, 17-19h chiều)
df['Có phải giờ cao điểm'] = df['Giờ'].apply(lambda x: 1 if (7 <= x <= 9 or 17 <= x <= 19) else 0)

# Hàm kiểm tra khoảng cách giữa hai điểm liên tiếp
def check_distance(row, next_row):
    # Khoảng cách tính từ cột 'Khoảng cách đến điểm tiếp theo (km)'
    km_actual = row['Km']
    km_calculated = next_row['Km']

    # Nếu khoảng cách thực tế không khớp với khoảng cách tính toán
    if np.isclose(km_actual, km_calculated, atol=0.5):  # Cho phép sai số 0.5 km
        return True  # Khoảng cách khớp
    else:
        return False  # Khoảng cách không khớp

# Kiểm tra khoảng cách giữa các dòng liên tiếp
df['Khoảng cách hợp lệ'] = df.apply(
    lambda row: check_distance(row, df.shift(-1).loc[row.name]) if row.name < len(df)-1 else False, axis=1)

# Tạo cột thời gian chờ giả định (tính từ sự chênh lệch giữa thời gian của các lần đo)
df['Thời gian chờ'] = df['Ngày giờ'].diff().dt.total_seconds() / 60  # Thời gian chờ tính bằng phút

# Kiểm tra và xử lý NaN trong dữ liệu
print(df.isna().sum())  # Kiểm tra số lượng NaN trong từng cột

# Xử lý NaN: Loại bỏ các dòng có NaN hoặc thay thế NaN bằng giá trị trung bình
df = df.dropna()  # Loại bỏ các dòng chứa NaN

# Hoặc bạn có thể thay thế NaN bằng giá trị trung bình của từng cột như sau:
# df = df.fillna(df.mean())  # Thay thế NaN bằng giá trị trung bình của cột đó

# In ra 5 dòng đầu của dữ liệu sau khi xử lý để kiểm tra
print(df.head())

# Giả sử bạn có cột 'Thời gian chờ' trong dữ liệu (Cột này là mục tiêu dự đoán)
# X là tập các đặc trưng đầu vào (features) gồm giờ, ngày trong tuần, giờ cao điểm, vận tốc GPS và khoảng cách
X = df[['Giờ', 'Ngày trong tuần', 'Có phải giờ cao điểm', 'V?n t?c GPS', 'Km']]

# y là biến mục tiêu 'Thời gian chờ'
y = df['Thời gian chờ']  # Thay 'Thời gian chờ' bằng tên cột phù hợp trong dữ liệu của bạn

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra (80% cho huấn luyện, 20% cho kiểm tra)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)
# Tính toán sai số trung bình tuyệt đối (MAE) và căn bậc hai của sai số trung bình bình phương (RMSE)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Vẽ 2 đồ thị so sánh giá trị thực tế và giá trị dự đoán với MAE và RMSE

# Đồ thị so sánh MAE
plt.figure(figsize=(12, 6))

# Đồ thị 1: So sánh giá trị thực tế với giá trị dự đoán dựa trên MAE
plt.subplot(1, 2, 1)
plt.plot(y_test.index, y_test, label='Thực tế', marker='o', color='blue')
plt.plot(y_test.index, y_pred, label='Dự đoán', marker='x', color='red')
plt.title(f'So sánh Thực tế và Dự đoán (MAE: {mae:.2f})')
plt.xlabel('Chỉ số dòng')
plt.ylabel('Thời gian chờ (phút)')
plt.legend()
plt.grid(True)

# Đồ thị 2: So sánh giá trị thực tế với giá trị dự đoán dựa trên RMSE
plt.subplot(1, 2, 2)
plt.plot(y_test.index, y_test, label='Thực tế', marker='o', color='blue')
plt.plot(y_test.index, y_pred, label='Dự đoán', marker='x', color='red')
plt.title(f'So sánh Thực tế và Dự đoán (RMSE: {rmse:.2f})')
plt.xlabel('Chỉ số dòng')
plt.ylabel('Thời gian chờ (phút)')
plt.legend()
plt.grid(True)

# Hiển thị đồ thị
plt.tight_layout()
plt.show()