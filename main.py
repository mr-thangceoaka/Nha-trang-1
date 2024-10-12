import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score, precision_score, accuracy_score
import matplotlib.pyplot as plt

# Đọc dữ liệu từ file CSV với xử lý encoding
file_path = r"C:\Users\MSI-PC\Downloads\Road bus CSV\79B00529_X.csv"

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
print("5 dòng đầu tiên của dữ liệu:")
print(df.head())

# Chuyển đổi cột 'Ngày giờ' thành kiểu thời gian để dễ xử lý
df['Ngày giờ'] = pd.to_datetime(df['Ngày gi?'])  # Sửa tên cột nếu cần

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
X = df[['Giờ', 'Ngày trong tuần', 'Có phải giờ cao điểm', 'V?n t?c GPS', 'Km']]  # Sửa 'V?n t?c GPS' thành 'Vận tốc GPS'
y = df['Thời gian chờ']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Sử dụng Random Forest để dự đoán
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Tạo DataFrame từ kết quả
results_df = pd.DataFrame({'Thời gian chờ thực tế (phút)': y_test, 'Thời gian chờ dự đoán (phút)': y_pred})

# Tính toán các chỉ số
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Xác định ngưỡng để phân loại (Ví dụ: 10 phút là ngưỡng)
threshold = 10
y_test_class = (y_test > threshold).astype(int)  # Nhãn thực tế
y_pred_class = (y_pred > threshold).astype(int)  # Nhãn dự đoán

# Tính toán Accuracy, Precision và F1 Score
accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)

# In ra các kết quả
print("\nKết quả:")
print(f"MAE: {mae:.2f} phút")

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"F1 Score: {f1:.2f}")

# Vẽ đồ thị
plt.figure(figsize=(14, 6))

# Đồ thị 1: Thời gian chờ trung bình thực tế và dự đoán
plt.subplot(1, 2, 1)
plt.bar(['Thực tế', 'Dự đoán'], [results_df['Thời gian chờ thực tế (phút)'].mean(), results_df['Thời gian chờ dự đoán (phút)'].mean()], color=['blue', 'orange'])
plt.title('Thời gian chờ trung bình')
plt.ylabel('Thời gian chờ (phút)')
plt.grid(axis='y')

# Đồ thị 2: So sánh thời gian chờ thực tế và thời gian dự đoán
plt.subplot(1, 2, 2)
x_ticks = results_df.index  # Chỉ số dòng
plt.plot(x_ticks, results_df['Thời gian chờ thực tế (phút)'].values, label='Thực tế', marker='o', color='blue')
plt.plot(x_ticks, results_df['Thời gian chờ dự đoán (phút)'].values, label='Dự đoán', marker='x', color='orange')
plt.title('So sánh thời gian chờ thực tế và dự đoán')
plt.xlabel('Chỉ số dòng')
plt.ylabel('Thời gian chờ (phút)')
plt.xticks(rotation=45)  # Xoay nhãn trục x nếu cần
plt.legend()
plt.grid(True)

plt.tight_layout()  # Điều chỉnh khoảng cách giữa các đồ thị
plt.show()
