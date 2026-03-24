# 📄 Tài liệu Kỹ thuật — Mô hình Dự đoán Lượt Bán Sản Phẩm

---

## 1. Bài toán

**Mục tiêu:** Dự đoán số lượng bán được (`Sales Count`) của một sản phẩm trên sàn e-commerce dựa trên các thông tin sẵn có **trước khi** sản phẩm ra thị trường hoặc trong quá trình bán.

**Loại bài toán:** Regression (hồi quy)

**Dữ liệu đầu vào:** File `final_dataset.csv` — tổng hợp từ hai nguồn:
- Thông tin sản phẩm: giá, rating, tỷ lệ giảm giá
- Kết quả phân tích cảm xúc (sentiment analysis) từ review khách hàng: số lượng nhận xét tích cực / tiêu cực / trung tính

---

## 2. Pipeline Xử lý ([train_model.ipynb](file:///f:/project/NCKH/train_model.ipynb))

```
raw CSV
  │
  ├─ [1] Làm sạch: bỏ null, duplicate
  ├─ [2] Loại outlier: IQR × 3 trên Sales Count
  ├─ [3] Feature Engineering (xem chi tiết mục 3)
  ├─ [4] Chọn features & định nghĩa target
  ├─ [5] Train/Test Split (80/20)
  ├─ [6] Scale features: MinMaxScaler (fit trên train, transform cả hai)
  ├─ [7] Hyperparameter Tuning: RandomizedSearchCV (50 iter × 5-fold CV)
  └─ [8] Đánh giá & So sánh mô hình
```

---

## 3. Thiết kế Feature

### 3.1 Nguyên tắc: Nhân quả, không phải Hệ quả

> Chỉ dùng thông tin mà người bán **biết trước** — tránh dùng thứ là *kết quả* của việc bán được.

| Feature | Giữ/Bỏ | Lý do |
|---------|---------|-------|
| `Price (After Discount)` | ✅ Giữ (→ `log_price`) | Nguyên nhân ảnh hưởng quyết định mua |
| `Discount Rate` | ✅ Giữ | Kích thích mua hàng |
| `Rating Average` | ✅ Giữ | Chỉ số chất lượng sản phẩm |
| `pos_ratio`, `neg_ratio` | ✅ Giữ | Phản ánh chất lượng cảm xúc review |
| `Number Reviews` / `log_reviews` | ❌ Bỏ | **Target leakage**: sản phẩm bán nhiều → nhiều review → circular |
| `neu_ratio` | ❌ Bỏ | Redundant: pos + neg + neu = 1, suy ra được từ 2 cái kia |
| `sentiment_score` | ❌ Bỏ | Corr = 1.00 với `pos_ratio`, hoàn toàn trùng lặp |

### 3.2 Các Feature được tạo mới

| Feature | Công thức | Ý nghĩa |
|---------|-----------|---------|
| `log_price` | `log1p(Price)` | Giảm skewness phân phối giá |
| `pos_ratio` | `pos_count / (pos+neg+neu+1)` | Tỷ lệ review tích cực thuần |
| `neg_ratio` | `neg_count / (pos+neg+neu+1)` | Tỷ lệ review tiêu cực thuần |
| `price_x_discount` | `log_price × Discount Rate` | Giá thấp + giảm mạnh → thu hút hơn |
| `rating_x_pos` | `Rating × pos_ratio` | Chất lượng kép: điểm cao + review tốt |
| `discount_x_pos` | `Discount × pos_ratio` | Giảm giá + feedback tốt → hiệu quả bán cao |

### 3.3 Log-transform Target

```python
y = log1p(Sales Count)    # train trên log scale
y_pred_actual = expm1(y_pred)   # inverse khi đánh giá
```

**Tại sao?** Sales Count thường có phân phối lệch phải rất mạnh (vài sản phẩm bán hàng triệu, đa số bán vài trăm). Log-transform giúp:
- Phân phối target gần chuẩn hơn → model học hiệu quả hơn
- Giảm ảnh hưởng của outlier
- Đặc biệt quan trọng với Linear/Ridge Regression

---

## 4. Chiến lược Hyperparameter Tuning

**Phương pháp:** `RandomizedSearchCV`

| Tham số | Giá trị |
|---------|---------|
| `n_iter` | 50 tổ hợp ngẫu nhiên |
| `cv` | 5-fold cross-validation |
| `scoring` | `neg_mean_squared_error` (trên log scale) |

**Tại sao không dùng GridSearchCV?**
- Grid Search phải thử **mọi tổ hợp** → với 5 tham số × 5 giá trị mỗi tham số = 5⁵ = 3125 lần chạy
- Random Search thử 50 lần ngẫu nhiên → nhanh hơn ~60x, kết quả tương đương trong hầu hết trường hợp

---

## 5. Các Chỉ số Đánh giá Mô hình

### 5.1 R² — Hệ số xác định

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**Ý nghĩa:** Tỷ lệ phương sai của target mà model giải thích được.

| Giá trị | Đánh giá |
|---------|----------|
| `> 0.9` | 🟢 Xuất sắc |
| `0.7 – 0.9` | 🟡 Tốt |
| `0.5 – 0.7` | 🟠 Trung bình |
| `0.3 – 0.5` | 🔴 Yếu |
| `< 0` | ❌ Tệ hơn đoán trung bình |

> **Kỳ vọng bài toán này:** R² ≥ 0.5 là tốt. Doanh số phụ thuộc nhiều vào yếu tố bên ngoài không quan sát được (quảng cáo, trend, thuật toán sàn...) nên R² thấp là tự nhiên — đây là giới hạn của **bài toán**, không phải model.

---

### 5.2 MAE — Mean Absolute Error

$$MAE = \frac{1}{n}\sum|y_i - \hat{y}_i|$$

**Ý nghĩa:** Trung bình sai lệch tuyệt đối giữa dự đoán và thực tế (cùng đơn vị với target = **lượt bán**).

- **Dễ hiểu nhất** trong 3 chỉ số
- Không phạt nặng các sai số lớn
- Ví dụ: MAE = 150 nghĩa là trung bình dự đoán lệch 150 lượt bán

> **Ngưỡng tốt:** MAE / Mean(Sales Count) < 20%

---

### 5.3 RMSE — Root Mean Squared Error

$$RMSE = \sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$$

**Ý nghĩa:** Tương tự MAE nhưng **phạt nặng hơn** các sai số lớn (do bình phương).

- Luôn RMSE ≥ MAE
- Nếu `RMSE >> MAE`: model đang sai rất lớn ở một số điểm → cần kiểm tra
- Nếu `RMSE ≈ MAE`: sai số phân bố đều, ổn định

| So sánh | Ý nghĩa |
|---------|---------|
| RMSE / MAE ≈ 1.0–1.5 | ✅ Sai số phân bố đều |
| RMSE / MAE > 2.5 | ⚠️ Có outlier prediction nghiêm trọng |

---

### 5.4 CV R² vs Test R²

| Chỉ số | Tính trên | Mục đích |
|--------|-----------|---------|
| `CV R²` | Train set (5-fold) | Đánh giá khả năng tổng quát hóa |
| `Test R²` | Test set (hold-out) | Đánh giá hiệu suất thực tế |

**Dấu hiệu cần chú ý:**
- `CV R² >> Test R²` (chênh > 0.1): **Overfitting** — model học thuộc dữ liệu train
- `CV R² ≈ Test R²`: ✅ Model ổn định, tổng quát tốt

---

## 6. Các Model Được Sử dụng

| Model | Loại | Ưu điểm |
|-------|------|---------|
| **Ridge Regression** | Linear | Nhanh, interpretable, baseline tốt |
| **Random Forest** | Ensemble (bagging) | Ổn định, ít overfit, feature importance dễ đọc |
| **XGBoost** | Ensemble (boosting) | Mạnh nhất với tabular data, nhiều regularization |
| **LightGBM** | Ensemble (boosting) | Nhanh hơn XGBoost, tốt với dataset lớn |

---

## 7. Lưu ý Thực tế

1. **Target leakage là lỗi khó phát hiện nhất** — luôn hỏi "feature này có phải *kết quả* của target không?"
2. **Scale sau khi split** — không được fit scaler trên toàn bộ dữ liệu trước khi split
3. **Log-transform target** — nhớ inverse (`expm1`) khi tính metrics trên original scale
4. **R² thấp ≠ model kém** — có thể bài toán vốn khó do thiếu features quan trọng
