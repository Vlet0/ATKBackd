# Backdoor Defense & Stealthiness Evaluation Suite

Bộ công cụ đánh giá **Tính ẩn (Stealthiness)** và **Khả năng chống đỡ (Defense Resilience)** cho hệ thống Backdoor Attack trên dữ liệu CSI Pose Estimation (Person-in-WiFi-3D & MMFI).

---

## 🎯 1. Thiết Lập Tối Ưu (Best Setting) & Scenario Thực Tế Nhất

Dựa trên kết quả sweep thực nghiệm trên dataset Person-in-WiFi 3D:

* **Scenario thực tế nhất:** **`bend` (Cúi người / Gập người)**
  * **Lý do khoa học & thực tiễn:** Trong ứng dụng nhận diện hành vi qua sóng WiFi (WiFi human sensing, elderly care, fall detection), chuyển động cúi người xảy ra thường xuyên và tự nhiên nhất. Khác với gật đầu (`nod`) hay giữ yên vị trí (`cross`), cúi người tạo ra phổ MicroDoppler mở rộng liên tục, giúp trigger hòa lẫn vào dao động sóng WiFi tự nhiên mà người quan sát không thể phát hiện.
* **Tham số tấn công tối ưu (Best Hyperparameters):**
  * `theta_max_deg = 20.0°`: Góc xoay joint nhỏ ($20^\circ$) $\rightarrow$ dáng người không bị biến dạng bất thường, ẩn tuyệt đối nhưng vẫn đạt ASR tối đa (**74.8%** trên Person-in-WiFi 3D, **72.2%** trên MMFI).
  * `rho = 0.3`: Tỉ lệ đầu độc 30% dataset.
  * `eps = 0.3`: Cường độ nhiễu MicroDoppler tối ưu cho scenario `bend`.
  * `pivot`: `6` cho Person-in-WiFi 3D, `7` cho MMFI (khớp theo cấu trúc giải phẫu xương L-elbow/R-elbow).

---

## 🛡️ 2. 4 Phương Pháp Defense Được Tích Hợp

| Defense | Nguyên lý | Chỉ số ẩn (Stealthiness Index) |
|---|---|---|
| **STRIP** | Runtime perturbation: Trộn input với clean samples và đo độ phân tán (dispersion) dự đoán | **Detection Rate** càng **THẤP** ($\le 35\%$) $\rightarrow$ Attack càng ẩn |
| **NoiSec** | Denoising Autoencoder: Reconstruct CSI và tính Mahalanobis distance trên residual feature | **Detection Rate** càng **THẤP** ($\le 25\%$) $\rightarrow$ Trigger giống nhiễu tự nhiên |
| **Neural Cleanse** | Reverse engineering: Đảo ngược tìm universal trigger nhỏ nhất qua các pivot joint | **Anomaly Index** near 2.0 (ngưỡng MAD) $\rightarrow$ Khó bị phát hiện |
| **Fine-Pruning** | Model repair: Prune 20% dormant neurons trên clean data và đo lại ASR | **Post-Pruning ASR** còn giữ được cao $\rightarrow$ Backdoor phân tán sâu trong model |

---

## 🚀 3. Hướng Dẫn Chạy Đánh Giá Defense

```bat
# Chạy đánh giá cả 4 defenses trên checkpoint model đã train:
run_defenses_best.bat --checkpoint experiments_out/victim_a/hpeli_bend_micro_dropper_s0/best.pt
```

### Kết quả thu được:
* `defend_out/summary.json`: Báo cáo chỉ số đầy đủ của 4 phương pháp.
* `defend_out/defenses_summary.png`: Biểu đồ trực quan hóa tổng hợp (ASR trước/sau Pruning, STRIP, NoiSec, Neural Cleanse).
