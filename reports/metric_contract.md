# Metric contract (khai báo bắt buộc — chốt ở phút 30, trước khi chạy eval)

> Các metric chuẩn đã được cài trong `src/metrics/tracks.py`. File này là nơi nhóm **cam kết** cách dùng chúng.
> Không đổi testset, threshold hay trọng số sau khi đã thấy kết quả eval.

| Mục | Khai báo của nhóm |
|---|---|
| Track | T02 · Raw QC Triage |
| Primary metric (tên đúng như evaluator in ra) | `Defect Recall@B` (kèm `Precision@B`) |
| Tăng tốt hay giảm tốt? | higher |
| Đơn vị đánh giá (image / region / pair / release / batch / case …) | image |
| Công thức + numerator / denominator | Recall@B = #ảnh lỗi thật (GT `is_defect=1`) nằm trong top-B theo `score` / #ảnh lỗi thật trong split eval. Precision@B = cùng numerator / B |
| Cách ghép prediction ↔ GT (IoU? id? tolerance?) | ghép theo `image_id`; chỉ tính ảnh thuộc split `eval`; sắp xếp `score` giảm dần (mergesort, ổn định) |
| Candidate pool cố định | toàn bộ 40 ảnh split `eval` trong `fixtures/t02_raw_qc/images.csv` (pipeline chấm cả 96 ảnh, evaluator lọc eval) |
| Review budget B / K | B = 20 |
| Seed | 42 |
| Threshold policy (chọn trên dev như thế nào) | Mọi hằng số trong `configs/lab.yaml → params` chọn trên **dev** bằng nhãn tự gán `reports/t02_dev_selflabels.csv` (56 ảnh, 28 lỗi), kiểm bằng `python -m tools.t02_dev_check`. Ví dụ ngưỡng nén `block_base + block_scale = 1.8` nằm giữa max ảnh sạch (1.70) và min ảnh nén (1.88) trên dev. Trung vị theo nhóm (camera_id, light) tính trên pool **không nhãn** |
| Eval split (không dùng để tune) | `split = eval` (40 ảnh). Không xem ảnh eval để gán nhãn, không chỉnh tham số sau khi chạy |
| Chỉ số lỗi xấu (miss / false alarm / false reject) | **Miss** (ảnh lỗi lọt khỏi top-B → vào dataset) là lỗi xấu nhất; thứ hai là false RECAPTURE ảnh đêm/cam_B tốt (mất dữ liệu hiếm) |
| Slices sẽ báo cáo (≥2) | `light` ∈ {day, dusk, night} (false alarm in top-B, có sẵn trong evaluator); `defect_type` ∈ {blur, glare, dark, compression} (recall in top-B); thêm `camera_id` (cam_A vs cam_B) trong report |
| Cách báo độ bất định khi N nhỏ (Wilson CI / bootstrap) | Wilson 95% CI cho Recall@B và Precision@B (N eval = 40, B = 20) |
| Ngày giờ chốt + người chốt | 2026-11-14 · Đoàn Quang Minh (G01) |

Nếu baseline = 0 thì ghi rõ "baseline = 0" thay vì phần trăm cải thiện.
