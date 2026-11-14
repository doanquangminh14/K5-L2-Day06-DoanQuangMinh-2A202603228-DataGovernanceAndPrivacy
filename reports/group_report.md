# Group Report — Day 06 Data Lab (Field-to-Release PoC)

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G01 |
| Challenge | T02 · Raw QC Triage: ảnh mờ, lóa và thiếu sáng |
| Thành viên + vai trò | Đoàn Quang Minh — làm một mình: thuật toán, pipeline, test, review, báo cáo |
| Git URL | https://github.com/doanquangminh14/K5-L2-Day06-DoanQuangMinh-2A202603228-DataGovernanceAndPrivacy |
| Tag `demo-180m` (commit) | `858da9c` |
| Tag `final-24h` (commit) | commit mà tag `final-24h` trỏ tới (`git rev-list -n1 final-24h`) |
| Deadline (giờ VN) | 2026-11-15 12:00 (GMT+7) |

## 01 · Problem statement

- **Ai chịu hậu quả khi hệ thống sai?** Người bị ảnh hưởng trước tiên là đội train model perception. Nếu ảnh mờ, lóa, tối hay vỡ nén lọt vào dataset thì nhãn bị nhiễu và model học kém đi. Ngược lại, nếu loại nhầm ảnh tốt, nhất là ảnh ban đêm hoặc ảnh từ cam_B, thì mất dữ liệu khó thu và tốn công chụp lại.
- **Điểm nghẽn:** mỗi lô người review chỉ xem được 20 ảnh (B = 20). Việc của tôi là xếp hạng sao cho 20 ảnh đó chứa nhiều ảnh lỗi nhất.
- **Đầu vào và đầu ra:** đầu vào là `fixtures/t02_raw_qc/images.csv` cùng 96 ảnh JPEG 320×180. Đầu ra là `out/poc/ranked_review.csv` (`image_id, score, pred_defect_type, pred_action`), kèm `review_queue.csv`, `contact_sheet.html`, `audit_log.jsonl` và `decision.json`.
- **Tiêu chí GO / HOLD / REWORK** (tôi chốt trong `configs/lab.yaml` trước khi chạy eval):
  - Có ảnh bị reviewer escalate thì HOLD (`ESCALATED_ITEMS`).
  - Top-20 chưa có người review thì HOLD (`HUMAN_REVIEW_PENDING`).
  - Hơn 50% ảnh phải chụp lại thì REWORK (`RECAPTURE_RATE_HIGH`).
  - Các trường hợp còn lại là GO (`QC_PASSED_WITH_RECAPTURES`), ảnh bị đánh RECAPTURE sẽ bị loại khỏi lô.
  - Với từng ảnh: `score ≥ 1.0` thì RECAPTURE, còn lại KEEP.
- **Giả định cần người phụ trách dữ liệu xác nhận:** ảnh đêm và ảnh cam_B vốn tối hơn hoặc mềm hơn nhưng vẫn là ảnh bình thường. Mỗi lô cũng phải có đủ ảnh sạch trong từng nhóm (camera, ánh sáng) để trung vị của nhóm phản ánh đúng mức "bình thường".

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | 56 (24 cam_A day, 12 cam_A dusk, 12 cam_B day, 4 cam_B dusk, 4 cam_B night) | `fixtures/t02_raw_qc` (tools/make_fixtures.py, seed 42) | CC0-1.0 | có |
| eval / heldout | 40 (20 cam_A day, 8 cam_A night, 4 cam_B dusk, 8 cam_B night) | như trên | CC0-1.0 | có |

- **Schema:** `image_id, path, camera_id, light, split`. Các loại lỗi tôi dự đoán: `none, blur, glare, dark, compression`.
- **Seed và cách chia split:** seed 42. Split chia theo cảnh gốc, tức 4 biến thể của cùng một cảnh luôn nằm chung một split. Cách chia này có sẵn trong fixture.
- **Kiểm tra `manifest.csv` (sha256):** đã kiểm bằng `python -m tools.verify_manifests`. Pipeline cũng kiểm lại ở bước load (`strict_provenance: true`). Test `test_tampered_input_fails_provenance` cho thấy chỉ cần sửa 1 byte trong một ảnh là pipeline dừng với exit 2.
- **Dữ liệu thật:** không dùng.
- **Nhãn tự gán:** repo không có nhãn nên tôi tự xem và gán nhãn 56 ảnh dev, lưu ở `reports/t02_dev_selflabels.csv`. Có 28 ảnh lỗi: 8 blur, 10 compression, 6 dark, 4 glare. Tôi không gán nhãn ảnh eval.
- **Lưu ý về split:** nhóm cam_A night chỉ có trong eval, nên các ngưỡng chưa được kiểm trên nhóm này (xem §09).

## 03 · Research & design

- **Tài liệu tham khảo:** xem §11.
- **Luồng xử lý:** đọc `images.csv` và kiểm sha256. Với mỗi ảnh, tôi tính 4 đặc trưng: phương sai Laplacian, độ sáng trung bình, tỉ lệ pixel cháy sáng lớn nhất trên lưới 4×4, và độ vỡ khối tại biên 8 px. Các đặc trưng được so với trung vị của nhóm (camera_id, light) để ra 4 điểm lỗi, mỗi điểm đã chia tỉ lệ sao cho 1.0 là ngưỡng. `score` là điểm lớn nhất trong 4 điểm, và loại lỗi là loại có điểm lớn nhất đó. Sau đó lấy top-20 ra hàng đợi review kèm contact sheet HTML. Người review chọn approve/edit/reject/escalate, pipeline áp quyết định, ghi `audit_log.jsonl`, rồi ra quyết định GO/HOLD/REWORK.
- **Công thức điểm** (`src/t02_features.py`):
  - blur = log10(trung vị lap của nhóm / lap). Bằng 1.0 khi ảnh kém nét hơn nhóm 10 lần.
  - dark = log2(trung vị độ sáng của nhóm / độ sáng) / 2. Bằng 1.0 khi ảnh tối hơn nhóm 4 lần.
  - glare = max(sat_local / 0.05, log2(độ sáng / trung vị độ sáng của nhóm)). Vế đầu bắt lóa cục bộ như trên biển số, vế sau bắt lóa phủ cả ảnh.
  - compression = (blockiness − 1.0) / 0.8. Bằng 1.0 khi blockiness = 1.8.
- **Thành phần offline:** không dùng model pretrained, chỉ dùng numpy/scipy/Pillow chạy trên CPU.
- **Các đánh đổi tôi đã cân nhắc:**
  - So theo trung vị nhóm thay vì một ngưỡng chung giúp không báo nhầm ảnh đêm và ảnh cam_B. Cái giá là kết quả phụ thuộc vào thành phần của lô: nếu lô có quá nhiều ảnh lỗi thì trung vị bị kéo lệch.
  - Tôi viết luật thủ công thay vì train classifier, vì không có nhãn để train và luật thì giải thích được (cột `reason` ghi rõ giá trị từng đặc trưng). Nhược điểm là luật dựa trên ảnh synthetic có thể không dùng tốt cho ảnh thật.
  - Lưới 4×4 đủ nhạy để bắt lóa trên biển số, vì vùng lóa chiếm khoảng một ô, mà vẫn tính nhanh.

## 04 · Baseline vs PoC

- **Baseline** (`src/baseline_t02.py`, làm theo §4 của đề): `score = −log10(phương sai Laplacian)` trên toàn ảnh, dùng một ngưỡng chung `baseline_lap_threshold = 100` (chọn trên dev) để quyết định KEEP/RECAPTURE.
- **PoC** (`src/pipeline.py` + `src/t02_features.py`): 4 đặc trưng, so theo nhóm (camera, light), `score` là điểm lớn nhất trong 4 điểm.
- **Điều kiện so sánh:** hai bên dùng cùng B = 20, cùng 40 ảnh eval, cùng seed 42 và cùng lệnh `src.main --offline`.

**Bảng 1 · Baseline vs PoC & hard slices.** Bảng này cần output của `python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json`. Lệnh đó cần thư mục `gt/` của giảng viên, mà repo học viên không có (chạy sẽ ra exit 2), nên tôi để trống chờ giảng viên chấm.

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Primary: Defect Recall@B (eval) | chờ giảng viên (cần gt/) | chờ giảng viên (cần gt/) | higher is better |
| Hard-slice: False alarm in top-B, light=night | chờ giảng viên | chờ giảng viên | lower is better |
| Precision@B | chờ giảng viên | chờ giảng viên | B = 20 |
| Runtime / run (96 ảnh) | 0.27 s | 0.32 s | `out/*/run_info.json`; Intel64 Family 6 Model 151, Windows 11 |

**Tự kiểm tra trên dev.** Đây là số tôi tự chấm bằng nhãn tự gán, không phải số chính thức. Lệnh: `python -m tools.t02_dev_check out/baseline/ranked_review.csv out/poc/ranked_review.csv`.

| Metric (dev, N = 56, 28 lỗi, B = 20) | Baseline | PoC |
|---|---|---|
| Recall@B (cao nhất có thể là 20/28 = 0.714) | 0.571 | 0.714 |
| Precision@B | 0.800 | 1.000 |
| Recall của quyết định RECAPTURE | 0.321 | 1.000 |
| Precision của quyết định RECAPTURE | 1.000 | 1.000 |
| Defect-type accuracy | 0.286 | 1.000 |

Số của PoC trên dev đẹp hơn thực tế, vì tôi chọn ngưỡng ngay trên dev. Số chính thức trên eval nhiều khả năng sẽ thấp hơn.

## 05 · Evaluation protocol

- Công thức chi tiết nằm trong `reports/metric_contract.md`. Đơn vị đánh giá là từng ảnh. Một ảnh được tính là TP nếu GT ghi là lỗi và ảnh nằm trong top-20. Mẫu số của recall là tổng số ảnh lỗi trong eval.
- **Ngưỡng và budget:** B = 20. Toàn bộ hằng số nằm trong `configs/lab.yaml → params`, tôi chốt ngày 2026-11-14, trước khi có kết quả eval.
- **Độ bất định:** eval chỉ có 40 ảnh nên khi có số chính thức, tôi sẽ báo kèm Wilson 95% CI cho Recall@B và Precision@B.
- **Lệnh evaluate:**
  `python -m src.evaluate --track t02 --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json`
  Baseline chạy tương tự với `out/baseline/...` và ghi ra `reports/metrics_baseline.json`.

## 06 · Result slices

| Slice (eval, chưa có nhãn) | N | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| Số ảnh RECAPTURE | 40 | 7 | 19 | baseline chỉ bắt được ảnh mờ |
| Top-20 theo ánh sáng (day / dusk / night) | 40 | 9 / 3 / 8 | 10 / 2 / 8 | recall và false alarm theo slice cần evaluator của giảng viên |

- **Failure case 1: baseline bỏ sót mọi lỗi không phải mờ.** Trên dev, baseline bỏ sót 19/28 ảnh lỗi, gồm toàn bộ ảnh lóa (img_02_2, img_18_2, img_20_3, img_22_3), ảnh tối (img_02_3, img_07_2, …) và ảnh vỡ nén (img_00_2, img_04_2, …). Nguyên nhân: Laplacian chỉ đo độ nét. Ảnh vỡ nén còn có Laplacian cao hơn do các cạnh ở biên block, còn ảnh lóa cục bộ thì vẫn nét.
- **Failure case 2: khoảng cách ngưỡng nén của PoC khá hẹp.** Trên dev, ảnh sạch có blockiness cao nhất là img_22_0 (1.69, cam_A dusk), ảnh nén nhẹ nhất là img_08_2 (1.88). Ngưỡng 1.8 nằm giữa hai giá trị này nhưng khoảng cách chỉ khoảng 0.2. Vì vậy trên eval, ảnh nén nhẹ hơn có thể bị bỏ sót, còn ảnh sạch nhiều chi tiết có thể bị báo nhầm. Ngoài ra, ảnh vừa tối vừa mờ như img_12_3 (lap = 43, mean = 13) chỉ được gán một loại lỗi là dark.
- **Failure case 3: kết quả phụ thuộc vào thành phần của lô.** Trung vị được tính trên chính lô đang xét. Nếu một nhóm, ví dụ cam_B night, có hơn nửa số ảnh bị cùng một lỗi thì trung vị bị kéo lệch theo, và lỗi đó sẽ bị coi là bình thường.
- **Thời gian chạy:** PoC mất 0.32 s cho 96 ảnh, khoảng 3.3 ms mỗi ảnh, đã tính cả việc tạo contact sheet. Baseline mất 0.27 s. Máy: Intel64 Family 6 Model 151, Windows 11, Python 3.11.9.

## 07 · Human & decision

- **Hàng đợi review:** `out/poc/review_queue.csv` gồm 20 ảnh xếp theo `score` giảm dần. Cột `reason` ghi loại lỗi và giá trị đặc trưng, cột `slice` ghi camera/ánh sáng. File `contact_sheet.html` mở offline được và có nút Export CSV.
- **Override và audit log:** người review lưu kết quả vào `out/poc/review_queue_reviewed.csv`, sau đó `review()` áp quyết định như sau:
  - `approve`: giữ nguyên.
  - `edit` + `override_value` (none / blur / dark / glare / compression): đổi loại lỗi và hành động.
  - `reject`: đảo KEEP ↔ RECAPTURE.
  - `escalate`: chuyển sang ESCALATE.

  Mỗi thay đổi được ghi một dòng vào `audit_log.jsonl` (ai sửa, trước, sau, lý do). Pipeline cũng ghi thêm một dòng khi tạo hàng đợi và một dòng khi ra quyết định. Dòng nào có quyết định mà thiếu tên người review thì pipeline dừng với exit 2 (có test cho trường hợp này). Tôi không bao giờ sửa `score`, để metric vẫn đo đúng kết quả của model.
- **Chính sách quyết định:** như §01. Owner là "Đoàn Quang Minh (Data QC owner)". Quyết định được ghi vào `decision.json` và `decision_log.csv`.
- **Khi nào không tự phát hành:** khi top-20 chưa có người review, khi có ảnh bị escalate, hoặc khi hơn 50% ảnh phải chụp lại. Trường hợp cuối thường là dấu hiệu camera hoặc quy trình có vấn đề, không còn là lỗi lẻ tẻ.
- **Bằng chứng review:** tôi đã tự review cả 20 ảnh trong hàng đợi (`reports/review_queue.csv`). Cả 20 ảnh tôi đều thấy đúng là ảnh lỗi và đúng loại lỗi máy báo, nên tôi approve hết, không có override nào. Mỗi ảnh tôi đều ghi chú điều tôi thấy, ví dụ "lóa đèn pha, vùng biển số cháy trắng". Log nằm ở `reports/audit_log.jsonl` (20 dòng) và `reports/decision_log.csv`. Quyết định cuối là **GO** (`QC_PASSED_WITH_RECAPTURES`): phát hành 49/96 ảnh, 47 ảnh phải chụp lại. Tỉ lệ chụp lại là 0.4896, chỉ thấp hơn ngưỡng REWORK 0.5 một chút.

## 08 · Reproduce & tests

```bash
python -m venv .venv && .venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.main --track t02 --pipeline src.baseline_t02 --output out/baseline --run-name baseline --offline --seed 42
.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t02 --output out/poc --run-name poc --offline --seed 42
.venv\Scripts\python.exe -m tools.t02_dev_check out/baseline/ranked_review.csv out/poc/ranked_review.csv
.venv\Scripts\python.exe -m src.evaluate --track t02 --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json   # cần gt/ của giảng viên
.venv\Scripts\python.exe -m pytest -q
```

- **Kết quả `pytest`:** 55 passed, 39 skipped. 39 test bị skip vì cần `gt/` của giảng viên. File test của tôi là `tests/test_t02_pipeline.py` với 8 test, tất cả đều pass. Các test kiểm:
  - chạy từ đầu đến cuối và cho kết quả lặp lại được;
  - PoC tốt hơn baseline trên dev;
  - ảnh cam_B và ảnh đêm không bị báo nhầm;
  - lóa cục bộ bị bắt;
  - ảnh một màu không làm chương trình lỗi;
  - override được áp dụng và có ghi audit;
  - review thiếu tên người review cho exit 2;
  - input bị sửa cho exit 2.
- **Máy:** Intel64 Family 6 Model 151 · Windows 11 · Python 3.11.9. Commit chứa code chính: `4b408e8`.
- **Một thay đổi trong test có sẵn:** `tests/test_cli.py::test_template_pipeline_fails_clearly_not_silently` vốn kiểm rằng pipeline chưa cài thì trả exit 3, và dùng t02 để kiểm. Vì tôi đã cài t02 nên tôi chuyển test này sang t03. Nội dung kiểm tra vẫn giữ nguyên.

## 09 · Limitations & ethics

- **PII và license:** fixture là ảnh synthetic CC0, không có người hay biển số thật. Nếu sau này áp dụng cho ảnh thật (BDD100K, biển số) thì ảnh có PII và không được commit. Riêng `contact_sheet.html` nhúng ảnh dạng base64, nên bản thân file HTML này cũng phải coi là dữ liệu nhạy cảm.
- **Những điều tôi chưa chắc:**
  - Nhóm cam_A night không có ở dev nên ngưỡng chưa được kiểm trên nhóm này.
  - Luật viết cho ảnh synthetic có thể không hợp với ảnh thật, nơi có nhiễu cảm biến, mờ do chuyển động cục bộ, mưa hay sương.
  - Nếu dữ liệu tăng gấp 10 lần thì thời gian chạy tăng tuyến tính (khoảng 3 ms mỗi ảnh). Nhưng giữ B = 20 cố định sẽ không đủ, khi đó cần cho B tăng theo tỉ lệ, hoặc tự động xử lý những ảnh có score rất cao.
- Tôi không khẳng định hệ thống "đạt GDPR" hay "đúng luật Việt Nam". Muốn dùng ảnh thật thì cần DPO hoặc bộ phận pháp lý xác nhận.
- **Hướng cải tiến:**
  - cho phép một ảnh mang nhiều loại lỗi;
  - chấm điểm riêng vùng biển số;
  - lấy trung vị từ một lô tham chiếu sạch thay vì từ chính lô đang xét;
  - kiểm thêm 10 ảnh out-of-domain theo mốc +24h của đề.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu với `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Đoàn Quang Minh | doanquangminh14 / dminh14102005@gmail.com | Toàn bộ T02: features, pipeline, baseline, dev check, tests, review, report | `4b408e8` (code + test + report), `858da9c` (slides), `c241a1a` (audit log) | có |

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|
| Sakaridis, Dai & Van Gool (2021), "ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding", ICCV 2021 · https://openaccess.thecvf.com/content/ICCV2021/html/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html · dataset: https://acdc.vision.ee.ethz.ch/ | 2026-11-14 | Dataset gồm 4006 ảnh lái xe chia đều cho sương, đêm, mưa và tuyết, ảnh nào cũng có nhãn. Tức là ảnh đêm vẫn là dữ liệu dùng được chứ không phải ảnh lỗi. Đây là lý do tôi so độ sáng và độ nét theo nhóm `light` thay vì dùng một ngưỡng chung. |
| ISO/IEC TR 5259-6:2026, "Artificial intelligence — Data quality for analytics and ML — Part 6: Visualization framework for data quality" · https://www.iso.org/standard/5259-6 (Part 1: https://www.iso.org/standard/81088.html) | 2026-11-14 | Bộ 5259 bàn về đo lường, quản lý và quản trị chất lượng dữ liệu cho ML. Part 6 nói về cách trình bày kết quả đo chất lượng cho người liên quan. Tôi áp dụng ý này qua contact sheet, cột `reason` ghi giá trị đặc trưng, và quyết định có `reason_code`, `owner`, audit log. Tôi chỉ đọc phần mô tả trên trang ISO, không đọc toàn văn vì phải trả phí. |
| Pech-Pacheco, Cristóbal, Chamorro-Martínez & Fernández-Valdivia (2000), "Diatom autofocusing in brightfield microscopy: a comparative study", ICPR 2000, vol. 3, pp. 3318–3321 · DOI: https://doi.org/10.1109/ICPR.2000.903548 | 2026-11-14 | Bài so sánh các cách đo độ nét khi lấy nét tự động. Biến thể dựa trên Laplacian cho kết quả ổn định, nhanh và ít bị nhiễu ảnh hưởng. Tôi dùng phương sai Laplacian làm đặc trưng blur, và baseline của đề cũng dùng đại lượng này. |
| Wang, Bovik & Evans (2000), "Blind measurement of blocking artifacts in images", IEEE ICIP 2000, vol. 3, pp. 981–984 · DOI: https://doi.org/10.1109/ICIP.2000.899622 · PDF: https://live.ece.utexas.edu/publications/2000/zw_icip_2000_blindblock.pdf | 2026-11-14 | Bài báo coi ảnh vỡ khối là ảnh gốc cộng với một tín hiệu khối lặp lại mỗi 8 px, rồi đo độ mạnh của tín hiệu đó mà không cần ảnh gốc. Tôi làm theo cách đơn giản hơn: lấy độ chênh lệch tại biên 8 px chia cho độ chênh lệch bên trong block, không dùng FFT như trong bài. |

---

**Bảng 3 · Checklist chốt khi nộp**

- [x] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [x] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [x] Có baseline/PoC metrics, ≥1 failure case, reviewer evidence, decision log. (Số dev tôi tự chấm bằng `tools/t02_dev_check`; `metrics_*.json` chính thức chờ giảng viên chạy với `gt/`.)
- [x] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [x] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên.
