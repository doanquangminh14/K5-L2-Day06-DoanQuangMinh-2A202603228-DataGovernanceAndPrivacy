# Group Report — Day 06 Data Lab (Field-to-Release PoC)

> Điền đầy đủ, cụ thể, không viết chung chung. Mọi con số phải lấy từ file do script sinh ra
> (`reports/metrics_*.json`, `reports/table1_baseline_vs_poc.md`) — không gõ tay, không làm tròn mất đơn vị.
> 3–5 trang (hoặc README + report.md tương đương).

## 00 · Header

| Mục | Giá trị |
|---|---|
| Group ID | G01 |
| Challenge | T02 · Raw QC Triage: ảnh mờ, lóa và thiếu sáng |
| Thành viên + vai trò | 1. Đoàn Quang Minh (Lead / Core algorithm / QA / Review + report) |
| Git URL | https://github.com/doanquangminh14/K5-L2-Day06-DoanQuangMinh-2A202603228-DataGovernanceAndPrivacy |
| Tag `demo-180m` (commit) | `858da9c` |
| Tag `final-24h` (commit) | commit mà tag `final-24h` trỏ tới — `git rev-list -n1 final-24h` |
| Deadline (giờ VN) | 2026-11-15 12:00 (GMT+7) |

## 01 · Problem statement

- **Ai bị ảnh hưởng / ai chịu hậu quả khi hệ thống sai?** Đội huấn luyện model perception: ảnh lỗi (mờ, lóa, tối, vỡ nén) lọt vào dataset làm nhiễu nhãn và giảm chất lượng model; ngược lại, loại nhầm ảnh tốt (đặc biệt ảnh đêm, ảnh cam_B) làm mất dữ liệu hiếm và tốn chi phí chụp lại.
- **Điểm nghẽn cụ thể:** người review chỉ xem được B = 20 ảnh mỗi lô; cần xếp hạng để 20 ảnh đó chứa nhiều lỗi nhất.
- **Đầu vào → đầu ra:** `fixtures/t02_raw_qc/images.csv` + 96 ảnh JPEG 320×180 → `out/poc/ranked_review.csv` (`image_id, score, pred_defect_type, pred_action`), `review_queue.csv`, `contact_sheet.html`, `audit_log.jsonl`, `decision.json`.
- **Tiêu chí GO / HOLD / REWORK (chốt trước khi chạy eval, `configs/lab.yaml`):**
  - HOLD `ESCALATED_ITEMS` nếu reviewer escalate ≥ 1 ảnh;
  - HOLD `HUMAN_REVIEW_PENDING` nếu top-B chưa được người review;
  - REWORK `RECAPTURE_RATE_HIGH` nếu tỉ lệ RECAPTURE > `max_recapture_rate` = 0.5;
  - còn lại GO `QC_PASSED_WITH_RECAPTURES` (ảnh RECAPTURE bị loại khỏi lô phát hành).
  - Mỗi ảnh: RECAPTURE nếu `score ≥ action_threshold` (1.0), ngược lại KEEP.
- **Giả định (cần domain owner xác nhận):** ảnh đêm/cam_B có phân bố độ sáng/độ nét riêng và là bình thường; mỗi lô có đủ ảnh sạch trong từng nhóm (camera, light) để trung vị nhóm đại diện cho "bình thường".

## 02 · Data & provenance

| Split | N | Nguồn / link | License | Synthetic? |
|---|---|---|---|---|
| dev | 56 (24 cam_A day, 12 cam_A dusk, 12 cam_B day, 4 cam_B dusk, 4 cam_B night) | `fixtures/t02_raw_qc` (tools/make_fixtures.py, seed 42) | CC0-1.0 | có |
| eval / heldout | 40 (20 cam_A day, 8 cam_A night, 4 cam_B dusk, 8 cam_B night) | như trên | CC0-1.0 | có |

- **Schema:** `image_id, path, camera_id, light, split`. Lớp lỗi dự đoán: `none, blur, glare, dark, compression`.
- **Seed:** 42 · **Cách chia split:** theo cảnh gốc (4 biến thể của một cảnh luôn cùng split) — do fixture cung cấp.
- **`manifest.csv` verified (sha256):** có — `python -m tools.verify_manifests`; pipeline kiểm lại ở bước load (`strict_provenance: true`), test `test_tampered_input_fails_provenance` chứng minh ảnh bị sửa 1 byte → exit 2.
- **Dữ liệu thật:** không dùng.
- **Nhãn nội bộ:** `reports/t02_dev_selflabels.csv` — 56 ảnh dev do nhóm tự xem và gán (28 lỗi: 8 blur, 10 compression, 6 dark, 4 glare). Không gán nhãn ảnh eval.
- **Lưu ý split:** cam_A night **chỉ có trong eval** → các ngưỡng không được kiểm trên nhóm này ở dev (rủi ro, xem §09).

## 03 · Research & design

- **References:** xem §11.
- **Data flow:**
  `images.csv + manifest (sha256)` → **features** (Laplacian variance, mean grey, max saturated fraction trên lưới 4×4, blockiness 8-px) → **chuẩn hoá theo trung vị nhóm (camera_id, light)** → 4 điểm lỗi, mỗi điểm scale sao cho 1.0 = ngưỡng → `score = max` + `pred_defect_type = argmax` → **top-20 queue + contact sheet HTML** → **reviewer** (approve/edit/reject/escalate) → override + `audit_log.jsonl` → **decision** GO/HOLD/REWORK.
- **Công thức điểm** (`src/t02_features.py`):
  - blur = log10(median_lap_nhóm / lap) / 1.0 → 1.0 khi độ nét thấp hơn 10× so với nhóm
  - dark = log2(median_mean_nhóm / mean) / 2.0 → 1.0 khi tối hơn 4×
  - glare = max( sat_local / 0.05 , log2(mean / median_mean_nhóm) / 1.0 ) → glare cục bộ trên biển số **hoặc** loá phủ cả ảnh
  - compression = (blockiness − 1.0) / 0.8 → 1.0 khi blockiness = 1.8
- **Thành phần offline:** không dùng model pretrained; chỉ numpy/scipy/Pillow (CPU).
- **Trade-offs:**
  - Chuẩn hoá theo nhóm thay vì ngưỡng tuyệt đối → không báo nhầm ảnh đêm/cam_B, đổi lại phụ thuộc vào thành phần lô (lô toàn ảnh lỗi sẽ kéo trung vị lệch).
  - Luật thủ công thay vì classifier học máy → giải thích được (`reason` ghi rõ feature), không cần nhãn; đổi lại có thể không tổng quát sang dữ liệu thật.
  - Lưới 4×4 cho glare: đủ nhạy với lóa trên biển số (chiếm ~1 ô), rẻ.

## 04 · Baseline vs PoC

- **Baseline** (`src/baseline_t02.py`, đúng §4 của đề): score = −log10(Laplacian variance) toàn ảnh, một ngưỡng chung `baseline_lap_threshold = 100` (chọn trên dev) cho KEEP/RECAPTURE.
- **PoC** (`src/pipeline.py` + `src/t02_features.py`): 4 feature, chuẩn hoá theo nhóm (camera, light), score = max của 4 điểm lỗi.
- **Cùng budget / cùng heldout / cùng seed:** xác nhận — B = 20, cùng 40 ảnh eval, seed 42, cùng lệnh `src.main --offline`.

**Bảng 1 · Baseline vs PoC & hard slices** — chờ giảng viên chấm: dán output của
`python -m src.compare reports/metrics_baseline.json reports/metrics_poc.json` (cần overlay `gt/` của giảng viên; repo học viên trả exit 2).

| Metric (unit) | Baseline | PoC | Δ & chú thích |
|---|---|---|---|
| Primary: Defect Recall@B (eval) | chờ giảng viên (cần gt/) | chờ giảng viên (cần gt/) | higher is better |
| Hard-slice: False alarm in top-B, light=night | chờ giảng viên | chờ giảng viên | lower is better |
| Precision@B | chờ giảng viên | chờ giảng viên | B = 20 |
| Runtime / run (96 ảnh) | 0.27 s | 0.32 s | `out/*/run_info.json`; Intel64 Family 6 Model 151, Windows 11 |

**Kiểm tra nội bộ trên dev (nhãn tự gán, KHÔNG phải số chấm chính thức)** — output của
`python -m tools.t02_dev_check out/baseline/ranked_review.csv out/poc/ranked_review.csv`:

| Metric (dev, N = 56, 28 lỗi, B = 20) | Baseline | PoC |
|---|---|---|
| Recall@B (tối đa có thể = 20/28 = 0.714) | 0.571 | 0.714 |
| Precision@B | 0.800 | 1.000 |
| Recall của quyết định RECAPTURE | 0.321 | 1.000 |
| Precision của quyết định RECAPTURE | 1.000 | 1.000 |
| Defect-type accuracy | 0.286 | 1.000 |

> Số PoC trên dev là **lạc quan**: các hằng số được chọn chính trên dev. Số eval chính thức có thể thấp hơn.

## 05 · Evaluation protocol

- Công thức primary metric, đơn vị, TP/FP/FN, denominator: xem `reports/metric_contract.md`. Đơn vị = ảnh; TP = ảnh GT lỗi trong top-20; denominator recall = số ảnh GT lỗi trong eval.
- **Threshold / budget:** B = 20; mọi hằng số trong `configs/lab.yaml → params`, chốt 2026-11-14 trước khi có kết quả eval.
- **Độ bất định:** N eval = 40 nhỏ → báo Wilson 95% CI cho Recall@B / Precision@B khi có số chính thức.
- **Lệnh evaluate chính xác:**
  `python -m src.evaluate --track t02 --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json`
  (tương tự `out/baseline/...` → `reports/metrics_baseline.json`).

## 06 · Result slices

| Slice (eval, không nhãn) | N | Baseline | PoC | Ghi chú |
|---|---|---|---|---|
| Số ảnh RECAPTURE | 40 | 7 | 19 | baseline chỉ bắt được blur |
| Top-20 theo light (day / dusk / night) | 40 | 9 / 3 / 8 | 10 / 2 / 8 | recall/false alarm theo slice: chờ evaluator của giảng viên |

- **Failure case 1 — baseline bỏ sót mọi lỗi không phải blur:** trên dev baseline bỏ sót 19/28 lỗi, gồm toàn bộ glare (img_02_2, img_18_2, img_20_3, img_22_3), dark (img_02_3, img_07_2, …) và compression (img_00_2, img_04_2, …). Root cause: Laplacian chỉ đo độ nét; ảnh vỡ nén còn **tăng** Laplacian ở biên block, ảnh lóa cục bộ vẫn nét.
- **Failure case 2 — PoC: biên an toàn hẹp ở lỗi nén nhẹ:** trên dev, ảnh sạch có blockiness cao nhất là img_22_0 (1.69, cam_A dusk) còn ảnh nén nhẹ nhất là img_08_2 (1.88). Ngưỡng 1.8 nằm giữa nhưng khe chỉ ~0.2 → ảnh nén nhẹ hơn trên eval có thể bị bỏ sót, ảnh sạch nhiều chi tiết có thể bị báo nhầm. Ngoài ra ảnh vừa tối vừa mờ (img_12_3, lap = 43, mean = 13) chỉ được gán **một** loại lỗi (dark).
- **Failure case 3 (rủi ro thiết kế) — phụ thuộc thành phần lô:** trung vị nhóm tính trên chính lô đầu vào; nếu một nhóm (vd. cam_B night) có > 50% ảnh lỗi cùng loại, trung vị bị kéo lệch và lỗi đó bị coi là "bình thường".
- **Latency / compute:** PoC 0.32 s cho 96 ảnh (~3.3 ms/ảnh, gồm cả contact sheet) · baseline 0.27 s · CPU Intel64 Family 6 Model 151, Windows 11, Python 3.11.9.

## 07 · Human & decision

- **Review queue:** `out/poc/review_queue.csv`, B = 20, ưu tiên theo `score` giảm dần; cột `reason` ghi loại lỗi + giá trị feature; `slice` = camera/light; `contact_sheet.html` (offline, có Export CSV).
- **Override + audit log:** reviewer lưu `out/poc/review_queue_reviewed.csv`; `review()` áp: `approve` giữ nguyên; `edit` + `override_value` ∈ {none, blur, dark, glare, compression} đổi loại lỗi và hành động; `reject` lật KEEP ↔ RECAPTURE; `escalate` → ESCALATE. Mỗi thay đổi ghi 1 dòng `out/poc/audit_log.jsonl` (actor, before, after, reason). Dòng có quyết định mà thiếu tên reviewer → fail rõ (exit 2, có test). `score` không bao giờ bị sửa → metric vẫn đo đúng model.
- **Decision policy + reason_code + owner:** xem §01; owner = "Đoàn Quang Minh (Data QC owner)"; ghi vào `decision.json` + `decision_log.csv`.
- **Khi nào HOLD/REWORK thay vì tự release:** chưa có người review top-B, có ảnh escalate, hoặc > 50% ảnh phải chụp lại (dấu hiệu lỗi camera/quy trình chứ không phải lỗi lẻ).
- **Bằng chứng review:** `reports/review_queue.csv` — 20/20 ảnh trong queue được reviewer (Đoàn Quang Minh) xác nhận `approve`, 0 override; `reports/audit_log.jsonl` (20 sự kiện); `reports/decision_log.csv`. Quyết định cuối: **GO** `QC_PASSED_WITH_RECAPTURES` — phát hành 49/96 ảnh, 47 ảnh RECAPTURE (tỉ lệ 0.4896, sát ngưỡng `max_recapture_rate` = 0.5).

## 08 · Reproduce & tests

```bash
python -m venv .venv && .venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m src.main --track t02 --pipeline src.baseline_t02 --output out/baseline --run-name baseline --offline --seed 42
.venv\Scripts\python.exe -m src.main --config configs/lab.yaml --track t02 --output out/poc --run-name poc --offline --seed 42
.venv\Scripts\python.exe -m tools.t02_dev_check out/baseline/ranked_review.csv out/poc/ranked_review.csv
.venv\Scripts\python.exe -m src.evaluate --track t02 --pred out/poc/ranked_review.csv --run-name poc --out reports/metrics_poc.json   # cần gt/ của giảng viên
.venv\Scripts\python.exe -m pytest -q
```

- **Output của `pytest`:** `55 passed, 39 skipped` (39 skipped = test cần `gt/` của giảng viên). Test của nhóm `tests/test_t02_pipeline.py`: 8 passed — E2E + tái lập, PoC > baseline trên dev, cam_B/đêm không bị báo nhầm, glare cục bộ bị bắt, ảnh hằng số không crash, override + audit, review thiếu reviewer → exit 2, input bị sửa → exit 2.
- **Máy:** Intel64 Family 6 Model 151 · Windows 11 · Python 3.11.9 · git sha: `4b408e8`.
- Thay đổi hạ tầng: `tests/test_cli.py::test_template_pipeline_fails_clearly_not_silently` chuyển từ t02 sang t03 vì t02 đã được cài (hành vi kiểm tra — exit 3 khi chưa cài — giữ nguyên).

## 09 · Limitations & ethics

- **Rủi ro pháp lý / PII / license:** fixture synthetic CC0, không có người/biển số thật. Nếu áp dụng lên ảnh thật (BDD100K, biển số), ảnh chứa PII → không commit, contact sheet nhúng base64 ảnh nên chính file HTML cũng là dữ liệu nhạy cảm.
- **Điều chưa biết:** cam_A night không có ở dev → ngưỡng chưa được kiểm trên nhóm này; luật thủ công trên ảnh synthetic có thể không tổng quát sang ảnh thật (nhiễu cảm biến, motion blur cục bộ, mưa/sương). Nếu scale 10×: chi phí tuyến tính (~3 ms/ảnh), nhưng B = 20 cố định sẽ không đủ — cần B theo tỉ lệ hoặc tự động hoá ảnh có score rất cao.
- Không tuyên bố "đạt GDPR / luật Việt Nam"; mọi việc dùng ảnh thật cần DPO/legal xác nhận.
- **Kế hoạch cải tiến:** gán nhiều nhãn lỗi cho một ảnh; ROI-based scoring (vùng biển số); trung vị nhóm tính trên lô tham chiếu sạch thay vì chính lô; kiểm 10 ảnh out-of-domain theo mốc +24h của đề.

## 10 · Team contribution

**Bảng 2 · Individual contribution evidence** (đối chiếu với `git shortlog -sne --all` và `reports/team_contributions.csv`)

| Học viên | Git author/email | Module / vai trò | Commit SHA / diff chứng minh | Trình bày? |
|---|---|---|---|---|
| Đoàn Quang Minh | doanquangminh14 / dminh14102005@gmail.com | Toàn bộ T02: features, pipeline, baseline, dev check, tests, report | `4b408e8` feat(t02): group-normalised QC scoring, Laplacian baseline, HITL review, tests, report | có |

## 11 · References

| URL | Ngày xem | 1–2 dòng ý nghĩa (không chép nguyên văn) |
|---|---|---|
| Sakaridis, Dai & Van Gool (2021), "ACDC: The Adverse Conditions Dataset with Correspondences for Semantic Driving Scene Understanding", ICCV 2021 · https://openaccess.thecvf.com/content/ICCV2021/html/Sakaridis_ACDC_The_Adverse_Conditions_Dataset_With_Correspondences_for_Semantic_Driving_ICCV_2021_paper.html · dataset: https://acdc.vision.ee.ethz.ch/ | 2026-11-14 | 4006 ảnh lái xe chia đều cho sương / đêm / mưa / tuyết, đều có nhãn — ảnh đêm là điều kiện vận hành hợp lệ chứ không phải lỗi → lý do chuẩn hoá độ sáng/độ nét theo nhóm `light` thay vì một ngưỡng tuyệt đối. |
| ISO/IEC TR 5259-6:2026, "Artificial intelligence — Data quality for analytics and ML — Part 6: Visualization framework for data quality" · https://www.iso.org/standard/5259-6 (cả bộ: Part 1 https://www.iso.org/standard/81088.html) | 2026-11-14 | Bộ 5259 định nghĩa đo lường, quản lý và quản trị chất lượng dữ liệu cho ML; Part 6 hướng dẫn trực quan hoá kết quả đo chất lượng cho các bên liên quan → contact sheet + `reason` hiển thị giá trị feature, decision có `reason_code`, `owner`, audit log. (Chỉ đọc mô tả trên trang ISO, không đọc toàn văn có phí.) |
| Pech-Pacheco, Cristóbal, Chamorro-Martínez & Fernández-Valdivia (2000), "Diatom autofocusing in brightfield microscopy: a comparative study", ICPR 2000, vol. 3, pp. 3318–3321 · DOI: https://doi.org/10.1109/ICPR.2000.903548 | 2026-11-14 | So sánh các thước đo lấy nét; biến thể dựa trên Laplacian cho độ đo độ nét tin cậy, nhanh, bền với nhiễu → dùng phương sai Laplacian làm feature blur (và là baseline của đề). |
| Wang, Bovik & Evans (2000), "Blind measurement of blocking artifacts in images", IEEE ICIP 2000, vol. 3, pp. 981–984 · DOI: https://doi.org/10.1109/ICIP.2000.899622 · PDF: https://live.ece.utexas.edu/publications/2000/zw_icip_2000_blindblock.pdf | 2026-11-14 | Coi ảnh vỡ khối = ảnh gốc + tín hiệu khối có chu kỳ 8 px; đo năng lượng tín hiệu đó mà không cần ảnh gốc. Nhóm dùng phiên bản đơn giản hoá: tỉ lệ chênh lệch tại biên 8 px / chênh lệch bên trong block (không dùng FFT như bài báo). |

---

**Bảng 3 · Checklist chốt khi nộp**

- [x] Link repo và tag cuối cùng trong README; input mẫu không chứa PII thật.
- [x] Đúng 1 lệnh run, 1 lệnh evaluate, 1 lệnh tests; không cần download khi chạy.
- [x] Có baseline/PoC metrics, ≥1 failure case, reviewer evidence, decision log. (metrics dev nội bộ qua `tools/t02_dev_check`; `metrics_*.json` chính thức chờ giảng viên chạy với `gt/`)
- [x] Có license/source provenance, data splits, seed, offline environment, GT/eval N.
- [x] Mỗi thành viên có ≥1 commit thực chất và được nêu minh bạch trong bảng trên.
