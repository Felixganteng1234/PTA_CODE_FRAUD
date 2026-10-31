"""Laporan lokal HTML/Markdown dari artefak eksperimen yang telah selesai.

Tidak melakukan pelatihan dan tidak menaksir hasil yang belum tersedia. Semua
nilai hasil berasal dari CSV/JSON run; gambar ditautkan relatif ke folder run.
"""
from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from urllib.parse import quote


CONDITIONS = ("pure", "smote")
FOLDS = ("fold1", "fold2", "fold3", "final_holdout")
LABELS = {"pure": "Pure LSTM", "smote": "LSTM + SMOTE", "delta_smote_minus_pure": "Selisih SMOTE - Pure"}
SCORES = ("precision", "recall", "f1", "average_precision", "pr_auc_trapezoid")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value, digits=6):
    if value is None or value == "":
        return "Tidak terdefinisi"
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Laporan menolak nilai hasil NaN atau infinity.")
    return f"{value:.{digits}f}"


def _integer(value):
    return f"{int(value):,}".replace(",", ".")


def _check_csv_json(csv_rows, json_rows, key_fields):
    """Cegah laporan merangkum CSV dan JSON yang berasal dari run berbeda."""
    indexed = {tuple(str(row[field]) for field in key_fields): row for row in csv_rows}
    if len(indexed) != len(csv_rows) or len(csv_rows) != len(json_rows):
        raise ValueError("CSV dan JSON ringkasan memiliki jumlah/baris kunci yang berbeda.")
    for expected in json_rows:
        key = tuple(str(expected[field]) for field in key_fields)
        if key not in indexed:
            raise ValueError(f"Ringkasan CSV tidak memiliki baris {key}.")
        actual = indexed[key]
        for field, value in expected.items():
            if field not in actual:
                raise ValueError(f"Kolom {field} tidak ada pada ringkasan CSV.")
            if value is None:
                matches = actual[field] == ""
            elif isinstance(value, (int, float)):
                try:
                    matches = math.isclose(float(actual[field]), value, rel_tol=1e-10, abs_tol=1e-12)
                except ValueError:
                    matches = False
            else:
                matches = actual[field] == str(value)
            if not matches:
                raise ValueError(f"CSV/JSON tidak konsisten pada {key}, kolom {field}.")


def _load_results(data_dir, output_dir):
    manifest = _read_json(output_dir / "run_manifest.json")
    if manifest.get("status") != "completed":
        raise ValueError("Laporan hanya dapat dibuat untuk run berstatus completed; run belum lengkap atau gagal.")
    config = manifest["config"]
    if config.get("profile") not in ("smoke", "full"):
        raise ValueError("Profil hasil harus smoke atau full.")
    metadata = _read_json(data_dir / "metadata.json")
    if metadata.get("status") != "complete":
        raise ValueError("Persiapan dataset belum lengkap.")
    if manifest.get("dataset_counts", {}).get("n") != metadata["sequences"]:
        raise ValueError("Jumlah sequence manifest tidak cocok dengan metadata dataset.")
    if manifest.get("dataset_counts", {}).get("fraud") != metadata["sequence_fraud"]:
        raise ValueError("Jumlah fraud manifest tidak cocok dengan metadata dataset.")
    rows = _read_csv(output_dir / "metrics.csv")
    expected = {(condition, fold) for condition in CONDITIONS for fold in FOLDS}
    actual = [(row["condition"], row["fold"]) for row in rows]
    if len(rows) != 8 or set(actual) != expected:
        raise ValueError("metrics.csv wajib memuat tepat 8 kombinasi kondisi/fold yang unik dan lengkap.")
    if set(manifest.get("completed_models", [])) != {f"{condition}/{fold}" for condition, fold in expected}:
        raise ValueError("Manifest belum mencatat seluruh 8 model selesai.")
    for row in rows:
        if row["profile"] != config["profile"]:
            raise ValueError("Profil metrics.csv berbeda dari manifest.")
        for field in SCORES:
            _number(row[field])
        if sum(int(row[field]) for field in ("tn", "fp", "fn", "tp")) != int(row["n_test"]):
            raise ValueError("Jumlah confusion matrix tidak sama dengan n_test.")
        if int(row["tp"]) + int(row["fn"]) != int(row["fraud_test"]):
            raise ValueError("Jumlah fraud confusion matrix tidak sama dengan fraud_test.")
    indexed = {(row["condition"], row["fold"]): row for row in rows}
    for fold in FOLDS:
        for field in ("n_test", "fraud_test", "test_start", "test_end", "train_end", "n_train_original"):
            if indexed[("pure", fold)][field] != indexed[("smote", fold)][field]:
                raise ValueError(f"Kedua kondisi tidak menggunakan populasi sama pada {fold}/{field}.")
    pooled = _read_json(output_dir / "pooled_summary.json")
    if len(pooled) != 2 or {row["condition"] for row in pooled} != set(CONDITIONS):
        raise ValueError("Ringkasan pooled harus berisi kedua kondisi.")
    _check_csv_json(_read_csv(output_dir / "pooled_summary.csv"), pooled, ("condition",))
    bootstrap = _read_json(output_dir / "bootstrap_uncertainty.json")
    bootstrap_rows = []
    for condition in (*CONDITIONS, "delta_smote_minus_pure"):
        if set(bootstrap[condition]) != set(SCORES):
            raise ValueError("Bootstrap harus berisi lima metrik untuk setiap kondisi/selisih.")
        for metric, values in bootstrap[condition].items():
            if values["n_valid"] + values["n_skipped"] != bootstrap["n_bootstrap"]:
                raise ValueError("Jumlah replikasi bootstrap tidak konsisten.")
            bootstrap_rows.append({"condition": condition, "metric": metric,
                                   "profile": config["profile"], **values})
    _check_csv_json(_read_csv(output_dir / "bootstrap_uncertainty.csv"), bootstrap_rows, ("condition", "metric"))
    if float(bootstrap["threshold"]) != float(config["threshold"]):
        raise ValueError("Threshold bootstrap berbeda dari konfigurasi.")
    selections = _read_json(output_dir / "selection_counts.json")
    figures = _read_json(output_dir / "figures.json")
    relative_figures = {}
    for name, raw_path in figures.items():
        path = Path(raw_path)
        path = path if path.is_absolute() else output_dir / path
        path = path.resolve()
        try:
            relative = path.relative_to(output_dir.resolve())
        except ValueError as exc:
            raise ValueError("Gambar laporan harus berada dalam folder run agar portabel.") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        relative_figures[name] = relative.as_posix()
    return manifest, metadata, rows, pooled, bootstrap, selections, relative_figures


def _render(blocks, title, smoke):
    """Satu struktur isi menghasilkan dua format dengan angka yang identik."""
    markdown, fragments = [f"# {title}"], []
    for kind, value in blocks:
        if kind == "heading":
            markdown.append(f"## {value}")
            fragments.append(f"<h2>{html.escape(value)}</h2>")
        elif kind == "paragraph":
            markdown.append(value)
            fragments.append(f"<p>{html.escape(value)}</p>")
        elif kind == "banner":
            markdown.append(f"> **{value}**")
            fragments.append(f'<div class="banner" role="note">{html.escape(value)}</div>')
        elif kind == "table":
            headers, rows = value
            escaped = lambda text: str(text).replace("|", "\\|").replace("\n", " ")
            lines = ["| " + " | ".join(map(escaped, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
            lines.extend("| " + " | ".join(map(escaped, row)) + " |" for row in rows)
            markdown.append("\n".join(lines))
            head = "".join(f"<th scope='col'>{html.escape(str(item))}</th>" for item in headers)
            body = "".join("<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>" for row in rows)
            fragments.append(f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>')
        elif kind == "code":
            code = json.dumps(value, ensure_ascii=False, indent=2)
            markdown.append(f"```json\n{code}\n```")
            fragments.append(f"<pre><code>{html.escape(code)}</code></pre>")
        elif kind == "image":
            label, path = value
            markdown.append(f"![{label}]({quote(path, safe='/')})")
            fragments.append(f'<figure><img src="{quote(path, safe="/")}" alt="{html.escape(label, quote=True)}"><figcaption>{html.escape(label)}</figcaption></figure>')
        elif kind == "links":
            markdown.append("\n".join(f"- [{label}]({quote(path, safe='/:#')})" for label, path in value))
            fragments.append("<ul>" + "".join(f'<li><a href="{html.escape(quote(path, safe="/:#"), quote=True)}">{html.escape(label)}</a></li>' for label, path in value) + "</ul>")
    css = """
    :root{color-scheme:light;font-family:Segoe UI,Arial,sans-serif;color:#17293b;background:#eef3f7;line-height:1.6}
    body{margin:0}main{max-width:1140px;margin:32px auto;padding:36px 42px;background:white;border:1px solid #dce4ed;border-radius:14px}
    header{border-bottom:3px solid #1d5774;padding-bottom:20px}h1{font-size:30px;line-height:1.3;margin:0 0 10px;color:#113d59}
    h2{font-size:21px;margin:32px 0 12px;color:#174961}p{margin:12px 0}.eyebrow{font-size:13px;letter-spacing:.06em;font-weight:700;color:#617486}
    .banner{background:#fff1d6;border:2px solid #d59722;border-radius:8px;padding:20px;font-size:23px;font-weight:800;color:#724300;margin:24px 0}
    .table-wrap{overflow-x:auto;margin:18px 0}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{border-bottom:1px solid #dce4ed;padding:10px 12px;text-align:left}th{background:#eaf2f8;color:#153e58}tbody tr:nth-child(even){background:#f7f9fb}
    pre{overflow:auto;background:#f0f4f7;padding:18px;border-radius:6px;font-size:12px}figure{margin:22px 0}img{max-width:100%;height:auto;border:1px solid #dce4ed;border-radius:6px}figcaption{font-size:13px;color:#607284}a{color:#125b84}footer{margin-top:40px;border-top:1px solid #dce4ed;padding-top:14px;font-size:12px;color:#607284}
    @media(max-width:700px){main{margin:0;padding:22px 16px;border:0;border-radius:0}h1{font-size:25px}.banner{font-size:19px}}
    @media print{body{background:white}main{max-width:none;margin:0;padding:0;border:0}.table-wrap{overflow:visible}table{white-space:normal;font-size:9px}th,td{padding:5px}h2{break-after:avoid}figure,tr{break-inside:avoid}pre{white-space:pre-wrap}.banner{font-size:18px}}
    """
    html_text = f'''<!doctype html>
<html lang="id"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style></head>
<body><main><header><div class="eyebrow">PAYSIM · PURE LSTM DAN LSTM + SMOTE · LAPORAN LOKAL</div><h1>{html.escape(title)}</h1><p>Profil {"SMOKE / uji teknis" if smoke else "FULL / eksperimen penelitian"} — status artefak: completed</p></header>
{"".join(fragments)}<footer>Laporan dibangun dari artefak eksperimen tersimpan. Tidak menggunakan layanan jaringan atau skrip eksternal.</footer></main></body></html>'''
    return "\n\n".join(markdown) + "\n", html_text


def build_report(data_dir: Path, output_dir: Path) -> Path:
    """Buat LAPORAN_HASIL.md dan laporan.html; kembalikan path absolut HTML.

    Run yang belum completed atau memiliki ringkasan tidak konsisten ditolak
    sebelum laporan ditulis. Memanggil ulang fungsi ini hanya membangun ulang
    laporan, tanpa mengubah hasil training.
    """
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    manifest, meta, results, pooled, boot, selections, figures = _load_results(data_dir, output_dir)
    config = manifest["config"]
    smoke = config["profile"] == "smoke"
    title = "Hasil Uji Teknis Pipeline Fraud" if smoke else "Hasil Eksperimen Deteksi Fraud"
    blocks = []
    add = lambda kind, value: blocks.append((kind, value))
    if smoke:
        add("banner", "UJI TEKNIS - BUKAN HASIL FINAL PENELITIAN")
        add("paragraph", "Profil full belum dijalankan dalam run ini. Angka berikut adalah hasil nyata uji teknis pada subset data latih. Hasil ini tidak menetapkan keunggulan akhir Pure LSTM atau LSTM+SMOTE dan tidak boleh disalin sebagai kesimpulan akhir skripsi. Jalankan konfigurasi full untuk eksperimen penelitian dengan seluruh data latih.")
    else:
        add("paragraph", "Run profil full telah selesai untuk kedua kondisi, tiga fold pengembangan, dan final holdout. Penafsiran hasil tetap dibatasi oleh populasi sequence yang memenuhi syarat, desain evaluasi, serta ketidakpastian statistik di bawah.")
    add("heading", "Status dan waktu eksekusi")
    runtime = manifest.get("runtime", {})
    elapsed = float(manifest.get("elapsed_seconds", 0))
    add("table", (["Informasi", "Nilai"], [
        ["Status", manifest["status"]], ["Profil", config["profile"]],
        ["Mulai (UTC)", manifest.get("started_at_utc", "Tidak dicatat")],
        ["Selesai (UTC)", manifest.get("finished_at_utc", "Tidak dicatat")],
        ["Durasi run total", f"{elapsed:.3f} detik ({elapsed / 60:.2f} menit)"],
        ["Perangkat TensorFlow", ", ".join(runtime.get("devices", [])) or "Tidak dicatat"],
        ["Thread CPU konfigurasi", config.get("cpu_threads", "Tidak dicatat")],
        ["Sistem", runtime.get("platform", "Tidak dicatat")],
        ["SHA-256 konfigurasi", manifest.get("config_sha256", "Tidak dicatat")],
    ]))
    add("paragraph", "Durasi total mencakup tuning, pemilihan epoch, pelatihan ulang, prediksi, dan bootstrap. Ini bukan pengukuran latency inferensi per transaksi; latency layanan produksi belum diuji. Perangkat dan jumlah thread di atas berasal dari manifest, sehingga kemampuan CPU/GPU tidak diasumsikan.")
    add("heading", "Audit dataset dan cakupan populasi")
    dataset_fields = [("raw_transactions", "Transaksi sumber"), ("raw_fraud", "Fraud sumber"),
                      ("customer_transactions", "Transaksi ke akun C"), ("merchant_transactions", "Transaksi ke merchant M"),
                      ("merchant_fraud", "Fraud ke merchant M"), ("customer_accounts", "Akun penerima C"),
                      ("eligible_accounts", "Akun yang membentuk sequence"), ("sequences", "Sequence terbentuk"),
                      ("sequence_fraud", "Sequence dengan target fraud")]
    add("table", (["Ukuran", "Jumlah"], [[label, _integer(meta[key])] for key, label in dataset_fields]))
    coverage = float(meta["fraud_coverage"]) * 100
    add("paragraph", f"Cakupan target fraud adalah {coverage:.2f}% dari fraud sumber. Hanya transaksi dengan sedikitnya dua transaksi pendahulu pada akun penerima C yang sama menjadi target. Karena itu, hasil ini tidak menggambarkan kemampuan mendeteksi seluruh fraud pada dataset mentah.")
    add("paragraph", f"Sequence sepanjang {meta['sequence_length']} transaksi menggunakan skema inclusive: dua transaksi sebelumnya dan transaksi target. Urutan ditentukan oleh {', '.join(meta['sort_keys'])}; indeks baris asli memecah kesamaan step, bukan bukti urutan waktu aktual di dalam jam yang sama. Label mengikuti {meta['label_rule']}. Target step berada pada {meta['min_target_step']}–{meta['max_target_step']}.")
    add("paragraph", "Fitur: " + ", ".join(meta["features"]) + ". nameDest hanya untuk pengelompokan; step hanya untuk urutan/pembagian; isFraud dan isFlaggedFraud tidak menjadi masukan.")
    add("paragraph", "Evaluasi bersifat setelah transaksi (post-transaction), karena fitur newbalanceOrig dan newbalanceDest mencakup saldo setelah transaksi target. Hasil tidak membuktikan kemampuan memblokir transaksi sebelum transaksi terjadi.")
    add("paragraph", f"Sumber: {meta['source_file']}; SHA-256: {meta['source_sha256']}. Baris sumber yang sama tidak dihapus otomatis; setiap transaksi dipertahankan sesuai kebijakan audit.")
    add("heading", "Skala pelatihan dan pembagian temporal")
    selection_rows = []
    for row in selections:
        original, selected = row["source"], row["selected"]
        fraction = selected["n"] / original["n"] * 100 if original["n"] else 0
        selection_rows.append([row["stage"], f"≤ {row['train_end']}", _integer(original["n"]),
                               _integer(selected["n"]), _integer(selected["fraud"]), f"{fraction:.3f}%", row["method"]])
    add("table", (["Tahap", "Step latih", "Populasi latih", "Latih dipilih", "Fraud dipilih", "Proporsi", "Metode"], selection_rows))
    add("paragraph", ("Untuk smoke, subset latih dipilih seragam tanpa pengembalian. Ukuran subset dapat digandakan bila jumlah fraud pada bagian awal latih belum cukup untuk SMOTE k=5. " if smoke else "Seluruh sequence pada masing-masing rentang latih digunakan. ") + "Subset/rentang latih yang sama dipakai kedua kondisi. Semua sequence validasi dan uji tetap digunakan secara lengkap dengan distribusi kelas asli; tabel di atas tidak merujuk pada ukuran data validasi atau uji.")
    if int(config["n_trials"]) == 1:
        add("paragraph", "Run ini memakai satu konfigurasi hyperparameter yang ditetapkan sebelum evaluasi (learning rate 0,001; dropout 0,2; batch size 512; patience 5) dan dijalankan identik pada kedua kondisi. Optuna hanya merekam satu trial; ini bukan pencarian atau optimasi TPE. Data tuning tetap step ≤340 dengan validasi bersih 341–400. Pada setiap fold, 60 step terakhir rentang latih memilih epoch melalui validation loss; final holdout tidak dipakai memilih parameter, epoch, atau threshold.")
    else:
        add("paragraph", "Optuna TPE menala kedua kondisi secara terpisah pada step ≤340, dengan validasi bersih 341–400. Pada setiap fold, 60 step terakhir rentang latih dipakai untuk memilih epoch melalui validation loss; model baru kemudian dilatih ulang pada seluruh data latih terpilih untuk jumlah epoch tersebut. StandardScaler di-fit hanya pada latih; SMOTE hanya diterapkan pada latih sesudah flatten dan sebelum reshape. Final holdout tidak dipakai memilih hyperparameter, epoch, atau threshold.")
    add("paragraph", f"Arsitektur sama: LSTM 64 unit, Dropout, LSTM 32 unit, Dropout, Dense sigmoid; Adam dan binary cross-entropy. SMOTE memakai sampling_strategy={config['smote_ratio']} dan k_neighbors={config['smote_k']}. Rasio tersebut adalah fraud/non-fraud setelah resampling, setara sekitar sepertiga total data latih, bukan 50% dari seluruh sampel.")
    add("heading", "Hasil terukur per fold dan final holdout")
    add("paragraph", f"Threshold klasifikasi ditetapkan sebelum evaluasi: {float(config['threshold']):g}; prediksi fraud bila probabilitas ≥ threshold. Threshold tidak dioptimasi pada final holdout. Precision, recall, dan F1 dilaporkan pada threshold ini. Semua skor berikut dalam rentang 0–1, bukan persentase.")
    ordered = sorted(results, key=lambda row: (FOLDS.index(row["fold"]), CONDITIONS.index(row["condition"])))
    headers = ["Kondisi", "Partisi", "Step uji", "Latih asli", "Latih sesudah perlakuan", "N uji", "Fraud uji", "Epoch", "Precision", "Recall", "F1", "AP", "PR-AUC trapezoid"]
    table = [[LABELS[row["condition"]], row["fold"], f"{row['test_start']}–{row['test_end']}",
              _integer(row["n_train_original"]), _integer(row["n_train_after_smote"]),
              _integer(row["n_test"]), _integer(row["fraud_test"]), row["selected_epoch"],
              *[_number(row[metric]) for metric in SCORES]] for row in ordered]
    add("table", (headers, table))
    objective_text = "menjadi objective Optuna" if int(config["n_trials"]) > 1 else "digunakan untuk mencatat performa validasi konfigurasi tunggal"
    add("paragraph", f"AP (average precision) menggunakan penjumlahan precision yang dibobot perubahan recall tanpa interpolasi dan {objective_text}. PR-AUC trapezoid adalah integrasi trapezoidal kurva precision-recall; nilainya dapat berbeda dari AP. Keduanya tidak bergantung pada satu threshold klasifikasi. ROC-AUC tambahan tersimpan dalam metrics.csv, tetapi bukan metrik utama penelitian ini.")
    add("table", (["Kondisi", "Partisi", "TN", "FP", "FN", "TP"], [[LABELS[row["condition"]], row["fold"], *[_integer(row[key]) for key in ("tn", "fp", "fn", "tp")]] for row in ordered]))
    add("heading", "Agregasi tiga fold pengembangan")
    add("paragraph", "Precision, recall, dan F1 pooled dihitung dari gabungan prediksi fold 1–3, bukan rata-rata skor fold. Mean AP dan simpangan baku sampel AP dihitung antar tiga fold (ddof=1). Final holdout tidak dimasukkan ke agregasi ini.")
    add("table", (["Kondisi", "N pooled", "Fraud pooled", "Precision pooled", "Recall pooled", "F1 pooled", "Mean AP fold", "SD AP fold"],
                  [[LABELS[row["condition"]], _integer(row["n_samples"]), _integer(row["n_fraud"]),
                    *[_number(row[metric]) for metric in ("precision", "recall", "f1", "pr_auc_fold_mean", "pr_auc_fold_std_sample")]] for row in pooled]))
    add("heading", "Ketidakpastian final holdout")
    add("paragraph", f"Bootstrap berpasangan sebanyak {boot['n_bootstrap']} replikasi, seed {boot['seed']}, metode {boot['method']}. Indeks sampel bootstrap yang sama digunakan pada kedua model. CI adalah percentile 95%; selisih didefinisikan sebagai SMOTE dikurangi Pure. Replikasi tanpa fraud dilewati untuk AP/PR-AUC dan jumlahnya dilaporkan.")
    ci_rows = []
    for condition in (*CONDITIONS, "delta_smote_minus_pure"):
        for metric in SCORES:
            row = boot[condition][metric]
            ci_rows.append([LABELS[condition], metric, _number(row["estimate"]),
                            _number(row["ci_lower"]), _number(row["ci_upper"]),
                            row["n_valid"], row["n_skipped"]])
    add("table", (["Kondisi/selisih", "Metrik", "Estimasi", "CI bawah", "CI atas", "Valid", "Dilewati"], ci_rows))
    add("paragraph", boot["assumption"])
    if boot.get("n_groups") is None:
        add("paragraph", "Bootstrap sequence mengasumsikan observasi independen. Sequence bertumpang tindih dan akun yang sama dapat saling bergantung, sehingga CI ini mungkin terlalu sempit. CI tidak mencakup ketidakpastian dari pengulangan seed pelatihan, pemilihan hyperparameter, atau variasi dataset.")
    if smoke:
        add("paragraph", "CI di atas tetap merupakan CI untuk model uji teknis yang dilatih pada subset. Banyaknya replikasi bootstrap tidak mengubah model smoke menjadi hasil final penelitian.")
    add("heading", "Grafik final holdout")
    for name, path in figures.items():
        add("image", ("Kurva precision-recall final holdout" if name == "precision_recall" else f"Confusion matrix final holdout: {name}", path))
    add("heading", "Konfigurasi dan reproduksibilitas")
    add("code", config)
    add("paragraph", "Versi pustaka runtime yang tercatat:")
    add("code", runtime.get("packages", {}))
    add("heading", "Berkas hasil pendukung")
    add("links", [("Metrik semua delapan model (CSV)", "metrics.csv"),
                  ("Agregasi fold (CSV)", "pooled_summary.csv"), ("Agregasi fold (JSON)", "pooled_summary.json"),
                  ("Bootstrap dan selisih berpasangan (CSV)", "bootstrap_uncertainty.csv"),
                  ("Bootstrap dan asumsi (JSON)", "bootstrap_uncertainty.json"),
                  ("Jumlah subset pelatihan", "selection_counts.json"), ("Manifest eksekusi dan konfigurasi", "run_manifest.json")])
    add("heading", "Dokumentasi metode perhitungan")
    add("links", [("Definisi average precision tanpa interpolasi — scikit-learn", "https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html"),
                  ("SMOTE: sampling_strategy dan k_neighbors — imbalanced-learn", "https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html")])
    markdown, html_text = _render(blocks, title, smoke)
    (output_dir / "LAPORAN_HASIL.md").write_text(markdown, encoding="utf-8")
    path = output_dir / "laporan.html"
    path.write_text(html_text, encoding="utf-8")
    return path.resolve()
