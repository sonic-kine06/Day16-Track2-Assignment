#!/bin/bash
set -e

echo "=== BƯỚC 1: Thêm pip vào PATH ==="
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

echo "=== BƯỚC 2: Cài thư viện ML ==="
pip3 install --quiet lightgbm scikit-learn pandas numpy kaggle

echo "=== BƯỚC 3: Setup Kaggle credentials ==="
mkdir -p ~/.kaggle
cat > ~/.kaggle/kaggle.json << 'KAGGLE_EOF'
{"username":"mydognameissu","key":"fca927a210e9605629639dec1b049102"}
KAGGLE_EOF
chmod 600 ~/.kaggle/kaggle.json

echo "=== BƯỚC 4: Tạo thư mục làm việc ==="
mkdir -p ~/ml-benchmark
cd ~/ml-benchmark

echo "=== BƯỚC 5: Tải dataset Credit Card Fraud ==="
~/.local/bin/kaggle datasets download -d mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/
echo "Dataset đã tải xong!"
ls -lh ~/ml-benchmark/

echo "=== BƯỚC 6: Tạo benchmark.py ==="
cat > ~/ml-benchmark/benchmark.py << 'PYTHON_EOF'
import time
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score

def run_benchmark():
    results = {}
    print("Đang tải dữ liệu...")
    start_time = time.time()
    df = pd.read_csv('creditcard.csv')
    load_time = time.time() - start_time
    results['data_load_time_seconds'] = round(load_time, 4)
    print(f"Tải dữ liệu xong trong {load_time:.2f} giây.")

    X = df.drop('Class', axis=1)
    y = df['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'verbose': -1
    }

    print("Đang huấn luyện mô hình LightGBM...")
    start_time = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'valid'],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(50)]
    )
    train_time = time.time() - start_time
    results['training_time_seconds'] = round(train_time, 4)
    results['best_iteration'] = model.best_iteration
    print(f"Huấn luyện xong trong {train_time:.2f} giây.")

    y_pred_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_prob > 0.5).astype(int)

    results['auc_roc'] = round(roc_auc_score(y_test, y_pred_prob), 4)
    results['accuracy'] = round(accuracy_score(y_test, y_pred), 4)
    results['f1_score'] = round(f1_score(y_test, y_pred), 4)
    results['precision'] = round(precision_score(y_test, y_pred), 4)
    results['recall'] = round(recall_score(y_test, y_pred), 4)

    single_row = X_test.iloc[[0]]
    start_time = time.time()
    model.predict(single_row, num_iteration=model.best_iteration)
    inf_1 = time.time() - start_time
    results['inference_latency_1_row_seconds'] = round(inf_1, 6)

    thousand_rows = X_test.iloc[:1000]
    start_time = time.time()
    model.predict(thousand_rows, num_iteration=model.best_iteration)
    inf_1000 = time.time() - start_time
    results['inference_time_1000_rows_seconds'] = round(inf_1000, 6)

    with open('benchmark_result.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("\n--- KẾT QUẢ BENCHMARK ---")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("Đã lưu kết quả vào benchmark_result.json")

if __name__ == '__main__':
    run_benchmark()
PYTHON_EOF

echo "=== BƯỚC 7: Chạy benchmark ==="
cd ~/ml-benchmark
python3 benchmark.py

echo "=== XONG! Kết quả: ==="
cat ~/ml-benchmark/benchmark_result.json
