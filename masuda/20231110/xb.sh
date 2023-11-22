#!/bin/bash

# ディレクトリのパスを設定
img_dir="img"

# img ディレクトリが存在することを確認
if [[ -d "$img_dir" ]]; then
    # img ディレクトリ内のファイルをループ処理
    for file in "$img_dir"/*.{png,pdf}; do
        if [[ -f "$file" ]]; then
            echo "Running for $file..."
            extractbb "$file"
        fi
    done
else
    echo "There is no img folder."
fi

echo "Done."