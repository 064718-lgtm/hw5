# AI / Human 文章偵測器 (Streamlit)

一個簡單的 AI vs Human 文章分類工具。輸入文本即可估計 AI / Human 機率，並可視覺化 token 權重。此版本採用啟發式特徵（無外部模型下載）。

- 線上 Demo: https://jtmgcfkvhjatjmrwvcdblm.streamlit.app/
- Token 權重視覺化：依頻率計算權重，透過長條圖呈現所有 token 的相對影響。
- 中英混排皆可使用。

## 安裝
```bash
pip install -r requirements.txt
```

## 執行
```bash
streamlit run app.py
```
- 第一次使用會自動下載模型；若離線則使用啟發式模式。

## 使用說明
1. 在文字框輸入/貼上待分析的文本；側邊可選擇範例（AI-like / Human-like）。
2. 按「分析 Analyze」開始推論，顯示 AI% / Human% 指標與柱狀圖。
3. 展開「特徵摘要」查看 token/句長等統計。
4. Token 權重視覺化：滑桿選擇前 N 個高權重 token，長條圖呈現頻率佔比，便於觀察重要詞彙。

## 檔案說明
- `app.py`: Streamlit 介面與偵測邏輯（Transformer + 啟發式、token 權重圖）。
- `requirements.txt`: 套件需求。
- `GPT_chat.md`: 對話紀錄。
- `.gitignore`: 版本控管忽略清單。

## 備註
- 權重定義：權重 = token 出現次數 / 總 token 數，供視覺化觀察高頻詞。
- 啟發式調整：偏向句長平滑、詞彙多樣度適中、停用詞比例適中的文本（常見於 AI 生成語句）。
