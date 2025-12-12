# AI / Human 文章偵測器 (Streamlit)

一個簡單的 AI vs Human 文章分類工具。輸入文本即可估計 AI / Human 機率，並可視覺化 token 權重。

- 線上 Demo: https://jtmgcfkvhjatjmrwvcdblm.streamlit.app/
- 內建 Transformer 模型 (`Hello-SimpleAI/chatgpt-detector-roberta`)，無法載入時改用啟發式特徵。
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
1. 在文字框輸入/貼上待分析的文本。
2. 勾選「Prefer transformer detector」即可使用模型推論；取消則強制啟發式。
3. 立即顯示 AI% / Human% 指標及長條圖。
4. 展開「模型分數」或「特徵摘要」查看細節。
5. Token 權重視覺化：可用滑桿選擇顯示前 N 個高權重 token，長條圖反映頻率佔比，便於觀察重要詞彙。

## 檔案說明
- `app.py`: Streamlit 介面與偵測邏輯（Transformer + 啟發式、token 權重圖）。
- `requirements.txt`: 套件需求。
- `GPT_chat.md`: 對話紀錄。
- `.gitignore`: 版本控管忽略清單。

## 備註
- 權重定義：權重 = token 出現次數 / 總 token 數，供視覺化觀察高頻詞。
- 若要自訂模型，修改 `load_detector()` 中的模型名稱即可。
