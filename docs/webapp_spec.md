# GECO2 カウンティング Web アプリ — 仕様書

## 1. 概要

画像内のオブジェクトを自動/半自動でカウントする Web アプリケーション。  
バックエンドとして既存の GECO2 API サーバー (`localhost:7860`) を呼び出す。

### 1.1 ユーザーストーリー

| # | ストーリー |
|---|---|
| U1 | 画像をアップロードし、**完全自動**でオブジェクトをカウントしたい |
| U2 | 画像上の対象オブジェクトを**クリック（ポイント指定）**してカウントしたい |
| U3 | 画像上に**バウンディングボックスを描画**してカウントしたい |
| U4 | 検出結果（BBox 付き画像・カウント数）を画面上で確認したい |
| U5 | 結果画像をダウンロードしたい |

---

## 2. システム構成

```
┌──────────────────────────────────────────────────┐
│  Web ブラウザ (フロントエンド)                      │
│  - HTML / CSS / JavaScript                        │
│  - Canvas ベースの画像操作 UI                      │
└──────────┬───────────────────────────────────────┘
           │ HTTP (multipart/form-data)
           ▼
┌──────────────────────────────────────────────────┐
│  GECO2 API サーバー (既存)                         │
│  http://<HOST>:7860                               │
│  ┌────────────────────────────────────────────┐   │
│  │ /predict         — ポイント指定カウント     │   │
│  │ /predict/image   — 同上 (画像レスポンス)    │   │
│  │ /predict_auto    — 完全自動カウント         │   │
│  │ /predict_auto/image — 同上 (画像レスポンス) │   │
│  │ /predict_bbox    — BB 指定カウント          │   │
│  │ /predict_bbox/image — 同上 (画像レスポンス) │   │
│  │ /health          — ヘルスチェック           │   │
│  └────────────────────────────────────────────┘   │
│           │ (内部通信)                             │
│           ▼                                       │
│  AODC サービス  http://aodc:7861                   │
│  (自動モード時のみ使用)                            │
└──────────────────────────────────────────────────┘
```

フロントエンドはブラウザ完結の**静的 SPA** とし、API サーバーを直接呼び出す。  
サーバーサイド (Node / Python) を別途立てる場合は、API プロキシのみ担当。

---

## 3. 画面構成

### 3.1 メイン画面 (単一ページ)

```
┌─────────────────────────────────────────────────────────────────┐
│ [ヘッダー] GECO2 Object Counter                                 │
├─────────┬───────────────────────────────────────────────────────┤
│         │                                                       │
│ サイド   │        画像表示エリア (Canvas)                         │
│ バー     │                                                       │
│         │  ┌──────────────────────────────────────────────┐     │
│ [モード] │  │                                              │     │
│ ○ 自動   │  │     アップロード画像 / 結果画像               │     │
│ ○ ポイント│  │                                              │     │
│ ○ BB    │  │     ← ポイントクリック / BB ドラッグ           │     │
│         │  │                                              │     │
│ [設定]   │  └──────────────────────────────────────────────┘     │
│ threshold│                                                       │
│ [0.33]  │  [実行ボタン]                [リセットボタン]          │
│         │                                                       │
│ [結果]   ├───────────────────────────────────────────────────────┤
│ Count:42│  結果パネル (検出 BBox リスト / 統計)                  │
│         │                                                       │
│ [DL]    │                                                       │
└─────────┴───────────────────────────────────────────────────────┘
```

### 3.2 UI コンポーネント一覧

| コンポーネント | 説明 |
|---|---|
| **画像ドロップゾーン** | ドラッグ&ドロップ or ファイル選択で画像アップロード。JPEG/PNG 対応 |
| **Canvas オーバーレイ** | アップロード画像の上に透明 Canvas を重ね、ポイント/BB 操作を実現 |
| **モード選択** | ラジオボタン: `自動` / `ポイント指定` / `BB 指定` |
| **Threshold スライダー** | 0.05〜0.95、デフォルト 0.33、ステップ 0.01 |
| **実行ボタン** | 選択モードに応じた API を呼び出す |
| **リセットボタン** | ポイント/BB の入力をクリア |
| **結果表示** | カウント数（大きなフォント）+ 結果画像 |
| **ダウンロードボタン** | 結果画像を PNG でダウンロード |
| **ステータスインジケータ** | API 接続状態を表示（`/health` をポーリング） |

---

## 4. モード別動作仕様

### 4.1 自動モード (Auto)

1. ユーザーが画像をアップロード
2. 「実行」ボタンで `POST /predict_auto/image` を呼び出す
3. レスポンスの PNG 画像を Canvas に表示
4. JSON でカウント数も取得したい場合は並行して `POST /predict_auto` も呼ぶ

**API 呼び出し:**
```
POST /predict_auto/image
Content-Type: multipart/form-data

image: <ファイル>
threshold: 0.33
```

**レスポンス:** `image/png` (BBox 描画済み画像)

### 4.2 ポイント指定モード (Point)

1. ユーザーが画像をアップロード
2. Canvas 上でクリック → クリック位置にマーカー描画
3. 複数ポイント左クリック = フォアグラウンド (label=1)
4. 右クリック = バックグラウンド (label=0) ※オプション
5. 「実行」ボタンで `POST /predict/image` を呼び出す

**API 呼び出し:**
```
POST /predict/image
Content-Type: multipart/form-data

image: <ファイル>
points: [[x1,y1],[x2,y2],...]    ← 元画像のピクセル座標
labels: [1,1,0,...]               ← 省略時は全て 1
threshold: 0.33
```

**レスポンス:** `image/png`

**重要:** `points` と `labels` は JSON 文字列としてフォーム送信する。

### 4.3 BB 指定モード (Bounding Box)

1. ユーザーが画像をアップロード
2. Canvas 上でドラッグで矩形を描画（複数可）
3. 「実行」ボタンで `POST /predict_bbox/image` を呼び出す

**API 呼び出し:**
```
POST /predict_bbox/image
Content-Type: multipart/form-data

image: <ファイル>
bboxes: [[x1,y1,x2,y2],...]    ← 元画像のピクセル座標 (左上, 右下)
threshold: 0.33
```

**レスポンス:** `image/png`

---

## 5. API リファレンス

### 5.1 ベース URL

```
http://<HOST>:7860
```

### 5.2 エンドポイント一覧

| メソッド | パス | 入力 | 出力 | 用途 |
|---|---|---|---|---|
| `GET` | `/health` | なし | `{"status":"ok"}` | ヘルスチェック |
| `POST` | `/predict` | image, points, labels?, threshold? | JSON | ポイント → カウント |
| `POST` | `/predict/image` | 同上 | PNG | ポイント → 画像 |
| `POST` | `/predict_auto` | image, threshold? | JSON | 全自動カウント |
| `POST` | `/predict_auto/image` | 同上 | PNG | 全自動 → 画像 |
| `POST` | `/predict_bbox` | image, bboxes, threshold? | JSON | BB → カウント |
| `POST` | `/predict_bbox/image` | 同上 | PNG | BB → 画像 |

### 5.3 共通パラメータ

| パラメータ | 型 | 必須 | デフォルト | 説明 |
|---|---|---|---|---|
| `image` | File | ✅ | — | JPEG または PNG 画像 |
| `threshold` | float | — | 0.33 | 検出信頼度閾値 (0.05〜0.95) |

### 5.4 JSON レスポンス形式

```json
{
  "count": 25,
  "pred_boxes": [
    [x1, y1, x2, y2],
    ...
  ],
  "exemplar_boxes": [
    [x1, y1, x2, y2],
    ...
  ]
}
```

| フィールド | 型 | 説明 |
|---|---|---|
| `count` | int | 検出されたオブジェクト数 |
| `pred_boxes` | float[][] | 検出 BBox 一覧 `[x1, y1, x2, y2]`（元画像ピクセル座標） |
| `exemplar_boxes` | float[][] | クエリに使用された Exemplar BBox 一覧 |

### 5.5 画像レスポンス (`/*/image`)

- Content-Type: `image/png`
- 元画像に以下を重畳描画:
  - **緑色ボックス**: 検出された各オブジェクト (`pred_boxes`)
  - **赤色ボックス**: Exemplar オブジェクト (`exemplar_boxes`)
  - **左上ラベル**: `Count: N`

### 5.6 エラーレスポンス

```json
{
  "detail": "エラーメッセージ"
}
```

| ステータス | 原因 |
|---|---|
| 400 | パラメータ不正 (JSON パースエラー、座標フォーマット不正等) |
| 502 | AODC サービスエラー (自動モードのみ) |
| 503 | AODC サービス未起動 (自動モードのみ) |

---

## 6. フロントエンド実装仕様

### 6.1 技術スタック (推奨)

| レイヤー | 推奨技術 | 理由 |
|---|---|---|
| フレームワーク | **React** or **Vue 3** or **バニラ JS** | 単一ページなので軽量で十分 |
| Canvas 操作 | **HTML5 Canvas API** or **Fabric.js** | ポイント/BB 描画に必要 |
| HTTP クライアント | **fetch API** | FormData 送信に対応 |
| スタイル | **Tailwind CSS** or **シンプル CSS** | 短時間で整った UI |

### 6.2 座標変換

Canvas 上のクリック/ドラッグ座標と元画像のピクセル座標を正しく変換する必要がある。

```javascript
// Canvas に表示する際のスケール比率
const scaleX = originalImage.naturalWidth  / canvas.clientWidth;
const scaleY = originalImage.naturalHeight / canvas.clientHeight;

// Canvas クリック座標 → API に送る元画像座標
function canvasToImage(canvasX, canvasY) {
  return [canvasX * scaleX, canvasY * scaleY];
}
```

### 6.3 API 呼び出し例

```javascript
// 自動モード
async function predictAuto(imageFile, threshold = 0.33) {
  const form = new FormData();
  form.append("image", imageFile);
  form.append("threshold", threshold);

  // JSON レスポンス (カウント数取得用)
  const res = await fetch("/predict_auto", { method: "POST", body: form });
  return await res.json();
}

// 自動モード (画像レスポンス)
async function predictAutoImage(imageFile, threshold = 0.33) {
  const form = new FormData();
  form.append("image", imageFile);
  form.append("threshold", threshold);

  const res = await fetch("/predict_auto/image", { method: "POST", body: form });
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

// ポイント指定モード
async function predictPoint(imageFile, points, labels, threshold = 0.33) {
  const form = new FormData();
  form.append("image", imageFile);
  form.append("points", JSON.stringify(points));    // [[x1,y1],[x2,y2]]
  form.append("labels", JSON.stringify(labels));     // [1,1,0]
  form.append("threshold", threshold);

  const res = await fetch("/predict/image", { method: "POST", body: form });
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

// BB 指定モード
async function predictBBox(imageFile, bboxes, threshold = 0.33) {
  const form = new FormData();
  form.append("image", imageFile);
  form.append("bboxes", JSON.stringify(bboxes));    // [[x1,y1,x2,y2]]
  form.append("threshold", threshold);

  const res = await fetch("/predict_bbox/image", { method: "POST", body: form });
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}
```

### 6.4 Canvas インタラクション

#### ポイントモード

```
mousedown → canvasToImage(e.offsetX, e.offsetY) → points 配列に追加
         → Canvas 上にマーカー (●) を描画
         → 左クリック: label=1 (緑), 右クリック: label=0 (赤)
```

#### BB モード

```
mousedown → ドラッグ開始座標を記録
mousemove → 仮矩形をリアルタイム描画
mouseup   → 確定 → canvasToImage で変換 → bboxes 配列に追加
          → Canvas 上に矩形を描画
```

### 6.5 CORS 対応

フロントエンドとバックエンドが別オリジンの場合、GECO2 API 側の CORS 設定が必要。  
既存の `api_server.py` に以下を追加:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # フロントエンドのオリジン
    allow_methods=["*"],
    allow_headers=["*"],
)
```

もしくは Nginx リバースプロキシで同一オリジンにまとめる。

---

## 7. 状態管理

```
AppState {
  mode: "auto" | "point" | "bbox"     // 現在のモード
  imageFile: File | null               // アップロード画像
  imageSrc: string | null              // 画像プレビュー URL
  points: [x, y][]                     // ポイントモード: クリック座標
  labels: number[]                     // ポイントモード: ラベル配列
  bboxes: [x1, y1, x2, y2][]          // BB モード: BBox 配列
  threshold: number                    // 検出閾値 (0.33)
  resultImageSrc: string | null        // 結果画像 URL
  resultCount: number | null           // 検出カウント
  resultBoxes: [x1,y1,x2,y2][]        // 検出結果 BBox
  isLoading: boolean                   // API 呼び出し中
  apiStatus: "connected" | "error"     // API 接続状態
}
```

---

## 8. 処理フロー

### 8.1 自動モード

```
[画像アップロード] → [実行ボタン押下]
    → POST /predict_auto         (JSON — カウント数取得)
    → POST /predict_auto/image   (PNG — 結果画像取得)
    → [結果表示: カウント数 + 結果画像]
```

※ 2 つの API を**並行呼び出し**して、カウント数と結果画像を同時に取得する。  
もしくは `/predict_auto` の JSON レスポンスの `pred_boxes` をフロント側で Canvas に描画すれば  
`/image` エンドポイントは不要。

### 8.2 ポイント指定モード

```
[画像アップロード] → [Canvas 上でポイントクリック] → [実行ボタン押下]
    → POST /predict/image
    → [結果表示]
```

### 8.3 BB 指定モード

```
[画像アップロード] → [Canvas 上で BB ドラッグ描画] → [実行ボタン押下]
    → POST /predict_bbox/image
    → [結果表示]
```

---

## 9. 非機能要件

| 項目 | 要件 |
|---|---|
| **レスポンス時間** | 自動モード: ~3–5秒、ポイント/BB モード: ~1–2秒 (GPU 依存) |
| **画像サイズ上限** | フロント側で 4096×4096 以上は警告。API は任意サイズ受付 |
| **対応ブラウザ** | Chrome / Firefox / Edge (最新版) |
| **同時接続** | 単一ユーザー想定 (GPU を 1 枚共有するため) |
| **ヘルスチェック** | 起動時 + 30 秒ごとに `GET /health` をポーリング |

---

## 10. ディレクトリ構成 (推奨)

```
counting-webapp/
├── index.html
├── style.css
├── app.js                  # メインロジック
├── api.js                  # API クライアント (fetch ラッパー)
├── canvas.js               # Canvas 操作 (座標変換, 描画)
├── package.json            # (npm 使用時)
└── README.md
```

React / Vue を使う場合:

```
counting-webapp/
├── public/
│   └── index.html
├── src/
│   ├── App.vue (or App.tsx)
│   ├── components/
│   │   ├── ImageCanvas.vue       # Canvas + 画像表示
│   │   ├── ModeSelector.vue      # モード切替
│   │   ├── ThresholdSlider.vue   # 閾値スライダー
│   │   ├── ResultPanel.vue       # 結果表示
│   │   └── StatusIndicator.vue   # API 接続状態
│   ├── composables/ (or hooks/)
│   │   ├── useApi.js             # API 呼び出し
│   │   └── useCanvas.js          # Canvas 操作
│   └── main.js
├── package.json
└── vite.config.js
```

---

## 11. CORS / デプロイ構成

### 11.1 開発時

```
[localhost:3000 — フロントエンド dev server]
         ↓ proxy
[localhost:7860 — GECO2 API]
```

Vite の場合:
```javascript
// vite.config.js
export default {
  server: {
    proxy: {
      '/predict': 'http://localhost:7860',
      '/health':  'http://localhost:7860',
    }
  }
}
```

### 11.2 本番 (Nginx リバースプロキシ)

```nginx
server {
    listen 80;

    # フロントエンド (静的ファイル)
    location / {
        root /var/www/counting-webapp;
        try_files $uri $uri/ /index.html;
    }

    # API プロキシ
    location /predict {
        proxy_pass http://localhost:7860;
        proxy_read_timeout 60s;
        client_max_body_size 20M;
    }

    location /health {
        proxy_pass http://localhost:7860;
    }
}
```

---

## 12. curl テスト例

開発中のバックエンド疎通確認用:

```bash
# ヘルスチェック
curl http://localhost:7860/health

# 自動モード (JSON)
curl -X POST http://localhost:7860/predict_auto \
  -F "image=@photo.jpg" \
  -F "threshold=0.33"

# 自動モード (画像)
curl -X POST http://localhost:7860/predict_auto/image \
  -F "image=@photo.jpg" \
  -o result.png

# ポイント指定 (JSON)
curl -X POST http://localhost:7860/predict \
  -F "image=@photo.jpg" \
  -F 'points=[[150,200],[300,400]]' \
  -F 'labels=[1,1]'

# ポイント指定 (画像)
curl -X POST http://localhost:7860/predict/image \
  -F "image=@photo.jpg" \
  -F 'points=[[150,200]]' \
  -o result.png

# BB 指定 (JSON)
curl -X POST http://localhost:7860/predict_bbox \
  -F "image=@photo.jpg" \
  -F 'bboxes=[[50,60,150,160]]'

# BB 指定 (画像)
curl -X POST http://localhost:7860/predict_bbox/image \
  -F "image=@photo.jpg" \
  -F 'bboxes=[[50,60,150,160]]' \
  -o result.png
```