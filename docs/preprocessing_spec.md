# 前処理システム仕様書: AODC + SAM2 → GECO2

> `/predict_auto` エンドポイントにおける、入力画像からGECO2検出パスへ渡す **exemplar bounding boxes** を生成するまでの全処理。

---

## 1. システム全体図

```
入力画像 (HWC uint8 RGB)
    │
    ├──[AODC サービス]──────────────────────────────────────┐
    │   ① リサイズ (384×576)                                │
    │   ② ImageNet正規化                                    │
    │   ③ 密度マップ推定 (自己類似性ベース)                   │
    │   ④ マルチピーク検出 (maximum_filter)                   │
    │   ⑤ ピーク座標 → 元画像空間へ変換                       │
    │   ⑥ 密度マップ → オブジェクト面積推定                    │
    │   出力: points[], peak_indices[], density_map, areas[]  │
    │                                                         │
    └──[GECO2 サービス]──────────────────────────────────────┘
        │
        │ Pass 1: SAM2 exemplar抽出
        │   ⑦ zero-shot前処理 (正規化 + 1024パディング)
        │   ⑧ Backbone特徴抽出 (Hiera + FPN)
        │   ⑨ SAM2 MaskDecoder (ポイント → 3段階マスク)
        │   ⑩ マスク選択 (expected_areas or 固定index=2)
        │   ⑪ マスク → BB変換
        │   ⑫ IoUフィルタリング
        │   ⑬ 穴BB除去フィルタ (_filter_exemplars_by_area)
        │   ⑭ BB座標を元画像空間に変換
        │
        │ Pass 2: GECO2検出 (本仕様書の範囲外だが接続点を記載)
        │   ⑮ adaptive前処理 (exemplar BBで再スケーリング)
        │   ⑯ Backbone + forward_detect
        │
        └→ 出力: pred_boxes[], count
```

---

## 2. AODC サービス (`aodc_server.py` → `AODCWrapper`)

### 2.1 サービス構成

| 項目 | 値 |
|---|---|
| ポート | 7861 |
| エンドポイント | `POST /predict_multi` |
| モデル | AODC (zero_shot=False, few_shot=False, reference-less) |
| チェックポイント | `aodc.pth` (環境変数 `AODC_CHECKPOINT` で変更可能) |
| 入力サイズ | **(384, 576)** — H×W (固定、`AODCWrapper.input_size`) |

### 2.2 画像前処理 (`AODCWrapper.run_multi()`)

```python
# Step 1: HWC uint8 → CHW float32 [0, 1]
img = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

# Step 2: (384, 576) にバイリニアリサイズ — アスペクト比は無視
img = F.interpolate(img.unsqueeze(0), size=(384, 576), mode="bilinear", align_corners=False)

# Step 3: ImageNet正規化
#   mean = [0.485, 0.456, 0.406]
#   std  = [0.229, 0.224, 0.225]
img = Normalize(mean, std)(img)
```

**精度上の注意点:**
- アスペクト比無視のリサイズにより、正方形や特殊アスペクト比の画像では密度マップ上のオブジェクト形状が歪む
- 入力サイズは AODC の学習時データ生成(`gendata384x576.py`)に合わせたもの

### 2.3 密度マップ推定

```python
den_map = model(img, dummy_boxes, mode="test", categories=[])
# dummy_boxes = zeros(1, 5) — reference-lessモードでは無視される
# categories = [] — 空リスト
den_map = F.relu(den_map.squeeze())  # [H_den, W_den], 負値をクリップ
```

**出力サイズ:** 密度マップの解像度はモデルのアーキテクチャ依存。入力 (384, 576) に対し、ネットワーク内のダウンサンプリングにより小さくなる。

**動作原理:** 自己類似性（self-similarity）ベース。画像内の繰り返しパターンを自動検出するため、**リファレンス画像やプロンプトは不要**。

**既知の問題:**
- 繰り返しパターンがあれば何でも検出する（例：錠剤画像に写った納品書の文字列パターンにもピークが立つ）
- 穴のあるオブジェクトでは、穴自体と外側オブジェクトの両方が繰り返しパターンとして検出される

### 2.4 マルチピーク検出

```python
from scipy.ndimage import maximum_filter

local_max = maximum_filter(den_map, size=min_distance * 2 + 1)
is_peak = (den_map == local_max) & (den_map > den_map.max() * rel_threshold)
```

| パラメータ | デフォルト値 | 意味 |
|---|---|---|
| `num_peaks` | **5** | 返すピーク最大数 |
| `min_distance` | **15** | ピーク間の最小距離（密度マップpx）。抑制カーネルサイズ = `2*15+1 = 31` |
| `rel_threshold` | **0.3** | グローバル最大値の何%以上をピークとみなすか |

**アルゴリズム:**
1. `maximum_filter` で局所最大値マスクを作成
2. `den_map == local_max` かつ `den_map > max * 0.3` の条件でフィルタ
3. 密度値降順でソートし、上位 `num_peaks` 個を取得

### 2.5 座標変換 (密度マップ空間 → 元画像空間)

```python
x = col * orig_w / den_w   # 水平
y = row * orig_h / den_h   # 垂直
```

**精度上の注意点:** 密度マップは低解像度のため、ピーク位置は元画像上でオブジェクト中心から数ピクセルずれうる。

### 2.6 面積推定 (`utils/area_estimation.py`)

AODC の密度マップ + ピーク座標から、各ピークのオブジェクト推定面積を計算。

```python
estimate_object_areas(density_map, peak_indices, orig_h, orig_w, half_max_ratio=0.5)
```

**アルゴリズム:**
1. 各ピークの密度値 `peak_val` を取得
2. `threshold = peak_val * 0.5` (半値)
3. `density_map >= threshold` で二値化
4. `scipy.ndimage.label()` で連結成分ラベリング
5. ピーク位置が属する連結成分のピクセル数を取得
6. 密度マップ空間のピクセル数 × `pixel_scale` で元画像空間の面積(px²)に変換
   - `pixel_scale = (orig_h / den_h) * (orig_w / den_w)`

| パラメータ | 値 | 意味 |
|---|---|---|
| `half_max_ratio` | **0.5** | 密度ピーク値に対する二値化しきい値の比率 |

**精度上の問題点:**
- **半値幅は実オブジェクト面積より小さくなりやすい** — 密度マップのピーク形状はガウシアン状であり、半値でクリップするとオブジェクトの実際の輪郭より狭い領域になる
- 隣接オブジェクトの密度ピークが融合している場合、一つの連結成分に複数オブジェクトが含まれ過大推定される
- **この面積推定値がSAM2のマスク選択に直接影響する** (§3.5)

### 2.7 APIレスポンス

```json
{
  "points": [[x1, y1], [x2, y2], ...],       // 元画像px座標
  "peak_indices": [[row1, col1], ...],         // 密度マップpx座標
  "density_map_b64": "base64-encoded float32", // 密度マップ全体
  "den_shape": [H_den, W_den]                  // 密度マップ形状
}
```

`density_map_b64`: float32配列をバイト列化 → base64エンコード。GECO2側で面積推定に使用。

---

## 3. GECO2 Pass 1: SAM2 Exemplar抽出 (`api_server.py` → `_run_inference()`)

### 3.1 入力

| パラメータ | ソース |
|---|---|
| `image` (np.ndarray) | 元の入力画像 HWC uint8 RGB |
| `points` (List[[x,y]]) | AODCの出力ポイント (元画像px座標) |
| `labels` (List[int]) | 全て `1` (前景) |
| `expected_areas` (List[float] or None) | 面積推定値 (§2.6) or None |

### 3.2 zero-shot前処理 (`_preprocess_image()`)

```python
# Step 1: HWC uint8 → CHW float32 [0, 1]
tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

# Step 2: ImageNet正規化
tensor = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor)

# Step 3: resize_and_pad (zero_shot=True)
dummy_bbox = [[0, 0, 1, 1]]
padded, _, scale = resize_and_pad(tensor, dummy_bbox, size=1024.0, zero_shot=True)
```

**`resize_and_pad()` (zero_shot=True) の動作:**
```python
longer_dimension = max(H, W)
scaling_factor = 1024.0 / longer_dimension
# zero_shot=True → exemplar BBによる追加スケーリングなし
resized = F.interpolate(img, scale_factor=scaling_factor, mode='bilinear')
# 右・下にゼロパディングして1024×1024に
padded = F.pad(resized, (0, pad_width, 0, pad_height), value=0)
```

**特性:**
- アスペクト比を保持
- 長辺が1024pxになるようスケーリング
- 短辺側はゼロパディング

**出力:** `[1, 3, 1024, 1024]` テンソル, `scale` (float)

### 3.3 ポイント座標変換

```python
point_coords = torch.tensor(points) * scale_zs  // 元画像px → 1024パディング空間px
```

### 3.4 Backbone特徴抽出

```python
feats = pipeline.cnt.forward_backbone(img_tensor_zs)
```

**Backbone:** sam2_hiera_base_plus (Hierachical Attention-based encoder + Feature Pyramid Network)

**出力 dict:**
- `vision_features`: `[1, 256, 64, 64]` — 最終段の特徴マップ
- `backbone_fpn`: 3レベルのFPN特徴 `[256×256, 128×128, 64×64]`
- `vision_pos_enc`: 各FPNレベルの位置エンコーディング

### 3.5 SAM2 マスク生成 (`MaskProcessor.predict_masks_from_points()`)

各ポイントに対して SAM2 MaskDecoder が **3段階のマスク** を出力する。

#### 3.5.1 PromptEncoder

```python
# ポイント座標 [N, 2] → [N, 1, 2] (1ポイント/プロンプト)
coords = batch_coords.unsqueeze(1)
labels = batch_labels.unsqueeze(1)
sparse_embeddings, dense_embeddings = prompt_encoder_sam(
    points=(coords, labels), boxes=None, masks=None
)
```

#### 3.5.2 MaskDecoder

```python
low_res_masks, iou_predictions, _, _ = mask_decoder(
    image_embeddings=features[-1],      # 最終FPNレベル
    image_pe=prompt_encoder.get_dense_pe(),
    sparse_prompt_embeddings=sparse_embeddings,
    dense_prompt_embeddings=dense_embeddings,
    multimask_output=True,              # ← 常にTrue: 3つのマスクを出力
    repeat_image=True,
    high_res_features=features[:-1],    # 高解像度FPNレベル
)
```

**出力:**
- `low_res_masks`: `[B, 4, H_low, W_low]` — 低解像度マスク (channel 0 は単一マスク用、1-3 が multimaskの3段階)
- `iou_predictions`: `[B, 4]` — 各マスクのIoU予測値

`multimask_output=True` の場合、チャネル 1, 2, 3 が使われる (MaskDecoder内部で channel 0 は除外されるため、実効的に `[B, 3, H, W]`)。

#### 3.5.3 マスクのアップスケール

```python
masks_full = F.interpolate(
    low_res_masks, (1024, 1024),
    mode="bilinear", align_corners=False
)  # [B, 3, 1024, 1024]
```

#### 3.5.4 マスク選択ロジック ★精度に最も影響する部分★

**SAM2の3段階マスクの意味:**
| Index | 粒度 | 典型例 |
|---|---|---|
| 0 | サブパーツ | 指の一つ、文字の一部 |
| 1 | パーツ | 手全体、単語 |
| 2 | 全体オブジェクト | 人全体、文章ブロック |

**現在のロジック（改善済み — Index 2 デフォルト + マージ検知フォールバック）:**

```python
if expected_areas is not None:
    mask_areas = (masks_full > 0).flatten(2).sum(dim=2).float()  # [B, 3]
    # デフォルト: 常にIndex 2（全体オブジェクト）を選択
    best_idx = torch.full((B,), 2, device=...)
    # マージ検知: mask2の面積が expected_area の10倍以上 → 隣接マージと判断
    merge_ratio = 10.0
    is_merged = mask_areas[:, 2] > (expected_areas * merge_ratio)
    # マージ時のみ面積最近接で小さいマスクにフォールバック
    if is_merged.any():
        diff = (mask_areas - expected_areas.unsqueeze(1)).abs()
        best_idx[is_merged] = diff.argmin(dim=1)[is_merged]
else:
    best_idx = 2  # 固定（全体マスク）
```

| パラメータ | 値 | 意味 |
|---|---|---|
| `merge_ratio` | **10.0** | mask2面積がexpected_areaの何倍以上でマージ判定 |

**設計思想:**
- **大多数のケースでIndex 2が正解。** SAM2のIndex 2は「ユーザーが意図したオブジェクト全体」を学習しており、最も信頼性が高い。
- 面積推定の精度に依存しない。推定が±2倍ずれても10倍判定には影響しない。
- `expected_areas` の役割は「正確な面積指定」ではなく「マージ異常検知のアンカー」に限定。

#### 3.5.5 マスク → バウンディングボックス変換

```python
@staticmethod
def _masks_to_bboxes(masks: torch.Tensor) -> torch.Tensor:
    # masks: [B, H, W] (binary)
    # 各軸方向の first/last True ピクセルを検出
    has_col = masks.any(dim=1)  # [B, W]
    has_row = masks.any(dim=2)  # [B, H]
    # x1 = first True col, x2 = last True col
    # y1 = first True row, y2 = last True row
    return bboxes  # [B, 4] as (x1, y1, x2, y2), 1024空間px
```

**出力:** BB座標は1024×1024パディング空間のピクセル座標。

### 3.6 IoUフィルタリング

```python
keep_mask = exemplar_ious >= iou_threshold  # iou_threshold = 0.7

if keep_mask.sum() == 0:
    # フォールバック: 最大IoUの50%以上を保持
    keep_mask = exemplar_ious >= exemplar_ious.max() * 0.5

exemplar_bboxes_px = exemplar_bboxes[keep_mask]
```

| パラメータ | 値 | 意味 |
|---|---|---|
| `iou_threshold` | **0.7** | SAM2のIoU予測値のフィルタ閾値 |
| フォールバック閾値 | **max * 0.5** | 全て0.7未満の場合の救済 |

**精度上の注意点:**
- IoU予測はSAM2の自己評価であり、実際のIoUではない
- 低品質マスク（背景をマスクしたもの等）はここで除去される
- フォールバックにより、品質が低くても最低1つは保持される

### 3.7 BB座標の元画像空間への変換

```python
exemplar_bboxes_orig = (exemplar_bboxes_px / scale_zs).cpu().tolist()
# 1024パディング空間px → 元画像px
```

### 3.8 穴BBフィルタ (`_filter_exemplars_by_area()`)

穴のあるオブジェクトでAODCが穴とオブジェクト両方にピークを立てた場合、SAM2が大小非常に異なるBBを生成する。この問題に対処するフィルタ。

```python
def _filter_exemplars_by_area(bboxes, area_ratio=3.0):
```

**アルゴリズム:**
1. 全BBの面積を計算: `area = (x2-x1) * (y2-y1)`
2. 面積の昇順でソート
3. 連続するBB間の面積比 (大/小) を計算
4. 最大の面積比ギャップを探す
5. `最大比 > area_ratio (3.0)` の場合、ギャップの大きい側（＝大面積グループ）のみを保持
6. `最大比 <= area_ratio` の場合、全てのBBを保持 (面積が均一)

| パラメータ | 値 | 意味 |
|---|---|---|
| `area_ratio` | **3.0** | このギャップ比を超えたら分割する閾値 |

### 3.9 マージBBフィルタ (`_filter_merged_exemplars()`)

SAM2 のマスクが複数の隣接オブジェクトを1つにまとめてしまった場合の検出・除去。

```python
def _filter_merged_exemplars(bboxes, points):
```

**アルゴリズム:**
1. 各 exemplar BB 内に含まれる AODC ピーク座標の数をカウント
2. BB内にピークが2個以上 → そのBBは複数オブジェクトをマージしている
3. マージBBを除去し、単一オブジェクトBBのみ保持
4. 全BBがマージ判定された場合はフォールバックとして全て保持

**設計思想:** AODCのピークは各オブジェクトの中心付近に位置する。1つのBBに2つのピークが含まれていれば、そのBBは2つのオブジェクトにまたがっている。

### 3.10 形状外れ値フィルタ (`_filter_exemplars_by_shape()`)

ノイズBB（計数対象と形状が全く異なるもの）の除去。

```python
def _filter_exemplars_by_shape(bboxes, max_aspect_dev=2.0):
```

**アルゴリズム:**
1. 各BBのアスペクト比を計算: `ar = max(w,h) / min(w,h)` (常に ≥ 1)
2. アスペクト比の中央値を計算
3. 各BBのアスペクト比と中央値の比率が `max_aspect_dev` を超えるBBを除去
4. 全て外れ値の場合はフォールバックとして全て保持

| パラメータ | 値 | 意味 |
|---|---|---|
| `max_aspect_dev` | **2.0** | 中央値からの許容倍率 |

**典型的な効果:** 丸い錠剤(ar≈1.0)の中に細長い文字列BB(ar≈5.0)が混ざった場合、中央値≈1.0に対して5.0/1.0=5.0 > 2.0 で文字列BBが除去される。

### 3.11 フィルタの適用順序

```
exemplar_bboxes_orig (SAM2出力、元画像座標)
    │
    ├→ _filter_exemplars_by_area()   … 穴BB除去（面積ギャップ）
    ├→ _filter_merged_exemplars()    … マージBB除去（ピーク数チェック）
    ├→ _filter_exemplars_by_shape()  … 形状外れ値除去（アスペクト比）
    │
    └→ exemplar_bboxes_orig (クリーンなexemplar) → Pass 2
```

各フィルタは独立に動作し、前段で除去されたBBは後段に渡らない。全て除去された場合は各フィルタ内のフォールバックで最低限のBBを保持。

---

## 4. GECO2 Pass 2 への接続 (参考)

Pass 1 で得られた `exemplar_bboxes_orig`（元画像px座標）が Pass 2 に渡される。

### 4.1 adaptive前処理 (`_preprocess_image_with_bboxes()`)

```python
def resize_and_pad(tensor, bboxes, size=1024.0, zero_shot=False):
    # zero_shot=False → exemplar BBの平均サイズから追加スケーリング
    scaling_factor = 1024.0 / max(H, W)
    scaled_bboxes = bboxes * scaling_factor
    a_dim = mean(scaled_bboxes の幅 + 高さ) / 2
    scaling_factor = min(1.0, 80 / a_dim) * scaling_factor
    #        ↑ exemplarの平均寸法が ~80px になるようスケーリング
```

**目的:** GECO2の学習時の想定（exemplarオブジェクトが ~80px程度）に合わせてスケーリングすることで、検出精度を最大化。

### 4.2 GECO2検出パイプライン (概要)

```
img_tensor [1, 3, 1024, 1024]
bboxes_scaled [1, M, 4] (1024空間px)
    │
    ├→ Backbone (Hiera+FPN) → feats
    │
    ├→ roi_align(feats, bboxes) → exemplar embeddings (3レベル)
    │
    ├→ shape_or_objectness(bbox_hw) → shape embedding
    │
    ├→ adapt_features(image_feats, prototypes) → adapted feature map
    │
    ├→ class_embed → centerness map
    ├→ bbox_embed → offset map
    │
    ├→ boxes_with_scores() → candidate boxes (NMS前)
    │       centerness を max/8 で閾値フィルタ
    │       local max (3×3 maxpool) でNMS的抑制
    │
    ├→ SAM2 MaskDecoder (候補BB → mask → 補正BB)
    │       ※ forward内のSAM2は固定index=2で最大マスクを使用
    │
    └→ 出力: pred_boxes [0,1]正規化, box_v (スコア)
```

### 4.3 最終後処理 (`_run_inference()` 内)

```python
# スコア閾値: max / (1/threshold) = max * threshold
thr_inv = 1.0 / threshold      # threshold=0.33 → thr_inv=3.03
sel = box_v > (box_v.max() / thr_inv)

# NMS
keep = ops.nms(pred_boxes[sel], box_v[sel], 0.5)

# [0,1]正規化 → 元画像px
final_boxes = (pred_boxes / scale * image_size).cpu().tolist()
```

---

## 5. パラメータ一覧

### 5.1 AODC側

| パラメータ | 場所 | 値 | 影響 |
|---|---|---|---|
| `input_size` | `AODCWrapper.__init__` | **(384, 576)** | 密度マップの解像度・精度 |
| `num_peaks` | `/predict_multi` | **5** | exemplar候補の最大数 |
| `min_distance` | `run_multi()` | **15** (密度map px) | ピーク間の最小距離 |
| `rel_threshold` | `run_multi()` | **0.3** | 弱いピークの除去閾値 |
| `half_max_ratio` | `estimate_object_areas()` | **0.5** | 面積推定の二値化閾値 |

### 5.2 SAM2側

| パラメータ | 場所 | 値 | 影響 |
|---|---|---|---|
| `image_size` | `MaskProcessor` | **1024** | 全処理の基準解像度 |
| `reduction` | `MaskProcessor` | **16** | backbone出力との縮小率 |
| `num_multimask_outputs` | `MaskDecoder` | **3** | マスク段階数 |
| `mask_select_index` | `predict_masks_from_points` | **2** | expected_areasなし時のデフォルト |
| `batch_step` | `predict_masks_from_points` | **50** | メモリ効率のためのバッチサイズ |

### 5.3 フィルタリング

| パラメータ | 場所 | 値 | 影響 |
|---|---|---|---|
| `iou_threshold` | `PointToCountPipeline` | **0.7** | SAM2マスク品質の閾値 |
| フォールバック閾値 | `_run_inference` | **max * 0.5** | IoU全低時の救済閾値 |
| `area_ratio` | `_filter_exemplars_by_area` | **3.0** | 穴フィルタの面積比閾値 |
| `merge_ratio` | `predict_masks_from_points` | **10.0** | mask2マージ検知の面積比閾値 |
| `max_aspect_dev` | `_filter_exemplars_by_shape` | **2.0** | アスペクト比外れ値の許容倍率 |

### 5.4 検出後処理

| パラメータ | 場所 | 値 | 影響 |
|---|---|---|---|
| `threshold` | `/predict_auto` Form | **0.33** | 検出スコア閾値 |
| NMS閾値 | `_run_inference` | **0.5** | NMSのIoU閾値 |
| adaptive target | `resize_and_pad` | **80px** | exemplar平均サイズの目標 |

---

## 6. 座標空間まとめ

```
元画像空間 (orig_h × orig_w px)
    │
    │  ×scale_zs (= 1024 / max(H,W))
    ▼
zero-shot 1024空間 (1024 × 1024 px, 右下パディング)
    │  ← SAM2 マスク生成・BB抽出はここで実行
    │
    │  ÷scale_zs
    ▼
元画像空間に戻す (exemplar_bboxes_orig)
    │
    │  ×scale_adaptive (= min(1.0, 80/a_dim) * 1024/max(H,W))
    ▼
adaptive 1024空間 (1024 × 1024 px, 右下パディング)
    │  ← GECO2検出はここで実行
    │
    │  ÷scale_adaptive * image_size
    ▼
[0, 1] 正規化空間 (GECO2出力)
    │
    │  ÷scale * image_size
    ▼
元画像空間 (最終出力)
```

**AODC密度マップ空間 (den_h × den_w)** は独立しており、元画像空間への変換は `col * orig_w / den_w` で行う。 `expected_areas` は `scale_zs²` でzero-shot 1024空間にスケーリングされてからSAM2に渡される。

---

## 7. 既知の問題と制約

### 7.1 重大 (精度に直接影響)

| # | 問題 | 原因 | 影響 | 現状 |
|---|---|---|---|---|
| 1 | **サブパーツマスク選択** | `expected_areas` が半値幅ベースで過小推定 → `diff.argmin` が index 0/1 を選択 | 物体の一部分だけのBBが生成 | **修正済み** (Index 2デフォルト + merge_ratio=10.0 フォールバック) |
| 2 | **AODC非対象ピーク** | 自己類似性が画像内の全繰り返しパターンに反応 | 納品書文字列等にもピークが立つ | **対策済み** (アスペクト比外れ値フィルタ §3.10) |
| 3 | **隣接オブジェクトのマージ** | SAM2 index 2 が密接した複数オブジェクトを一つのマスクにする | 1 exemplar BBが複数オブジェクトをカバー | **対策済み** (マージBBフィルタ §3.9 + マスク選択改善 §3.5.4) |

### 7.2 中程度

| # | 問題 | 原因 | 影響 |
|---|---|---|---|
| 4 | AODC アスペクト比歪み | 固定(384,576)リサイズ | 正方形画像でオブジェクト形状が歪み、面積推定に誤差 |
| 5 | 密度マップ低解像度 | ネットワーク内ダウンサンプリング | ピーク位置が数px精度でしか得られない |
| 6 | IoUフォールバックの寛容さ | `max * 0.5` は低品質マスクも通す | 背景マスクが exemplar として採用される可能性 |

### 7.3 軽微

| # | 問題 | 原因 | 影響 |
|---|---|---|---|
| 7 | `num_peaks=5` 固定 | endpoint パラメータ化されているが API呼出時デフォルト使用 | object密度が高い画像で exemplar 不足の可能性 |
| 8 | 穴フィルタが2グループのみ | 最大1ギャップで分割 | 3段階以上のサイズ群では中間が混入 |

---

## 8. データフロー詳細 (疑似コード)

```python
# === AODC サービス (/predict_multi) ===
image_np = read_image()                         # [H, W, 3] uint8

# 前処理
img = to_tensor(image_np) / 255.0               # [3, H, W] float32
img = resize(img, (384, 576))                   # [3, 384, 576]
img = imagenet_normalize(img)                   # [3, 384, 576]

# 密度マップ推定
den_map = relu(aodc_model(img))                 # [den_h, den_w]

# ピーク検出
local_max = maximum_filter(den_map, size=31)
peaks = where((den_map == local_max) & (den_map > max*0.3))
peaks = sort_by_value(peaks, descending)[:5]

# 座標変換
points = [(col * W/den_w, row * H/den_h) for row, col in peaks]
peak_indices = [(row, col) for row, col in peaks]

# 面積推定
for each peak (row, col):
    threshold = den_map[row, col] * 0.5
    region = connected_component_at(den_map >= threshold, row, col)
    area_orig = region.pixel_count * (H/den_h) * (W/den_w)

# === GECO2 サービス (/predict_auto → _run_inference) ===

# expected_areas デコード
density_map = decode_base64(aodc_response.density_map_b64)
expected_areas = estimate_object_areas(density_map, peak_indices, H, W)

# Pass 1: zero-shot前処理
img_tensor = normalize(to_tensor(image_np) / 255.0)
img_padded, scale_zs = resize_and_pad(img_tensor, dummy_bb, zero_shot=True)
                                                # [1, 3, 1024, 1024]

# ポイント座標変換
point_coords = points * scale_zs                # 1024空間

# Backbone
feats = backbone(img_padded)                    # FPN 3レベル

# expected_areas スケーリング
ea_1024 = expected_areas * (scale_zs ** 2)      # 1024空間の面積

# SAM2 マスク生成 (各ポイント → 3マスク)
for each point batch (step=50):
    sparse, dense = prompt_encoder(point)
    low_res_masks, ious = mask_decoder(feats, sparse, dense)  # [B, 3, H_low, W_low]
    masks = interpolate(low_res_masks, (1024, 1024))          # [B, 3, 1024, 1024]

    # マスク選択
    if ea_1024 is not None:
        mask_areas = (masks > 0).sum(dim=spatial)             # [B, 3]
        selected = argmin(|mask_areas - ea_1024|)             # ★問題のあるロジック
    else:
        selected = 2  # 固定（全体マスク）

    bboxes = masks_to_bboxes(masks[selected])                 # [B, 4] 1024空間px

# IoUフィルタ
keep = ious >= 0.7  (or fallback: >= max*0.5)
bboxes_kept = bboxes[keep]

# 元画像座標に変換
exemplar_bboxes_orig = bboxes_kept / scale_zs

# 穴フィルタ
areas = [bb.area for bb in exemplar_bboxes_orig]
sorted_by_area = sort(areas)
max_gap = max(sorted[i+1] / sorted[i])
if max_gap > 3.0:
    exemplar_bboxes_orig = keep_larger_group()

# === Pass 2 (本仕様書の主範囲外) ===
img_tensor2, bboxes_scaled, scale_adapt = resize_and_pad_adaptive(
    image_np, exemplar_bboxes_orig              # zero_shot=False → ~80px target
)
feats2 = backbone(img_tensor2)
results = forward_detect(feats2, bboxes_scaled)
# → pred_boxes, box_v → threshold → NMS → 最終出力
```

---

## 9. モデル重み・チェックポイント

| モデル | チェックポイント | ロード方法 |
|---|---|---|
| AODC | `aodc.pth` (環境変数) | `torch.load`, `weights_only=False` |
| Hiera Backbone (SAM2) | `sam2_hiera_base_plus.pt` (torch.hub自動DL) | `torch.hub.load_state_dict_from_url` |
| SAM2 MaskDecoder | 同上 (mask_decoder prefix) | `load_state_dict` |
| SAM2 PromptEncoder | 同上 (prompt_encoder prefix) | `load_state_dict` |
| GECO2 検出ヘッド | `CNTQG_multitrain_ca44.pth` | `torch.load`, `weights_only=True` |

GECO2チェックポイントに含まれるprefixキー: `adapt_features`, `backbone`, `bbox_embed`, `class_embed`, `sam_prompt_encoder`, `shape_or_objectness`

---

## 10. ファイル構成

| ファイル | 役割 |
|---|---|
| `aodc_server.py` | AODCサービス (独立コンテナ) |
| `models/aodc_wrapper.py` | AODC推論ラッパー (`run()`, `run_multi()`) |
| `api_server.py` | GECO2サービス (Pass 1 + Pass 2 + 全エンドポイント) |
| `models/sam_mask.py` | SAM2 MaskProcessor (`predict_masks_from_points()`, `forward()`) |
| `models/counter_infer.py` | GECO2 CNTモデル (`forward_backbone()`, `forward_detect()`) |
| `models/point_to_count.py` | Pipeline wrapper (api_serverからは直接使われず、`cnt`属性のみ利用) |
| `utils/data.py` | `resize_and_pad()` — zero-shot/adaptive両方のリサイズ+パディング |
| `utils/area_estimation.py` | `estimate_object_areas()` — 密度マップからの面積推定 |
| `utils/box_ops.py` | `boxes_with_scores()` — centreness + offset → 候補BB |
| `utils/arg_parser.py` | デフォルトハイパーパラメータ定義 |
