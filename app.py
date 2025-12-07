import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper

# ページ設定
st.set_page_config(page_title="Braille Reader", page_icon="🔍", layout="centered")

# ==========================================
# 関数: 画像の傾き補正
# ==========================================
def correct_skew(image, contours):
    if not contours: return image, 0
    all_points = np.concatenate(contours)
    rect = cv2.minAreaRect(all_points)
    angle = rect[-1]
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
    if abs(angle) > 10.0 or abs(angle) < 0.2: return image, 0

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), angle

# ==========================================
# 関数: 点字解析メインロジック
# ==========================================
def process_braille_image(image_array):
    # 1. 前処理
    if len(image_array.shape) == 3: gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else: gray_image = image_array
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 2. ドット検出 & 傾き補正
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    dot_contours = [cnt for cnt in contours if 10 < cv2.contourArea(cnt) < 5000]
    corrected_img, angle = correct_skew(gray_image, dot_contours)
    
    blurred_corr = cv2.GaussianBlur(corrected_img, (5, 5), 0)
    thresh_corr = cv2.adaptiveThreshold(blurred_corr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours_final, _ = cv2.findContours(thresh_corr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 3. ドット抽出
    raw_dots = []
    radii_list = []
    dot_id_counter = 0
    for contour in contours_final:
        area = cv2.contourArea(contour)
        if 3 < area < 5000:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            x_r, y_r, w_r, h_r = cv2.boundingRect(contour)
            if float(w_r)/h_r > 1.8 or float(w_r)/h_r < 0.5: continue # 形状フィルタ
            
            mask = np.zeros(thresh_corr.shape, dtype=np.uint8)
            cv2.circle(mask, center, int(radius), 255, -1)
            if cv2.mean(thresh_corr, mask=mask)[0] < 130: continue # 中身チェック

            raw_dots.append({'id': dot_id_counter, 'center': center, 'radius': radius})
            radii_list.append(radius)
            dot_id_counter += 1

    if not raw_dots: return corrected_img, "ドットなし", []
    
    median_radius = np.median(radii_list)
    braille_dots = [d for d in raw_dots if median_radius * 0.5 <= d['radius'] <= median_radius * 2.0]
    if not braille_dots: return corrected_img, "有効ドットなし", []
    avg_radius = np.mean([d['radius'] for d in braille_dots])

    # 4. 行分離
    braille_dots.sort(key=lambda d: d['center'][1])
    lines = []
    if braille_dots:
        curr_line = [braille_dots[0]]
        curr_y_sum = braille_dots[0]['center'][1]
        for i in range(1, len(braille_dots)):
            dot = braille_dots[i]
            if abs(dot['center'][1] - (curr_y_sum / len(curr_line))) < avg_radius * 3.0:
                curr_line.append(dot)
                curr_y_sum += dot['center'][1]
            else:
                lines.append(curr_line)
                curr_line = [dot]
                curr_y_sum = dot['center'][1]
        lines.append(curr_line)

    braille_cells = []
    used_dot_ids = set()

    for line_dots in lines:
        if not line_dots: continue
        line_cy = np.median([d['center'][1] for d in line_dots])
        line_dots.sort(key=lambda d: d['center'][0])
        dots_x = np.array([d['center'][0] for d in line_dots])

        # A. グループ化
        x_diffs = np.diff(dots_x)
        gap_thresh = avg_radius * 4.5
        groups = []
        curr_grp = [line_dots[0]]
        for i, diff in enumerate(x_diffs):
            if diff < gap_thresh: curr_grp.append(line_dots[i+1])
            else: groups.append(curr_grp); curr_grp = [line_dots[i+1]]
        groups.append(curr_grp)

        # --- 【重要】サイズとピッチの実測 ---
        # 1. 文字幅（Intra-Pitch）の実測: 2列あるグループの幅を集める
        widths = []
        for g in groups:
            gx = [d['center'][0] for d in g]
            w = max(gx) - min(gx)
            if w > avg_radius: widths.append(w)
        
        # 実測値の中央値を「文字内の列間隔」とする
        if widths:
            REAL_INTRA_PITCH = np.median(widths)
        else:
            REAL_INTRA_PITCH = avg_radius * 2.5 # デフォルト

        # 2. 文字間隔（Char-Pitch）の実測
        starts = [min([d['center'][0] for d in g]) for g in groups]
        if len(starts) > 1:
            diffs = np.diff(starts)
            valid = diffs[diffs > avg_radius * 4.0]
            if len(valid) > 0:
                REAL_CHAR_PITCH = np.percentile(valid, 25) # 小さい方の山
            else:
                REAL_CHAR_PITCH = REAL_INTRA_PITCH * 2.5
        else:
            REAL_CHAR_PITCH = REAL_INTRA_PITCH * 2.5

        # 3. 縦ピッチの実測
        y_dists = [abs(d['center'][1] - line_cy) for d in line_dots]
        valid_y = [dy for dy in y_dists if dy > avg_radius * 0.5]
        v_pitch = np.median(valid_y) if valid_y else avg_radius * 2.5

        # --- 固定セルサイズの定義 ---
        # 実測した文字幅 + マージン(半径分) をセルの幅とする
        FIXED_W = int(REAL_INTRA_PITCH + (avg_radius * 2.5))
        FIXED_H = int(v_pitch * 2 + avg_radius * 3)
        
        prev_right_edge = -1 # 衝突防止用

        for grp in groups:
            min_x = min([d['center'][0] for d in grp])
            max_x = max([d['center'][0] for d in grp])
            grp_cx = (min_x + max_x) / 2
            
            # --- 空白判定 ---
            if prev_right_edge != -1:
                # 前のセルの右端から、今の文字の左端までの距離
                # 実際の文字の左端は min_x だが、セルの左端はもっと左にあるはず
                current_cell_left_ideal = grp_cx - (FIXED_W / 2)
                
                gap = current_cell_left_ideal - prev_right_edge
                
                # ギャップが「文字ピッチの0.6倍」以上あればスペース
                if gap > REAL_CHAR_PITCH * 0.6:
                    steps = int(round(gap / REAL_CHAR_PITCH))
                    steps = min(steps, 3)
                    for k in range(steps):
                        sp_x = prev_right_edge + 2 + (k * REAL_CHAR_PITCH)
                        braille_cells.append({
                            'rect': (int(sp_x), int(line_cy - FIXED_H/2), FIXED_W, FIXED_H),
                            'pattern': [False]*6, 'has_dot': False, 'is_space': True
                        })
                        # スペース描画で右端更新
                        prev_right_edge = sp_x + FIXED_W

            # --- 1列/2列の判定と位置合わせ ---
            col1_x, col2_x = 0, 0
            cell_center_x = 0
            
            # グループ幅が「実測列間隔」の8割以上あれば2列とみなす
            if (max_x - min_x) > REAL_INTRA_PITCH * 0.8:
                # 2列文字 (中心合わせ)
                cell_center_x = grp_cx
                col1_x = grp_cx - (REAL_INTRA_PITCH / 2)
                col2_x = grp_cx + (REAL_INTRA_PITCH / 2)
            else:
                # 1列文字 (左列か右列か判定)
                # 前の文字からの距離で判定するのが確実
                # ...だが簡易的に、前の右端から「標準ピッチ」の距離にある場所を左列とする
                if prev_right_edge == -1:
                    # 行頭なら左列とみなす
                    col1_x = min_x
                    col2_x = min_x + REAL_INTRA_PITCH
                    cell_center_x = min_x + (REAL_INTRA_PITCH / 2)
                else:
                    ideal_left_col = prev_right_edge + (REAL_CHAR_PITCH - FIXED_W) # 概算
                    # ドットが理想の左列より明らかに右にあれば右列
                    if min_x - ideal_left_col > REAL_INTRA_PITCH * 0.6:
                        col1_x = min_x - REAL_INTRA_PITCH
                        col2_x = min_x
                        cell_center_x = min_x - (REAL_INTRA_PITCH / 2)
                    else:
                        col1_x = min_x
                        col2_x = min_x + REAL_INTRA_PITCH
                        cell_center_x = min_x + (REAL_INTRA_PITCH / 2)

            # アンカー計算 & 衝突防止
            anchor_x = int(cell_center_x - (FIXED_W / 2))
            if anchor_x < prev_right_edge:
                anchor_x = prev_right_edge + 2 # 重なるなら強制移動
                # ターゲットもずらす
                shift = anchor_x - int(cell_center_x - (FIXED_W / 2))
                col1_x += shift
                col2_x += shift
            
            anchor_y = int(line_cy - (FIXED_H / 2))

            targets = [
                (col1_x, line_cy - v_pitch), (col1_x, line_cy), (col1_x, line_cy + v_pitch),
                (col2_x, line_cy - v_pitch), (col2_x, line_cy), (col2_x, line_cy + v_pitch)
            ]
            
            pattern = [False] * 6
            for dot in grp:
                dx, dy = dot['center']
                best_idx = -1
                min_dist = float('inf')
                for idx, (tx, ty) in enumerate(targets):
                    d = np.sqrt((dx-tx)**2 + ((dy-ty)*0.8)**2)
                    if d < min_dist: min_dist = d; best_idx = idx
                if min_dist < avg_radius * 3.5:
                    pattern[best_idx] = True
                    used_dot_ids.add(dot['id'])

            braille_cells.append({
                'rect': (anchor_x, anchor_y, FIXED_W, FIXED_H),
                'pattern': pattern,
                'targets': targets, # 可視化用に追加
                'has_dot': True,
                'is_space': False
            })
            prev_right_edge = anchor_x + FIXED_W
        
        braille_cells.append({'is_newline': True})

    # 5. 翻訳 & 出力
    def get_dots_tuple(bool_pattern): return tuple(i + 1 for i, b in enumerate(bool_pattern) if b)
    
    jp_map = {(1,): "あ", (1, 2): "い", (1, 4): "う", (1, 2, 4): "え", (2, 4): "お", (1, 6): "か", (1, 2, 6): "き", (1, 4, 6): "く", (1, 2, 4, 6): "け", (2, 4, 6): "こ", (1, 5, 6): "さ", (1, 2, 5, 6): "し", (1, 4, 5, 6): "す", (1, 2, 4, 5, 6): "せ", (2, 4, 5, 6): "そ", (1, 3, 5): "た", (1, 2, 3, 5): "ち", (1, 3, 4, 5): "つ", (1, 2, 3, 4, 5): "て", (2, 3, 4, 5): "と", (1, 3): "な", (1, 2, 3): "に", (1, 3, 4): "ぬ", (1, 2, 3, 4): "ね", (2, 3, 4): "の", (1, 3, 6): "は", (1, 2, 3, 6): "ひ", (1, 3, 4, 6): "ふ", (1, 2, 3, 4, 6): "へ", (2, 3, 4, 6): "ほ", (1, 3, 5, 6): "ま", (1, 2, 3, 5, 6): "み", (1, 3, 4, 5, 6): "む", (1, 2, 3, 4, 5, 6): "め", (2, 3, 4, 5, 6): "も", (3, 4): "や", (3, 4, 6): "ゆ", (3, 4, 5): "よ", (1, 5): "ら", (1, 2, 5): "り", (1, 4, 5): "る", (1, 2, 4, 5): "れ", (2, 4, 5): "ろ", (3,): "わ", (3, 5): "を", (3, 5, 6): "ん", (2,): "っ", (2, 5): "ー", (2, 5, 6): "。", (5, 6): "、", (2, 6): "？", (2, 3, 5): "！"}
    num_map = {(1,): "1", (1, 2): "2", (1, 4): "3", (1, 2, 4): "4", (1, 5): "5", (1, 6): "6", (1, 2, 5): "7", (1, 2, 6): "8", (2, 4): "9", (2, 4, 5): "0"}
    yoon_map = {(1,): "a", (1, 6): "きゃ", (1, 4, 6): "きゅ", (2, 4, 6): "きょ", (1, 5, 6): "しゃ", (1, 4, 5, 6): "しゅ", (2, 4, 5, 6): "しょ", (1, 3, 5): "ちゃ", (1, 3, 4, 5): "ちゅ", (2, 3, 4, 5): "ちょ", (1, 3): "にゃ", (1, 3, 4): "にゅ", (2, 3, 4): "にょ", (1, 3, 6): "ひゃ", (1, 3, 4, 6): "ひゅ", (2, 3, 4, 6): "ひょ", (1, 3, 5, 6): "みゃ", (1, 3, 4, 5, 6): "みゅ", (2, 3, 4, 5, 6): "みょ", (1, 5): "りゃ", (1, 4, 5): "りゅ", (2, 4, 5): "りょ"}
    dakuten_map = {"か":"が","き":"ぎ","く":"ぐ","け":"げ","こ":"ご","さ":"ざ","し":"じ","す":"ず","せ":"ぜ","そ":"ぞ","た":"だ","ち":"ぢ","つ":"づ","て":"で","と":"ど","は":"ば","ひ":"び","ふ":"ぶ","へ":"べ","ほ":"ぼ","う":"ゔ"}
    handaku_map = {"は":"ぱ","ひ":"ぴ","ふ":"ぷ","へ":"ぺ","ほ":"ぽ"}

    text = ""
    mode_num, mode_dak, mode_han, mode_yoon = False, False, False, False
    details = []
    
    # 画像生成 (RGB)
    res_img = cv2.cvtColor(corrected_img, cv2.COLOR_GRAY2RGB)

    for cell in braille_cells:
        if cell.get('is_newline'): text += "\n"; continue
        
        rx, ry, rw, rh = cell['rect']
        if cell.get('is_space'):
            text += "　"; mode_num = False
            cv2.rectangle(res_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 1)
            continue
        
        # 描画
        cv2.rectangle(res_img, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
        if 'targets' in cell:
            for tx, ty in cell['targets']:
                cv2.circle(res_img, (int(tx), int(ty)), 2, (0, 0, 255), 1)

        dots = get_dots_tuple(cell['pattern'])
        label = "".join(map(str, dots))
        cv2.putText(res_img, label, (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 2)

        # 翻訳
        char = "?"
        spec = False
        if dots == (3, 4, 5, 6): mode_num=True; spec=True; char="[数]"
        elif dots == (5,): mode_dak=True; spec=True; char="[濁]"
        elif dots == (6,): mode_han=True; spec=True; char="[半]"
        elif dots == (4,): mode_yoon=True; spec=True; char="[拗]"
        elif dots == (4, 5): mode_yoon=True; mode_dak=True; spec=True; char="[拗濁]"
        elif dots == (4, 6): mode_yoon=True; mode_han=True; spec=True; char="[拗半]"

        if not spec:
            if mode_num: char = num_map.get(dots, "?")
            elif mode_yoon: char = yoon_map.get(dots, "?"); mode_yoon=False
            else: char = jp_map.get(dots, "?")
            
            if mode_dak: char = dakuten_map.get(char, char+"゛"); mode_dak=False
            elif mode_han: char = handaku_map.get(char, char+"゜"); mode_han=False
            text += char
        
        p = cell['pattern']
        vis = f"{'●' if p[0] else '○'} {'●' if p[3] else '○'}\n{'●' if p[1] else '○'} {'●' if p[4] else '○'}\n{'●' if p[2] else '○'} {'●' if p[5] else '○'}"
        details.append({'char': char, 'dots': dots, 'visual': vis})

    return res_img, text, details

# ==========================================
# Streamlit UI
# ==========================================
st.title("点字翻訳アプリ (Braille Reader)")
st.write("画像の点字部分をトリミングして翻訳します。")

uploaded_file = st.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    try: image = ImageOps.exif_transpose(image)
    except: pass

    st.subheader("1. 範囲指定")
    cropped_img = st_cropper(image, realtime_update=True, box_color='#0000FF', aspect_ratio=None)
    
    st.subheader("2. 翻訳")
    if st.button("この範囲を翻訳する"):
        if cropped_img is not None:
            img_array = np.array(cropped_img)
            if len(img_array.shape) == 3: img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else: img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

            with st.spinner("解析中..."):
                result_img, text, details = process_braille_image(img_cv)
                st.success("完了！")
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.image(result_img, caption="解析結果", use_column_width=True)
                with col2:
                    st.text_area("翻訳テキスト", text, height=200)
                
                with st.expander("詳細レポートを見る"):
                    for i, det in enumerate(details):
                        st.text(f"[{i+1:02d}] {det['char']} {det['dots']}")
                        st.text(det['visual'])
                        st.divider()
