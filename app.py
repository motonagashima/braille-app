import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper

# ページ設定
st.set_page_config(
    page_title="Braille Reader",
    page_icon="🔍",
    layout="centered"
)

# ==========================================
# 関数: 画像の傾き補正
# ==========================================
def correct_skew(image, contours):
    if not contours:
        return image, 0
    
    all_points = np.concatenate(contours)
    rect = cv2.minAreaRect(all_points)
    angle = rect[-1]
    
    if angle < -45: angle = -(90 + angle)
    else: angle = -angle
        
    if abs(angle) > 10.0: return image, 0
    if abs(angle) < 0.2: return image, 0

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
    if len(image_array.shape) == 3:
        gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image_array
        
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 2. ドット検出 & 傾き補正
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    dot_contours = [cnt for cnt in contours if 10 < cv2.contourArea(cnt) < 5000]
    
    corrected_img, angle = correct_skew(gray_image, dot_contours)
    
    # 補正後再処理
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
            
            # 形状フィルタ
            x_r, y_r, w_r, h_r = cv2.boundingRect(contour)
            aspect = float(w_r) / h_r
            if aspect > 1.8 or aspect < 0.5: continue

            # 充填率フィルタ
            mask = np.zeros(thresh_corr.shape, dtype=np.uint8)
            cv2.circle(mask, center, int(radius), 255, -1)
            mean_val = cv2.mean(thresh_corr, mask=mask)[0]
            if mean_val < 130: continue 

            raw_dots.append({'id': dot_id_counter, 'center': center, 'radius': radius})
            radii_list.append(radius)
            dot_id_counter += 1

    if not raw_dots: return corrected_img, "ドットが見つかりませんでした。", []

    median_radius = np.median(radii_list)
    braille_dots = []
    for dot in raw_dots:
        if median_radius * 0.5 <= dot['radius'] <= median_radius * 2.0:
            braille_dots.append(dot)
    
    if not braille_dots: return corrected_img, "有効なドットなし", []
    
    avg_radius = np.mean([d['radius'] for d in braille_dots])

    # 4. 行の分離
    braille_dots.sort(key=lambda d: d['center'][1])
    lines_of_dots = []
    if braille_dots:
        current_line = [braille_dots[0]]
        current_line_y_sum = braille_dots[0]['center'][1]
        for i in range(1, len(braille_dots)):
            dot = braille_dots[i]
            dy = dot['center'][1]
            avg_y = current_line_y_sum / len(current_line)
            if abs(dy - avg_y) < avg_radius * 3.0: 
                current_line.append(dot)
                current_line_y_sum += dy
            else:
                lines_of_dots.append(current_line)
                current_line = [dot]
                current_line_y_sum = dy
        lines_of_dots.append(current_line)

    braille_cells = []
    used_dot_ids = set()

    # --- 行ごとの処理 ---
    for line_dots in lines_of_dots:
        if not line_dots: continue
        
        line_center_y = np.median([d['center'][1] for d in line_dots])
        line_dots.sort(key=lambda d: d['center'][0])
        dots_x = np.array([d['center'][0] for d in line_dots])

        x_diffs = np.diff(dots_x)
        gap_threshold = avg_radius * 4.5
        groups = []
        current_group = [line_dots[0]]
        for i, diff in enumerate(x_diffs):
            if diff < gap_threshold:
                current_group.append(line_dots[i+1])
            else:
                groups.append(current_group)
                current_group = [line_dots[i+1]]
        groups.append(current_group)

        # グリッド計算
        group_starts = np.array([min([d['center'][0] for d in g]) for g in groups])
        
        estimated_pitch = avg_radius * 6.0
        if len(group_starts) > 1:
            diffs = np.diff(group_starts)
            valid_diffs = diffs[diffs > avg_radius * 4.0]
            if len(valid_diffs) > 0:
                estimated_pitch = np.percentile(valid_diffs, 25)

        base_x = group_starts[0]
        indices = np.round((group_starts - base_x) / estimated_pitch)
        
        # 線形回帰 (エラー回避付き)
        PERFECT_PITCH = estimated_pitch
        PERFECT_START = base_x
        
        if len(indices) >= 2:
            try:
                slope, intercept = np.polyfit(indices, group_starts, 1)
                PERFECT_PITCH = slope
                PERFECT_START = intercept
            except:
                pass

        y_dists = [abs(d['center'][1] - line_center_y) for d in line_dots]
        valid_y = [dy for dy in y_dists if dy > avg_radius * 0.5]
        v_pitch = np.median(valid_y) if valid_y else avg_radius * 2.5

        # セル生成
        start_idx = int(min(indices)) if len(indices)>0 else 0
        end_idx = int(max(indices)) if len(indices)>0 else 0
        
        FIXED_W = int(PERFECT_PITCH * 0.75)
        FIXED_H = int(v_pitch * 2 + avg_radius * 3)
        INTRA_PITCH = avg_radius * 2.5

        for idx in range(start_idx, end_idx + 1):
            cell_left = PERFECT_START + (idx * PERFECT_PITCH)
            cell_center_x = cell_left + (PERFECT_PITCH * 0.4) 
            
            t_col1 = cell_center_x - (INTRA_PITCH / 2)
            t_col2 = cell_center_x + (INTRA_PITCH / 2)
            
            t_y1 = line_center_y - v_pitch
            t_y2 = line_center_y
            t_y3 = line_center_y + v_pitch
            
            targets = [
                (t_col1, t_y1), (t_col1, t_y2), (t_col1, t_y3),
                (t_col2, t_y1), (t_col2, t_y2), (t_col2, t_y3)
            ]
            
            pattern = [False] * 6
            has_dot = False
            
            search_x_min = cell_left - (PERFECT_PITCH * 0.1)
            search_x_max = cell_left + (PERFECT_PITCH * 0.9)
            local_dots = [d for d in line_dots if search_x_min < d['center'][0] < search_x_max]
            
            for dot in local_dots:
                if dot['id'] in used_dot_ids: continue
                dx, dy = dot['center']
                min_dist = float('inf')
                best_idx = -1
                for ti, (tx, ty) in enumerate(targets):
                    dist = np.sqrt((dx - tx)**2 + ((dy - ty)*0.8)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = ti
                
                if min_dist < avg_radius * 3.5:
                    pattern[best_idx] = True
                    has_dot = True
                    used_dot_ids.add(dot['id'])

            anchor_x = int(cell_center_x - (FIXED_W / 2))
            anchor_y = int(line_center_y - (FIXED_H / 2))
            
            braille_cells.append({
                'rect': (anchor_x, anchor_y, FIXED_W, FIXED_H),
                'pattern': pattern,
                'has_dot': has_dot,
                'is_space': not has_dot
            })
            
        braille_cells.append({'is_newline': True})

    # 5. 翻訳処理
    def get_dots_tuple(bool_pattern):
        return tuple(i + 1 for i, b in enumerate(bool_pattern) if b)

    jp_map = {
        (1,): "あ", (1, 2): "い", (1, 4): "う", (1, 2, 4): "え", (2, 4): "お",
        (1, 6): "か", (1, 2, 6): "き", (1, 4, 6): "く", (1, 2, 4, 6): "け", (2, 4, 6): "こ",
        (1, 5, 6): "さ", (1, 2, 5, 6): "し", (1, 4, 5, 6): "す", (1, 2, 4, 5, 6): "せ", (2, 4, 5, 6): "そ",
        (1, 3, 5): "た", (1, 2, 3, 5): "ち", (1, 3, 4, 5): "つ", (1, 2, 3, 4, 5): "て", (2, 3, 4, 5): "と",
        (1, 3): "な", (1, 2, 3): "に", (1, 3, 4): "ぬ", (1, 2, 3, 4): "ね", (2, 3, 4): "の",
        (1, 3, 6): "は", (1, 2, 3, 6): "ひ", (1, 3, 4, 6): "ふ", (1, 2, 3, 4, 6): "へ", (2, 3, 4, 6): "ほ",
        (1, 3, 5, 6): "ま", (1, 2, 3, 5, 6): "み", (1, 3, 4, 5, 6): "む", (1, 2, 3, 4, 5, 6): "め", (2, 3, 4, 5, 6): "も",
        (3, 4): "や", (3, 4, 6): "ゆ", (3, 4, 5): "よ",
        (1, 5): "ら", (1, 2, 5): "り", (1, 4, 5): "る", (1, 2, 4, 5): "れ", (2, 4, 5): "ろ",
        (3,): "わ", (3, 5): "を", (3, 5, 6): "ん",
        (2,): "っ", (2, 5): "ー", (2, 5, 6): "。", (5, 6): "、", (2, 6): "？", (2, 3, 5): "！"
    }
    num_map = {(1,): "1", (1, 2): "2", (1, 4): "3", (1, 2, 4): "4", (1, 5): "5", (1, 6): "6", (1, 2, 5): "7", (1, 2, 6): "8", (2, 4): "9", (2, 4, 5): "0"}
    yoon_map = {
        (1,): "a", (1, 6): "きゃ", (1, 4, 6): "きゅ", (2, 4, 6): "きょ",
        (1, 5, 6): "しゃ", (1, 4, 5, 6): "しゅ", (2, 4, 5, 6): "しょ",
        (1, 3, 5): "ちゃ", (1, 3, 4, 5): "ちゅ", (2, 3, 4, 5): "ちょ",
        (1, 3): "にゃ", (1, 3, 4): "にゅ", (2, 3, 4): "にょ",
        (1, 3, 6): "ひゃ", (1, 3, 4, 6): "ひゅ", (2, 3, 4, 6): "ひょ",
        (1, 3, 5, 6): "みゃ", (1, 3, 4, 5, 6): "みゅ", (2, 3, 4, 5, 6): "みょ",
        (1, 5): "りゃ", (1, 4, 5): "りゅ", (2, 4, 5): "りょ",
    }
    dakuten_map = {"か":"が","き":"ぎ","く":"ぐ","け":"げ","こ":"ご","さ":"ざ","し":"じ","す":"ず","せ":"ぜ","そ":"ぞ","た":"だ","ち":"ぢ","つ":"づ","て":"で","と":"ど","は":"ば","ひ":"び","ふ":"ぶ","へ":"べ","ほ":"ぼ","う":"ゔ"}
    handaku_map = {"は":"ぱ","ひ":"ぴ","ふ":"ぷ","へ":"ぺ","ほ":"ぽ"}

    final_text = ""
    mode_num, mode_dak, mode_han, mode_yoon = False, False, False, False
    cell_details = []

    # --- 【追加】結果画像作成 ---
    # ここが抜けていたため NameError になっていました
    result_img = cv2.cvtColor(corrected_img, cv2.COLOR_GRAY2RGB)

    for cell in braille_cells:
        if cell.get('is_newline'): 
            final_text += "\n"
            continue
        
        rx, ry, rw, rh = map(int, cell['rect'])
        
        if cell.get('is_space'):
            final_text += "　"
            mode_num = False
            # 空白セル描画 (黄色)
            cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 1)
            continue
        
        # 文字セル描画 (青)
        cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
        
        dots = get_dots_tuple(cell['pattern'])
        char_raw = "?"
        is_special = False
        
        if dots == (3, 4, 5, 6): mode_num = True; is_special = True; char_raw="[数]"
        elif dots == (5,): mode_dak = True; is_special = True; char_raw="[濁]"
        elif dots == (6,): mode_han = True; is_special = True; char_raw="[半]"
        elif dots == (4,): mode_yoon = True; is_special = True; char_raw="[拗]"
        elif dots == (4, 5): mode_yoon=True; mode_dak=True; is_special=True; char_raw="[拗濁]"
        elif dots == (4, 6): mode_yoon=True; mode_han=True; is_special=True; char_raw="[拗半]"

        if not is_special:
            if mode_num: char_raw = num_map.get(dots, "?")
            elif mode_yoon: char_raw = yoon_map.get(dots, "?"); mode_yoon = False
            else: char_raw = jp_map.get(dots, "?")
            
            if mode_dak: char_raw = dakuten_map.get(char_raw, char_raw + "゛"); mode_dak = False
            elif mode_han: char_raw = handaku_map.get(char_raw, char_raw + "゜"); mode_han = False
            final_text += char_raw
        
        # ドット番号描画
        label = "".join(map(str, dots))
        cv2.putText(result_img, label, (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 2)
        
        p = cell['pattern']
        vis = f"{'●' if p[0] else '○'} {'●' if p[3] else '○'}\n{'●' if p[1] else '○'} {'●' if p[4] else '○'}\n{'●' if p[2] else '○'} {'●' if p[5] else '○'}"
        cell_details.append({'char': char_raw, 'dots': dots, 'visual': vis})

    return result_img, final_text, cell_details

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
                    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="解析結果", use_column_width=True)
                with col2:
                    st.text_area("翻訳テキスト", text, height=200)
                
                with st.expander("詳細レポートを見る"):
                    for i, det in enumerate(details):
                        st.text(f"[{i+1:02d}] {det['char']} {det['dots']}")
                        st.text(det['visual'])
                        st.divider()
