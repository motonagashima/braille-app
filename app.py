import streamlit as st
import cv2
import numpy as np
from PIL import Image
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
    # すべての輪郭点を含む最小矩形を取得
    if not contours:
        return image, 0
    
    all_points = np.concatenate(contours)
    rect = cv2.minAreaRect(all_points)
    angle = rect[-1]
    
    # 角度の正規化 (OpenCVのバージョンにより異なる場合があるため調整)
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    # わずかな角度なら補正しない（ノイズ対策）
    if abs(angle) < 0.5:
        return image, 0

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 回転後の画像サイズ計算（見切れ防止）
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated, angle

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

    # 2. 一次ドット検出 (傾き検出用)
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # --- 【追加】傾き補正 ---
    # ドットと思われる輪郭だけを集めて角度を計算
    dot_contours = []
    for cnt in contours:
        if 10 < cv2.contourArea(cnt) < 5000:
            dot_contours.append(cnt)
            
    corrected_img, angle = correct_skew(gray_image, dot_contours)
    
    # 補正後の画像で再処理
    blurred_corr = cv2.GaussianBlur(corrected_img, (5, 5), 0)
    thresh_corr = cv2.adaptiveThreshold(blurred_corr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours_final, _ = cv2.findContours(thresh_corr, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # 3. 本番ドット検出
    raw_dots = []
    radii_list = []
    dot_id_counter = 0

    for contour in contours_final:
        area = cv2.contourArea(contour)
        if 3 < area < 5000: 
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            
            # 白抜き・ハイフンチェック
            x_r, y_r, w_r, h_r = cv2.boundingRect(contour)
            aspect_ratio = float(w_r) / h_r
            if aspect_ratio > 1.8 or aspect_ratio < 0.5: continue

            mask = np.zeros(thresh_corr.shape, dtype=np.uint8)
            cv2.circle(mask, center, int(radius), 255, -1)
            mean_val = cv2.mean(thresh_corr, mask=mask)[0]
            if mean_val < 130: continue 

            raw_dots.append({'id': dot_id_counter, 'center': center, 'radius': radius})
            radii_list.append(radius)
            dot_id_counter += 1

    if not raw_dots:
        return corrected_img, "ドットが見つかりませんでした。", []

    # 基準半径
    median_radius = np.median(radii_list)
    braille_dots = []
    valid_radii = []
    for dot in raw_dots:
        if median_radius * 0.5 <= dot['radius'] <= median_radius * 2.0:
            braille_dots.append(dot)
            valid_radii.append(dot['radius'])
    
    if not braille_dots:
        return corrected_img, "有効なドットが見つかりませんでした。", []

    avg_radius = np.mean(valid_radii)

    # 4. グリッド解析 (行認識強化版)
    
    # --- 【修正】行のクラスタリング ---
    # Y座標でソート
    braille_dots.sort(key=lambda d: d['center'][1])
    
    lines_of_dots = []
    if braille_dots:
        current_line = [braille_dots[0]]
        current_line_y_sum = braille_dots[0]['center'][1]
        
        for i in range(1, len(braille_dots)):
            dot = braille_dots[i]
            dy = dot['center'][1]
            
            # 現在の行の平均Y座標
            current_line_avg_y = current_line_y_sum / len(current_line)
            
            # 平均との差が 半径*2.5 以内なら同じ行とみなす
            if abs(dy - current_line_avg_y) < avg_radius * 2.5:
                current_line.append(dot)
                current_line_y_sum += dy
            else:
                # 新しい行へ
                lines_of_dots.append(current_line)
                current_line = [dot]
                current_line_y_sum = dy
        lines_of_dots.append(current_line)

    braille_cells = []
    used_dot_ids = set()

    # 各行ごとの処理
    for line_dots in lines_of_dots:
        if not line_dots: continue
        
        # 行のY中心を再計算
        line_center_y = np.median([d['center'][1] for d in line_dots])
        
        # X座標でソート
        line_dots.sort(key=lambda d: d['center'][0])
        dots_x = np.array([d['center'][0] for d in line_dots])

        # 文字グループ化
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

        # ピッチ推定
        group_starts = np.array([min([d['center'][0] for d in g]) for g in groups])
        estimated_pitch = avg_radius * 6.0
        if len(group_starts) > 1:
            start_diffs = np.diff(group_starts)
            valid_diffs = start_diffs[start_diffs > avg_radius * 4.0]
            if len(valid_diffs) > 0:
                estimated_pitch = np.percentile(valid_diffs, 25)

        # 縦ピッチ
        y_dists = [abs(d['center'][1] - line_center_y) for d in line_dots]
        valid_y = [dy for dy in y_dists if dy > avg_radius * 0.5]
        v_pitch = np.median(valid_y) if valid_y else avg_radius * 2.5

        FIXED_CELL_WIDTH = estimated_pitch * 0.75 
        FIXED_CELL_HEIGHT = (v_pitch * 2) + (avg_radius * 3)
        intra_pitch = avg_radius * 2.5
        cursor_x = group_starts[0]
        
        # 尺取り虫ロジック
        for grp in groups:
            min_x = min([d['center'][0] for d in grp])
            max_x = max([d['center'][0] for d in grp])
            grp_width = max_x - min_x
            
            dist_from_cursor = min_x - cursor_x
            if dist_from_cursor < -avg_radius: cursor_x = min_x
            
            gap_steps = int(round(dist_from_cursor / estimated_pitch))
            gap_steps = min(gap_steps, 5)

            for _ in range(gap_steps):
                sp_anchor_x = int(cursor_x - (FIXED_CELL_WIDTH / 2) + (intra_pitch / 2))
                braille_cells.append({
                    'rect': (sp_anchor_x, int(line_center_y - (FIXED_CELL_HEIGHT/2)), int(FIXED_CELL_WIDTH), int(FIXED_CELL_HEIGHT)),
                    'pattern': [False]*6, 'targets': [], 'has_dot': False, 'is_space': True
                })
                cursor_x += estimated_pitch

            col1_x, col2_x = 0, 0
            cell_center_x = 0
            
            if grp_width > avg_radius * 1.5:
                cell_center_x = (min_x + max_x) / 2
                col1_x = min_x
                col2_x = max_x
            else:
                ideal_left = cursor_x
                diff = min_x - ideal_left
                if diff > intra_pitch * 0.6:
                    col1_x = min_x - intra_pitch; col2_x = min_x
                    cell_center_x = min_x - (intra_pitch/2)
                else:
                    col1_x = min_x; col2_x = min_x + intra_pitch
                    cell_center_x = min_x + (intra_pitch/2)

            targets = [
                (col1_x, line_center_y - v_pitch), (col1_x, line_center_y), (col1_x, line_center_y + v_pitch),
                (col2_x, line_center_y - v_pitch), (col2_x, line_center_y), (col2_x, line_center_y + v_pitch)
            ]
            
            pattern = [False] * 6
            matched_dots = []
            
            for dot in grp:
                dx, dy = dot['center']
                min_dist = float('inf')
                best_idx = -1
                for idx, (tx, ty) in enumerate(targets):
                    dist = np.sqrt((dx - tx)**2 + ((dy - ty)*0.9)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                
                if min_dist < avg_radius * 4.0:
                    pattern[best_idx] = True
                    matched_dots.append({'target_idx': best_idx, 'dot_center': (dx, dy)})
                    used_dot_ids.add(dot['id'])

            anchor_x = int(cell_center_x - (FIXED_CELL_WIDTH / 2))
            anchor_y = int(line_center_y - (FIXED_CELL_HEIGHT / 2))

            braille_cells.append({
                'rect': (anchor_x, anchor_y, int(FIXED_CELL_WIDTH), int(FIXED_CELL_HEIGHT)),
                'pattern': pattern,
                'targets': targets,
                'has_dot': True,
                'is_space': False,
                'matched_dots': matched_dots
            })
            
            cursor_x = min_x + estimated_pitch

        braille_cells.append({'is_newline': True})

    # 5. 翻訳と可視化
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
    num_map = {
        (1,): "1", (1, 2): "2", (1, 4): "3", (1, 2, 4): "4", (1, 5): "5",
        (1, 6): "6", (1, 2, 5): "7", (1, 2, 6): "8", (2, 4): "9", (2, 4, 5): "0"
    }
    yoon_map = {
        (1,): "a", (1, 6): "きゃ", (1, 4, 6): "きゅ", (2, 4, 6): "きょ",
        (1, 5, 6): "しゃ", (1, 4, 5, 6): "しゅ", (2, 4, 5, 6): "しょ",
        (1, 3, 5): "ちゃ", (1, 3, 4, 5): "ちゅ", (2, 3, 4, 5): "ちょ",
        (1, 3): "にゃ", (1, 3, 4): "にゅ", (2, 3, 4): "にょ",
        (1, 3, 6): "ひゃ", (1, 3, 4, 6): "ひゅ", (2, 3, 4, 6): "ひょ",
        (1, 3, 5, 6): "みゃ", (1, 3, 4, 5, 6): "みゅ", (2, 3, 4, 5, 6): "みょ",
        (1, 5): "りゃ", (1, 4, 5): "りゅ", (2, 4, 5): "りょ",
    }
    dakuten_char_map = {"か":"が","き":"ぎ","く":"ぐ","け":"げ","こ":"ご","さ":"ざ","し":"じ","す":"ず","せ":"ぜ","そ":"ぞ","た":"だ","ち":"ぢ","つ":"づ","て":"で","と":"ど","は":"ば","ひ":"び","ふ":"ぶ","へ":"べ","ほ":"ぼ","う":"ゔ"}
    handakuten_char_map = {"は":"ぱ","ひ":"ぴ","ふ":"ぷ","へ":"ぺ","ほ":"ぽ"}

    final_text = ""
    mode_number = False; mode_dakuten = False; mode_handakuten = False; mode_yoon = False
    
    # 補正後の画像をRGB変換して使用
    result_img = cv2.cvtColor(corrected_img, cv2.COLOR_GRAY2RGB)
    cell_details = []

    for cell in braille_cells:
        if cell.get('is_newline'):
            final_text += "\n"
            continue
        
        rx, ry, rw, rh = map(int, cell['rect'])
        
        if cell.get('is_space'):
            final_text += "　"
            mode_number = False
            cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 2)
            continue
        
        cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
        
        if not cell['has_dot']: continue
        
        dots = get_dots_tuple(cell['pattern'])
        char_raw = "?"
        is_special = False
        
        if dots == (3, 4, 5, 6): mode_number = True; is_special=True; char_raw="[数]"
        elif dots == (5,): mode_dakuten = True; is_special=True; char_raw="[濁]"
        elif dots == (6,): mode_handakuten = True; is_special=True; char_raw="[半]"
        elif dots == (4,): mode_yoon = True; is_special=True; char_raw="[拗]"
        elif dots == (4, 5): mode_yoon = True; mode_dakuten = True; is_special=True; char_raw="[拗濁]"
        elif dots == (4, 6): mode_yoon = True; mode_handakuten = True; is_special=True; char_raw="[拗半]"

        if not is_special:
            if mode_number: char_raw = num_map.get(dots, "?")
            elif mode_yoon: char_raw = yoon_map.get(dots, "?"); mode_yoon = False
            else: char_raw = jp_map.get(dots, "?")
            
            if mode_dakuten: char_raw = dakuten_char_map.get(char_raw, char_raw + "゛"); mode_dakuten = False
            elif mode_handakuten: char_raw = handakuten_char_map.get(char_raw, char_raw + "゜"); mode_handakuten = False
            
            final_text += char_raw
        
        label = "".join(map(str, dots))
        cv2.putText(result_img, label, (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 2)
        
        for tx, ty in cell['targets']:
            cv2.circle(result_img, (int(tx), int(ty)), 2, (0, 0, 255), 1)

        p = cell['pattern']
        dot_visual =  f" {'●' if p[0] else '○'} {'●' if p[3] else '○'}\n {'●' if p[1] else '○'} {'●' if p[4] else '○'}\n {'●' if p[2] else '○'} {'●' if p[5] else '○'}"
        cell_details.append({'char': char_raw, 'dots': dots, 'visual': dot_visual})

    return result_img, final_text, cell_details

# ==========================================
# Streamlit UI
# ==========================================
st.title("点字翻訳アプリ (Braille Reader)v1")
st.write("画像の点字部分をトリミングして翻訳します。")

uploaded_file = st.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.subheader("1. 範囲指定")
    st.write("翻訳したい点字の部分を枠で囲んでください。")
    
    # トリミングコンポーネント (box_color='blue'で見やすく)
    cropped_img = st_cropper(image, realtime_update=True, box_color='#0000FF', aspect_ratio=None)
    
    st.subheader("2. 翻訳")
    if st.button("この範囲を翻訳する"):
        if cropped_img is not None:
            # PIL -> OpenCV (numpy) 変換
            img_array = np.array(cropped_img)
            # RGB -> BGR (OpenCV用)
            if len(img_array.shape) == 3:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                img_cv = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)

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
                        st.text(f"[{i+1:02d}] {det['char']} (ドット: {det['dots']})")
                        st.text(det['visual'])
                        st.divider()
