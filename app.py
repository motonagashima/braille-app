import streamlit as st
import cv2
import numpy as np

# ページ設定
st.set_page_config(page_title="点字翻訳アプリ", layout="wide")
st.title("📷 点字翻訳アプリ")
st.write("点字の画像をアップロードすると、日本語に翻訳します。")

# ==========================================
# 1. 画像のアップロード
# ==========================================
uploaded_file = st.file_uploader("画像ファイルを選択してください", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 画像をOpenCV形式に変換
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    braille_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if braille_image is None:
        st.error("画像を読み込めませんでした。")
    else:
        # 画像を表示（サイドバーなどに元画像を表示するのもありですが、ここではメインに）
        st.image(braille_image, caption="アップロードされた画像", channels="BGR", use_container_width=True)

        with st.spinner("解析中..."):
            # ==========================================
            # 2. 前処理
            # ==========================================
            gray_image = cv2.cvtColor(braille_image, cv2.COLOR_BGR2GRAY)
            blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
            thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # ==========================================
            # 3. ドット検出
            # ==========================================
            contours, _ = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            raw_dots = []
            radii_list = []
            dot_id_counter = 0

            for contour in contours:
                area = cv2.contourArea(contour)
                if 3 < area < 3000: 
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    
                    # 白抜きチェック
                    mask = np.zeros(thresh_image.shape, dtype=np.uint8)
                    cv2.circle(mask, center, int(radius), 255, -1)
                    mean_val = cv2.mean(thresh_image, mask=mask)[0]
                    if mean_val < 120: continue 

                    raw_dots.append({
                        'id': dot_id_counter,
                        'center': center, 
                        'radius': radius
                    })
                    radii_list.append(radius)
                    dot_id_counter += 1

            if not raw_dots:
                st.warning("ドットが見つかりませんでした。もっと鮮明な画像を試してください。")
            else:
                median_radius = np.median(radii_list)
                braille_dots = []
                valid_radii = []
                
                for dot in raw_dots:
                    if median_radius * 0.5 <= dot['radius'] <= median_radius * 2.0:
                        braille_dots.append(dot)
                        valid_radii.append(dot['radius'])
                
                avg_radius = np.mean(valid_radii) if valid_radii else median_radius
                
                # デバッグ情報の表示
                with st.expander("検出パラメータ"):
                    st.write(f"検出ドット数: {len(braille_dots)}")
                    st.write(f"基準半径: {avg_radius:.1f} px")

                # ==========================================
                # 4. グリッド解析 (尺取り虫方式 + 固定枠表示)
                # ==========================================
                dots_y = np.array([d['center'][1] for d in braille_dots])
                dots_y_sorted = np.sort(dots_y)
                y_diffs = np.diff(dots_y_sorted)
                line_separators = np.where(y_diffs > avg_radius * 3.5)[0]
                
                line_y_centers = []
                start_idx = 0
                for sep_idx in line_separators:
                    end_idx = sep_idx + 1
                    line_y_centers.append(np.median(dots_y_sorted[start_idx:end_idx]))
                    start_idx = end_idx
                line_y_centers.append(np.median(dots_y_sorted[start_idx:]))

                braille_cells = []
                used_dot_ids = set()

                for line_center_y in line_y_centers:
                    line_dots = [d for d in braille_dots if abs(d['center'][1] - line_center_y) < avg_radius * 4]
                    if not line_dots: continue
                    
                    line_dots.sort(key=lambda d: d['center'][0])
                    dots_x = np.array([d['center'][0] for d in line_dots])

                    # --- A. グループ化 ---
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

                    # --- B. ピッチ推定 ---
                    group_starts = np.array([min([d['center'][0] for d in g]) for g in groups])
                    estimated_pitch = avg_radius * 6.0
                    
                    if len(group_starts) > 1:
                        start_diffs = np.diff(group_starts)
                        valid_diffs = start_diffs[start_diffs > avg_radius * 4.0]
                        if len(valid_diffs) > 0:
                            q25 = np.percentile(valid_diffs, 25)
                            estimated_pitch = q25

                    # 縦ピッチ
                    y_dists = [abs(d['center'][1] - line_center_y) for d in line_dots]
                    valid_y = [dy for dy in y_dists if dy > avg_radius * 0.5]
                    v_pitch = np.median(valid_y) if valid_y else avg_radius * 2.5

                    # 固定セルサイズ
                    FIXED_CELL_WIDTH = estimated_pitch * 0.75 
                    FIXED_CELL_HEIGHT = (v_pitch * 2) + (avg_radius * 3)
                    intra_pitch = avg_radius * 2.5

                    # --- C. 尺取り虫ロジック ---
                    cursor_x = group_starts[0]
                    
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
                            col1_x = min_x; col2_x = max_x
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

                # --- 救済処理 ---
                for dot in braille_dots:
                    if dot['id'] not in used_dot_ids:
                        dx, dy = dot['center']
                        min_dist = float('inf')
                        best_cell = None
                        best_target_idx = -1
                        for cell in braille_cells:
                            if cell.get('is_newline') or cell.get('is_space'): continue
                            for idx, (tx, ty) in enumerate(cell['targets']):
                                dist = np.sqrt((dx - tx)**2 + (dy - ty)**2)
                                if dist < min_dist:
                                    min_dist = dist
                                    best_cell = cell
                                    best_target_idx = idx
                        if min_dist < avg_radius * 5.0:
                            best_cell['pattern'][best_target_idx] = True
                            best_cell['has_dot'] = True
                            best_cell['matched_dots'].append({'target_idx': best_target_idx, 'dot_center': (dx, dy)})
                            used_dot_ids.add(dot['id'])

                # ==========================================
                # 5. 翻訳処理
                # ==========================================
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
                    (1,): "a", 
                    (1, 6): "きゃ", (1, 4, 6): "きゅ", (2, 4, 6): "きょ",
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
                cell_details = []

                for cell in braille_cells:
                    if cell.get('is_newline'): final_text += "\n"; continue
                    if cell.get('is_space'): final_text += "　"; mode_number = False; continue
                    if not cell['has_dot']: continue
                    
                    dots = get_dots_tuple(cell['pattern'])
                    char_raw = "?"; is_special = False
                    
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
                    
                    p = cell['pattern']
                    dot_visual =  f" {'●' if p[0] else '○'} {'●' if p[3] else '○'}\n {'●' if p[1] else '○'} {'●' if p[4] else '○'}\n {'●' if p[2] else '○'} {'●' if p[5] else '○'}"
                    cell_details.append({'char': char_raw, 'dots': dots, 'visual': dot_visual})

                # ==========================================
                # 結果表示
                # ==========================================
                st.subheader("📝 翻訳結果")
                st.success(final_text)

                # ==========================================
                # 6. 結果可視化
                # ==========================================
                result_img = braille_image.copy()
                for cell in braille_cells:
                    if cell.get('is_newline'): continue
                    
                    rx, ry, rw, rh = map(int, cell['rect'])
                    if cell.get('is_space'):
                         cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (0, 255, 255), 1)
                    else:
                         cv2.rectangle(result_img, (rx, ry), (rx+rw, ry+rh), (255, 0, 0), 2)
                         for tx, ty in cell['targets']:
                            cv2.circle(result_img, (int(tx), int(ty)), 2, (0, 0, 255), 1)
                         if 'matched_dots' in cell:
                            for match in cell['matched_dots']:
                                 t_idx = match['target_idx']
                                 tx, ty = cell['targets'][t_idx]
                                 dx, dy = match['dot_center']
                                 cv2.line(result_img, (int(tx), int(ty)), (int(dx), int(dy)), (0, 255, 0), 1)
                         dots = get_dots_tuple(cell['pattern'])
                         label = "".join(map(str, dots))
                         cv2.putText(result_img, label, (rx, ry-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,200), 2)

                st.subheader("🔍 認識結果の可視化")
                # BGRからRGBに変換して表示
                st.image(result_img, channels="BGR", caption="解析オーバーレイ画像", use_container_width=True)

                with st.expander("詳細認識レポートを見る"):
                    for i, det in enumerate(cell_details):
                         st.text(f"[Cell {i+1:02d}] 文字: {det['char']}  ドット: {det['dots']}\n{det['visual']}\n" + "-" * 20)
