N_REL = 8
REL_NONE, REL_SELF, REL_RIGHT, REL_LEFT, REL_DOWN, REL_UP, REL_ROW, REL_COL = range(8)

def group_words_into_lines(word_boxes, gap_ratio=0.5, vshare=0.5):
    order = sorted(range(len(word_boxes)),
                   key=lambda i: (word_boxes[i][1], word_boxes[i][0]))
    seg_of = [-1] * len(word_boxes)
    cur, prev = -1, None
    for i in order:
        x0, y0, x1, y1 = word_boxes[i]
        h = max(1, y1 - y0)
        new = True
        if prev is not None:
            px0, py0, px1, py1 = word_boxes[prev]
            v = min(y1, py1) - max(y0, py0)
            if v > vshare * min(h, py1 - py0) and (x0 - px1) < gap_ratio * h and x0 >= px0:
                new = False
        if new:
            cur += 1
        seg_of[i] = cur
        prev = i
    return seg_of

def _ov(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))

def build_edges(seg_boxes):
    S = len(seg_boxes)
    src, dst, rel = [], [], []
    for i in range(S):
        src.append(i); dst.append(i); rel.append(REL_SELF)

    for i, (x0, y0, x1, y1) in enumerate(seg_boxes):
        h, w = max(1, y1 - y0), max(1, x1 - x0)
        best = {REL_RIGHT: (1e9, -1), REL_LEFT: (1e9, -1),
                REL_DOWN: (1e9, -1), REL_UP: (1e9, -1)}
        for j, (a0, b0, a1, b1) in enumerate(seg_boxes):
            if i == j:
                continue
            vo = _ov(y0, y1, b0, b1)
            ho = _ov(x0, x1, a0, a1)
            
            if vo > 0.5 * min(h, b1 - b0):
                if a0 >= x1 and a0 - x1 < best[REL_RIGHT][0]:
                    best[REL_RIGHT] = (a0 - x1, j)
                if a1 <= x0 and x0 - a1 < best[REL_LEFT][0]:
                    best[REL_LEFT] = (x0 - a1, j)
                if vo > 0.7 * min(h, b1 - b0):
                    src.append(i); dst.append(j); rel.append(REL_ROW)
                    
            if ho > 0.3 * min(w, a1 - a0):
                if b0 >= y1 and b0 - y1 < best[REL_DOWN][0]:
                    best[REL_DOWN] = (b0 - y1, j)
                if b1 <= y0 and y0 - b1 < best[REL_UP][0]:
                    best[REL_UP] = (y0 - b1, j)
                    
            if abs(a0 - x0) < 20:
                src.append(i); dst.append(j); rel.append(REL_COL)
                
        for r, (d, j) in best.items():
            if j >= 0:
                src.append(i); dst.append(j); rel.append(r)
    return src, dst, rel