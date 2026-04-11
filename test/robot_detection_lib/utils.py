

def suppress_duplicate_boxes(
    boxes: list[dict],
    iou_threshold: float = 0.65,
) -> list[dict]:
    suppressed = set()

    for i, box_i in enumerate(boxes):
        if i in suppressed:
            continue
        for j, box_j in enumerate(boxes):
            if j <= i or j in suppressed:
                continue
            if iou(box_i["box"], box_j["box"]) > iou_threshold:
                bi, bj = box_i["box"], box_j["box"]
                area_i = (bi[2] - bi[0]) * (bi[3] - bi[1])
                area_j = (bj[2] - bj[0]) * (bj[3] - bj[1])
                suppressed.add(j if area_i >= area_j else i)

    return [r for i, r in enumerate(boxes) if i not in suppressed]


def iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0