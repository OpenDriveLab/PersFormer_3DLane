from . import details

def nms(boxes, scores, overlap, top_k):
    return details.nms_forward(boxes, scores, overlap, top_k)
