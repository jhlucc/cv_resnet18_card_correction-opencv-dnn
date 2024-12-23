import numpy as np
import cv2


def max_pool2d(input, kernel_size, stride=1, padding=0, return_indices=False):
    batch_size, channels, in_height, in_width = input.shape
    k_height, k_width = kernel_size
    
    out_height = int((in_height + 2 * padding - k_height) / stride) + 1
    out_width = int((in_width + 2 * padding - k_width) / stride) + 1
    out = np.zeros((batch_size, channels, out_height, out_width), dtype=np.float32)
    index = np.zeros((batch_size, channels, out_height, out_width), dtype=np.int64)
    if padding > 0:
        input_ = np.zeros((batch_size, channels, in_height + 2 * padding, in_width + 2 * padding), dtype=np.float32)
        input_[:, :, padding:padding + in_height, padding:padding + in_width] = input
        input = input_

    for b in range(batch_size):
        for c in range(channels):
            for i in range(out_height):
                for j in range(out_width):
                    start_i = i * stride
                    start_j = j * stride
                    end_i = start_i + k_height
                    end_j = start_j + k_width
                    Xi = input[b, c, start_i: end_i, start_j: end_j]
  
                    max_value = np.max(Xi)
                    k = np.argmax(Xi)
                    Ia = k // k_height + start_i - padding
                    Ib = k % k_width + start_j - padding
                    Ia = Ia if Ia > 0 else 0
                    Ib = Ib if Ib > 0 else 0
                    max_index = Ia * in_width + Ib
                    out[b, c, i, j] = max_value
                    index[b, c, i, j] = max_index

    if return_indices:
        return out, index
    else:
        return out
    
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2
    hmax = max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).astype(np.float32)
    return heat * keep, keep

def numpy_topk(scores, K, axis=-1):
    indices = np.argsort(-scores, axis=axis).take(np.arange(K), axis=axis)  ### 从大到小排序，取出前K个
    sort_scores = np.take(scores, indices)
    return sort_scores, indices

def _gather_feat(feat, ind, mask=None):
    # print("_gather_feat input shape:", feat.shape, ind.shape)
    dim = feat.shape[2]
    ind = np.tile(np.expand_dims(ind, axis=2), (1, 1, dim))
    feat = np.take_along_axis(feat, ind, axis=1)
    # print("_gather_feat output shape:", feat.shape, ind.shape)
    if mask is not None:
        mask = np.tile(np.expand_dims(mask, axis=2), (1, 1, feat.shape[-1]))
        feat = feat[mask]
        feat = feat.reshape((-1, dim))
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = np.transpose(feat, (0, 2, 3, 1))
    feat = feat.reshape((feat.shape[0], -1, feat.shape[3]))
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=40):
    batch, cat, height, width = scores.shape

    topk_scores, topk_inds = numpy_topk(scores.reshape((batch, cat, -1)), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).astype(np.float32)
    topk_xs = (topk_inds % width).astype(np.float32)

    topk_score, topk_ind = numpy_topk(topk_scores.reshape((batch, -1)), K)
    topk_clses = (topk_ind / K).astype(np.int32)
    topk_inds = _gather_feat(topk_inds.reshape((batch, -1, 1)),topk_ind).reshape((batch, K))
    topk_ys = _gather_feat(topk_ys.reshape((batch, -1, 1)), topk_ind).reshape((batch, K))
    topk_xs = _gather_feat(topk_xs.reshape((batch, -1, 1)), topk_ind).reshape((batch, K))

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def bbox_decode(heat, wh, reg=None, K=100):
    batch, cat, height, width = heat.shape

    heat, keep = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
    if reg is not None:
        reg = _tranpose_and_gather_feat(reg, inds)
        reg = reg.reshape((batch, K, 2))
        xs = xs.reshape((batch, K, 1)) + reg[:, :, 0:1]
        ys = ys.reshape((batch, K, 1)) + reg[:, :, 1:2]
    else:
        xs = xs.reshape((batch, K, 1)) + 0.5
        ys = ys.reshape((batch, K, 1)) + 0.5
    wh = _tranpose_and_gather_feat(wh, inds)
    wh = wh.reshape((batch, K, 8))
    clses = clses.reshape((batch, K, 1)).astype(np.float32)
    scores = scores.reshape((batch, K, 1))
    bboxes = np.concatenate(
        [
            xs - wh[..., 0:1],
            ys - wh[..., 1:2],
            xs - wh[..., 2:3],
            ys - wh[..., 3:4],
            xs - wh[..., 4:5],
            ys - wh[..., 5:6],
            xs - wh[..., 6:7],
            ys - wh[..., 7:8],
        ],
        axis=2,
    )
    detections = np.concatenate([bboxes, scores, clses, xs, ys], axis=2)

    return detections, inds

def decode_by_ind(heat, inds, K=100):
    batch, cat, height, width = heat.shape
    score = _tranpose_and_gather_feat(heat, inds)
    score = score.reshape((batch, K, cat))
    Type = np.max(score, axis=2)
    return Type

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.0], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size, rot=0):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, rot, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def bbox_post_process(bbox, c, s, h, w):
    for i in range(bbox.shape[0]):
        bbox[i, :, 0:2] = transform_preds(bbox[i, :, 0:2], c[i], s[i], (w, h))
        bbox[i, :, 2:4] = transform_preds(bbox[i, :, 2:4], c[i], s[i], (w, h))
        bbox[i, :, 4:6] = transform_preds(bbox[i, :, 4:6], c[i], s[i], (w, h))
        bbox[i, :, 6:8] = transform_preds(bbox[i, :, 6:8], c[i], s[i], (w, h))
        bbox[i, :, 10:12] = transform_preds(bbox[i, :, 10:12], c[i], s[i], (w, h))
    return bbox

def nms(dets, thresh):
    '''
    len(dets)是batchsize,在推理时batchsize通常等于1, 那就意味着这个函数执行到开头的if就返回了
    即使在推理时输入多张图片, batchsize大于1,也不应该做nms的呀.因为计算多个目标的重叠关系只是在一张图片内做的,多张图片之间计算目标框的
    重叠关系,这个是什么意思呢?
    '''

    if len(dets) < 2:
        return dets
    index_keep = []
    keep = []
    for i in range(len(dets)):
        box = dets[i]
        if box[8] < thresh:
            break
        max_score_index = -1
        ctx = (dets[i][0] + dets[i][2] + dets[i][4] + dets[i][6]) / 4
        cty = (dets[i][1] + dets[i][3] + dets[i][5] + dets[i][7]) / 4
        for j in range(len(dets)):
            if i == j or dets[j][8] < thresh:
                break
            x1, y1 = dets[j][0], dets[j][1]
            x2, y2 = dets[j][2], dets[j][3]
            x3, y3 = dets[j][4], dets[j][5]
            x4, y4 = dets[j][6], dets[j][7]
            a = (x2 - x1) * (cty - y1) - (y2 - y1) * (ctx - x1)
            b = (x3 - x2) * (cty - y2) - (y3 - y2) * (ctx - x2)
            c = (x4 - x3) * (cty - y3) - (y4 - y3) * (ctx - x3)
            d = (x1 - x4) * (cty - y4) - (y1 - y4) * (ctx - x4)
            if (a > 0 and b > 0 and c > 0 and d > 0) or (a < 0 and b < 0
                                                         and c < 0 and d < 0):
                if dets[i][8] > dets[j][8] and max_score_index < 0:
                    max_score_index = i
                elif dets[i][8] < dets[j][8]:
                    max_score_index = -2
                    break
        if max_score_index > -1:
            index_keep.append(max_score_index)
        elif max_score_index == -1:
            index_keep.append(i)
    for i in range(0, len(index_keep)):
        keep.append(dets[index_keep[i]])
    return np.array(keep)


def draw_show_img(img, result, savepath):
    polys = result['POLYGONS']
    centers = result['CENTER']
    angle_cls = result['LABELS']
    bbox = result['BBOX']
    color = (0,0,255)
    for idx, poly in enumerate(polys):
        poly = poly.reshape(4, 2).astype(np.int32)
        ori_center = ((bbox[idx][0]+bbox[idx][2])//2,(bbox[idx][1]+bbox[idx][3])//2)
        img = cv2.drawContours(img,[poly],-1,color,2)
        img = cv2.circle(img,tuple(centers[idx].astype(np.int64).tolist()),5,color,thickness=2)
        img = cv2.circle(img,ori_center,5,color,thickness=2)
        img = cv2.putText(img,str(angle_cls[idx]),ori_center,cv2.FONT_HERSHEY_SIMPLEX,2,color,2)
    cv2.imwrite(savepath,img)

def merge_images_horizontal(images, output_path="./show.jpg"):   
    # 确定目标高度（所有图像的目标高度）
    target_height = min(img.shape[0] for img in images)
    
    # 调整所有图像的大小，以使它们的高度一致
    resized_images = []
    for img in images:
        aspect_ratio = img.shape[1] / img.shape[0]  # 计算原始图像的宽高比
        new_width = int(target_height * aspect_ratio)  # 计算调整后的宽度
        resized_img = cv2.resize(img, (new_width, target_height))  # 调整图像大小
        resized_images.append(resized_img)
    
    # 计算合并后的总宽度
    total_width = sum(img.shape[1] for img in resized_images)
    
    # 创建一个新的空白图像
    merged_image = np.zeros((target_height, total_width, 3), dtype=np.uint8)
    
    # 将所有调整大小后的图像粘贴到新的图像上
    x_offset = 0
    for img in resized_images:
        merged_image[:, x_offset:x_offset + img.shape[1]] = img
        x_offset += img.shape[1]
    
    # 保存合并后的图像
    cv2.imwrite(output_path, merged_image)