import cv2
import numpy as np
import math
from utils import bbox_decode, decode_by_ind, bbox_post_process, nms, draw_show_img, merge_images_horizontal


def ResizePad(img, target_size):
    h, w = img.shape[:2]
    m = max(h, w)
    ratio = target_size / m
    new_w, new_h = int(ratio * w), int(ratio * h)
    img = cv2.resize(img, (new_w, new_h), cv2.INTER_LINEAR)
    top = (target_size - new_h) // 2
    bottom = (target_size - new_h) - top
    left = (target_size - new_w) // 2
    right = (target_size - new_w) - left
    img1 = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    return img1, new_w, new_h, left, top

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class cv_resnet18_card_correction:
    def __init__(self, model_path):
        self.model = cv2.dnn.readNet(model_path)
        self.resize_shape = [768, 768]
        self.outlayer_names = self.model.getUnconnectedOutLayersNames()
        self.mean = np.array([0.408, 0.447, 0.470],dtype=np.float32).reshape((1, 1, 3))
        self.std = np.array([0.289, 0.274, 0.278],dtype=np.float32).reshape((1, 1, 3))
        self.K = 10
        self.obj_score = 0.5
        self.out_height = self.resize_shape[0] // 4
        self.out_width = self.resize_shape[1] // 4

    def infer(self, srcimg):
        self.image = srcimg.copy()
        ori_h, ori_w = srcimg.shape[:-1]
        self.c = np.array([ori_w / 2., ori_h / 2.], dtype=np.float32)
        self.s = max(ori_h, ori_w) * 1.0
        blob, new_w, new_h, left, top = self.preprocess(srcimg, self.resize_shape)
        self.model.setInput(blob)
        pre_out = self.model.forward(self.outlayer_names)
        
        out = self.postprocess(pre_out)
        return out

    def preprocess(self, img, resize_shape):
        im, new_w, new_h, left, top = ResizePad(img, resize_shape[0])
        im = (im.astype(np.float32) / 255.0 - self.mean) / self.std
        im = np.expand_dims(im.transpose((2, 0, 1)), axis=0)
        return im.astype(np.float32), new_w, new_h, left, top

    def distance(self, x1, y1, x2, y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    def crop_image(self, img, position):
        x0, y0 = position[0][0], position[0][1]
        x1, y1 = position[1][0], position[1][1]
        x2, y2 = position[2][0], position[2][1]
        x3, y3 = position[3][0], position[3][1]

        img_width = self.distance((x0 + x3) / 2, (y0 + y3) / 2, (x1 + x2) / 2,
                                  (y1 + y2) / 2)
        img_height = self.distance((x0 + x1) / 2, (y0 + y1) / 2, (x2 + x3) / 2,
                                   (y2 + y3) / 2)

        corners_trans = np.zeros((4, 2), np.float32)
        corners_trans[0] = [0, 0]
        corners_trans[1] = [img_width, 0]
        corners_trans[2] = [img_width, img_height]
        corners_trans[3] = [0, img_height]

        transform = cv2.getPerspectiveTransform(position, corners_trans)
        dst = cv2.warpPerspective(img, transform,
                                  (int(img_width), int(img_height)))
        return dst
    
    def postprocess(self, output):
        reg = output[3]
        wh = output[2]
        hm = output[4]
        angle_cls = output[0]
        ftype_cls = output[1]
        
        hm = sigmoid(hm)
        angle_cls = sigmoid(angle_cls)
        ftype_cls = sigmoid(ftype_cls)

        bbox, inds = bbox_decode(hm, wh, reg=reg, K=self.K)
        angle_cls = decode_by_ind(angle_cls, inds, K=self.K)
        ftype_cls = decode_by_ind(ftype_cls, inds,K=self.K).astype(np.float32)

        for i in range(bbox.shape[1]):
            bbox[0][i][9] = angle_cls[0][i]
        bbox = np.concatenate((bbox, np.expand_dims(ftype_cls, axis=-1)),axis=-1)
        # bbox = nms(bbox, 0.3)
        bbox = bbox_post_process(bbox.copy(), [self.c], [self.s], self.out_height, self.out_width)
        res = []
        angle = []
        sub_imgs = []
        ftype = []
        score = []
        center = []
        corner_left_right = []
        for idx, box in enumerate(bbox[0]):
            if box[8] > self.obj_score:
                angle.append(int(box[9]))
                res.append(box[0:8])
                box8point = np.array(box[0:8]).reshape(4,2).astype(np.int32)
                corner_left_right.append([box8point[:,0].min(),box8point[:,1].min(),box8point[:,0].max(),box8point[:,1].max()])
                sub_img = self.crop_image(self.image,res[-1].copy().reshape(4, 2))
                if angle[-1] == 1:
                    sub_img = cv2.rotate(sub_img, 2)
                if angle[-1] == 2:
                    sub_img = cv2.rotate(sub_img, 1)
                if angle[-1] == 3:
                    sub_img = cv2.rotate(sub_img, 0)
                sub_imgs.append(sub_img)
                ftype.append(int(box[12]))
                score.append(box[8])
                center.append([box[10],box[11]])

        result = {
            "POLYGONS": np.array(res),
            "BBOX": np.array(corner_left_right),
            "SCORES": np.array(score),
            "OUTPUT_IMGS": sub_imgs,
            "LABELS": np.array(angle),
            "LAYOUT": np.array(ftype),
            "CENTER": np.array(center)
        }
        return result
    
if __name__ == "__main__":
    imgpath = 'testimgs/demo3.jpg'
    model_path = 'cv_resnet18_card_correction.onnx'
    mynet = cv_resnet18_card_correction(model_path)
    
    srcimg = cv2.imread(imgpath)
    out = mynet.infer(srcimg)

    draw_show_img(srcimg.copy(), out, "show.jpg")
    merge_images_horizontal([srcimg] + out['OUTPUT_IMGS'],"pp4_rotate_show.jpg")
    # cv2.imwrite('rotate_img.jpg',out['OUTPUT_IMGS'][0])