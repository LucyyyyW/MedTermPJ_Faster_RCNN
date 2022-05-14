from utils import compute_ap
import numpy as np
import itertools
import torch

CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

def convert_targets(target):
    anno = target['annotation']
    h, w = anno['size']['height'], anno['size']['width']
    boxes = []
    classes = []
    area = []
    iscrowd = []
    objects = anno['object']
    if not isinstance(objects, list):
        objects = [objects]
    for obj in objects:
        bbox = obj['bndbox']
        bbox = [int(bbox[n]) - 1 for n in ['xmin', 'ymin', 'xmax', 'ymax']]
        boxes.append(bbox)
        classes.append(CLASSES.index(obj['name']))
        iscrowd.append(int(obj['difficult']))
        area.append((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    classes = torch.as_tensor(classes)
    area = torch.as_tensor(area)
    iscrowd = torch.as_tensor(iscrowd)

    image_id = anno['filename'][5:-4]
    image_id = torch.as_tensor([int(image_id)])

    target = {}
    target["boxes"] = boxes
    target["labels"] = classes
    target["image_id"] = image_id

    # for conversion to coco api
    target["area"] = area
    target["iscrowd"] = iscrowd
    return target

def evaluate(model, dataloader, device):
    AP = []
    model.eval()
    for images, targets in dataloader:
        
        images = list(image.to(device) for image in images)
        targets = [convert_targets(t) for t in targets]
        targets = [{k: v.to(device) for k, v in t.items() if not isinstance(v, dict)} for t in targets]
        preds = model(images, targets)
        
        for gt, pred in zip(targets,preds):
            gt_boxes = gt['boxes'].cpu().numpy()
            gt_labels = gt['labels'].cpu().numpy()
            boxes = pred['boxes'].detach().cpu().numpy()
            labels = pred['labels'].detach().cpu().numpy()
            scores = pred['scores'].detach().cpu().numpy()
        
            AP.append(compute_ap(gt_boxes,gt_labels,boxes,labels,scores))
     
    return np.mean(AP)

