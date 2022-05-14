from utils import compute_ap
import utils
import numpy as np
import itertools
import torch
from dataset import get_voc
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import torch
from PIL import Image
import numpy as np
import os

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

def evaluate(model, dataloader, device, threshold):
    AP_25, AP_50, AP_75= [], [], []

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
        
            AP_25.append(compute_ap(gt_boxes,gt_labels,boxes,labels,scores, 0.25))
            AP_50.append(compute_ap(gt_boxes,gt_labels,boxes,labels,scores, 0.5))
            AP_75.append(compute_ap(gt_boxes,gt_labels,boxes,labels,scores, 0.75))
     
    return np.mean(AP_25),np.mean(AP_50),np.mean(AP_75)
num_classes = 21
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
dataset = get_voc('./data', 'test')
data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)
threshold = 0.75

backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

model = FasterRCNN(backbone,
                   num_classes=21,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes).to(device)

model.load_state_dict(torch.load('model_MobileNet.pth',map_location='cpu'))
model.to(device)

value = evaluate(model, data_loader, 0, 25)
print(value)