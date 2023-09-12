import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()
import rospy
import numpy as np
import os, json, cv2, random, sys
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets import register_coco_instances
from matplotlib import pyplot as plt
from geometry_msgs.msg import Point


def listener():
    rospy.Subscriber('/fan/image', Point, callback, queue_size=1)


def callback(data):
    flag = data
    if flag.x == 1:
        for d in ["val"]:
            register_coco_instances(f"gown_{d}", {}, f"./gown_{d}/gown_{d}.json", f"./gown_{d}")

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.DATASETS.TRAIN = ("gown_val",)
        cfg.DATASETS.TEST = ()
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.00025
        cfg.SOLVER.MAX_ITER = 500
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        # trainer = DefaultTrainer(cfg)
        # trainer.resume_or_load(resume=False)
        # trainer.train()

        cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_1.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
        cfg.DATASETS.TEST = ("gown_val",)
        predictor = DefaultPredictor(cfg)

        dataset_dicts = DatasetCatalog.get("gown_val")
        gown_val_metadata = MetadataCatalog.get("gown_val")
        for d in random.sample(dataset_dicts, 1):
            im = cv2.imread(d["file_name"])
            im_or = cv2.imread(d["file_name"])
            print(d["file_name"])
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],
                           metadata=gown_val_metadata,
                           scale=1.0,
                           instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels
                           )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            mask = outputs['instances'].pred_masks.to('cpu').numpy()[0, :, :]
            tmp = np.where(mask == True)

            # plt.imshow(out.get_image()[:, :, ::-1])
            # mask = outputs['instances'].pred_masks.to('cpu').numpy()[0,:,:]
            # tmp = np.where(mask==True)
            # print(tmp[1][0])
            # print(tmp[0][0])
            # plt.imshow(mask, alpha=0.5)
            # plt.show()

            ############################################# random
            msg_pixel = Point()
            msg_pixel.x = tmp[1][0]
            msg_pixel.y = tmp[0][0]
            msg_pixel.z = 1
            pub_pixel.publish(msg_pixel)


if __name__ == '__main__':
    np.set_printoptions(precision=6, suppress=True)
    rospy.init_node("maskrcnn")

    pub_pixel = rospy.Publisher('/fan/pixel', Point, queue_size=1)

    listener()

    rospy.spin()
