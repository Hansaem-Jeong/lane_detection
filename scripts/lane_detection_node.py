#!/usr/bin/env python3

import rospy
import ros_numpy 

import logging
import argparse
import os
import torch

from lib.config import Config
from lib.experiment import Experiment
from lib.runner import Runner
import yaml
from natsort import natsorted
import re
import numpy as np
import cv2

import sys
sys.path.append("/home/avees/hansaem/lane_ws/src/lane_detection/")
sys.path.append("/home/avees/hansaem/lane_ws/src/lane_detection/scripts")


class LaneDetection:
    def __init__(self): #parse_args():
        print("Init()")
        rospy.init_node('lane_detection_node', anonymous = True)
    
        namesp = argparse.Namespace()
    
        setattr(namesp, 'mode', rospy.get_param("mode", 'test'))
        setattr(namesp, 'exp_name', rospy.get_param("~exp_name"))
        setattr(namesp, 'cfg', rospy.get_param("cfg", None))
        setattr(namesp, 'resume', rospy.get_param("resume", False))
        setattr(namesp, 'epoch', rospy.get_param("epoch", None))
        setattr(namesp, 'cpu', rospy.get_param("cpu", False))
        setattr(namesp, 'save_predictions', rospy.get_param("save_predictions", False))
        setattr(namesp, 'view', rospy.get_param("view", all))
        setattr(namesp, 'test_first_dir', rospy.get_param("~test_first_dir"))
        setattr(namesp, 'test_second_dir', rospy.get_param("~test_second_dir"))
        setattr(namesp, 'test_dataset', rospy.get_param("~test_dataset"))
        setattr(namesp, 'video_name', rospy.get_param("~video_name"))
        setattr(namesp, 'conf_threshold', rospy.get_param("conf_threshold", 0.4))
        setattr(namesp, 'nms_thres', rospy.get_param("nms_thres", 45.))
        setattr(namesp, 'max_lane', rospy.get_param("max_lane", 2))
        setattr(namesp, 'data_dir', rospy.get_param("~data_dir"))
        setattr(namesp, 'deterministic', rospy.get_param("deterministic", False))
        setattr(namesp, 'exps_basedir', rospy.get_param("~exps_basedir"))
        setattr(namesp, 'package_path', rospy.get_param("~package_path"))
    
        #args = parser.parse_args()
        args = namesp
    
        if args.cfg is None and args.mode == "train":
            raise Exception("If you are training, you have to set a config file using --cfg /path/to/your/config.yaml")
        if args.resume and args.mode == "test":
            raise Exception("args.resume is set on `test` mode: can't resume testing")
        if args.epoch is not None and args.mode == 'train':
            raise Exception("The `epoch` parameter should not be set when training")
        if args.view is not None and args.mode != "test":
            raise Exception('Visualization is only available during evaluation')
        if args.cpu:
            raise Exception("CPU training/testing is not supported: the NMS procedure is only implemented for CUDA")
        self.args = args
    
        #return args
    def main(self):
        print("--------------------")
        print("Init()")
        print(self.args)
        args=self.args
        print("--------------------")
        #os.environ["CUDA_VISIBLE_DEVICES"] = '1'
        hyper="conf_threshold_{}_nms_thres_{}_max_lane_{}".format(args.conf_threshold,args.nms_thres,args.max_lane)
        print("{}_{}".format(args.exp_name,hyper))
        hyper_param = [args.conf_threshold, args.nms_thres, args.max_lane, args.test_dataset]
        exp = Experiment(args.exp_name, args, mode=args.mode, exps_basedir=args.exps_basedir)
        if args.cfg is None:
            cfg_path = exp.cfg_path
        else:
            cfg_path = args.cfg
        args.video_name=hyper
        cfg = Config(cfg_path)
        exp.set_cfg(cfg, override=False)
        device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')
        runner = Runner(cfg, exp, device,args.test_dataset,args.test_first_dir,args.test_second_dir,args.exp_name,hyper,hyper_param, args.video_name,args.data_dir, view=args.view, resume=args.resume, deterministic=args.deterministic)
        if args.mode == 'train':
            try:
                runner.train()
            except KeyboardInterrupt:
                logging.info('Training interrupted.')
        runner.eval(epoch=args.epoch or exp.get_last_checkpoint_epoch(), save_predictions=args.save_predictions)
        #runner._eval(epoch=args.epoch or exp.get_last_checkpoint_epoch(), save_predictions=args.save_predictions)
    

if __name__ == '__main__':
    #main()
    lane_detection = LaneDetection()
    lane_detection.main()
