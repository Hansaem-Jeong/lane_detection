import pickle
import random
import logging

import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
import os
from natsort import natsorted
import re

#hansaem
import time


class Runner:
    def __init__(self, cfg, exp, device, test_dataset,test_first_dir,test_second_dir,exp_name,hyper,hyper_param,video_name,root_path,resume=False, view=None, deterministic=False):
        self.cfg = cfg
        self.exp = exp
        self.device = device
        self.resume = resume
        self.view = view
        self.test_dataset=test_dataset
        self.test_first_dir=test_first_dir
        self.test_second_dir=test_second_dir
        self.logger = logging.getLogger(__name__)

        print("4444444444444444444444444444444444444444444444444444")
        print(self.test_dataset)
        self.dataset_type=hyper_param[3]
        self.conf_threshold=hyper_param[0]
        self.nms_thres=hyper_param[1]
        self.nms_topk=hyper_param[2]

        self.root=root_path
        self.video_name=video_name
        self.hyper=hyper
        print(self.root)
        self.exp_name = "/{}/{}/".format(exp_name,self.hyper)
        self.name = test_first_dir + test_second_dir +test_dataset
        print(self.name)
        self.log_dir = self.name+self.exp_name#os.path.join(self.name,self.exp_name)
        print(self.log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # Fix seeds
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        random.seed(cfg['seed'])

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def train(self):
        self.exp.train_start_callback(self.cfg)
        starting_epoch = 1
        model = self.cfg.get_model()
        model = model.to(self.device)
        optimizer = self.cfg.get_optimizer(model.parameters())
        scheduler = self.cfg.get_lr_scheduler(optimizer)
        if self.resume:
            last_epoch, model, optimizer, scheduler = self.exp.load_last_train_state(model, optimizer, scheduler)
            starting_epoch = last_epoch + 1
        max_epochs = self.cfg['epochs']
        train_loader = self.get_train_dataloader()
        loss_parameters = self.cfg.get_loss_parameters()
        for epoch in trange(starting_epoch, max_epochs + 1, initial=starting_epoch - 1, total=max_epochs):
            self.exp.epoch_start_callback(epoch, max_epochs)
            model.eval()
            pbar = tqdm(train_loader)
            for i, (images, labels, _) in enumerate(pbar):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = model(images, **self.cfg.get_train_parameters())
                loss, loss_dict_i = model.loss(outputs, labels, **loss_parameters)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Scheduler step (iteration based)
                scheduler.step()

                # Log
                postfix_dict = {key: float(value) for key, value in loss_dict_i.items()}
                postfix_dict['lr'] = optimizer.param_groups[0]["lr"]
                self.exp.iter_end_callback(epoch, max_epochs, i, len(train_loader), loss.item(), postfix_dict)
                postfix_dict['loss'] = loss.item()
                pbar.set_postfix(ordered_dict=postfix_dict)
            self.exp.epoch_end_callback(epoch, max_epochs, model, optimizer, scheduler)

            # Validate
            if (epoch + 1) % self.cfg['val_every'] == 0:
                self.eval(epoch, on_val=True)
        self.exp.train_end_callback()

    def eval(self, epoch, on_val=False, save_predictions=False):
        #prediction_name="predictions_r34_culane"#
        model = self.cfg.get_model()
        model_path = self.exp.get_checkpoint_path(epoch)
        self.logger.info('Loading model %s', model_path)
        model.load_state_dict(self.exp.get_epoch_model(epoch))
        model = model.to(self.device)
        model.eval()
        if on_val and self.test_dataset ==None:
            dataloader = self.get_val_dataloader()
            print("1111111111111111111111111111111111111111111111\n")
        elif self.test_dataset !=None:
            dataloader =self.get_kodas_test_dataloader()
            print("2222222222222222222222222222222222222222222222\n")
            print(self.test_dataset)
        else:
            dataloader = self.get_test_dataloader()
            print("3333333333333333333333333333333333333333333333\n")
        test_parameters = self.cfg.get_test_parameters()
        predictions = []
        self.exp.eval_start_callback(self.cfg)
        with torch.no_grad():
            for idx, (images, _, _) in enumerate(tqdm(dataloader)):
#hansaem
#                start_times = time.time()
                images = images.to(self.device)
                output = model(images, **test_parameters)
                prediction = model.decode(output, as_lanes=True)
                predictions.extend(prediction)
#hansaem
#                cycle_times = time.time() - start_times
#                print(f"{cycle_times:.5f} sec")
                if self.view:
                    img = (images[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    img, fp, fn = dataloader.dataset.draw_annotation(idx, img=img, pred=prediction[0])
                    if self.view == 'mistakes' and fp == 0 and fn == 0:
                         continue

                    __name=self.log_dir+str(idx) + '.jpg'

                    cv2.imwrite(__name, img)
                    cv2.imshow("KODAS", img)
                    cv2.waitKey(1)
        print("hansaem hansaem hansaem\n")
        image_folder = self.log_dir
        video_name =  self.log_dir+self.video_name+'.avi'
        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images = natsorted(images)
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        video = cv2.VideoWriter(video_name, 0, 30, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()

        # if save_predictions:
        #     with open("/data2/lane_data/LaneATT/prediction/8_7/"+prediction_name+'.pkl', 'wb') as handle:
        #         pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # self.exp.eval_end_callback(dataloader.dataset.dataset, predictions, epoch)

#hansaem
#    def ros_run(self):



    def get_train_dataloader(self):
        train_dataset = self.cfg.get_dataset('train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=self.cfg['batch_size'],
                                                   shuffle=True,
                                                   num_workers=8,
                                                   worker_init_fn=self._worker_init_fn_)
        return train_loader

    def get_kodas_test_dataloader(self):
        self.cfg.set_kodas('test', self.dataset_type,self.conf_threshold, self.nms_thres, self.nms_topk,self.root)
        test_dataset = self.cfg.get_dataset('test')
        print("55555555555555555555555555555555555555555555")
        print(test_dataset)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_test_dataloader(self):
        test_dataset = self.cfg.get_dataset('test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=self.cfg['batch_size'] if not self.view else 1,
                                                  shuffle=False,
                                                  num_workers=8,
                                                  worker_init_fn=self._worker_init_fn_)
        return test_loader

    def get_val_dataloader(self):
        val_dataset = self.cfg.get_dataset('val')
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=self.cfg['batch_size'],
                                                 shuffle=False,
                                                 num_workers=8,
                                                 worker_init_fn=self._worker_init_fn_)
        return val_loader

    @staticmethod
    def _worker_init_fn_(_):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2**32 - 1
        random.seed(torch_seed)
        np.random.seed(np_seed)
