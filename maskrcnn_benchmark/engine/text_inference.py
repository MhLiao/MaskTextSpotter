# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import tempfile
import time
import os
from collections import OrderedDict

import torch

from tqdm import tqdm

from ..structures.bounding_box import BoxList
from ..utils.comm import is_main_process
from ..utils.comm import scatter_gather
from ..utils.comm import synchronize
import torch.distributed as dist

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.utils.chars import getstr_grid, get_tight_rect, char2num
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pickle

# def compute_on_dataset(model, data_loader, device):
# 	model.eval()
# 	results_dict = {}
# 	cpu_device = torch.device("cpu")
# 	for i, batch in tqdm(enumerate(data_loader)):
# 		images, targets, image_paths = batch
# 		images = images.to(device)
# 		with torch.no_grad():
# 			predictions = model(images)
# 			if predictions is not None:
# 				global_predictions = predictions[0]
# 				char_predictions = predictions[1]
# 				words = char_predictions['texts']
# 				rec_scores = char_predictions['rec_scores']
# 				char_scores = char_predictions['rec_char_scores']
# 				seq_words = char_predictions['seq_outputs']
# 				seq_scores = char_predictions['seq_scores']
# 				detailed_seq_scores = char_predictions['detailed_seq_scores']
# 				global_predictions = [o.to(cpu_device) for o in global_predictions]
# 				results_dict.update(
# 					{image_paths[0]: [global_predictions[0], words, rec_scores, char_scores, seq_words, seq_scores, detailed_seq_scores]}
# 				)
# 	return results_dict

def compute_on_dataset(model, data_loader, device):
	model.eval()
	results_dict = {}
	cpu_device = torch.device("cpu")
	for i, batch in tqdm(enumerate(data_loader)):
		images, targets, image_paths = batch
		images = images.to(device)
		with torch.no_grad():
			predictions = model(images)
			if predictions is not None:
				global_predictions = predictions[0]
				char_predictions = predictions[1]
				char_mask = char_predictions['char_mask']
				boxes = char_predictions['boxes']
				seq_words = char_predictions['seq_outputs']
				seq_scores = char_predictions['seq_scores']
				detailed_seq_scores = char_predictions['detailed_seq_scores']
				global_predictions = [o.to(cpu_device) for o in global_predictions]
				results_dict.update(
					{image_paths[0]: [global_predictions[0], char_mask, boxes, seq_words, seq_scores, detailed_seq_scores]}
				)
	return results_dict

def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points,key = lambda x:x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    if px1<0:
        px1=1
    if px1>image_width:
        px1 = image_width - 1
    if px2<0:
        px2=1
    if px2>image_width:
        px2 = image_width - 1
    if px3<0:
        px3=1
    if px3>image_width:
        px3 = image_width - 1
    if px4<0:
        px4=1
    if px4>image_width:
        px4 = image_width - 1

    if py1<0:
        py1=1
    if py1>image_height:
        py1 = image_height - 1
    if py2<0:
        py2=1
    if py2>image_height:
        py2 = image_height - 1
    if py3<0:
        py3=1
    if py3>image_height:
        py3 = image_height - 1
    if py4<0:
        py4=1
    if py4>image_height:
        py4 = image_height - 1
    return [px1, py1, px2, py2, px3, py3, px4, py4]

def mask2polygon(mask, box, im_size, threshold=0.5, output_folder=None):
	# mask 32*128
	image_width, image_height = im_size[0], im_size[1]
	box_h = box[3] - box[1]
	box_w = box[2] - box[0]
	cls_polys = (mask*255).astype(np.uint8)
	poly_map = np.array(Image.fromarray(cls_polys).resize((box_w, box_h)))
	poly_map = poly_map.astype(np.float32) / 255
	poly_map=cv2.GaussianBlur(poly_map,(3,3),sigmaX=3)
	ret, poly_map = cv2.threshold(poly_map,0.5,1,cv2.THRESH_BINARY)
	if 'total_text' in output_folder or 'cute80' in output_folder:
		SE1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
		poly_map = cv2.erode(poly_map,SE1) 
		poly_map = cv2.dilate(poly_map,SE1);
		poly_map = cv2.morphologyEx(poly_map,cv2.MORPH_CLOSE,SE1)
		try:
			 _, contours, _ = cv2.findContours((poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		except:
			contours, _ = cv2.findContours((poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		if len(contours)==0:
			print(contours)
			print(len(contours))
			return None
		max_area=0
		max_cnt = contours[0]
		for cnt in contours:
			area=cv2.contourArea(cnt)
			if area > max_area:
				max_area = area
				max_cnt = cnt
		perimeter = cv2.arcLength(max_cnt,True)
		epsilon = 0.01*cv2.arcLength(max_cnt,True)
		approx = cv2.approxPolyDP(max_cnt,epsilon,True)
		pts = approx.reshape((-1,2))
		pts[:,0] = pts[:,0] + box[0]
		pts[:,1] = pts[:,1] + box[1]
		polygon = list(pts.reshape((-1,)))
		polygon = list(map(int, polygon))
		if len(polygon)<6:
			return None     
	else:      
		SE1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
		poly_map = cv2.erode(poly_map,SE1) 
		poly_map = cv2.dilate(poly_map,SE1);
		poly_map = cv2.morphologyEx(poly_map,cv2.MORPH_CLOSE,SE1)
		idy,idx=np.where(poly_map == 1)
		xy=np.vstack((idx,idy))
		xy=np.transpose(xy)
		hull = cv2.convexHull(xy, clockwise=True)
		#reverse order of points.
		if  hull is None:
			return None
		hull=hull[::-1]
		#find minimum area bounding box.
		rect = cv2.minAreaRect(hull)
		corners = cv2.boxPoints(rect)
		corners = np.array(corners, dtype="int")
		pts = get_tight_rect(corners, box[0], box[1], image_height, image_width, 1)
		polygon = [x * 1.0 for x in pts]
		polygon = list(map(int, polygon))
	return polygon


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
	all_predictions = scatter_gather(predictions_per_gpu)
	if not is_main_process():
		return
	# merge the list of dicts
	predictions = {}
	for p in all_predictions:
		predictions.update(p)
	return predictions

def format_output(out_dir, boxes, img_name):
	res = open(os.path.join(out_dir, 'res_' + img_name.split('.')[0] + '.txt'), 'wt')
	## char score save dir
	ssur_name = os.path.join(out_dir, 'res_' + img_name.split('.')[0])
	for i, box in enumerate(boxes):
		save_name = ssur_name + '_' + str(i) + '.pkl'
		save_dict = {}
		if 'total_text' in out_dir or 'cute80' in out_dir:
			# np.save(save_name, box[-2])
			save_dict['seg_char_scores'] = box[-3]
			save_dict['seq_char_scores'] = box[-2]
			box = ','.join([str(x) for x in box[:4]]) + ';' + ','.join([str(x) for x in box[4:4+int(box[-1])]]) + ';' + ','.join([str(x) for x in box[4+int(box[-1]):-3]]) + ',' + save_name
		else:
			save_dict['seg_char_scores'] = box[-2]
			save_dict['seq_char_scores'] = box[-1]
			np.save(save_name, box[-1])
			box = ','.join([str(x) for x in box[:-2]]) + ',' + save_name
		with open(save_name,'wb') as f:
			pickle.dump(save_dict, f, protocol = 2)
		res.write(box + '\n')
	res.close()

def process_char_mask(char_masks, boxes, threshold=192):
    texts, rec_scores, rec_char_scores, char_polygons = [], [], [], []
    for index in range(char_masks.shape[0]):
        box = list(boxes[index])
        box = list(map(int, box))
        text, rec_score, rec_char_score, char_polygon = getstr_grid(char_masks[index,:,:,:].copy(), box, threshold=threshold)
        texts.append(text)
        rec_scores.append(rec_score)
        rec_char_scores.append(rec_char_score)
        char_polygons.append(char_polygon)
        # segmss.append(segms)
    return texts, rec_scores, rec_char_scores, char_polygons

# def prepare_results_for_evaluation(predictions, output_folder, model_name):
# 	results_dir = os.path.join(output_folder, model_name+'_results')
# 	if not os.path.isdir(results_dir):
# 		os.mkdir(results_dir)
# 	for image_path, prediction in predictions.items():
# 		im_name = image_path.split('/')[-1]
# 		global_prediction, words, rec_scores, char_scores, seq_words, seq_scores, detailed_seq_scores = prediction[0], prediction[1], prediction[2], prediction[3], prediction[4], prediction[5], prediction[6]
# 		# print(detailed_seq_scores.shape)
# 		img = Image.open(image_path)
# 		width, height = img.size
# 		global_prediction = global_prediction.resize((width, height))
# 		boxes = global_prediction.bbox.tolist()
# 		scores = global_prediction.get_field("scores").tolist()
# 		masks = global_prediction.get_field("mask").cpu().numpy()
# 		result_logs = []
# 		for k, box in enumerate(boxes):
# 			box = list(map(int, box))
# 			mask = masks[k,0,:,:]
# 			polygon = mask2polygon(mask, box, img.size, threshold=0.5, output_folder=output_folder)
# 			if polygon is None:
# 				polygon = [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]
# 			score = scores[k]
# 			word = words[k]
# 			rec_score = rec_scores[k]
# 			char_score = char_scores[k]
# 			seq_word = seq_words[k]
# 			seq_char_scores = seq_scores[k]
# 			seq_score = sum(seq_char_scores) / float(len(seq_char_scores))
# 			# detailed_seq_score = detailed_seq_scores[k]
# 			# print(detailed_seq_score.shape)
# 			if 'total_text' in output_folder or 'cute80' in output_folder:
# 				result_log = [int(x * 1.0) for x in box[:4]] + polygon + [word] + [seq_word] + [score] + [rec_score] + [seq_score] + [char_score] + [detailed_seq_score] + [len(polygon)]
# 			else:
# 				result_log = [int(x * 1.0) for x in box[:4]] + polygon + [word] + [seq_word] + [score] + [rec_score] + [seq_score] + [char_score] + [detailed_seq_score] 
# 			result_logs.append(result_log)
# 		format_output(results_dir, result_logs, im_name)

def creat_color_map(n_class, width):
    splits = int(np.ceil(np.power((n_class * 1.0), 1.0 / 3)))
    maps = []
    for i in range(splits):
        r = int(i * width * 1.0 / (splits-1))
        for j in range(splits):
            g = int(j * width * 1.0 / (splits-1))
            for k in range(splits-1):
                b = int(k * width * 1.0 / (splits-1))
                maps.append((r, g, b, 200))
    return maps

def visualization(image, polygons, char_polygons, words, resize_ratio, colors):
	draw = ImageDraw.Draw(image, 'RGBA')
	for polygon in polygons:
		draw.polygon(polygon, fill=None, outline=(0, 255, 0, 255))
	for i, char_polygon in enumerate(char_polygons):
		for j, polygon in enumerate(char_polygon):
			polygon = [int(x*resize_ratio) for x in polygon]
			char = words[i][j]
			color = colors[char2num(char)]
			draw.polygon(polygon, fill=color, outline=color)

def prepare_results_for_evaluation(predictions, output_folder, model_name, vis=False):
	results_dir = os.path.join(output_folder, model_name+'_results')
	if not os.path.isdir(results_dir):
		os.mkdir(results_dir)
	if vis:
		visu_dir = os.path.join(output_folder, model_name+'_visu')
		if not os.path.isdir(visu_dir):
			os.mkdir(visu_dir)
	for image_path, prediction in predictions.items():
		im_name = image_path.split('/')[-1]
		global_prediction, char_mask, boxes_char, seq_words, seq_scores, detailed_seq_scores = prediction[0], prediction[1], prediction[2], prediction[3], prediction[4], prediction[5]
		words, rec_scores, rec_char_scoress, char_polygons = process_char_mask(char_mask, boxes_char)
		test_image_width, test_image_height = global_prediction.size
		img = Image.open(image_path)
		width, height = img.size
		resize_ratio = float(height) / test_image_height
		global_prediction = global_prediction.resize((width, height))
		boxes = global_prediction.bbox.tolist()
		scores = global_prediction.get_field("scores").tolist()
		masks = global_prediction.get_field("mask").cpu().numpy()
		result_logs = []
		polygons = []
		for k, box in enumerate(boxes):
			box = list(map(int, box))
			mask = masks[k,0,:,:]
			polygon = mask2polygon(mask, box, img.size, threshold=0.5, output_folder=output_folder)
			if polygon is None:
				polygon = [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]
			polygons.append(polygon)
			score = scores[k]
			word = words[k]
			rec_score = rec_scores[k]
			char_score = rec_char_scoress[k]
			seq_word = seq_words[k]
			seq_char_scores = seq_scores[k]
			seq_score = sum(seq_char_scores) / float(len(seq_char_scores))
			detailed_seq_score = detailed_seq_scores[k]
			detailed_seq_score = np.squeeze(np.array(detailed_seq_score), axis=1)
			if 'total_text' in output_folder or 'cute80' in output_folder:
				result_log = [int(x * 1.0) for x in box[:4]] + polygon + [word] + [seq_word] + [score] + [rec_score] + [seq_score] + [char_score] + [detailed_seq_score] + [len(polygon)]
			else:
				result_log = [int(x * 1.0) for x in box[:4]] + polygon + [word] + [seq_word] + [score] + [rec_score] + [seq_score] + [char_score] + [detailed_seq_score] 
			result_logs.append(result_log)
		if vis:
			colors = creat_color_map(37, 255)
			visualization(img, polygons, char_polygons, words, resize_ratio, colors)
			img.save(os.path.join(visu_dir,im_name))
		format_output(results_dir, result_logs, im_name)

def inference(
	model,
	data_loader,
	iou_types=("bbox",),
	box_only=False,
	device="cuda",
	expected_results=(),
	expected_results_sigma_tol=4,
	output_folder=None,
	model_name=None,
	cfg=None
):

	# convert to a torch.device for efficiency
	model_name = model_name.split('.')[0] + '_' + str(cfg.INPUT.MIN_SIZE_TEST)
	predictions_path = os.path.join(output_folder, model_name + '_predictions.pth')
	if os.path.isfile(predictions_path):
		predictions = torch.load(predictions_path)
	else:
		device = torch.device(device)
		num_devices = (
			dist.get_world_size()
			if dist.is_initialized()
			else 1
		)
		logger = logging.getLogger("maskrcnn_benchmark.inference")
		dataset = data_loader.dataset
		logger.info("Start evaluation on {} images".format(len(dataset)))
		start_time = time.time()
		predictions = compute_on_dataset(model, data_loader, device)
		# wait for all processes to complete before measuring the time
		# synchronize()
		total_time = time.time() - start_time
		total_time_str = str(datetime.timedelta(seconds=total_time))
		logger.info(
			"Total inference time: {} ({} s / img per device, on {} devices)".format(
				total_time_str, total_time * num_devices / len(dataset), num_devices
			)
		)

		# predictions = _accumulate_predictions_from_multiple_gpus(predictions)
		# if not is_main_process():
		# 	return

		if output_folder:
			torch.save(predictions, predictions_path)

	prepare_results_for_evaluation(predictions, output_folder, model_name, vis=cfg.TEST.VIS)