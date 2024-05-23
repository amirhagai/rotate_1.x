from numbers import Number
from typing import List, Optional, Union
import random
import cv2
import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmdet.structures.bbox import BaseBoxes, get_box_type
from mmdet.structures.mask import PolygonMasks
from mmengine.utils import is_list_of
import torch
import os
import re
from PIL import Image

from torchvision.transforms import Compose, ToTensor, ColorJitter, ToPILImage
from mmrotate.registry import TRANSFORMS
from pathlib import Path




@TRANSFORMS.register_module()
class BboxColorJitter(BaseTransform):
    """Color jitter only over bboxes

    """

    def __init__(self, prob, brightness=0, contrast=0, saturation=0, hue=0.5, class_num=5 ) -> None:
        self.prob = prob
        # self.path = path
        self.class_num = class_num 
        self.brightness, self.contrast, self.saturation, self.hue = \
            brightness, contrast, saturation, hue
        self.transform_im = \
                Compose([
                ColorJitter(
                brightness=self.brightness, contrast=self.contrast,\
                      saturation=self.saturation, hue=self.hue),

            ])
        self.transform_done_im = \
            ToPILImage()
    
    def ann_to_bbox(anno_list):
        """"
        this is how the boxes that i get looks like if anno_list is the bbox given by the annotation file
        """
        cnt = np.int0(np.array(anno_list).reshape(4, 2))
        cnt = cnt.reshape((4, 2))
        rect = cv2.minAreaRect(cnt)
        return torch.tensor([rect[0][0], rect[0][1], rect[1][0], rect[1][1], np.pi * rect[2] / 180])


    def transform(self, results: dict) -> dict:
    
        recs = results['gt_bboxes'].tensor
        labels = results['gt_bboxes_labels']
        # print(labels, results["img_path"])
        im_shape = results['img'].shape
        boxes = []
        for i, rec in enumerate(recs):
            if labels[i] == self.class_num and random.random() < self.prob:
                rec = (
                    (rec[0].item(), rec[1].item()),
                    (rec[2].item(), rec[3].item()),
                        (rec[4].item() * 180 / np.pi)
                    )         
                box = cv2.boxPoints(rec)
                box = np.int0(box)
                boxes.append(box)
        if len(boxes) > 0:
            mask = np.ones(im_shape).astype(np.uint8)
            cv2.drawContours(mask, boxes, -1, (255, 0, 0), thickness=cv2.FILLED)
            indecis = np.where(mask == 255)
            bbox_to_transform = results['img'][indecis[0], indecis[1], :].astype(np.float32) / 255.
            transformed_tensor = self.transform_im(torch.tensor(bbox_to_transform.T[:, :, None]))
            results['img'][indecis[0], indecis[1], :] = (transformed_tensor[:, :, 0].T.numpy() * 255).astype(np.uint8)
            name = results['file_name']
            Image.fromarray(results['img']).save(f'/app/data/mids/jitter_{name}')
 
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class RemoveClassesAnnotation(BaseTransform):
    """
    Remove parts of the classes

    """

    def __init__(self, classes_to_remove) -> None:
        
        self.classes = {
            'plane' : 0,
            'baseball-diamond' : 1,
            'bridge' : 2, 
            'ground-track-field' : 3,
            'small-vehicle' : 4,
            'large-vehicle' : 5,
            'ship' : 6,
            'tennis-court' : 7,
            'basketball-court' : 8, 
            'storage-tank' : 9,
            'soccer-ball-field' : 10,
            'roundabout' : 11,
            'harbor' : 12,
            'swimming-pool' : 13,
            'helicopter' :14
        }

        self.classes_to_remove = np.array([self.classes[class_name] for class_name in classes_to_remove])


    def transform(self, results: dict) -> dict:
        labels = results['gt_bboxes_labels']
        mask = np.isin(labels, self.classes_to_remove)
        indices = ~mask

        results['gt_bboxes'].tensor = results['gt_bboxes'].tensor[indices]
        results['gt_bboxes_labels'] = results['gt_bboxes_labels'][indices]
        results['gt_ignore_flags'] = results['gt_ignore_flags'][indices]
        results['instances'] = [results['instances'][i] for i in range(len(results['instances'])) if indices[i]]

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class InjectLargeVehicleData(BaseTransform):
    """Inject the 3D injection results.
       the results should be located as explain at base_path
       this transformaiton does not change the annotations at all
    Args:
        prob - probability to inject the data
        base_path - where does the data located, images should be under 
                    base_path/results['img_path']/{i}.png
                    base_path/results['img_path']/{i}_seg.png
                    
        type - the type of injection
    """

    def __init__(self, prob, base_path, injection_type) -> None:
        self.prob = prob
        self.base_path = base_path
        self.injection_type = injection_type

    def get_files(self, folder_path):

        files = os.listdir(folder_path)

        pattern_images = re.compile(r'^-?\d+(\.\d+)?\.png$')
        matching_images = [file for file in files if re.match(pattern_images, file)]

        pattern_segs = re.compile(r'^-?\d+(\.\d+)?\_seg.png$')
        matching_segs = [file for file in files if re.match(pattern_segs, file)]

        sampled_images = []
        sampled_segs = []

        # Iterate through each path in the list
        for i in range(len(matching_images)):
            # With probability p, add the path to the sampled list
            if random.random() < self.prob:
                sampled_images.append(matching_images[i])
                sampled_segs.append(matching_segs[i])
        

        return sampled_images, sampled_segs
        

    def transform(self, results: dict) -> dict:


        folder_path = f"{self.base_path }/mid_reults/{results['file_name'][:-4]}"
        if not os.path.exists(folder_path):
            return results
        sampled_images, sampled_segs = self.get_files(folder_path)

        sampled_images, sampled_segs = [np.array(Image.open(f"{folder_path}/{im}"))[:, :, ::-1] for im in sampled_images], [np.array(Image.open(f"{folder_path}/{im}"))[:, :, None] / 255 for im in sampled_segs]


        dota_np = results['img'].astype(np.float32)
        if self.injection_type == 'ycbcr':
            for i in range(len(sampled_images)):

                yuv_img = np.array(Image.fromarray(sampled_images[i]).convert('YCbCr'))
                yuv_origin = np.array(
                    Image.fromarray((sampled_segs[i] * dota_np).astype(np.uint8)).convert('YCbCr')
                )

                new_obj = np.concatenate(
                    [
                        yuv_origin[:, :, 0][:, :, None],
                        yuv_img[:, :, 1][:, :, None],
                        yuv_img[:, :, 2][:, :, None],
                    ],
                    axis=2,
                ).astype(np.uint8)

                new_obj_im = Image.fromarray(new_obj, 'YCbCr').convert('BGR')
                dota_np = (1 - sampled_segs[i]) * dota_np + sampled_segs[i] * new_obj_im

        elif self.injection_type == "simple":

            for i in range(len(sampled_images)):
                dota_np = (1 - sampled_segs[i]) * dota_np + sampled_segs[i] * sampled_images[i]

        results['img'] = dota_np.astype(np.uint8)
        name = results['file_name']
        Image.fromarray(results['img']).save(f'/app/data/mids/inject_{self.injection_type}_{name}')

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class ConvertBoxType(BaseTransform):
    """Convert boxes in results to a certain box type.

    Args:
        box_type_mapping (dict): A dictionary whose key will be used to search
            the item in `results`, the value is the destination box type.
    """

    def __init__(self, box_type_mapping: dict) -> None:
        self.box_type_mapping = box_type_mapping

    def transform(self, results: dict) -> dict:
        """The transform function."""
        for key, dst_box_type in self.box_type_mapping.items():
            if key not in results:
                continue
            assert isinstance(results[key], BaseBoxes), \
                f"results['{key}'] not a instance of BaseBoxes."
            results[key] = results[key].convert_to(dst_box_type)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(box_type_mapping={self.box_type_mapping})'
        return repr_str


@TRANSFORMS.register_module()
class Rotate(BaseTransform):
    """Rotate the images, bboxes, masks and segmentation map by a certain
    angle. Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)
    Modified Keys:
    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    Added Keys:
    - homography_matrix
    Args:
        rotate_angle (int): An angle to rotate the image.
        img_border_value (int or float or tuple): The filled values for
            image border. If float, the same fill value will be used for
            all the three channels of image. If tuple, it should be 3 elements.
            Defaults to 0.
        mask_border_value (int): The fill value used for masks. Defaults to 0.
        seg_ignore_label (int): The fill value used for segmentation map.
            Note this value must equals ``ignore_label`` in ``semantic_head``
            of the corresponding config. Defaults to 255.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 rotate_angle: int,
                 img_border_value: Union[int, float, tuple] = 0,
                 mask_border_value: int = 0,
                 seg_ignore_label: int = 255,
                 interpolation: str = 'bilinear') -> None:
        if isinstance(img_border_value, (float, int)):
            img_border_value = tuple([float(img_border_value)] * 3)
        elif isinstance(img_border_value, tuple):
            assert len(img_border_value) == 3, \
                f'img_border_value as tuple must have 3 elements, ' \
                f'got {len(img_border_value)}.'
            img_border_value = tuple([float(val) for val in img_border_value])
        else:
            raise ValueError(
                'img_border_value must be float or tuple with 3 elements.')
        self.rotate_angle = rotate_angle
        self.img_border_value = img_border_value
        self.mask_border_value = mask_border_value
        self.seg_ignore_label = seg_ignore_label
        self.interpolation = interpolation

    def _get_homography_matrix(self, results: dict) -> np.ndarray:
        """Get the homography matrix for Rotate."""
        img_shape = results['img_shape']
        center = ((img_shape[1] - 1) * 0.5, (img_shape[0] - 1) * 0.5)
        cv2_rotation_matrix = cv2.getRotationMatrix2D(center,
                                                      -self.rotate_angle, 1.0)
        return np.concatenate(
            [cv2_rotation_matrix,
             np.array([0, 0, 1]).reshape((1, 3))],
            dtype=np.float32)

    def _record_homography_matrix(self, results: dict) -> None:
        """Record the homography matrix for the geometric transformation."""
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = self.homography_matrix
        else:
            results['homography_matrix'] = self.homography_matrix @ results[
                'homography_matrix']

    def _transform_img(self, results: dict) -> None:
        """Rotate the image."""
        results['img'] = mmcv.imrotate(
            results['img'],
            self.rotate_angle,
            border_value=self.img_border_value,
            interpolation=self.interpolation)

    def _transform_masks(self, results: dict) -> None:
        """Rotate the masks."""
        results['gt_masks'] = results['gt_masks'].rotate(
            results['img_shape'],
            self.rotate_angle,
            border_value=self.mask_border_value,
            interpolation=self.interpolation)

    def _transform_seg(self, results: dict) -> None:
        """Rotate the segmentation map."""
        results['gt_seg_map'] = mmcv.imrotate(
            results['gt_seg_map'],
            self.rotate_angle,
            border_value=self.seg_ignore_label,
            interpolation='nearest')

    def _transform_bboxes(self, results: dict) -> None:
        """Rotate the bboxes."""
        if len(results['gt_bboxes']) == 0:
            return
        img_shape = results['img_shape']
        center = (img_shape[1] * 0.5, img_shape[0] * 0.5)
        results['gt_bboxes'].rotate_(center, self.rotate_angle)
        results['gt_bboxes'].clip_(img_shape)

    def _filter_invalid(self, results: dict) -> None:
        """Filter invalid data w.r.t `gt_bboxes`"""
        # results['img_shape'] maybe (h,w,c) or (h,w)
        height, width = results['img_shape'][:2]
        if 'gt_bboxes' in results:
            if len(results['gt_bboxes']) == 0:
                return
            bboxes = results['gt_bboxes']
            valid_index = results['gt_bboxes'].is_inside([height,
                                                          width]).numpy()
            results['gt_bboxes'] = bboxes[valid_index]

            # ignore_flags
            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_index]

            # labels
            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                    valid_index]

            # mask fields
            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_index.nonzero()[0]]

    def transform(self, results: dict) -> dict:
        """The transform function."""
        self.homography_matrix = self._get_homography_matrix(results)
        self._record_homography_matrix(results)
        self._transform_img(results)
        if results.get('gt_bboxes', None) is not None:
            self._transform_bboxes(results)
        if results.get('gt_masks', None) is not None:
            self._transform_masks(results)
        if results.get('gt_seg_map', None) is not None:
            self._transform_seg(results)
        self._filter_invalid(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(rotate_angle={self.rotate_angle}, '
        repr_str += f'img_border_value={self.img_border_value}, '
        repr_str += f'mask_border_value={self.mask_border_value}, '
        repr_str += f'seg_ignore_label={self.seg_ignore_label}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class RandomRotate(BaseTransform):
    """Random rotate image & bbox & masks. The rotation angle will choice in.

    [-angle_range, angle_range). Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)
    Modified Keys:
    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    Added Keys:
    - homography_matrix
    Args:
        prob (float): The probability of whether to rotate or not. Defaults
            to 0.5.
        angle_range (int): The maximum range of rotation angle. The rotation
            angle will lie in [-angle_range, angle_range). Defaults to 180.
        rect_obj_labels (List[int], Optional): A list of labels whose
            corresponding objects are alwags horizontal. If
            results['gt_bboxes_labels'] has any label in ``rect_obj_labels``,
            the rotation angle will only be choiced from [90, 180, -90, -180].
            Defaults to None.
        rotate_type (str): The type of rotate class to use. Defaults to
            "Rotate".
        **rotate_kwargs: Other keyword arguments for the ``rotate_type``.
    """

    def __init__(self,
                 prob: float = 0.5,
                 angle_range: int = 180,
                 rect_obj_labels: Optional[List[int]] = None,
                 rotate_type: str = 'Rotate',
                 **rotate_kwargs) -> None:
        assert 0 < angle_range <= 180
        self.prob = prob
        self.angle_range = angle_range
        self.rect_obj_labels = rect_obj_labels
        self.rotate_cfg = dict(type=rotate_type, **rotate_kwargs)
        self.rotate = TRANSFORMS.build({'rotate_angle': 0, **self.rotate_cfg})
        self.horizontal_angles = [90, 180, -90, -180]

    @cache_randomness
    def _random_angle(self) -> int:
        """Random angle."""
        return self.angle_range * (2 * np.random.rand() - 1)

    @cache_randomness
    def _random_horizontal_angle(self) -> int:
        """Random horizontal angle."""
        return np.random.choice(self.horizontal_angles)

    @cache_randomness
    def _is_rotate(self) -> bool:
        """Randomly decide whether to rotate."""
        return np.random.rand() < self.prob

    def transform(self, results: dict) -> dict:
        """The transform function."""
        if not self._is_rotate():
            return results

        rotate_angle = self._random_angle()
        if self.rect_obj_labels is not None and 'gt_bboxes_labels' in results:
            for label in self.rect_obj_labels:
                if (results['gt_bboxes_labels'] == label).any():
                    rotate_angle = self._random_horizontal_angle()
                    break

        self.rotate.rotate_angle = rotate_angle
        return self.rotate(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'rotate_angle={self.angle_range}, '
        repr_str += f'rect_obj_labels={self.rect_obj_labels}, '
        repr_str += f'rotate_cfg={self.rotate_cfg})'
        return repr_str


@TRANSFORMS.register_module()
class RandomChoiceRotate(BaseTransform):
    """Random rotate image & bbox & masks from a list of angles. Rotation angle
    will be randomly choiced from ``angles``. Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)
    Modified Keys:
    - img
    - gt_bboxes
    - gt_masks
    - gt_seg_map
    Added Keys:
    - homography_matrix
    Args:
        angles (list[int]): Angles for rotation. 0 is the default value for
            non-rotation and shouldn't be included in ``angles``.
        prob (float or list[float]): If ``prob`` is a float, it is the
            probability of whether to rotate. If ``prob`` is a list, it is
            the probabilities of each rotation angle in ``angles``.
        rect_obj_labels (List[int]): A list of labels whose corresponding
            objects are alwags horizontal. If results['gt_bboxes_labels'] has
            any label in ``rect_obj_labels``, the rotation angle will only be
            choiced from [90, 180, -90, -180].
        rotate_type (str): The type of rotate class to use. Defaults to
            "Rotate".
        **rotate_kwargs: Other keyword arguments for the ``rotate_type``.
    """

    def __init__(self,
                 angles,
                 prob: Union[float, List[float]] = 0.5,
                 rect_obj_labels=None,
                 rotate_type='Rotate',
                 **rotate_kwargs) -> None:
        if isinstance(prob, list):
            assert is_list_of(prob, Number)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, Number):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f'probs must be number or list of number, but \
                              got `{type(prob)}`.')
        self.prob = prob

        assert isinstance(angles, list) and is_list_of(angles, int)
        assert 0 not in angles
        self.angles = angles
        if isinstance(self.prob, list):
            assert len(self.prob) == len(self.angles)

        self.rect_obj_labels = rect_obj_labels
        self.rotate_cfg = dict(type=rotate_type, **rotate_kwargs)
        self.rotate = TRANSFORMS.build({'rotate_angle': 0, **self.rotate_cfg})
        self.horizontal_angles = [90, 180, -90, -180]

    @cache_randomness
    def _choice_angle(self) -> int:
        """Choose the angle."""
        angle_list = self.angles + [0]
        if isinstance(self.prob, list):
            non_prob = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        else:
            non_prob = 1. - self.prob
            single_ratio = self.prob / (len(angle_list) - 1)
            prob_list = [single_ratio] * (len(angle_list) - 1) + [non_prob]
        angle = np.random.choice(angle_list, p=prob_list)
        return angle

    @cache_randomness
    def _random_horizontal_angle(self) -> int:
        """Random horizontal angle."""
        return np.random.choice(self.horizontal_angles)

    def transform(self, results: dict) -> dict:
        """The transform function."""
        rotate_angle = self._choice_angle()
        if rotate_angle == 0:
            return results

        if self.rect_obj_labels is not None and 'gt_bboxes_labels' in results:
            for label in self.rect_obj_labels:
                if (results['gt_bboxes_labels'] == label).any():
                    rotate_angle = self._random_horizontal_angle()
                    break

        self.rotate.rotate_angle = rotate_angle
        return self.rotate(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(angles={self.angles}, '
        repr_str += f'prob={self.prob}, '
        repr_str += f'rect_obj_labels={self.rect_obj_labels}, '
        repr_str += f'rotate_cfg={self.rotate_cfg})'
        return repr_str


@TRANSFORMS.register_module()
class ConvertMask2BoxType(BaseTransform):
    """Convert masks in results to a certain box type.

    Required Keys:

    - ori_shape
    - gt_bboxes (BaseBoxes[torch.float32])
    - gt_masks (BitmapMasks | PolygonMasks)
    - instances (List[dict]) (optional)
    Modified Keys:
    - gt_bboxes
    - gt_masks
    - instances

    Args:
        box_type (str): The destination box type.
        keep_mask (bool): Whether to keep the ``gt_masks``.
            Defaults to False.
    """

    def __init__(self, box_type: str, keep_mask: bool = False) -> None:
        _, self.box_type_cls = get_box_type(box_type)
        assert hasattr(self.box_type_cls, 'from_instance_masks')
        self.keep_mask = keep_mask

    def transform(self, results: dict) -> dict:
        """The transform function."""
        assert 'gt_masks' in results.keys()
        masks = results['gt_masks']
        results['gt_bboxes'] = self.box_type_cls.from_instance_masks(masks)
        if not self.keep_mask:
            results.pop('gt_masks')

        # Modify results['instances'] for RotatedCocoMetric
        converted_instances = []
        for instance in results['instances']:
            m = np.array(instance['mask'][0])
            m = PolygonMasks([[m]], results['ori_shape'][1],
                             results['ori_shape'][0])
            instance['bbox'] = self.box_type_cls.from_instance_masks(
                m).tensor[0].numpy().tolist()
            if not self.keep_mask:
                instance.pop('mask')
            converted_instances.append(instance)
        results['instances'] = converted_instances

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(box_type_cls={self.box_type_cls}, '
        repr_str += f'keep_mask={self.keep_mask})'
        return repr_str
