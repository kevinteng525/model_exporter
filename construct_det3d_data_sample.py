import torch
from mmengine.structures import InstanceData

from mmdet3d.structures import Det3DDataSample
from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes

data_sample = Det3DDataSample()
meta_info = dict(
     img_shape=(800, 1196, 3),
     pad_shape=(800, 1216, 3))
gt_instances_3d = InstanceData(metainfo=meta_info)
gt_instances_3d.bboxes_3d = BaseInstance3DBoxes(torch.rand((5, 7)))
gt_instances_3d.labels_3d = torch.randint(0, 3, (5,))
data_sample.gt_instances_3d = gt_instances_3d
assert 'img_shape' in data_sample.gt_instances_3d.metainfo_keys()
len(data_sample.gt_instances_3d)
5
print(data_sample)

pred_instances = InstanceData(metainfo=meta_info)
pred_instances.bboxes = torch.rand((5, 4))
pred_instances.scores = torch.rand((5, ))
data_sample = Det3DDataSample(pred_instances=pred_instances)
assert 'pred_instances' in data_sample

pred_instances_3d = InstanceData(metainfo=meta_info)
pred_instances_3d.bboxes_3d = BaseInstance3DBoxes(
     torch.rand((5, 7)))
pred_instances_3d.scores_3d = torch.rand((5, ))
pred_instances_3d.labels_3d = torch.rand((5, ))
data_sample = Det3DDataSample(pred_instances_3d=pred_instances_3d)
assert 'pred_instances_3d' in data_sample

data_sample = Det3DDataSample()
gt_instances_3d_data = dict(
     bboxes_3d=BaseInstance3DBoxes(torch.rand((2, 7))),
     labels_3d=torch.rand(2))
gt_instances_3d = InstanceData(**gt_instances_3d_data)
data_sample.gt_instances_3d = gt_instances_3d
assert 'gt_instances_3d' in data_sample
assert 'bboxes_3d' in data_sample.gt_instances_3d

from mmdet3d.structures import PointData
data_sample = Det3DDataSample()
gt_pts_seg_data = dict(
     pts_instance_mask=torch.rand(2),
     pts_semantic_mask=torch.rand(2))
data_sample.gt_pts_seg = PointData(**gt_pts_seg_data)
print(data_sample)