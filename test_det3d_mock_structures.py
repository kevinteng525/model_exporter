#!/usr/bin/env python3
"""
Tests for InputFlattener with Mock MMDetection3D Data Structures

This file uses mock implementations to avoid NumPy compatibility issues.
"""

import sys
import torch
import numpy as np
from typing import Dict, List, Any
import traceback

# Add the refined directory to path
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/refined')

# Import InputFlattener
from improved_exporter import InputFlattener


# Mock implementations of MMDetection3D structures
class MockLiDARInstance3DBoxes:
    """Mock LiDAR 3D bounding boxes"""
    def __init__(self, tensor):
        self.tensor = tensor
        # Generate all the properties that real LiDARInstance3DBoxes would have
        with torch.no_grad():
            self.dims = torch.abs(tensor[:, 3:6])  # w, l, h
            self.yaw = tensor[:, 6].fmod(2 * np.pi)
            self.bottom_center = tensor[:, :3]
            self.center = self.bottom_center + torch.stack([torch.zeros_like(tensor[:, 0]),
                                                          torch.zeros_like(tensor[:, 0]),
                                                          self.dims[:, 2] / 2], dim=1)
            self.gravity_center = self.center
            self.bottom_height = tensor[:, 2] - self.dims[:, 2] / 2
            self.top_height = tensor[:, 2] + self.dims[:, 2] / 2
            self.volume = self.dims.prod(dim=1)
            # BEV (bird's eye view) corners
            self.bev = torch.stack([self.bottom_center[:, 0] - self.dims[:, 1]/2,
                                   self.bottom_center[:, 1] - self.dims[:, 0]/2,
                                   self.bottom_center[:, 0] + self.dims[:, 1]/2,
                                   self.bottom_center[:, 1] + self.dims[:, 0]/2,
                                   self.yaw], dim=1)
            self.nearest_bev = self.bev[:, :4]
            # Generate 3D corners (8 corners per box)
            corners = []
            for i in range(tensor.shape[0]):
                x, y, z, w, l, h, yaw = tensor[i]
                # Simple corner generation
                corners_i = torch.tensor([
                    [x - w/2, y - l/2, z - h/2], [x + w/2, y - l/2, z - h/2],
                    [x + w/2, y + l/2, z - h/2], [x - w/2, y + l/2, z - h/2],
                    [x - w/2, y - l/2, z + h/2], [x + w/2, y - l/2, z + h/2],
                    [x + w/2, y + l/2, z + h/2], [x - w/2, y + l/2, z + h/2]
                ])
                corners.append(corners_i)
            self.corners = torch.stack(corners)


class MockInstanceData:
    """Mock InstanceData class"""
    def __init__(self, **kwargs):
        # Store attributes directly on the object to be found by InputFlattener
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Add some dummy attributes to increase dir() length
        self.__dummy1 = None
        self.__dummy2 = None
        self.__dummy3 = None
        self.__dummy4 = None
        self.__dummy5 = None
        self.__dummy6 = None
        self.__dummy7 = None
        self.__dummy8 = None
        self._method = lambda: None  # A dummy method that should be skipped


class MockPointData:
    """Mock PointData class"""
    def __init__(self, **kwargs):
        # Store attributes directly on the object to be found by InputFlattener
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Add some dummy attributes to increase dir() length
        self.__dummy1 = None
        self.__dummy2 = None
        self.__dummy3 = None
        self.__dummy4 = None
        self.__dummy5 = None
        self._method = lambda: None  # A dummy method that should be skipped


class MockDet3DDataSample:
    """Mock Det3DDataSample class"""
    def __init__(self):
        self.gt_instances_3d = None
        self.pred_instances_3d = None
        self.pts_pred_instances_3d = None
        self.img_pred_instances_3d = None
        self.gt_instances = None
        self.pred_instances = None
        self.gt_pts_seg = None
        self.pred_pts_seg = None
        self.proposals = None
        self.ignored_instances = None
        self._metainfo = {}

    def set_metainfo(self, metainfo):
        self._metainfo = metainfo

    def __contains__(self, key):
        return hasattr(self, key) and getattr(self, key) is not None


def test_mock_instance_data_structures():
    """Test mock InstanceData structures with various 3D detection fields"""
    print("\n=== Test 1: Mock InstanceData Structures ===")

    # Test 1.1: GT instances 3D
    print("\n1.1 Testing gt_instances_3d...")
    data_sample = MockDet3DDataSample()

    # Create ground truth 3D instances
    gt_instances_3d = MockInstanceData(
        bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(5, 7)),
        labels_3d=torch.randint(0, 10, (5,)),
        scores_3d=torch.ones(5),
        attrs_3d=torch.randint(0, 4, (5,))
    )
    data_sample.gt_instances_3d = gt_instances_3d

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(data_sample)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"Found {len(flattened)} tensors:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: shape={info['shape']}")

    # Check that we found the expected tensors
    expected_min_count = 4  # bboxes_3d.tensor, labels_3d, scores_3d, attrs_3d
    assert len(flattened) >= expected_min_count, f"Expected at least {expected_min_count} tensors, got {len(flattened)}"
    print("✅ gt_instances_3d test passed")

    # Test 1.2: Multi-modality instances
    print("\n1.2 Testing multi-modality instances...")
    data_sample = MockDet3DDataSample()

    # Point cloud predictions
    pts_pred = MockInstanceData(
        bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(2, 7)),
        scores_3d=torch.rand(2),
        labels_3d=torch.randint(0, 10, (2,))
    )
    data_sample.pts_pred_instances_3d = pts_pred

    # Image predictions
    img_pred = MockInstanceData(
        bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(3, 7)),
        scores_3d=torch.rand(3),
        labels_3d=torch.randint(0, 10, (3,))
    )
    data_sample.img_pred_instances_3d = img_pred

    # 2D predictions
    pred_2d = MockInstanceData(
        bboxes=torch.rand(4, 4),
        scores=torch.rand(4),
        labels=torch.randint(0, 10, (4,))
    )
    data_sample.pred_instances = pred_2d

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(data_sample)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"Found {len(flattened)} tensors:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: shape={info['shape']}")

    expected_min_count = 8  # Each InstanceData should have at least 2-3 tensors
    assert len(flattened) >= expected_min_count, f"Expected at least {expected_min_count} tensors, got {len(flattened)}"
    print("✅ Multi-modality instances test passed")

    return True


def test_mock_point_data_structures():
    """Test Mock PointData structures for point cloud segmentation"""
    print("\n=== Test 2: Mock PointData Structures ===")

    # Test 2.1: GT point segmentation
    print("\n2.1 Testing gt_pts_seg...")
    data_sample = MockDet3DDataSample()

    gt_pts_seg = MockPointData(
        pts_semantic_mask=torch.randint(0, 20, (1000,)),
        pts_instance_mask=torch.randint(0, 100, (1000,)),
        pts_bbox_mask=torch.randint(0, 2, (1000,), dtype=torch.bool)
    )
    data_sample.gt_pts_seg = gt_pts_seg

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(data_sample)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"Found {len(flattened)} tensors:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: shape={info['shape']}")

    assert len(flattened) == 3, f"Expected 3 tensors, got {len(flattened)}"
    print("✅ gt_pts_seg test passed")

    # Test 2.2: Pred point segmentation
    print("\n2.2 Testing pred_pts_seg...")
    data_sample = MockDet3DDataSample()

    pred_pts_seg = MockPointData(
        pts_semantic_mask=torch.rand(1000, 20),
        pts_instance_mask=torch.rand(1000, 50),
        pts_offset=torch.rand(1000, 3)
    )
    data_sample.pred_pts_seg = pred_pts_seg

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(data_sample)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"Found {len(flattened)} tensors:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: shape={info['shape']}")

    assert len(flattened) == 3, f"Expected 3 tensors, got {len(flattened)}"
    print("✅ pred_pts_seg test passed")

    return True


def test_complete_mock_det3d_data_sample():
    """Test a complete MockDet3DDataSample with all fields"""
    print("\n=== Test 3: Complete MockDet3DDataSample ===")

    print("\n3.1 Creating complete MockDet3DDataSample...")
    data_sample = MockDet3DDataSample()

    # Add metainfo
    data_sample.set_metainfo({
        'img_shape': (800, 1196, 3),
        'pad_shape': (800, 1216, 3),
        'batch_input_shape': (800, 1216)
    })

    # GT 3D instances
    gt_instances_3d = MockInstanceData(
        bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(5, 7)),
        labels_3d=torch.randint(0, 10, (5,)),
        scores_3d=torch.ones(5),
        difficults=torch.randint(0, 2, (5,), dtype=torch.bool)
    )
    data_sample.gt_instances_3d = gt_instances_3d

    # Pred 3D instances (point cloud based)
    pts_pred = MockInstanceData(
        bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(3, 7)),
        labels_3d=torch.randint(0, 10, (3,)),
        scores_3d=torch.rand(3)
    )
    data_sample.pts_pred_instances_3d = pts_pred

    # Pred 3D instances (image based)
    img_pred = MockInstanceData(
        bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(2, 7)),
        labels_3d=torch.randint(0, 10, (2,)),
        scores_3d=torch.rand(2)
    )
    data_sample.img_pred_instances_3d = img_pred

    # 2D predictions
    pred_2d = MockInstanceData(
        bboxes=torch.rand(10, 4),
        scores=torch.rand(10),
        labels=torch.randint(0, 10, (10,))
    )
    data_sample.pred_instances = pred_2d

    # Point segmentation
    gt_pts_seg = MockPointData(
        pts_semantic_mask=torch.randint(0, 20, (2000,)),
        pts_instance_mask=torch.randint(0, 50, (2000,))
    )
    data_sample.gt_pts_seg = gt_pts_seg

    pred_pts_seg = MockPointData(
        pts_semantic_mask=torch.rand(2000, 20),
        pts_instance_mask=torch.rand(2000, 50)
    )
    data_sample.pred_pts_seg = pred_pts_seg

    # Proposals
    proposals = MockInstanceData(
        bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(20, 7)),
        scores_3d=torch.rand(20)
    )
    data_sample.proposals = proposals

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(data_sample)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"\nFound {len(flattened)} tensors in complete MockDet3DDataSample:")
    for i, info in enumerate(flattener.tensor_info[:20]):  # Show first 20
        print(f"  [{i}] {info['path']}: shape={info['shape']}")
    if len(flattened) > 20:
        print(f"  ... and {len(flattened) - 20} more tensors")

    # We expect many tensors from all the fields (each LiDARInstance3DBoxes has many properties)
    expected_min_count = 15
    assert len(flattened) >= expected_min_count, f"Expected at least {expected_min_count} tensors, got {len(flattened)}"

    # Check some key paths exist
    paths = [info['path'] for info in flattener.tensor_info]
    assert any('gt_instances_3d' in path for path in paths), "Missing gt_instances_3d tensors"
    assert any('pts_pred_instances_3d' in path for path in paths), "Missing pts_pred_instances_3d tensors"
    assert any('img_pred_instances_3d' in path for path in paths), "Missing img_pred_instances_3d tensors"
    assert any('pred_instances' in path for path in paths), "Missing pred_instances tensors"
    assert any('gt_pts_seg' in path for path in paths), "Missing gt_pts_seg tensors"
    assert any('pred_pts_seg' in path for path in paths), "Missing pred_pts_seg tensors"

    print("\n✅ Complete MockDet3DDataSample test passed")

    # Test 3.2: Test batch of MockDet3DDataSamples
    print("\n3.2 Testing batch of MockDet3DDataSamples...")

    batch_size = 3
    batch = []
    for i in range(batch_size):
        ds = MockDet3DDataSample()
        gt = MockInstanceData(
            bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(4, 7)),
            labels_3d=torch.randint(0, 10, (4,))
        )
        ds.gt_instances_3d = gt

        pred = MockInstanceData(
            bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(6, 7)),
            scores_3d=torch.rand(6),
            labels_3d=torch.randint(0, 10, (6,))
        )
        ds.pred_instances_3d = pred

        batch.append(ds)

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(batch)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"\nFound {len(flattened)} tensors in batch of {batch_size} MockDet3DDataSamples:")
    for i, info in enumerate(flattener.tensor_info[:10]):  # Show first 10
        print(f"  [{i}] {info['path']}: shape={info['shape']}")
    if len(flattened) > 10:
        print(f"  ... and {len(flattened) - 10} more tensors")

    expected_min_count = batch_size * 5
    assert len(flattened) >= expected_min_count, f"Expected at least {expected_min_count} tensors, got {len(flattened)}"
    print("✅ Batch of MockDet3DDataSamples test passed")

    return True


def test_mixed_mock_structures():
    """Test mixing MockDet3DDataSample with regular tensors and dictionaries"""
    print("\n=== Test 4: Mixed Mock Structures ===")

    print("\n4.1 Testing mixed inputs (tensors + MockDet3DDataSample)...")

    # Create a mixed input structure
    input_data = {
        'points': [torch.randn(5000, 5) for _ in range(2)],  # Raw point cloud
        'images': torch.randn(2, 6, 3, 224, 224),  # Multi-view images
        'data_samples': [],  # Will be filled with MockDet3DDataSample
        'metadata': {
            'timestamp': torch.tensor([1234567890]),
            'sensor_ids': torch.tensor([0, 1])
        }
    }

    # Add MockDet3DDataSamples to the list
    for i in range(2):
        ds = MockDet3DDataSample()
        gt = MockInstanceData(
            bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(3, 7)),
            labels_3d=torch.randint(0, 10, (3,))
        )
        ds.gt_instances_3d = gt

        pred = MockInstanceData(
            bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(2, 7)),
            scores_3d=torch.rand(2),
            labels_3d=torch.randint(0, 10, (2,))
        )
        ds.pred_instances_3d = pred

        input_data['data_samples'].append(ds)

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"\nFound {len(flattened)} tensors in mixed structure:")
    for i, info in enumerate(flattener.tensor_info[:15]):  # Show first 15
        print(f"  [{i}] {info['path']}: shape={info['shape']}")
    if len(flattened) > 15:
        print(f"  ... and {len(flattened) - 15} more tensors")

    # Check we have tensors from all parts
    paths = [info['path'] for info in flattener.tensor_info]
    assert any('points' in path for path in paths), "Missing point cloud tensors"
    assert any('images' in path for path in paths), "Missing image tensors"
    assert any('data_samples' in path for path in paths), "Missing data sample tensors"
    assert any('metadata' in path for path in paths), "Missing metadata tensors"

    expected_min_count = 10
    assert len(flattened) >= expected_min_count, f"Expected at least {expected_min_count} tensors, got {len(flattened)}"
    print("✅ Mixed structure test passed")

    return True


def test_mock_forward_results_format():
    """Test ForwardResults format with mock structures"""
    print("\n=== Test 5: Mock ForwardResults Format ===")

    # Test 5.1: List[MockDet3DDataSample] format
    print("\n5.1 Testing List[MockDet3DDataSample] format...")

    forward_results = []
    for i in range(3):
        ds = MockDet3DDataSample()
        pred = MockInstanceData(
            bboxes_3d=MockLiDARInstance3DBoxes(torch.rand(4, 7)),
            scores_3d=torch.rand(4),
            labels_3d=torch.randint(0, 10, (4,))
        )
        ds.pred_instances_3d = pred
        forward_results.append(ds)

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(forward_results)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"Found {len(flattened)} tensors in List[MockDet3DDataSample] format:")
    for i, info in enumerate(flattener.tensor_info[:10]):
        print(f"  [{i}] {info['path']}: shape={info['shape']}")
    if len(flattened) > 10:
        print(f"  ... and {len(flattened) - 10} more tensors")

    expected_min_count = 12  # 3 samples * 4 tensors each minimum
    assert len(flattened) >= expected_min_count, f"Expected at least {expected_min_count} tensors, got {len(flattened)}"
    print("✅ List[MockDet3DDataSample] format test passed")

    return True


def run_mock_det3d_tests():
    """Run all mock Det3DDataSample-related tests"""
    print("=" * 60)
    print("InputFlattener Mock Det3DDataStructure Test Suite")
    print("=" * 60)

    tests = [
        ("Mock InstanceData Structures", test_mock_instance_data_structures),
        ("Mock PointData Structures", test_mock_point_data_structures),
        ("Complete MockDet3DDataSample", test_complete_mock_det3d_data_sample),
        ("Mixed Mock Structures", test_mixed_mock_structures),
        ("Mock ForwardResults Format", test_mock_forward_results_format)
    ]

    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running Test: {test_name}")
        print(f"{'='*60}")

        try:
            result = test_func()
            if result:
                print(f"\n✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"\n❌ {test_name}: FAILED")
                failed += 1
        except Exception as e:
            print(f"\n❌ {test_name}: FAILED with exception")
            print(f"Error: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print("Mock Det3DDataStructure Test Summary")
    print(f"{'='*60}")
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 All Mock Det3DDataStructure tests passed!")
    else:
        print(f"\n⚠️ {failed} test(s) failed")

    return failed == 0


if __name__ == '__main__':
    success = run_mock_det3d_tests()
    sys.exit(0 if success else 1)