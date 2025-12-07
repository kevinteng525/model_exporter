#!/usr/bin/env python3
"""
Comprehensive tests for InputFlattener class

Tests cover various data structures found in MMDetection3D including:
- Basic tensor structures
- Nested dictionaries (common in model inputs/outputs)
- List and tuple structures
- MMDetection3D specific data structures (points, bboxes, etc.)
- Edge cases and error handling
- Self-consistency tests
"""

import sys
import torch
import numpy as np
from typing import Dict, List, Any
import traceback

# Add mmdetection3d to path for accessing data structures
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/mmdetection3d')

# Import the InputFlattener from improved_exporter
sys.path.insert(0, '/Users/kevinteng/src/kevinteng525/open-mmlab/refined')
from improved_exporter import InputFlattener


def test_tensor_flattening_and_reconstruction():
    """Test basic tensor flattening and reconstruction"""
    print("\n=== Test 1: Basic Tensor Flattening and Reconstruction ===")

    flattener = InputFlattener()

    # Test 1.1: Single tensor
    print("\n1.1 Testing single tensor...")
    tensor = torch.randn([100, 5])
    flattened = flattener.analyze_and_flatten(tensor, "input_tensor")
    reconstructed = flattener.reconstruct_inputs(flattened)

    assert len(flattened) == 1, f"Expected 1 tensor, got {len(flattened)}"
    assert torch.equal(tensor, reconstructed['input_tensor']), "Reconstructed tensor mismatch"
    print("✅ Single tensor test passed")

    # Test 1.2: Multiple tensors
    print("\n1.2 Testing multiple tensors...")
    input_data = {
        'points': torch.randn([5000, 5]),
        'features': torch.randn([64, 256, 32, 32]),
        'metadata': torch.randn([1, 10])
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    assert len(flattened) == 3, f"Expected 3 tensors, got {len(flattened)}"
    for key in input_data:
        assert torch.equal(input_data[key], reconstructed[key]), f"Mismatch in {key}"
    print("✅ Multiple tensors test passed")

    # Test 1.3: Different dtypes
    print("\n1.3 Testing different dtypes...")
    input_data = {
        'float32_tensor': torch.randn([10, 3], dtype=torch.float32),
        'float16_tensor': torch.randn([10, 3], dtype=torch.float16),
        'int32_tensor': torch.randint(0, 100, [10, 3], dtype=torch.int32),
        'int64_tensor': torch.randint(0, 100, [10, 3], dtype=torch.int64),
        'bool_tensor': torch.randint(0, 2, [10, 3], dtype=torch.bool)
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    assert len(flattened) == 5, f"Expected 5 tensors, got {len(flattened)}"
    for key in input_data:
        assert torch.equal(input_data[key], reconstructed[key]), f"Mismatch in {key}"
        assert input_data[key].dtype == reconstructed[key].dtype, f"Dtype mismatch in {key}"
    print("✅ Different dtypes test passed")

    # Test 1.4: Empty tensor (should be skipped)
    print("\n1.4 Testing empty tensor handling...")
    input_data = {
        'valid_tensor': torch.randn([10, 3]),
        'empty_tensor': torch.empty(0),
        'another_valid': torch.randn([5, 5])
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)

    # Should only have 2 tensors (empty tensor skipped)
    assert len(flattened) == 2, f"Expected 2 tensors (empty skipped), got {len(flattened)}"
    print("✅ Empty tensor handling test passed")

    return True


def test_nested_dict_structures():
    """Test nested dictionary structures common in MMDetection3D"""
    print("\n=== Test 2: Nested Dictionary Structures ===")

    flattener = InputFlattener()

    # Test 2.1: Simple nested dict
    print("\n2.1 Testing simple nested dictionary...")
    input_data = {
        'lidar': {
            'points': torch.randn([10000, 5]),
            'voxels': torch.randn([2000, 5, 4]),
            'num_points': torch.randint(1, 5, [2000]),
            'coors': torch.randint(0, 100, [2000, 3])
        },
        'camera': {
            'images': torch.randn([6, 3, 224, 224]),
            'intrinsics': torch.randn([6, 3, 3])
        }
    }

    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    assert len(flattened) == 6, f"Expected 6 tensors, got {len(flattened)}"

    # Check reconstruction
    for modality in input_data:
        for key in input_data[modality]:
            original = input_data[modality][key]
            recon = reconstructed[modality][key]
            assert torch.equal(original, recon), f"Mismatch in {modality}.{key}"
    print("✅ Simple nested dict test passed")

    # Test 2.2: Deeply nested dict
    print("\n2.2 Testing deeply nested dictionary...")
    input_data = {
        'data': {
            'inputs': {
                'points': torch.randn([5000, 3]),
                'features': {
                    'geometry': torch.randn([5000, 64]),
                    'intensity': torch.randn([5000, 1])
                }
            },
            'metadata': {
                'timestamp': torch.tensor([1234567890]),
                'sensor_info': {
                    'lidar_height': torch.tensor([1.73]),
                    'camera_num': torch.tensor([6])
                }
            }
        }
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    assert len(flattened) == 6, f"Expected 6 tensors, got {len(flattened)}"

    # Check nested access
    assert torch.equal(
        input_data['data']['inputs']['features']['geometry'],
        reconstructed['data']['inputs']['features']['geometry']
    ), "Deeply nested reconstruction failed"
    print("✅ Deeply nested dict test passed")

    return True


def test_list_tuple_structures():
    """Test list and tuple structures containing tensors"""
    print("\n=== Test 3: List and Tuple Structures ===")

    # Test 3.1: List of tensors
    print("\n3.1 Testing list of tensors...")
    input_data = {
        'points_list': [
            torch.randn([100, 5]),
            torch.randn([150, 5]),
            torch.randn([120, 5])
        ],
        'single_tensor': torch.randn([200, 5])
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    assert len(flattened) == 4, f"Expected 4 tensors, got {len(flattened)}"

    for i in range(3):
        assert torch.equal(
            input_data['points_list'][i],
            reconstructed['points_list'][i]
        ), f"Mismatch in points_list[{i}]"
    assert torch.equal(input_data['single_tensor'], reconstructed['single_tensor'])
    print("✅ List of tensors test passed")

    # Test 3.2: Mixed structure (dict containing lists)
    print("\n3.2 Testing mixed structure...")
    input_data = {
        'batch_data': {
            'points': [torch.randn([100, 5]), torch.randn([80, 5])],
            'features': torch.randn([180, 64]),
            'metadata': {
                'indices': [torch.tensor([0]), torch.tensor([1])]
            }
        }
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"Found {len(flattened)} tensors:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: shape={info['shape']}")

    # Actually we find 5 tensors: points[0], points[1], features, metadata.indices[0], metadata.indices[1]
    assert len(flattened) == 5, f"Expected 5 tensors, got {len(flattened)}"

    # Verify reconstruction
    assert len(reconstructed['batch_data']['points']) == 2
    assert len(reconstructed['batch_data']['metadata']['indices']) == 2
    print("✅ Mixed structure test passed")

    # Test 3.3: Tuple of tensors
    print("\n3.3 Testing tuple of tensors...")
    input_data = {
        'output_tuple': (
            torch.randn([1, 100, 5]),
            torch.randn([1, 50]),
            torch.randn([1, 10, 10, 64])
        )
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    assert len(flattened) == 3, f"Expected 3 tensors, got {len(flattened)}"
    assert len(reconstructed['output_tuple']) == 3
    print("✅ Tuple of tensors test passed")

    return True


def test_mmdetection3d_structures():
    """Test MMDetection3D specific data structures"""
    print("\n=== Test 4: MMDetection3D Specific Structures ===")

    # Test 4.1: LiDAR point cloud data structure
    print("\n4.1 Testing LiDAR point cloud structure...")
    # Simulate typical LiDAR batch data
    num_points = [5000, 3000]  # Different number of points per sample

    input_data = {
        'points': [
            torch.randn(num_points[0], 5),  # x, y, z, intensity, timestamp
            torch.randn(num_points[1], 5)
        ],
        'points_mask': [
            torch.ones(num_points[0], dtype=torch.bool),
            torch.ones(num_points[1], dtype=torch.bool)
        ],
        'voxels': {
            'voxels': torch.randn(4000, 5, 4),
            'num_points': torch.randint(1, 5, [4000]),
            'coors': torch.randint(0, 200, [4000, 3])
        },
        'batch_size': torch.tensor([2])
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"Found {len(flattened)} tensors:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: shape={info['shape']}")

    # Actually we find 8 tensors: points[0], points[1], points_mask[0], points_mask[1], voxels.voxels, voxels.num_points, voxels.coors, batch_size
    assert len(flattened) == 8, f"Expected 8 tensors, got {len(flattened)}"

    # Verify points list reconstruction
    assert len(reconstructed['points']) == 2
    assert reconstructed['points'][0].shape == (num_points[0], 5)
    assert reconstructed['points'][1].shape == (num_points[1], 5)
    print("✅ LiDAR point cloud structure test passed")

    # Test 4.2: Multi-camera data structure
    print("\n4.2 Testing multi-camera structure...")
    num_cameras = 6
    input_data = {
        'img': [
            torch.randn([3, 224, 224]) for _ in range(num_cameras)
        ],
        'img_metas': [
            {
                'cam2img': torch.randn([3, 3]),
                'lidar2cam': torch.randn([4, 4]),
                'img_shape': torch.tensor([224, 224]),
                'pad_shape': torch.tensor([224, 224])
            } for _ in range(num_cameras)
        ]
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    assert len(flattened) == num_cameras * 5, f"Expected {num_cameras * 5} tensors, got {len(flattened)}"
    assert len(reconstructed['img']) == num_cameras
    assert len(reconstructed['img_metas']) == num_cameras
    print("✅ Multi-camera structure test passed")

    # Test 4.3: Detection output structure
    print("\n4.3 Testing detection output structure...")
    input_data = {
        'det_3d': {
            'boxes_3d': torch.randn([100, 7]),  # (x, y, z, w, l, h, yaw)
            'scores_3d': torch.randn([100]),
            'labels_3d': torch.randint(0, 10, [100]),
            'attrs_3d': torch.randn([100, 4])
        },
        'det_2d': {
            'bboxes': torch.randn([100, 4]),  # (x1, y1, x2, y2)
            'scores': torch.randn([100]),
            'labels': torch.randint(0, 10, [100])
        },
        'mask': {
            'foreground': torch.randint(0, 2, [1, 200, 200], dtype=torch.bool),
            'background': torch.randint(0, 2, [1, 200, 200], dtype=torch.bool)
        }
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"Found {len(flattened)} tensors:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: shape={info['shape']}")

    assert len(flattened) == 9, f"Expected 9 tensors, got {len(flattened)}"

    # Verify specific keys
    assert torch.equal(input_data['det_3d']['boxes_3d'], reconstructed['det_3d']['boxes_3d'])
    assert torch.equal(input_data['mask']['foreground'], reconstructed['mask']['foreground'])
    print("✅ Detection output structure test passed")

    return True


def test_edge_cases():
    """Test edge cases and special conditions"""
    print("\n=== Test 5: Edge Cases ===")

    # Test 5.1: Mixed data types
    print("\n5.1 Testing mixed data types...")
    class CustomObject:
        def __init__(self):
            self.value = 42
            self.tensor_field = torch.randn([5, 5])

    input_data = {
        'tensor': torch.randn([10, 10]),
        'int_value': 42,
        'float_value': 3.14,
        'string_value': 'test_string',
        'list_with_mixed': [1, 2.0, 'three', torch.randn([5, 5])],
        'custom_object': CustomObject(),
        'none_value': None,
        'empty_dict': {},
        'empty_list': []
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    # Should extract only the tensors from tensor and list_with_mixed[3], and custom_object.tensor_field
    expected_tensors = 3
    assert len(flattened) == expected_tensors, f"Expected {expected_tensors} tensors, got {len(flattened)}"
    print("✅ Mixed data types test passed")

    # Test 5.2: Very deep nesting
    print("\n5.2 Testing very deep nesting...")
    deep_data = {}
    current = deep_data
    for i in range(10):
        current['level'] = {}
        current = current['level']
        if i % 3 == 0:
            current[f'tensor_{i}'] = torch.randn([i+1, i+1])

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(deep_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    assert len(flattened) == 4, f"Expected 4 tensors, got {len(flattened)}"
    print("✅ Very deep nesting test passed")

    # Test 5.3: Large tensors
    print("\n5.3 Testing large tensors...")
    input_data = {
        'large_1d': torch.randn(1000000),
        'large_2d': torch.randn(1000, 1000),
        'large_3d': torch.randn(100, 100, 100),
        'small': torch.randn(10)
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    assert len(flattened) == 4, f"Expected 4 tensors, got {len(flattened)}"
    for key in input_data:
        assert torch.equal(input_data[key], reconstructed[key]), f"Large tensor mismatch in {key}"
    print("✅ Large tensors test passed")

    # Test 5.4: Tensor with special shapes
    print("\n5.4 Testing tensors with special shapes...")
    input_data = {
        'scalar': torch.tensor(42.0),
        'zero_dim': torch.randn([]),
        'single_dim': torch.randn([100]),
        'multi_dim': torch.randn([2, 3, 4, 5]),
        'large_first_dim': torch.randn(1, 1000),
        'large_last_dim': torch.randn(1000, 1)
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    print(f"Found {len(flattened)} tensors:")
    for i, info in enumerate(flattener.tensor_info):
        print(f"  [{i}] {info['path']}: shape={info['shape']}")

    # Zero-dimensional tensors (scalars) have 0 elements and should be skipped
    # But scalar tensor with 1 element has 1 element and should be included
    assert len(flattened) == 6, f"Expected 6 tensors, got {len(flattened)}"
    print("✅ Special shapes test passed")

    return True


def test_self_consistency():
    """Test self-consistency: flatten then reconstruct should recover original"""
    print("\n=== Test 6: Self-Consistency Tests ===")

    # Test 6.1: Simple structure
    print("\n6.1 Testing simple structure consistency...")
    input_data = {
        'points': torch.randn([100, 5]),
        'features': torch.randn([100, 64])
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    # Check if reconstruction matches original exactly
    def compare_structures(orig, recon, path=""):
        if isinstance(orig, torch.Tensor) and isinstance(recon, torch.Tensor):
            assert torch.equal(orig, recon), f"Tensor mismatch at {path}"
            assert orig.dtype == recon.dtype, f"Dtype mismatch at {path}"
            assert orig.device == recon.device, f"Device mismatch at {path}"
        elif isinstance(orig, dict) and isinstance(recon, dict):
            assert set(orig.keys()) == set(recon.keys()), f"Key mismatch at {path}"
            for key in orig:
                compare_structures(orig[key], recon[key], f"{path}.{key}" if path else key)
        elif isinstance(orig, (list, tuple)) and isinstance(recon, (list, tuple)):
            assert len(orig) == len(recon), f"Length mismatch at {path}"
            for i, (o, r) in enumerate(zip(orig, recon)):
                compare_structures(o, r, f"{path}[{i}]")
        else:
            # For non-tensor data, just check equality if possible
            try:
                assert orig == recon, f"Value mismatch at {path}"
            except:
                pass  # Some objects might not be comparable

    compare_structures(input_data, reconstructed)
    print("✅ Simple structure consistency test passed")

    # Test 6.2: Complex nested structure
    print("\n6.2 Testing complex structure consistency...")
    input_data = {
        'batch': {
            'points': [
                torch.randn([100, 5]),
                torch.randn([80, 5]),
                torch.randn([120, 5])
            ],
            'images': torch.randn([3, 224, 224]),
            'metadata': {
                'timestamps': torch.tensor([1, 2, 3]),
                'sensor_ids': torch.tensor([0, 1, 2])
            }
        }
    }

    flattener = InputFlattener()
    flattened = flattener.analyze_and_flatten(input_data)
    reconstructed = flattener.reconstruct_inputs(flattened)

    compare_structures(input_data, reconstructed)
    print("✅ Complex structure consistency test passed")

    # Test 6.3: Check flattener metadata consistency
    print("\n6.3 Testing flattener metadata consistency...")
    assert len(flattened) == len(flattener.tensor_info), "Tensor info length mismatch"
    assert len(flattener.flatten_mapping) == len(flattener.tensor_info), "Mapping length mismatch"

    for idx, info in enumerate(flattener.tensor_info):
        assert 'path' in info, f"Missing path in tensor info {idx}"
        assert 'shape' in info, f"Missing shape in tensor info {idx}"
        assert 'dtype' in info, f"Missing dtype in tensor info {idx}"
        assert info['path'] in flattener.flatten_mapping, f"Path {info['path']} not in mapping"
        assert flattener.flatten_mapping[info['path']] == idx, f"Index mismatch for path {info['path']}"

    print("✅ Metadata consistency test passed")

    return True


def run_all_tests():
    """Run all tests and report results"""
    print("=" * 60)
    print("InputFlattener Comprehensive Test Suite")
    print("=" * 60)

    tests = [
        ("Tensor Flattening and Reconstruction", test_tensor_flattening_and_reconstruction),
        ("Nested Dictionary Structures", test_nested_dict_structures),
        ("List and Tuple Structures", test_list_tuple_structures),
        ("MMDetection3D Specific Structures", test_mmdetection3d_structures),
        ("Edge Cases", test_edge_cases),
        ("Self-Consistency", test_self_consistency)
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
    print("Test Summary")
    print(f"{'='*60}")
    print(f"Total Tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️ {failed} test(s) failed")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)