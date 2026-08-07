def build_model(name, num_keypoints=14, num_coor=3, num_person=1,
                subcarrier_num=180, dataset='person-in-wifi-3d', pretrained=False):
    """
    Build model with proper configuration for dataset.
    
    Args:
        name: Model name ('hpeli', 'metafiplusplus', 'graphposefi')
        num_keypoints: Number of keypoints (14 for PWIF3D, 17 for MMFI)
        num_coor: Coordinates per keypoint (default 3 for xyz)
        num_person: Number of people (default 1)
        subcarrier_num: Number of CSI subcarriers (default 180)
        dataset: Dataset name ('person-in-wifi-3d' or 'mmfi')
        pretrained: Whether to use pretrained weights
    """
    name = name.lower()
    
    # Auto-adjust num_keypoints based on dataset if not explicitly set
    if dataset == 'mmfi' and num_keypoints == 14:
        num_keypoints = 17
        print(f"[build_model] Auto-adjusted num_keypoints to 17 for MMFI dataset")
    
    if name == 'hpeli':
        from models.hpeli import HPELiNet, hpeli_init
        m = HPELiNet(num_keypoints, num_coor, subcarrier_num, num_person, dataset)
        m.apply(hpeli_init)
        return m
    if name == 'metafiplusplus':
        from models.metafiplusplus import MetaFiPlusPlusNet, metafiplusplus_init   # requires torchvision
        m = MetaFiPlusPlusNet(num_keypoints, num_coor, num_person, dataset, pretrained)
        m.apply(metafiplusplus_init)
        return m
    if name == 'graphposefi':
        from models.graphposefi import GraphPoseFiNet, graphposefi_init
        m = GraphPoseFiNet(num_keypoints, num_coor, num_person,
                           subcarrier_num, dataset, pretrained)
        m.apply(graphposefi_init)
        return m
    raise ValueError(f'Unknown model: {name}')
