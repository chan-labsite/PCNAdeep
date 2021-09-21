def inspect_PCNA_data(json_path, image_path, out_dir='../../../inspect/test'):
    """Inspect PCNA training data.
    """
    import cv2
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import DatasetCatalog, MetadataCatalog
    prefix = os.path.basename(image_path)
    DatasetCatalog.register("pcna", lambda d: load_PCNA_from_json(json_path, image_path))
    metadata = MetadataCatalog.get("pcna").set(thing_classes=['G1/G2', 'S', 'M', 'E'])

    dataset_dicts = load_PCNA_from_json(json_path, image_path)
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img, metadata=metadata, scale=0.5)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imwrite(os.path.join(out_dir, prefix + d["image_id"] + '.png'), vis.get_image())