# pcnaDeep: a deep-learning based single-cell cycle profiler with PCNA signal

Welcome! pcnaDeep integrates cutting-edge detection techniques with tracking and cell cycle resolving models.
Using the Mask R-CNN model under FAIR's Detectron2 framework, pcnaDeep is able to detect and resolve very dense cell tracks with __PCNA fluorescence__.

<img src="/tutorial/assets/overview.jpg" alt="overview" width="800" />

## Installation
1. PyTorch (torch >= 1.7.1) installation and CUDA GPU support are essential. Visit [PyTorch homepage](https://pytorch.org/) for specific installation schedule.

- Check the GPU and PyTorch are available:
   ```
   import torch
   print(torch.cuda.is_available())
   ```

2. Install modified __Detectron2 v0.4__ in this directory ([original package homepage](https://github.com/facebookresearch/detectron2))

   ```angular2html
      cd detectron2-04_mod
      pip install .
   ```

   <details>
   <summary>Building detectron2 on Windows? Click here.
   </summary>

      - Before building detectron2, you must install <a title="Microsoft Visual C++" href="https://visualstudio.microsoft.com/vs/features/cplusplus/">Microsoft Visual C++</a> (please use the standard installation).
      After installation, please restart your system.
      - If your torch version is old, the following changes of the `torch` package may be required. <a title="Ref" href="https://blog.csdn.net/weixin_42644340/article/details/109178660">Reference (Chinese)</a>.

         ```angular2html
            In torch\include\torch\csrc\jit\argument_spec.h,
            static constexpr size_t DEPTH_LIMIT = 128;
               change to -->
            static const size_t DEPTH_LIMIT = 128;
         ```
   </details>

   ---

   In pcnaDeep, the detectron2 v0.4 dependency has been modified in two ways:
      1. To generate confidence score output of the instance classification, the method `detectron2.modeling.roi_heads.fast_rcnn.fast_rcnn_inference_single_image` has been modified.
      2. A customized dataset mapper function has been implemented as `detectron2.data.dataset_mapper.read_PCNA_training`.


3. Install pcnaDeep from source in this directory
   ```
   cd bin
   python setup.py install
   ```
4. (optional, for training data annotation only) Download [VGG Image Annotator 2](https://www.robots.ox.ac.uk/~vgg/software/via/) software.
5. (optional, for visualisation only) Install [Fiji (ImageJ)](https://fiji.sc/) with [TrackMate CSV Importer](https://github.com/tinevez/TrackMate-CSVImporter) plugin.


## Demo data download

All demo data are stored at [Zenodo](https://zenodo.org/record/5515771#.YqAISRNBxxg).

### Download pre-trained Mask R-CNN weights

The Mask R-CNN is trained on 60X microscopic images sized 1200X1200 square pixels. [Download here](https://zenodo.org/record/5515771/files/mrcnn_sat_rot_aug.pth?download=1).

You must download pre-trained weights and save it under `~/models/` for running tutorials.

### Download example datasets

You may need to download [some example datasets](https://github.com/chan-labsite/PCNAdeep/tree/main/examples) to run tutorials (like the quick-start guide below).

## Getting started

See [a quick tutorial](tutorial/getting_started.ipynb) to get familiar with pcnaDeep.

You may also go through other tutorials for advanced usages.

## API Documentation

API documentation is available [here](https://pcnadeep.readthedocs.io/en/latest/index.html).

## Reference

Please cite our paper if you found this package useful. 
```
Gui Y, Xie S, Wang Y, Wang P, Yao R, Gao X, Dong Y, Wang G, Chan KY. pcnaDeep: a fast and robust single-cell tracking method using deep-learning mediated cell cycle profiling. Bioinformatics. 2022 Oct 14;38(20):4846-4847. doi: 10.1093/bioinformatics/btac602. PMID: 36047834.
```

## Licence

pcnaDeep is released under the [Apache 2.0 license](LICENSE).
