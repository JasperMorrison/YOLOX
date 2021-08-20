# Prepare the dataset to ./datasets/coco
# Edit exps/example/custom/yolox_s.py
python tools/train.py -f exps/example/custom/yolox_s.py -c weights/yolox_s.pth --devices 0 -b 8

# To Caffe
git clone https://github.com/JasperMorrison/PytorchToCaffe -b yolov5
python tools/export_caffe.py -f exps/example/custom/yolox_s.py -c YOLOX_outputs/yolox_s/latest_ckpt.pth

