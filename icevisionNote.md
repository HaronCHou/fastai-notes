## icevision Notes
#### 学习笔记：
- 网址：https://airctic.com/dev/getting_started_object_detection/
- icevision
  - parsers/coco_parser.py voc_parser.py via_parser.py parser.py
1. 解析数据集：自动加载注释文件并解析；parser.parse() 支持VOC格式和COCO格式
  recored是icevision中的一个关键的概念，包含图像和注释的信息；是可以扩展的。
  record的类的继承关系：`__mro__`
  ```python
  print(BaseRecord.__mro__)
(<class 'icevision.core.record.BaseRecord'>, <class 'icevision.core.components.composite.TaskComposite'>, <class 'object'>)
  ```
  record先把所有的标签信息给处理完毕：bbox label，分验证集等，路径，ID编号，等工作全部完成
  ![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/bd347a11-b37f-48c1-b4ef-7227f0b35376)
