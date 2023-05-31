# fastai-notes
fastai part1 &amp; part2 notes, part1的中文笔记来自其他作者对hiromis笔记的翻译，part2的中文笔记为英文笔记翻译而成

## fastai part1 notes
- https://github.com/hiromis/notes 参考hiromis的笔记，有中文版，质量较高。
  - [chinese](/chinese)文件夹，来源于https://github.com/hiromis/notes 
- [lesson1.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson1.pdf)  - [lesson2.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson2.pdf)
- [lesson3.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson3.pdf)  - [lesson4.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson4.pdf)
- [lesson5.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson5.pdf)  - [lesson6.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson6.pdf)
- [lesson7.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson7.pdf)  - [lesson8.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson8.pdf)
- [lesson9.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson9.pdf)  - [lesson10.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson10.pdf)
- [lesson11.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson11.pdf) - [lesson12.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson12%20%E7%BD%91%E7%BB%9C%E7%AC%94%E8%AE%B0.pdf)   - [lesson12.pdf详细](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson%2012%20Notes%20Advanced%20training%20techniques%3B%20ULMFiT%20from%20scratch.pdf)

## fastai part2 notes
- lesson8 markdown 笔记源文件： [lesson8 中文版笔记 md版本 ](./lesson8/lesson8.md)    [lesson8.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson8.pdf)    [fastai 2019 lesson8 notes 笔记_hello world-CSDN博客](https://blog.csdn.net/haronchou/article/details/120541922)
- lesson9 markdown ：[lesson9 中文版笔记 md版本 ](./lesson9/lesson9.md)     [lesson9.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson9.pdf)
- lessson10 markdown: [lesson10 中文版笔记 md版本 ](./lesson10/lesson10.md)  [lesson10.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson10.pdf)
- lessson11 markdown: [lesson11 中文版笔记 md版本 ](https://github.com/HaronCHou/fastai-notes/blob/main/lesson11/lesson11.md)   [lesson11.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson11.pdf)
- lesson12 markdown: [lesson12 中文版笔记 md版本](https://github.com/HaronCHou/fastai-notes/blob/main/lesson12/Lesson%2012%20Notes%20Advanced%20training%20techniques%3B%20ULMFiT%20from%20scratch.md)  [lesson12.pdf详细](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson%2012%20Notes%20Advanced%20training%20techniques%3B%20ULMFiT%20from%20scratch.pdf)
------------------
## 每日更新
❌UnDo  ✔️Done   ⭕ToDo
### 2023年5月31日 星期三
- RetinaNet fastai1的nb学习，下载了cocosample的数据集，学习其内容
- 感触比较大的是：anchor很多，anchor和gt-box的情况，可视化，可以发现VOC数据集，在189个锚框的时候，能框住的目标真的很少，这会导致怎样的问题？
- https://github.com/ChristianMarzahl/ObjectDetection/tree/master  这个代码中的mAP计算，是scores的阈值为0.3，然后计算这个下面的precision和recall，只有一个pr值，相当于；计算样本的pr曲线感觉不太对劲哦，应该是里面的固定scores阈值后，变化iou阈值？或者反着来的mAP值，才是对的；怎么回事对样本的pr曲线呢，这个有啥意义呢？
- 和之前的mAP计算代码差距太大，必须加以对比。
- 之前的mAP计算逻辑如下：
  - （1）nms_scores_thresh=0.05 ---> nms， nms_iou_thresh=0.5
  - （2）`tps[k,c] ` k为置信度阈值；c为某个类别； `fps[k,c], fns[k,c]` 

> Hi @ChristianMarzahl , thank you for your work. It was helpful for me, especially anchors representations.
> I adapted your code to fastai2, feel free to use it https://colab.research.google.com/drive/1ZA6yWj8wHwKUj3HT_LK3rsKYUPIkwnzZ
> Once I get good results I'll update this repo https://github.com/manycoding/signatures-detection with the V2 code.

v2 code of this repo, 值得一看

![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/30ae8420-9545-4f28-911c-bb0423057ac9)
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/a79ecb19-b70c-4785-b37f-a9ee6440e0e2)

1. https://github.com/ChristianMarzahl/ObjectDetection/tree/master
2. https://github.com/wushuang01/Focal_loss_for_object_detection RetinaNet focal loss voc2007 mAP 69%
3. longcw/yolo2-pytorch#23 中 https://github.com/cory8249/yolo2-pytorch/tree/master cory8249在上面的基础上获得了 71%的VOC mAP
4. https://nbviewer.org/gist/daveluo/2ab83da32e623864e543d7251e9beef4 详细的MAP计算
 
 > daveluo参考了Sylvain Gugger的代码，Sylvain的mAP代码地址为： https://github.com/sgugger/Deep-Learning/blob/master/mAP/Computing%20the%20mAP%20metric.ipynb
 > fastai 论坛的讨论地址为： http://forums.fast.ai/t/mean-average-precision-map/14345 
 > 
 > ————————————————
 >	版权声明：本文为CSDN博主「_helen_520」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
 >	原文链接： https://blog.csdn.net/haronchou/article/details/127976769
  5. 我用4的代码，进行的mAP计算：voc07 ssd_fastai 189anchor， mAP=0.45 https://github.com/HaronCHou/fastai-notes/blob/main/SSD_mAP.ipynb

![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/d8eb1aba-6082-4689-86ab-e8d3fe95a071)



### 2023年5月29日 星期一
- 上周五看到两个好的代码：①yolov3的pytorch源码，我已经可以完全看懂了；需要尝试，并了解其中的一些loss，等差别；mAP等；关键在于mAP的计算是怎样的
- VOC2007_mAP在yolov v2的时候也不太高；v3也不太高；
- RetinaNet fastai1，有比较好的metric，仔细阅读其代码，然后，看能否在这个基础上优化，做有意义的测试。https://github.com/ChristianMarzahl/ObjectDetection/tree/master fastai1 有详细的metric debug代码，并复用
- https://github.com/wushuang01/Focal_loss_for_object_detection  RetinaNet focal loss voc2007 mAP 69%
- https://github.com/longcw/yolo2-pytorch/tree/master yolov2 pytorch 没有mAP方面的说明
- https://github.com/longcw/yolo2-pytorch/issues/23 中https://github.com/cory8249/yolo2-pytorch/tree/master cory8249在上面的基础上获得了 71%的VOC mAP

### 2023年5月26日14:50:11 周五
- yolov3去使用voc数据集，并计算voc2007的mAP

```bash
# yolov3唐宇迪代码环境配置：在fastai1的环境上安装tensorflow
pip install --index-url https://pypi.douban.com/simple tensorflow
or
pip install --index-url http://mirrors.aliyun.com/pypi/simple/ tensorflow
```

### 2023年5月25日15:55:38 周四
- 仔细阅读了yolov3的源码和唐宇迪的注释；环境为：A100, root环境下，zhr_fastai1的环境；代码也在root环境下；数据集是coco数据集。
- 心得：这个loss里面最大的组成是置信度loss，这个挺奇怪的；有目标的和没目标的，这个loss的计算极其奇怪；v2和v3的loss组成是一毛一样的。
- yolov3跑VOC2007数据集试一下，看一下mAP的结果，现在是两边的mAP定义不一样，不知道Yolov3的是不是coco map；要区分一下结果。才能有一个一直的认识
- https://pjreddie.com/darknet/yolo/ yolo跑VOC数据集的demo在此处
- 还有Yolov2的源码使用的VOC mAP，可以去看下源码的实现


![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/ee9e390a-67be-4635-b88b-d2a129aecfe9)


### 2023年5月24日09:43:07 周三
- clas类别概率图可视化：意义很大；
- 特征提取hooks，可视化，意义也比较大；至少也是指示anchor的类别；
- ①骨干网络特征提取图可视化；②类别概率图可视化，两者结合更能体现出骨干网络和检测头的价值
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/67977a04-5b80-4462-a8e8-792e3e2bcff3)



### 2023年5月23日14:52:09
> 
> - 精读Radek的《menta Learning》
>   - 建立baseline，然后阅读paper（正确的方法阅读），验证是否对自己的任务有用；读论文，复现，看结果，分析是否work？
>   - 每天都一点点改进，让自己的模型表现更好
>   - 不要单纯调参去运行实验，改进网络更重要。
>   - 要写Debug代码来帮助诊断和分析模型，理解模型到底在干什么，是不是work？
- 读yolo的paper，并看涨点多的地方，去改进SSD，看看效果怎么样？

- 读以前写得《目标检测.docx》的文档，温故而知新！看到voc2007数据集的目标数量，再看训练结果，一起分析。
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/b21a1aa9-185f-4b3f-a104-a934cdc596ff)
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/e7ae7021-7529-4e1d-8aa1-2f816ea43c53)
- VOC mAP和COCO mAP的差异，控制的自变量不一样。来源：https://blog.csdn.net/c2250645962/article/details/106476147
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/27fe13ef-4283-4f5f-a2d0-8cb1408e4dea)
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/89d9beb2-66a3-424f-b5f2-1450fe544a85)
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/da43960b-4512-4bc8-a472-115fd16d0c81)



### 2023年5月18日10:36:57
- [x] focal_loss和bce_loss的差异在哪里体现？ softmax在哪里体现？
- bce_loss里面也是要自己实现softmax的 [自己实现的](https://blog.csdn.net/code_plus/article/details/115739343)
- [很详细的focal loss的介绍](https://blog.csdn.net/BIgHAo1/article/details/121783011)
  - 既然focal loss对分类任务影响较大，可以在分类任务上尝试，看看focal loss对acc的影响有多大，同样的数据，同样的epoch来做一个参考和说明
  - 在pets的数据集上看下效果。
  - 测试后，发现4个epoch并不能看到太多；同时一定要Lr_find看下合适的学习率是多少；loss_func竟然可以自定义；关键在于里面没有One_hot_embedding，pets数据集没有自定义做这一步
  - 测试三个loss后发现，没啥变化，在pets上。讲道理focal loss要优秀一些；结论是：数据集太完美了，不存在不平衡，只要有一个类别是不平衡的话，应该就可以看到效果，所以下一步是制造不平衡的数据集，然后看下效果。
  - 2023年5月23日10:59:28 补充：focal loss在分类问题上，对loss的改变，差别不大；但是在目标检测任务中，由于anchor是189个，数量较多；一个epoch下来，loss的改变是比较大的；focal loss的绝对值比BCE loss低了10倍多左右，绝对值差异；但是对于map的结果影响不大，还是0.35左右的mAP，所以在ssd网络中，要提升mAP的话，loss可能不是最重要的，还要寻找优化点最高的地方！！！🌟🌟🌟🌟🌟

![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/7801bcce-ac59-4c1b-8c10-59d881972cdc)
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/3566558b-2b5b-401f-b77c-aaa907de5dd8)
- 将其中一个类别从200张图，减少90张后；差异
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/8c10c1f6-3da8-42bb-abdf-d325033c28d6)
[笔记本为pet.ipynb](https://github.com/HaronCHou/fastai-notes/blob/main/pets.ipynb)

![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/72e1f419-7414-42ba-8b22-105c28bb09f4)
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/4857fe3b-036f-4de5-b70f-7c1cb93b2b0d)
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/a04693f4-fb20-405a-9783-6c588781f239)


#### 2023年5月17日 ToDo
  - [**如何打造高质量的深度学习数据集**？](https://www.zhihu.com/question/333074061)
  - [ ] 之前使用的是BCE_loss，改为focal_loss，试一下loss下降的情况，然后，再看下训练完成后mAP有怎样的变化呢？
     - 之前使用的就已经是focal_loss了，改为bce_loss回去看看 
     - 同时为了分析label_loss和bbox_loss，参考https://github.com/HaronCHou/fastai-notes/blob/main/SSD_mAP.ipynb 里面的做法，把label_loss和bbox_loss分别显示
     - bce_loss和focal_loss差别如下所示：
- 与focal loss相比，loss的值增加了很多。应该是anchor box 189个比较大。189*bs*所有的，累积下来，focal loss得比重就降低了很多。
- focal loss在类别不平衡问题面前是很有效的，以及在易区分的物体上降低了一些难度大
- 对于目标检测而言，anchor189个太多，是一个典型的类别不平衡问题，所以加了focal loss后，效果好很多！ 
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/f56e9b60-166d-4e65-85ac-a2822237b19e)
![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/eb460adb-5580-4825-9d9b-878f7cec1ac9)

  - [ ] 再对照Jeremy之前的内容看一下效果，看看改为focal_loss之后的收益到底有多大？
  - [ ] 还有Yolov5的效果为什么好那么多，CoCo_map都到了90%多了，why?指标不一样？还是网络真的有那么大的改善？结论和答案是什么？icevision的yolov5的v0c2007，效果好太多了，我都不知道为啥？
  - [ ] 自动驾驶qq群，有关slam的JD，以及一些行业交流，这方面也需要更open的来了解这写行业；极氪36有行业发展、分析、咨询类的文章，观点认识，需要认识整个行业，并开拓视野。
  - [ ] 『背后的利益是指挥棒』，前端是表象/现象，不是本质的运行逻辑。（遇到困惑需要上升一个维度来思考）
  - [ ] zeku解散，突然裁员，2000员工；达摩院自动驾驶100人并入菜鸟，裁员200，70%；再看校招薪水的公众号，全球自动驾驶缩水。（不知道结论可不可靠，需要再次多方证实）。并入菜鸟，说明赛道选的是送快递，有局限的应用领域。同时，变现能力在紧缩时代很重要。热钱。『问了几个人，大哥：熬过冬天，适合囤积；chenyang；自动驾驶群看到高仙机器人的深度学习总监，在开课，卖课。单位和买课结合；联想很多人进入AI培训领域赚热钱，怎么个想法』

#### 2023年5月16日09:20:11 周二
  - [x] ✔️成功的经验：mAP计算的每一步要像Jeremy的pascal_muliti.ipynb那样一步步的去走；当时只是运行了Jeremy的笔记本，但是没有自己尝试去复制，所以不知道啥意思。这个很重要。『自己从头制作笔记本，并且不要作弊』，在Jeremy的课程中也有强调这一点。
  - 由于SSD训练的效果不太好，所以一开始就计算mAP意义不太大。反而是训练的差不多的时候再来计算mAP比较有意义。然后再自己去微调。
  - 学习深度碎片：https://github.com/EmbraceLife/My_Journey_on_Kaggle
  - wasimlorgat在[fastai](https://forums.fast.ai/t/introduce-yourself-here/99261/106) twitter为：[wasimlorgat](https://twitter.com/wasimlorgat/media )
  - 博客位置：https://wasimlorgat.com/ 
  - 笔记位置：[wasimlorgat学习笔记](https://github.com/HaronCHou/fastai-notes/blob/main/wasimlorgat.md)
> 1. 遇到困难，不要放弃；分解为更简单的任务来增强技能，并稍后再回来！——这个真的超级有用
  - [x] ✔️[完成ssd fastaiv2 + mAP.ipynb](https://github.com/HaronCHou/fastai-notes/blob/main/SSD_jav.ipynb)
    - [x] ✔️其中，mAP还是总的计算，速度还可以的，很快，可能在于batched_nms的函数替换，效果很快；里面也有很多验证性代码，过程值得学习
    - [x] ✔️mAP只到 0.3671，还很小；且由于统计规律，10%的小框框根本就无法输出 见笔记本的统计结果 https://github.com/HaronCHou/fastai-notes/blob/main/SSD_Object_Detection.ipynb

#### 2023年5月11日11:46:41
  - 看到[深度碎片](https://github.com/EmbraceLife/My_Journey_on_Kaggle/tree/main)的github和twitter
  - 2023年5月11日16:52:50 在daniel的推荐下，再次去看了Radek的《mata learning》，很神奇的只花了三个小时就完成了，翻译为中文的笔记见：[meata learning 中文翻译 How To Learn Deep Learning And Thrive In The Digital Age](https://note.youdao.com/s/N8ZKdqlo)
  - 看到icevision在Fastai上的内容；看到muller在推荐使用icevision做object detection，更新了blog[icevision相关笔记](https://blog.csdn.net/haronchou/article/details/130557309)
  - 多学习和记录


------------------
以下是深度碎片的原文：
 ![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/df1755bf-e0b7-460e-8bd6-8180ac3e191c)

