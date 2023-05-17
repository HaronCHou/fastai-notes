# fastai-notes
fastai part1 &amp; part2 notes, part1的中文笔记来自其他作者对hiromis笔记的翻译，part2的中文笔记为英文笔记翻译而成

## fastai part1 notes
- https://github.com/hiromis/notes 参考hiromis的笔记，有中文版，质量较高。
  - [chinese](/chinese)文件夹，来源于https://github.com/hiromis/notes 
- [lesson1.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson1.pdf)
- [lesson2.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson2.pdf)
- [lesson3.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson3.pdf)
- [lesson4.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson4.pdf)
- [lesson5.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson5.pdf)
- [lesson6.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson6.pdf)
- [lesson7.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson7.pdf)
- [lesson8.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson8.pdf)
- [lesson9.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson9.pdf)
- [lesson10.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson10.pdf)
- [lesson11.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson11.pdf)
- [lesson12.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson12%20%E7%BD%91%E7%BB%9C%E7%AC%94%E8%AE%B0.pdf)
  - [lesson12.pdf详细](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson%2012%20Notes%20Advanced%20training%20techniques%3B%20ULMFiT%20from%20scratch.pdf)

## fastai part2 notes
- lesson8 markdown 笔记源文件： [lesson8 中文版笔记 md版本 ](./lesson8/lesson8.md)    [lesson8.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson8.pdf)    [fastai 2019 lesson8 notes 笔记_hello world-CSDN博客](https://blog.csdn.net/haronchou/article/details/120541922)
- lesson9 markdown ：[lesson9 中文版笔记 md版本 ](./lesson9/lesson9.md)     [lesson9.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson9.pdf)
- lessson10 markdown: [lesson10 中文版笔记 md版本 ](./lesson10/lesson10.md)  [lesson10.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson10.pdf)
- lessson11 markdown: [lesson11 中文版笔记 md版本 ](https://github.com/HaronCHou/fastai-notes/blob/main/lesson11/lesson11.md)   [lesson11.pdf](https://github.com/HaronCHou/fastai-notes/blob/main/lesson11.pdf)
- lesson12 markdown: [lesson12 中文版笔记 md版本](https://github.com/HaronCHou/fastai-notes/blob/main/lesson12/Lesson%2012%20Notes%20Advanced%20training%20techniques%3B%20ULMFiT%20from%20scratch.md)  [lesson12.pdf详细](https://github.com/HaronCHou/fastai-notes/blob/main/Lesson%2012%20Notes%20Advanced%20training%20techniques%3B%20ULMFiT%20from%20scratch.pdf)
------------------
## 每日更新
❌UnDo  ✔️Done   ⭕ToDo

#### 2023年5月17日 ToDo
  - [ ] 之前使用的是BCE_loss，改为focal_loss，试一下loss下降的情况，然后，再看下训练完成后mAP有怎样的变化呢？
  - [ ] 再对照Jeremy之前的内容看一下效果，看看改为focal_loss之后的收益到底有多大？
  - [ ] 还有Yolov5的效果为什么好那么多，CoCo_map都到了90%多了，why?指标不一样？还是网络真的有那么大的改善？结论和答案是什么？icevision的yolov5的v0c2007，效果好太多了，我都不知道为啥？
  - [ ] 自动驾驶qq群，有关slam的JD，以及一些行业交流，这方面也需要更open的来了解这写行业；极氪36有行业发展、分析、咨询类的文章，观点认识，需要认识整个行业，并开拓视野。
  - [ ] 『背后的利益是指挥棒』，前端是表象/现象，不是本质的运行逻辑。（遇到困惑需要上升一个维度来思考）
  - [ ] zeku解散，突然裁员，2000员工；达摩院自动驾驶100人并入菜鸟，裁员200，70%；再看校招薪水的公众号，全球自动驾驶缩水。（不知道结论可不可靠，需要再次多方证实）。并入菜鸟，说明赛道选的是送快递，有局限的应用领域。同时，变现能力在紧缩时代很重要。热钱。『问了几个人，大哥：熬过冬天，适合囤积；chenyang；自动驾驶群看到高仙机器人的深度学习总监，在开课，卖课。单位和买课结合；联想很多人进入AI培训领域赚热钱，怎么个想法』

#### 2023年5月11日11:46:41
  - 看到[深度碎片](https://github.com/EmbraceLife/My_Journey_on_Kaggle/tree/main)的github和twitter
  - 2023年5月11日16:52:50 在daniel的推荐下，再次去看了Radek的《mata learning》，很神奇的只花了三个小时就完成了，翻译为中文的笔记见：[meata learning 中文翻译 How To Learn Deep Learning And Thrive In The Digital Age](https://note.youdao.com/s/N8ZKdqlo)
  - 看到icevision在Fastai上的内容；看到muller在推荐使用icevision做object detection，更新了blog[icevision相关笔记](https://blog.csdn.net/haronchou/article/details/130557309)
  - 多学习和记录
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

------------------
以下是深度碎片的原文：
 ![image](https://github.com/HaronCHou/fastai-notes/assets/22512646/df1755bf-e0b7-460e-8bd6-8180ac3e191c)

