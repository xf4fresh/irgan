## ltr-gan-pointwise
+ You should download the data set from [here](https://drive.google.com/drive/folders/0B-dulzPp3MmCM01kYlhhNGQ0djA?usp=sharing), and then put in `MQ2008-semi/`.
+ Run `python run.py` to evaluate the stored models.

http://www.bigdatalab.ac.cn/benchmark/bm/dd?data=MQ2007
http://www.bigdatalab.ac.cn/benchmark/bm/dd?data=MQ2007-semi
http://www.bigdatalab.ac.cn/benchmark/bm/dd?data=MQ2008-semi
[Introducing LETOR 4.0 Datasets](https://arxiv.org/pdf/1306.2597.pdf)
[learning_to_rank-BM25](https://people.cs.umass.edu/~jpjiang/cs646/16_learning_to_rank.pdf)
[期刊-面向排序学习的特征分析研究](http://www.docin.com/p-223510084.html)
[硕士论文-半监督排序算法](http://www.docin.com/p-1357970752.html?docfrom=rrela)

[博客-Tensorflow一些常用基本概念与函数（1）](http://blog.csdn.net/lenbow/article/details/52152766)

[wiki-NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG)

[Optimizing Top-N Collaborative Filtering via Dynamic Negative Item Sampling](http://wnzhang.net/papers/lambdarankcf-sigir.pdf)


tf Using GPUs 教程
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/docs_src/tutorials/using_gpu.md
中文-http://wiki.jikexueyuan.com/project/tensorflow-zh/how_tos/using_gpu.html

安装gpu版本
pip install tensorflow-gpu

sudo pip install --upgrade tensorflow-gpu -i  https://pypi.mirrors.ustc.edu.cn/simple/

ImportError: libcudnn.so.6: cannot open shared object file: No such file or directory
参考:
https://www.tensorflow.org/install/install_linux#InstallingNativePip
https://stackoverflow.com/questions/41991101/importerror-libcudnn-when-running-a-tensorflow-program
https://gist.github.com/mjdietzx/0ff77af5ae60622ce6ed8c4d9b419f45

sudo ldconfig -v |grep libcudnn.so.