# 移动计算摄影

Task：

1. PPT总结：里面涉及到的算法，解决的问题，包括一些扩展阅读，有意思的技术来。
2. 知识形成系统：包括收集一些论文，数据集，代码，连接。
3. 开源的代码，可以自己拍些图片跑跑实验。

### 1. 模式识别理解

**基本方法**：数据获取 -> 模式分割 -> 模式分类 -> 后处理

- 模式分类是模式识别的核心技术（分割、检测也常采用模式分类方法）
- 基本方法：模版匹配
  - 计算每一类模版的匹配距离
  - 模式样本表示：图像，特征矢量，结构（graph、tree、sequence）
  - 模型：保留部分训练样本，或从样本合成模版
  - 拓展：
    - 不变性特征，不变性距离度量
    - 贝叶斯分类：最大后验概率决策：$p(\omega_i|x)=\frac{p(x|\omega_i)p(\omega_i)}{p(x)}=\frac{p(x|\omega_i)p(\omega_i)}{\sum_{j=1}^cp(x|\omega_j)p(\omega_j)}$
    - 判别函数：相当于广义相似度，如神经网络：$f_i(x)=f(x,\theta_0,\theta_i)$

### 2. 模式识别方法分类

- 按模式/模型表示方法分类
  - Statistical：特征矢量
    - Parametric (Gaussian)
    - Non-parametric (Parzen window, KNN)
    - Semi-parametric (GM)
    - Neural network
    - Logistic regression
    - Decision tree
    - Kernel (SVM)
    - Ensemble (Boosting)
  - Structural:句法、结构
    - Syntactic parsing
    - String matching, tree
    - Graph matching
    - Hidden Markov model (HMM)
    - Markov random field (MRF)
    - Structured prediction 
    - Graph neural network (GNN)
- 生成/判别模型
  - 生成模型Generative Model：表示各个类别内部结构或特征分布$p(x|c)$
  - 判别模型Discriminative Model：表示不同类别之间的区别，一般为判别函数(Discriminant function)、边界函数或后验概率$P(c|x)$
  - 生成学习:得到每个类别的结构描述或分布函数，不同类别分别学习
  - 判别学习:得到判别函数或边界函数的参数，所有类别样本同时学习
  - Generative Models
    - Template (prototype)-based classifier
    - Parametric probability density (Gaussian, GM)： $p(x|C)=f(x,\theta)$
    - Bayesian network (directed graph)：概率密度函数树近似 $p(x)=\prod_i^n p(x_i|pa_i)$
    - Hidden Markov model (HMM) 特征矢量序列密度函数近似: $p(O|\lambda)=\sum_QP(O|Q,\lambda)P(Q|\lambda)$
    - Undirected graphs
      - Attributed relational graph (ARG)
      - Markov random field (MRF)
  - Discriminative Models
    - Artificial neural networks (ANN): discriminant function regardless of probability distribution 神经网络输出近似后验概率 $y_i(x)=P(\omega_i|x)$
    - Support vector machine (SVM): hyperplane classifier (2-class)
      Decision boundary $w·x+b><0$
    - Boosting: weighted combination of multi-discriminators (2-class)
      Boosting 判别函数是多个分类器加权和 $F(x)=\sum_t^T\alpha_th_t(x)$
    - Conditional random field (CRF): Labeling by minimizing energy function, without assumption of conditional independence.

### 3. 模式识别现状总结

- 深度神经网络在几乎所有任务中超越了传统方法的性能
  - 分类：end-to-end feature extraction + classification
  - 检测：foreground- background classification, boundary regrassion
  - 分割：pixel classification
  - 描述：end-to-end image-to-text mapping（多模态）





### Introduction

- 问题：现有长曝光摄影解决方案无法帮助用户获得移动和静态场景元素分别模糊/清晰的效果

- 传统长曝光摄影：
  - 使用长时间曝光，在高锐度背景上产生前景模糊的效果（瀑布、光迹等），但即使轻微的晃动也会导致背景锐度下降、此外必须在镜头上添加中性密度以避免传感器过度曝光
  - 平移摄影：背景相对于移动主体会变得模糊，而移动的主体会变得清晰，实现方法是用相机跟踪移动的主体，同时保持快门打开，适度增加曝光时间，并略为缩小光圈，以避免图像曝光过度。必须尽可能精确追踪拍摄对象的运动轨迹，以避免拍摄对象锐度的意外下降，同时需要在正确的时刻按下快门按钮。
  - 问题：需要高超的技能和实践操作，手动选择相机快门速度，同时考虑到场景移动的速度，以达到所需的效果。本文的主要贡献在于一个可计算的长曝光移动摄影系统，用户选择希望产生的前景或背景模糊效果后，只需轻按快门即可生成长曝光的1200万像素照片，同时补偿相机和主体的运动，从而保留所需的背景和主体。

- 系统组成：
  - 捕捉时间表和帧选择，产生与场景或摄像机速度无关的归一化模糊轨迹长度
  - 主体检测结合了注视显著性、人物和宠物面部区域预测以及对其运动的跟踪
  - 对输入图像进行对齐、以消除相机震动，在前景元素移动的情况下稳定背景，或消除主体运动，同事产生令人愉悦的背景运动模糊轨迹
  - 密集运动预测和模糊合成。跨越多个高分辨率输入帧，生成平滑的曲线运动模糊轨迹，并保留高光

### Implementation

- Burst capture 

  - https://research.google/pubs/pub45586/

  - https://dl.acm.org/doi/pdf/10.1145/2980179.2980254

  - 连拍捕捉通常是指快速连续捕捉多个帧的过程，通常是为了以某种方式提高图像质量或性能。这个想法是利用可以从图像序列中提取的附加数据，而不是依赖于单个图像。Burst Capture 广泛应用于计算机视觉和图像处理的各个方面，用于不同的目的：

    1. **高动态范围 (HDR)**：通过捕获具有不同曝光的多个图像，可以生成比任何单次曝光图像具有更高亮度范围的 HDR 图像。
    2. **降噪**：快速连续拍摄多张照片可以使算法平均掉传感器噪声，从而提高最终图像质量。
    3. **超分辨率**：利用帧之间轻微的自然变化，算法可以生成比任何单个捕获的帧更高分辨率的图像。
    4. **运动分析**：在理解运动至关重要的应用中，突发捕捉可以帮助跨帧跟踪特征。
    5. **深度映射**：连拍捕捉可以通过使用帧之间视角的细微变化来帮助生成更准确的深度图。
    6. **焦点堆叠**：在微距摄影或显微镜中，可以组合具有不同焦距的多个图像以产生具有更大景深的最终图像。
    7. **视频分析**：以非常高的帧速率进行连拍捕捉
    8. 对于运动分析、对象跟踪和其他时间敏感的应用程序非常有用。

    在计算机视觉中，突发捕捉通常由以下内容来补充：

    1. **机器学习算法**：可以训练深度学习模型从突发捕获的数据中提取更细致的信息。
    2. **时间分析**：由于帧在时间上相关，因此时间相干性和其他基于时间的度量可用于改进分析。
    3. **光流算法**：这些算法可以提供有关帧之间对象移动的附加数据，有助于完成对象跟踪或稳定等任务。
    4. **数据融合技术**：突发捕获的数据可以与激光雷达、雷达或热图像等其他数据类型融合，以进行更全面的分析。
    5. **校准技术**：由于突发捕捉可能涉及曝光、焦点或其他设置的变化，因此为了准确解释数据，校准可能是必要的。
    6. **计算摄影方法**：执行图像对齐、拼接和其他复杂转换以充分利用突发捕获数据的算法。

    利用突发捕获的一些流行模型和方法包括：

    - 遮蔽 Hasinoff 等人的连拍摄影。(2016) - 使用连拍从图像中删除对象。
    - DAIN 和 BMBC 等视频加速方法 - 从突发中合成慢动作视频。
    - Mildenhall 等人的 EBSR 和 MISR。(2018) - 通过连拍实现超分辨率。
    - 循环方法，如 RDN - 使用突发进行去模糊。

- Automatic subject detection

  - 目标: 在保持主要拍摄对象清晰的同时实现背景模糊效果。

  - 预测主体代理方法：attention saliency的proxy task

    - attention saliency：旨在通过计算建模和预测图像/视频中视觉上突出并捕获人类目光的区域和对象，从而提供对感知和注意力的洞察。
    - 根据颜色、对比度、边缘、对象、面部、文本等特征，识别图像中引起注意的视觉突出区域。显着区域是的眼睛在观看场景时首先自然关注和处理的区域。模型尝试使用影响人类注意力和注视模式的特征通过计算来预测显着性。显着性的应用包括图像裁剪、重定向、压缩、对象检测、视频摘要等。卷积神经网络等深度学习方法最近在注意力显着性预测方面取得了很高的准确性。

  - 预测模型：从一个在salicon数据集[Jiang et al. 2015]上训练的更大模型中提炼出来的一个mobile-friendly的3-level U-Net with skip connections [Bazarevsky. 2019]模型
    Encoder：15 BlazeBlock with 10 base channels
    Decoder：separable convolutions可分离卷积 and bi-linear 采样层

    为了关注signal中的saliency峰值，将预测值re-normalize，并将低于阈值0.43的清零。

  - saliency signal的峰值往往在subject center，因此用face signal进行补充，有助于保持subject faces的sharp，这对于subjects with complex articulated motion尤其重要。通过首先预测人类、猫和狗的face region来计算face signal，然后使用smootherstep falloff对生成的区域进行feathering[Ebert et al. 2003],，最后使用类似于[Wadhwa et al. 2018]的whole-subject segmentation进行masking。

  - Saliency 和face signals进行如下组合生成subject weight map，每个像素weight $w=s(1+f)$, where $s\in[0,1],f\in[0,1]$分别是saliency signal value和face signal value，然后re-normalization到$[0,1]$区间。在合成步骤也会使用facial signal来保持face sharpness。

  - 要看总结的论文：

    [Bazarevsky et al. 2019]：用于预测对象的模型

    Valentin Bazarevsky, Andrey Vakunov, Andrei Tkachenka, George Sung, Changcheng Li, and Matthias Grundmann. "BlazeFace: Sub-millisecond neural face detection on mobile GPUs." arXiv preprint arXiv:1907.05047 (2019).

    [Jiang et al. 2015]：SALICON数据集

    Ming Jiang, Shengsheng Huang, Juanyong Duan, and Qi Zhao. "SALICON: Saliency in context." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

    [Ebert et al. 2003]：区域羽化方法

    David S. Ebert, F. Kenton Musgrave, Darwyn Peachey, Ken Perlin, and Steven Worley. Texturing & modeling: a procedural approach. Morgan Kaufmann, 2003.

    [Wadhwa et al. 2018]：

    Neal Wadhwa, Rahul Garg, David E. Jacobs, Bryan E. Feldman, Nori Kanazawa, Robert Carroll, Yair Movshovitz-Attias, Jonathan T. Barron, Yael Pritch, and Marc Levoy. "Synthetic depth-of-field with a single-camera mobile phone." ACM Transactions on Graphics 37, 4 (2018): 1-13.

  - 要找的方法：

    - 可以代替的用于背景模糊效果模型，例如：

      - 编码器-解码器网络 - 编码器网络将输入图像压缩为低维表示，然后是解码器网络来重建图像。可以操纵编码器输出来分离前景和背景。
      - 生成对抗网络 (GAN) - GAN 可以被训练来根据输入图像生成逼真的模糊背景。清晰的前景可以合成在顶部。
      - 深度预测网络 - 从 RGB 图像预测深度图的网络可以根据深度不连续性来分离前景和背景。然后可以将模糊应用于较低深度的区域。
      - 视频帧插值网络 - 基于运动向量在帧之间进行插值可能会提供中间流以实现模糊效果。
      - 光流网络 - 估计的场景流可以类似地提供像素运动来合成背景模糊。
      - 对象剪切网络 - 针对检测和分割对象而优化的网络可以隔离前景以进行合成。
      - 注意力模型 - 图像上的软注意力图可以隔离显着的前景区域。
      - 元学习模型 - 从转移到新场景的示例中学习通用模糊滤镜。

      核心思想是使用神经网络来隔离前景/背景、估计像素运动并生成模糊效果。许多网络架构可以通过不同的方式和自己的权衡来实现这一点

    - 可代替U-Net结构进行attention saliency预测的模型：

      - ResNet - 具有跳跃连接的残差网络可以很好地用于显着性预测，同时又轻量级。残差块允许训练更深的模型。
      - MobileNet - 使用深度可分离卷积构建适合移动应用的非常轻量级的模型。可以配置用于密集预测任务，例如显着性。
      - EfficientNet - 使用复合系数扩展基线模型的宽度、深度和分辨率。可以平衡准确性和效率。
      - 深度可分离卷积网络 - 用深度可分离卷积替换常规卷积可以减少移动使用的计算量。
      - ShuffleNet - 使用逐点组卷积和通道洗牌来降低计算成本，同时保持准确性。非常适合移动设备。
      - SqueezeNet - 依靠 1x1 卷积来压缩和扩展网络内的通道，从而减少参数。
      - 神经架构搜索网络 - 自动搜索可以设计适合移动平台的非常高效且轻量级的网络。

      核心思想是使用高效的构建块，最大限度地减少计算和参数，同时保持任务所需的表示能力。最初的 U-Net 实现了这种平衡，但许多其他移动优先架构可以替代并实现类似的结果。

- 运动跟踪：

  - 使用基于 [Grundmann et al. 2011]的feature tracking library提取运动轨迹，用于后续的image alignment。运动轨迹统计数据还用于select frames，以确定在场景中是否捕捉到了足够的motion。

    Background blur中的subject tracking需要subject上的high concentration of tracks，以实现stable、高质量的alignment。为了优化latency，在每个grid的cell尺寸为 5× 5 pixels each的image grid上使用rejection sampling，以生成density与subject weight map成正比的feature tracks。只尝试extract cells中的feature tracks，其中采样的uniform random variable $v \in [0, 1]$ 小于corresponding grid位置的平均track-weight。

  - 要看的论文：[Grundmann et al. 2011]
  - 要想的方法：rejection sampling

- 图像对齐

  - 根据feature tracks的对应关系，首先估算global transforms将所有frames对齐到reference frame。这样可以消除摄像机的overall motion，包括用于追踪subject的handshake和sweeping motion。其余的图像alignment阶段则针对所需的motion blur效果：前景或背景模糊。为了便于说明，选取了一个示例场景，如图3所示，一辆出租车驶过繁忙的城市十字路口。

  - 一：前景模糊 - [Zaragoza et al. 2013]
  - 二：背景模糊 - [Porikli 2004]
  - 上面两个方法内容需要展开了解看看

- 选择帧

  - 系统使用frame selection来计算motion-blur轨迹的估计长度，以决定incremental frame处理outer-loop什么时候需要停止。
    首先，使用alignment solver计算出的transformations将运动特征轨迹变换到base frame的reference space，使其与输出图像中相应的跟踪特征运动模糊轨迹在空间上对齐。
    然后，可以计算出每个alignment track的长度，使用轨迹长度分布的高百分位数来估算整体模糊轨迹长度。最后将该估计值与恒定目标设置进行比较，以确定是否满足帧选择标准。

    以image diagonal的百分比来衡量track length，这一指标对image resolution或aspect-ratio基本不敏感。在前景模糊的情况下，使用第98 百分位数达到 30% 的目标值，为移动速度最快的object生成相对较long和smooth的模糊轨迹。在背景模糊的情况下，使用第 80 百分位数目标值为 2.8%，为更大面积的背景生成较短的模糊轨迹，目的是保持主体的清晰度，避免丢失周围场景的context。这些设置都是在大量输入bursts中反复试验derived empirically。

  - 可能的ideas，可实现估计运动模糊长度并使用它来确定何时停止处理其他帧的类似效果：

    - 光流 - 计算帧之间的密集光流。阈值流矢量长度来估计具有显着运动的区域。使用流动长度的百分位数作为模糊估计。
    - 特征跟踪 - 正如原始方法中所做的那样，跨帧跟踪特征点。使用参考帧空间中的轨道长度作为模糊估计。
    - pose估计 - 对于人类受试者，估计跨帧的 3D 姿势。查看关节速度和位移来预测运动模糊。
    - 基于学习的预测 - 训练神经网络直接预测输入帧的预期运动模糊。使用其输出作为模糊估计。
    - 生成模型 - 训练生成模型（如 GAN）来合成运动模糊。使用匹配真实模糊所需的生成器迭代作为估计。
    - 模拟 - 通过累积跨帧的变换来物理模拟模糊。当模拟模糊与目标匹配时停止。
    - 混合方法 - 结合光流、特征跟踪和姿态估计等技术来获得稳健的运动模糊估计。
    - Optical Flow - Compute dense optical flow between frames. Threshold flow vector lengths to estimate regions with significant motion. Use percentile of flow lengths as blur estimate.
    - Feature Tracking - As done in the original method, track feature points across frames. Use track lengths in reference frame space as blur estimate.
    - Pose Estimation - For human subjects, estimate 3D poses across frames. Look at joint velocity and displacement to predict motion blur.
    - Learning-based Prediction - Train a neural network to directly predict expected motion blur from input frames. Use its output as blur estimate.
    - Generative Modeling - Train a generative model like GAN to synthesize motion blur. Use generator iterations needed to match real blur as estimate.
    - Simulation - Physically simulate blur by accumulating transformations across frames. Stop when simulated blur matches target.
    - Hybrid Approaches - Combine techniques like optical flow, feature tracking and pose estimation to get robust motion blur estimates.

    核心思想是使用经典视觉技术或基于学习的预测来分析帧到帧的运动。运动量充当预期模糊的代理。这允许智能地选择要处理的帧数以实现所需的模糊效果。

- 运动预测（需要加网络结构的思考）

  - 对输入的low-resolution图像进行alignment后，将其输入motion-blur kernel-prediction神经网络，每次输入一对input frame pair，每次iteration预测a pair of line和weight kernel maps。如第 4.7 节所述，低分辨率核映射用于合成半分辨率的运动模糊片段，跨越相应的输入帧。

    运动预测模型负责预测沿线段的两个空间积分的参数，这两个空间积分近似定义了在相应时间间隔内通过每个运动模糊输出像素看到的颜色平均值的时间积分。使用的模型基于[Brooks and Barron 2019]，并做了进一步修改，以改善性能和圈像质量之问的权衡，使我们能够在移动设备上获得合理的内存和计算预算。

    他们的数学公式预测了给定图像对$k$中每个输入帧$i$的权重图$y_i$，共有$N=17 $个通道，用于杈衡预测线段上的每个相应纹理样本。本文简化了这一模型，只预测一个通道，用于权衡每个输入帧的积分结果。从图2中可以看到灰度图的示例，该示例显示，网络预测的输入图像各处的权重大致相同，只有在闭塞区域，权重偏向于两个输入中的一个。这种简化大大降低了系统的复杂性和内存使用量，并允许将更多的网络容量用于预测线段。

    此外，我们还消除了由于预测线段的端点误差[Zhang et al. 2016] 而产生的伪影，这种误差会导致预测线段在跨时间问隔的末端不完美地相遇，从而在模糊轨迹的中间产生非常明显的伪影，如图6所示。为避免这一问题，采用归一化的递减线性斜坡函数进一步缩放输入图像纹理样本，使样本更靠近输出像素，并逐渐降低每个预测线段上较远样本的权重。输入帧对$k$的输出像素$(x,y)$的强度为$IK(x,y)=\sum_{i\in\{k,k+1\}}y_i(x,y)/N-1$...

    我们还对网络架构进行了如下修改。首先，用参数化 ReLU [He et al. 2015] 替换了整个网络中的泄漏 ReLU 卷积激活，其中斜率系数是通过学习获得的。接下来，为了避免常见的棋盘式假象 [Odena et al. 2016]，替换了 2 倍重采样层，使用平均池化进行下采样，并在 2倍卷积后进行双线性 上采样。这就产生了第5节中分析的"Ours-large"模型。此外，为了改善浮动操作数、参数数和感受野之间的平衡，我们进一步将 U-Net 模型拓扑结构缩减到只有3层，其中每层使用 1x1 卷积，然后是一个具有4个3x3 卷积层的ResNet 块 [He et al. 2016]。这使得标为"Ours "的模型的学习参数大大减少。

    如图6所示，斜坡函数wn 为学习的单权重模型带来了显著的好处，因为它使预测的线段在每幅输入图像中都具有空间跨度，相当于整合了整个时间间隔。当的模型在训练时删去了这一项，即模型"Ours-abl."时，网络预测的线段每边大约跨越一半的时间间隔，从而导致模糊轨迹中间出现明显的不连续性。更多实例可参见第5节中的模型对比分析。

- 渲染（不太了解）

  - 运动预测网络输出的线条和权重内核映射由渲染器使用，该渲染器可合成运动模糊图像。渲染器由OpenCL 内核实现，可在移动设备的GPU 上高效运行，充分利用硬件纹理单元，同时在半分辨率输入图像中进行自适应纹理采样查找（纹理采样 N 的数量根据预测线向量的长度按比例调整）。运动预测和渲染迭代可一次执行一对输入帧，产生片断线性运动模糊轨迹。通过使用双线性纹理查找，将核映射从低分辨率向上采样到半分辨率。

- 合成（不太了解）

  - Section 4.7的合成blurred image是以half resolution计算的，以满足设备的memory and latency constraints. Accordingly, even perfectly aligned, zero-motion regions of the blurred image will lose detail due to the upsampling of the result computed at half resolution. 

  - 为了保留details,我们将 blurred image与maximally sharp regular exposure图像合成， 以保证image的sharp清晰度。

  - 两类内容需要保护：

    1. stationary scene content 静止场景内容
    2. semantically important subjects with little movement, as shown in Figure 9.语义上很重要但移动很少的主体

  - 对于类别1，我们会生成一个掩码，掩码中的像素在整组帧对中的运动非常小。。。

    ​	![image-20230830151601081](/Users/qiaochufeng/Library/Application Support/typora-user-images/image-20230830151601081.png)

  - 对于类别2，它打破了光学运动模糊的物理行为，更注重美感。例如，如果一个场景中有两个以不同轨迹运动的主体，就不可能同时对准这两个主体。即使是单个主体，也可能因为主体内部的运动（如面部表情变化等）而无法对齐。主体面部模糊的图像就是（糟糕的）模糊图像。我们的解決方案是重新使用4.2中描述的语义人脸信号，并对其进行修改，使其只包含对齐参考帧中平均特征移动较小的人脸。

  - 最后，我们用一个简单的最大值运算符将流动遮罩和剪切脸部遮罩结合起来。图9显示了两种遮罩类型对最终合成效果的累积影响。



















