[toc]
 
# homework2 solutions

## 补充材料学习: 矩阵微商的知识

### 微商的向量化
假设存在一个函数 $f:\R^n \rightarrow \R^m$, 将一个$n$维向量映射为$m$维向量。
$$f(x)=[f_1(x_1, \cdots, x_n),f_2(x_1, \cdots, x_n), \cdots, f_m(x_1, \cdots, x_n)]$$
那么它的Jacobian 矩阵是$m\times n$维度:

$$\frac {\partial f}{\partial x}= 
\begin{bmatrix}
\frac{\partial f_1}{\partial x_1} &  \cdots & \frac{\partial f_1}{\partial x_n}\\ 
 \vdots & \ddots & \vdots \\
 \frac{\partial f_n}{\partial x_1}& \cdots & \frac{\partial f_m}{\partial x_n}
\end{bmatrix}
$$



