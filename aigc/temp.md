$$
p(x_1 | y) = p(x|y) - p(x_2 | y)

\\

\nabla log(p(x_1 | y)) = \nabla log(p(x|y)) - \nabla log(p(x_2 | y))
\\=
\nabla log(\frac{p(x|y)}{p(x_2 | y)})
$$

其中，
$$
p(x|y) = \frac{p(x)p(y|x)}{p(y)}
\\
p(x_2|y) = \frac{p(x_2)p(y|x_2)}{p(y)}
$$
所以
$$
\nabla log(p(x_1 | y)) = \nabla log(\frac{p(x|y)}{p(x_2 | y)})

\\=

\nabla log(\frac{p(x)p(y|x)}{p(x_2)p(y|x_2)})

\\=

\nabla log(p(x)p(y|x)) - \nabla log(p(x_2)p(y|x_2))

\\=

\nabla log(p(x)) + \nabla log(p(y|x)) - \nabla log(p(x_2)) - \nabla log(p(y|x_2))
$$
用 $\gamma, \beta$ 表示有条件和负条件的权重：
$$
\nabla log(p(x_1 | y)) = 
\nabla log(p(x)) + \gamma \nabla log(p(y|x)) - \nabla log(p(x_2)) - \beta \nabla log(p(y|x_2))
$$
替换：
$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}

\\

p(y|x_2) = \frac{p(x_2|y)p(y)}{p(x_2)}
$$
即：
$$
\nabla log(p(x_1 | y)) = 
\nabla log(p(x)) + \gamma \nabla log(p(y|x)) - \nabla log(p(x_2)) - \beta \nabla log(p(y|x_2))

\\=

\nabla log(p(x)) + \gamma \nabla log(\frac{p(x|y)p(y)}{p(x)}) - \nabla log(p(x_2)) - \beta \nabla log(\frac{p(x_2|y)p(y)}{p(x_2)})

\\=

\nabla log(p(x)) + \gamma \nabla log(p(x|y)) + \gamma \nabla log(p(y)) - \gamma \nabla log(p(x)) - \nabla log(p(x_2)) - \beta \nabla log(p(x_2|y)) - \beta \nabla log(p(y)) + \beta \nabla log(p(x_2))

\\=

(1 - \gamma) \nabla log(p(x))
+
(\gamma - \beta) \nabla log(p(y))
+
\gamma \nabla log(p(x|y))
+ 
(\beta - 1) \nabla log(p(x_2))
- 
\beta \nabla log(p(x_2|y))
$$
因为 $p(x_1 | y) = p(x|y) - p(x_2 | y)$ 且 $log(p(y))$ 的对数似然是常数，因此：
$$
\nabla log(p(x_1 | y)) 

\\=

(1 - \gamma) \nabla log(p(x))
+
\gamma \nabla log(p(x|y))
+ 
(\beta - 1) \nabla log(p(x_2))
- 
\beta (\nabla p(x|y) - \nabla p(x_1 | y))


\\=

(1 - \gamma) \nabla log(p(x))
+
\gamma \nabla log(p(x|y))
+ 
(\beta - 1) \nabla log(p(x_2))
- 
\beta \nabla log(p(x|y)) + \beta \nabla log(p(x_1 | y)))
$$
移项 && 合并：
$$
(1 - \beta)\nabla log(p(x_1 | y))

=

(1 - \gamma) \nabla log(p(x))
+
(\gamma - \beta) \nabla log(p(x|y))
+ 
(\beta - 1) \nabla log(p(x_2))
$$
即：
$$
\nabla log(p(x_1 | y))

=
\frac{1}{(1 - \beta)}
[
(1 - \gamma) \nabla log(p(x))
+
(\gamma - \beta) \nabla log(p(x|y))
+ 
(\beta - 1) \nabla log(p(x_2))
]
$$

