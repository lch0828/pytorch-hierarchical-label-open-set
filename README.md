# Hierarchical Multi-Label Open-Set Classification

This repo provides the code inspired by the paper [Hierarchical Text Classification with Reinforced Label Assignment](https://proceedings.mlr.press/v80/wehrmann18a/wehrmann18a.pdf) [1] and [Deep Open Set Recognition Using Dynamic Intra-class Splitting](https://link.springer.com/article/10.1007/s42979-020-0086-9) [2].

* See ```HMLOSC.pdf``` for details.

## Run

```
python3 main.py 
[--epoch] 
[--batch] 
[--train_div]
[--test_div] 
[--in_dim]
[--global_weight_dim]
[--transition_weight_dim]
[--total_classes_at_level]
[--total_levels] [--B]
[--encoder {vit_b_16,vit_b_32,cnn}] 
[--test_model]
[--deep_residual]
```

## Reference
[1] Wehrmann, J., Cerri, R. & Barros, R. Hierarchical Multi-Label Classification Networks. Proceedings Of The 35th International Conference On Machine Learning. 80 pp. 5075-5084 (2018,7,10), https://proceedings.mlr.press/v80/wehrmann18a.html

[2] P. Schlachter, Y. Liao, and B. Yang, “Deep Open Set Recognition Using Dynamic Intra-class Splitting,” SN COMPUT. SCI., vol. 1, p. 77, 2020.