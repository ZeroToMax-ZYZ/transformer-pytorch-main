# 训练时间记录

单卡3080 laptop 16G
2 小时 37 分 一个epoch

4卡3080 桌面端 10G
49分 一个epoch

# 在mutil30K数据集上训练

```cmd
python train_multi30k_base.py
```

# val in mutil30k
## 标准推理，最后五次取平均
```cmd
python evaluate_multi30k_bleu.py --experiment-dir experiments\transformer_multi30k_en_de_base_20260331_191821 --checkpoint-paths 
```

- 结果为：BLEU = 38.24

## 只使用指定的权重进行推理
```cmd
python evaluate_multi30k_bleu.py --experiment-dir experiments\transformer_multi30k_en_de_base_20260331_191821 --checkpoint-paths experiments\transformer_multi30k_en_de_base_20260331_191821\checkpoints\best.pth
```
- 结果为：BLEU = 37.64