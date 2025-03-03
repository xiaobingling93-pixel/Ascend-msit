## DitCacheConfig

### 功能说明
DiT缓存配置类，保存优化后的缓存参数。

### 函数原型
```python
class DitCacheConfig(use_cache=True, cache_step_start=None, cache_step_interval=None, 
                    cache_block_start=None, cache_num_blocks=None)
```

### 参数说明
| 参数名 | 输入/返回值 | 含义 | 使用限制 |
| ------ | ------ | ------ | ------ |
| use_cache | 输入 | 是否启用缓存 | 可选。<br>数据类型：bool。<br>默认值：True |
| cache_step_start | 输入 | 缓存起始时间步 | 可选。<br>数据类型：int。<br>默认值：None |
| cache_step_interval | 输入 | 缓存时间步间隔 | 可选。<br>数据类型：int。<br>默认值：None |
| cache_block_start | 输入 | 缓存起始块索引 | 可选。<br>数据类型：int。<br>默认值：None |
| cache_num_blocks | 输入 | 缓存块数量 | 可选。<br>数据类型：int。<br>默认值：None |

### 主要属性

#### to_dict()
将配置转换为字典格式

**返回值：**
- 数据类型：dict
- 含义：包含所有配置参数的字典

### 调用示例
```python
from msmodelslim.pytorch.multi_modal.dit_cache import DitCacheConfig

# 创建缓存配置
config = DitCacheConfig(
    cache_step_start=10,
    cache_step_interval=5,
    cache_block_start=2,
    cache_num_blocks=4
)

# 转换为字典
config_dict = config.to_dict()
```

### 使用说明

1. cache_step_start表示从哪个时间步开始应用缓存
2. cache_step_interval表示缓存计算的间隔步数
3. cache_block_start和cache_num_blocks定义缓存应用的块范围

### 注意事项

1. 缓存配置应与模型结构匹配
2. 缓存范围过大会导致性能下降
3. 建议根据校准集来搜索调整配置参数
