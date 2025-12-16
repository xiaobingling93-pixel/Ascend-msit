# msguard

## 简介

msguard：安全校验库，提供了丰富的入参校验规则和安全工具函数，用于在模型训练、推理等场景下对文件路径、环境变量、目录遍历等操作进行安全检查，防止路径遍历、软链接攻击、动态库劫持等安全风险。

**基本概念**

-   Rule：规则集合，包含了一系列预定义的路径和安全约束（`Constraint`）条件，如 `input_dir_traverse`、`input_file_read` 等。
-   Constraint：约束条件，是Rule中的具体校验项，每个`Constraint`都有一个`is_satisfied_by`方法用于判断输入是否满足约束。

## 使用前准备

**环境准备**

执行以下命令安装 msguard：

```
pip3 install msguard
```

成功安装后，在Python脚本中通过 `import msguard` 即可使用。

## 快速入门

以下示例展示了如何使用装饰器方式快速为函数添加路径校验。

1.  在Python脚本中导入所需模块。

    ```python
    from msguard import validate_params, Rule
    ```

2.  定义您的目标函数，并使用`@validate_params`装饰器。

    ```python
    @validate_params({'input_dir': Rule.input_dir_traverse})
    def find_csv_path(input_dir):
        # 在 input_dir 下找到 csv 文件
        # xxx
    ```

3.  调用函数`find_csv_path`。在函数执行前，装饰器会自动校验入参`input_dir`是否符合`input_dir_traverse`规则。如果校验失败，将抛出异常。

以下示例展示如何在校验命令行输入的目录参数。

```python
import argparse
from msguard import validate_args, Rule

parser = argparse.ArgumentParser(description='Example with secure path argument.')
parser.add_argument('--input-path',
                    type=validate_args(Rule.input_dir_traverse),  # 校验输入路径
                    help='输入目录路径，将进行安全校验。')

args = parser.parse_args()
print(f"校验通过的输入路径: {args.input_path}")
```

运行脚本时，如果传入的路径不符合`Rule.input_dir_traverse`规则，`argparse`会报错。


## 功能说明

msguard 提供了一组 Python API，允许开发者在函数调用前、函数内部或包裹第三方函数时，对其参数进行安全约束校验。

## 注意事项

1.  使用装饰器`@validate_params`时，其参数字典的 key 必须与目标函数中待校验参数的变量名严格一致，否则校验不会生效。
2.  `Rule`中提供的`Constraint`在校验失败时默认会抛出异常。如果不想中断程序，可在函数内部使用`is_satisfied_by`方法进行判断。

## 使用指导

### 装饰器用法

此方法通过在函数定义前添加装饰器，在函数被调用时自动校验指定参数。

```python
from msguard import validate_params, Rule

@validate_params({'input_dir': Rule.input_dir_traverse})  # 校验 input_dir 参数
def find_csv_path(input_dir):
    # 函数逻辑
    # 仅在 input_dir 通过 Rule.input_dir_traverse 校验后，此处的代码才会执行
    pass
```

### 函数内部判断用法

此方法在函数内部手动调用校验，可更灵活地控制校验失败后的行为（如不抛异常，直接返回）。

```python
from msguard import Rule

def find_csv_path(input_dir):
    if not Rule.input_dir_traverse.is_satisfied_by(input_dir):  # 手动校验
        print("路径不合法")
        return  # 校验失败，直接返回
    # 校验成功，继续执行函数逻辑
    pass
```

### 任意函数包裹用法

此方法可以动态地对任何函数（包括第三方库函数）的调用进行参数校验。

```python
import os
import pandas as pd
from msguard import validate_params, Rule

def read_csv_from_dir(input_dir):
    csv_file = os.path.join(input_dir, "a.csv")
    # 包裹 pandas.read_csv 函数，校验其第一个参数（名称为'filepath_or_buffer'）
    safe_read_csv = validate_params({"filepath_or_buffer": Rule.input_file_read})(pd.read_csv)
    df = safe_read_csv(csv_file)  # 调用被包裹后的安全函数
    return df
```

**注意**：包裹函数时，`validate_params`参数字典的 key 必须与被包裹函数签名中的形参名完全一致（例如`pd.read_csv`的第一个参数名为`filepath_or_buffer`）。

## 输出说明

API 调用本身无固定输出格式。
-   校验失败时，相关函数（如装饰器或`validate_args`）会抛出异常（如`ValueError`, `argparse.ArgumentTypeError`）。
-   校验成功时，程序正常执行。

## 扩展功能

### 其他安全工具函数

msguard 除了参数校验，还提供了一些安全的工具函数，用于替代标准库中的高风险操作。

**open_s**

安全版的`open`函数，在打开文件前进行路径校验，返回文件句柄。

```python
from msguard import open_s

with open_s("/path/to/file.txt", "r") as f:
    content = f.read()
```

**walk_s**

安全版的`os.walk`函数，在遍历目录时自动进行最大文件数和遍历深度判断，防止拒绝服务攻击。

```python
from msguard import walk_s

for root, dirs, files in walk_s("/safe/directory", max_files=10000, max_depth=5):
    for name in files:
        print(os.path.join(root, name))
```

**update_env_s**

安全的环境变量更新函数，在向`PATH`、`LD_LIBRARY_PATH`等环境变量添加搜索路径时，自动判断路径是否为绝对路径，并避免添加重复的路径分隔符(`:`)，防护动态库劫持。

```python
from msguard import update_env_s

new_path = update_env_s("PATH", "/usr/local/safe/bin")
```

## 附录

### Rule 约束参考

`Rule` 对象包含多种预定义的约束条件 (`Constraint`)，以下列出部分常用项：

| 约束名 | 说明 |
| :--- | :--- |
| `input_dir_traverse` | 校验输入目录是否可用于安全遍历。 |
| `input_file_read` | 校验输入文件是否可安全读取。 |
| `output_dir` | 校验输出目录是否可安全创建和写入。 |
| `output_file` | 校验输出文件是否可安全创建和写入。 |

更多约束请参考 msguard 库的官方文档或源码。

### 示例代码

**综合使用示例代码**

```python
#!/usr/bin/env python3
import argparse
import os
from msguard import validate_params, validate_args, Rule, open_s, walk_s

# 1. 装饰器用法示例
@validate_params({'source': Rule.input_dir_traverse, 'dest': Rule.output_dir})
def copy_validated_data(source, dest):
    """一个需要校验源目录和目标目录的函数"""
    print(f"准备从 {source} 复制数据到 {dest}")
    # ... 实际的复制逻辑

# 2. argparse 集成示例
parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    type=validate_args(Rule.input_file_read),
                    help='配置文件路径')

# 3. 函数内部判断示例
def process_single_file(filepath):
    if not Rule.input_file_read.is_satisfied_by(filepath):
        return {"status": "error", "message": "Invalid file path"}
    # 使用安全 open
    with open_s(filepath, 'r') as f:
        data = f.read()
    return {"status": "success", "data": data}

# 4. 安全遍历示例
def list_safe_files(directory):
    file_list = []
    for root, dirs, files in walk_s(directory, max_depth=3):
        for f in files:
            file_list.append(os.path.join(root, f))
    return file_list

if __name__ == '__main__':
    # 测试装饰器函数
    copy_validated_data('/safe/input', '/safe/output')

    # 测试命令行解析 (假设运行: python script.py --config /etc/app/config.yaml)
    args = parser.parse_args()
    print(f"使用配置文件: {args.config}")

    # 测试文件处理
    result = process_single_file(args.config)
    print(result)

    # 测试安全遍历
    files = list_safe_files('/var/log')
    print(f"Found {len(files)} safe files.")
```

## FAQ

1.  **Q: 使用 `@validate_params` 装饰器后，为什么参数校验没有生效？**
   
    A: 请检查装饰器参数字典的 key 是否与您要校验的函数参数名**完全一致**（包括大小写）。例如，要校验参数 `data_path`，装饰器必须写为 `@validate_params({'data_path': Rule.input_dir_traverse})`。

2.  **Q: 如何查看 `Rule` 中所有可用的校验规则？**
   
    A: 您可以在 Python 交互环境中导入 msguard 后，使用 `dir(Rule)` 查看所有属性，或查阅 msguard 的官方文档。

3.  **Q: 校验失败时抛出的异常信息不够清晰，如何自定义错误信息？**
   
    A: 当前版本的校验规则使用预定义的错误信息。如需自定义，可在函数内部使用 `is_satisfied_by` 方法进行判断，然后在条件分支中抛出包含自定义信息的异常。
