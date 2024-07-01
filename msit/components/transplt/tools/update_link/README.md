# msit transplt API映射表格自动更新脚本使用指南

## 简介
该工具用来自动更新API映射表格中的AscendAPILink属性。

## 使用方式
### 预备工作
该脚本依赖chrome浏览器以及chromedriver，其中chromedriver的下载地址
在[这里](http://chromedriver.storage.googleapis.com/index.html)。
1. 下载完的压缩包需要放在`msit/msit/components/transplt/tools/update_link/driver`目录并解压。
需要注意的是chrome浏览器与chromedriver的版本要匹配才能正常运行。

2. 需要使用pip安装`requirements.txt`中的python组件
```commandline
pip3 install -r requirements.txt
```

### 运行命令
需要在命令行中进入到`update_link`目录用以下命令运行：
```commandline
python3 update_link.py <original_excel_dir> config.json [MODE]
```
其中EXCEL_DIR是映射表格excel文件所在目录。
MODE为可选项，可选值是1、2、3，默认值是3。
1) 当MODE为1时，该脚本会从昇腾社区爬取AscendCL 与MindX SDK的API，
并存储到当前文件夹的`ascend_apis.xlsx`表格中去，下载比较久，需要几个小时。
2) 当MODE为2时，该脚本会根据下载好的`ascend_apis.xlsx`去更新API映射表格中的AscendAPILink属性。
3) 当MODE为3时，会执行上述2个步骤

